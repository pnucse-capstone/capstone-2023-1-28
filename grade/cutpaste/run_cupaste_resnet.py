import os
import shutil

import cv2
import numpy as np
import matplotlib.pyplot as plt
import easydict
import pandas as pd
from pathlib import WindowsPath

from typing import Iterable, List, Optional
from PIL import Image

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.feature_extraction import get_graph_node_names
from torch.utils.data import Dataset

import captum.attr

from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

from cutpaste.dataset import MVTecAT
from cutpaste.cutpaste import CutPaste, cut_paste_collate_fn
from cutpaste.model import ProjectionNet
from pathlib import Path
from collections import defaultdict
from cutpaste.density import GaussianDensitySklearn, GaussianDensityTorch
from cutpaste.utils import str2bool

class MVTecAT(Dataset):

    def __init__(self, root_dir, size, transform=None, mode="train"):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode
        self.size = size
        self.image_names = list(self.root_dir.glob("*.png"))


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        filename = self.image_names[idx]
        label = filename.parts[-2]
        img = Image.open(filename)
        img = img.resize((self.size,self.size)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label != "good"

class GradCam(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, name_layer: str) -> None:
        super().__init__()
        self.model = model
        self.model.eval()

        names_mid = name_layer.split(".")
        layer = model
        for name in names_mid:
            layer = layer.__getattr__(name)
        self.layer = layer

        self.cam = captum.attr.LayerGradCam(self.model, self.layer)
        return


def combine_img(file_dir, file_groups, final_dir):
    # if not os.path.exists(result_dir):
    #     os.makedirs(result_dir)
    for group_name, image_filenames in file_groups.items():
        # 각 그룹 내의 이미지를 가로 방향으로 이어붙여 빈 이미지를 생성합니다.
        total_width = 0
        max_height = 0

        for filename in image_filenames:
            image = Image.open(os.path.join(file_dir,filename))
            total_width += image.width
            max_height = max(max_height, image.height)

        result_image = Image.new("RGB", (total_width, max_height))

        # 이미지를 오른쪽으로 이어붙입니다.
        x_offset = 0
        for filename in image_filenames:
            image = Image.open(os.path.join(file_dir,filename))
            result_image.paste(image, (x_offset, 0))
            x_offset += image.width
            os.remove(os.path.join(file_dir,filename))

        # (리사이즈 왜곡이 심하다 싶으면 수정하기)
        # result_image = result_image.resize((512,512))
        result_image.save(os.path.join(final_dir, f"{group_name}.png"))

def run(size_url):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #모델 경로
    model_dir = "./cutpaste/model/" + size_url
    # 얘네로 평균을 내서 비교후 거리를 구함
    distance_dir = "./static/cutpaste/aver_distance/" + size_url + "/good"
    input_dir = "./static/cutpaste/datasets/" + size_url
    result_dir = './static/cutpaste/results/' + size_url
    final_input_dir = './static/final/inputs/' + size_url
    final_result_dir = './static/final/results/' + size_url



    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(final_input_dir):
        os.makedirs(final_input_dir)
    if not os.path.exists(final_result_dir):
        os.makedirs(final_result_dir)

    input_lst = []
    for filename in os.listdir(input_dir):
        input_lst.append(filename)
    if len(input_lst) == 0:
        print("비어있음")
        return      #안되면 None 추가

    model = os.path.join(model_dir,os.listdir(model_dir)[0] if os.path.exists(model_dir) else None)
    if model_dir is None:
        print("가중치를 넣어주세요.")
        return
    #모델 로드
    head_layer = 1
    head_layers = [512]*head_layer+[128]
    weights = torch.load(model, map_location=device)
    classes = weights["out.weight"].shape[0]
    model = ProjectionNet(pretrained=False, head_layers=head_layers, num_classes=classes)
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    #값 설정



    size = 512
    batch_size = 64

    test_transform = transforms.Compose([])
    test_transform.transforms.append(transforms.Resize((size,size)))
    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225]))

    dataloader_train = DataLoader(MVTecAT(distance_dir, size, transform = test_transform, mode="test"), batch_size, shuffle=False, num_workers=0)
    dataloader_test = DataLoader(MVTecAT(input_dir, size, transform = test_transform, mode="test"), batch_size, shuffle=False, num_workers=0)

    #트레인 임베딩 제작
    trains = []
    with torch.no_grad():
        for x, label in dataloader_train:
            train, logit = model(x.to(device))

            trains.append(train.cpu())

    trains = torch.cat(trains)
    trains = torch.nn.functional.normalize(trains, p=2, dim=1)

    #테스트 임베딩 제작
    embeds = []
    with torch.no_grad():
        for x, label in dataloader_test:
            embed, logit = model(x.to(device))

            embeds.append(embed.cpu())

    embeds = torch.cat(embeds)
    embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)


    density = GaussianDensityTorch()
    density.fit(trains)

    distances = density.predict(embeds)

    # 딱 맞게 하면 안되고 여유를 조금 주어야 한다
    # 소형관 정상 이미지들의 평균 거리 = 20.5350, 10.5 또는 4.5
    # 중형관 정상 이미지들의 평균 거리 = 20.9062, 6.5
    # 대형관 정상 이미지들의 평균 거리 = 22.5396, 9.5

    if size_url == 'small':
        Good_value = 20.5350
        dis_value = 4.5
    elif size_url == 'middle':
        Good_value = 20.9062
        dis_value = 6.5
    else:
        Good_value = 22.5396
        dis_value = 9.5

    ## 정상/결함 파일 분류를 위한 변수선언
    anomal_lst = []


    for i in distances:
      if i < Good_value - dis_value or i > Good_value + dis_value:
        print("비정상,", i)
        anomal_lst.append(0)
      else:
        print("정상,", i)
        anomal_lst.append(1)


    ## GradCam 모델 초기화
    name_layer = 'resnet18'
    gradcam = GradCam(model, name_layer)

    ## 모델 돌리기
    for filename in input_lst:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_dir, filename)

            # 이미지 불러오기
            # 히트맵은 그레이스케일, 두 이미지를 겹치는 cv2.addWeighted 하려면 RGB 이미지로 로드
            image = Image.open(input_path).convert("L")
            RGBimage = Image.open(input_path).convert("RGB")

            # 이미지를 텐서로 변환
            preprocess = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ])
            image = preprocess(image)
            RGBimage = preprocess(RGBimage)

            # 배치 크기를 추가
            batch_size = batch_size
            image = image.unsqueeze(0).expand(batch_size, -1, -1, -1)  # 배치 크기를 추가하고 복제
            RGBimage = RGBimage.unsqueeze(0).expand(batch_size, -1, -1, -1)

            heatmap = None

            #히트맵
            (B, _, H, W) = image.shape
            featuremaps = image.squeeze(1).detach().cpu().numpy()
            heatmaps = np.zeros((B, H, W, 3), dtype=np.uint8)
            for (i_map, fmap) in enumerate(featuremaps):
                hmap = cv2.normalize(fmap, None, 0, 1, cv2.NORM_MINMAX)
                hmap = cv2.convertScaleAbs(hmap, None, 255, None)
                hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
                heatmap = heatmaps[i_map, :, :] = hmap

            # cv2.imwrite("hello.png",heatmap)

            #원본 이미지와 히트맵을 겹쳐 표시
            images_show = np.zeros((B, H, W, 3), dtype=np.uint8)
            images_raw  = RGBimage.permute((0, 2, 3, 1))[..., [2, 1, 0]].detach().cpu().numpy()
            images_raw  = (images_raw * 255).astype(np.uint8)
            images_raw = images_raw[0]
            images_show = cv2.addWeighted(images_raw, 0.8, heatmap, 0.2, 0)

            # 저장
            cv2.imwrite(os.path.join(result_dir, filename),images_show)

    ## 그룹별로 분류해서 비정상 /정상 판단 및 이미지 붙이기
    # 여기에 그룹별 정상/비정상이 담김
    # 중복처리 방지 여기서 해야하나?
    result_dict = {}

    if size_url != 'small':
        anomal_groups = defaultdict(list)
        file_groups = defaultdict(list)
        for filename, anom in zip(input_lst, anomal_lst):
            common_part = filename.rsplit('_', 1)[0]
            anomal_groups[common_part].append(anom)
            file_groups[common_part].append(filename)

        for group, anomalies in anomal_groups.items():
            if 0 in anomalies:
                result_dict[group] = 0
            else:
                result_dict[group] = 1
        print(result_dict)
        # 이미지 합치기
        combine_img(input_dir, file_groups, final_input_dir)
        combine_img(result_dir, file_groups, final_result_dir)
    else:
        for i in range(len(anomal_lst)):
            if anomal_lst[i]:
                result_dict[input_lst[i].rsplit('.',1)[0]] = 1
            else:
                result_dict[input_lst[i].rsplit('.',1)[0]] = 0
        for filename in os.listdir(input_dir):
            src_path = os.path.join(input_dir, filename)
            dst_path = os.path.join(final_input_dir, filename)
            shutil.move(src_path, dst_path)
        for filename in os.listdir(result_dir):
            src_path = os.path.join(result_dir, filename)
            dst_path = os.path.join(final_result_dir, filename)
            shutil.move(src_path, dst_path)
    # todo : result_dict를 사용해서 해당 key값이 불량이 아니면 출력하기.
    # result_dict에 해당 파일 불량 여부.
    print(result_dict)
    return result_dict