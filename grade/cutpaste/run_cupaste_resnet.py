import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easydict
import pandas as pd

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


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #모델 경로
    model="cutpaste/model/small/model-small-2023-10-31_11_31_19.tch"
    # 얘네로 평균을 내서 비교후 거리를 구함
    FolderPath_train="cutpaste/aver_distance/small/train/good"  #small

    # UNet 이미지가 위치한 폴더 <-- 이거 전처리 과정을 거친 데이터를 넣는 위치도 생각을 해야겠네
    # 전처리 해서 cutpaste/datasets에 넣자
    FolderPath_test="cutpaste/datasets/small"


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


    dataloader_train = DataLoader(MVTecAT(FolderPath_train, size, transform = test_transform, mode="test"), batch_size, shuffle=False, num_workers=0)
    dataloader_test = DataLoader(MVTecAT(FolderPath_test, size, transform = test_transform, mode="test"), batch_size, shuffle=False, num_workers=0)


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

    print(distances)

    #딱 맞게 하면 안되고 여유를 조금 주어야 한다
    #소형관 정상 이미지들의 평균 거리 = 20.5350, 10.5 또는 4.5
    #중형관 정상 이미지들의 평균 거리 = 20.9062, 6.5
    # 대형관 정상 이미지들의 평균 거리 = 22.5396, 9.5

    Good_value = 21.0671
    dis_value = 10.5

    for i in distances:
      if i < Good_value - dis_value or i > Good_value + dis_value:
        print("비정상,", i)
      else:
        print("정상,", i)


    # GradCam 모델 초기화
    name_layer = 'resnet18'
    gradcam = GradCam(model, name_layer)

    input_directory = FolderPath_test

    for filename in os.listdir(input_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_directory, filename)

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

            cv2.imwrite("hello.png",heatmap)

            #원본 이미지와 히트맵을 겹쳐 표시
            images_show = np.zeros((B, H, W, 3), dtype=np.uint8)
            images_raw  = RGBimage.permute((0, 2, 3, 1))[..., [2, 1, 0]].detach().cpu().numpy()
            images_raw  = (images_raw * 255).astype(np.uint8)
            images_raw = images_raw[0]
            images_show = cv2.addWeighted(images_raw, 0.8, heatmap, 0.2, 0)

            # 이미지를 화면에 표시
            cv2.imwrite("her.png",images_show)
