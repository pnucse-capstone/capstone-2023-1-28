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

from dataset import MVTecAT
from cutpaste import CutPaste, cut_paste_collate_fn
from model import ProjectionNet
from pathlib import Path
from collections import defaultdict
from density import GaussianDensitySklearn, GaussianDensityTorch
from utils import str2bool

class MVTecAT(Dataset):

    def __init__(self, root_dir, size, transform=None, mode="train"):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode
        self.size = size
        # output_ 만 들고오는걸로 수정
        self.image_names = list(filter(lambda x: x.name.startswith("output_"), self.root_dir.glob("*.png")))
        print(self.image_names)

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## 경로 설정
model_dir="./static/cutpaste/model/" + size_url + "/"
input_dir = "./static/results/" + size_url + "/png"
result_dir = "./static/cutpaste/" + size_url

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 가중치랑 테스트임베딩 들고오기
model = None
embed_pt = None
for file in os.listdir(model_dir):
    if file.endswith('.tch'):
        model = file
    if file.endswith('.pt'):
        embed_pt = file


#모델 로드
head_layer = 1
head_layers = [512]*head_layer+[128]
weights = torch.load(model, map_location=device)
classes = weights["out.weight"].shape[0]
model = ProjectionNet(pretrained=False, head_layers=head_layers, num_classes=classes)
model.load_state_dict(weights)
model.to(device)
model.eval()



## 부가 설정
size = 512
batch_size = 64

test_transform = transforms.Compose([])
test_transform.transforms.append(transforms.Resize((size,size)))
test_transform.transforms.append(transforms.ToTensor())
test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]))

test_data_eval = MVTecAT(input_dir, size, transform = test_transform, mode="test")
dataloader_test = DataLoader(test_data_eval, batch_size, shuffle=False, num_workers=0)

#테스트 임베딩 제작
embeds = []
with torch.no_grad():
    for x, label in dataloader_test:
        embed, logit = model(x.to(device))

        #print(embed.shape,logit)
        embeds.append(embed.cpu())

embeds = torch.cat(embeds)
embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)


lt = torch.load(embed_pt)
distances = lt.predict(embeds)
print(distances)

#딱 맞게 하면 안되고 여유를 조금 주어야 한다, 약 15 정도
#소형관 정상 이미지들의 평균 거리 = 39.3880, 54
#중형관 정상 이미지들의 평균 거리 = 143.0075, 158
Good_value = 39
dis_value = 15

for i in distances:
    if i < Good_value - dis_value or i > Good_value + dis_value:
        #DB에 넣는걸로 수정 필요
        print("비정상,", i)
    else:
        #너도
        print("정상,", i)


# GradCam 모델 초기화
name_layer = 'resnet18'
gradcam = GradCam(model, name_layer)

input_directory = input_dir

for filename in os.listdir(input_directory):
    if filename.startswitch('output_') and (filename.endswith(".jpg") or filename.endswith(".png")):
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

        # cv2.imwrite("heatmap.png",heatmap)

        #원본 이미지와 히트맵을 겹쳐 표시
        images_show = np.zeros((B, H, W, 3), dtype=np.uint8)
        images_raw  = RGBimage.permute((0, 2, 3, 1))[..., [2, 1, 0]].detach().cpu().numpy()
        images_raw  = (images_raw * 255).astype(np.uint8)
        images_raw = images_raw[0]
        images_show = cv2.addWeighted(images_raw, 0.8, heatmap, 0.2, 0)

        # 이미지 저장 <-- 내가 필요한건 이것만 있으면 됨
        # 파일 저장후 DB에 넣기
        cv2.imwrite(os.path.join(result_dir,filename),heatmap)

print("작업끝!")