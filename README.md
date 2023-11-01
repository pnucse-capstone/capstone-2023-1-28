[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/fnZ3vxy8)

### 1. 프로젝트 소개
### 인공지능 비전 기반 RT필름 불량검출

#### 1.1) 배경
&nbsp;조선소 및 기타 산업 분야에서의 용접 작업은 구조물 및 선박 제조에 있어 핵심적인 과정입니다. 이러한 용접 부위에서의 결함은 구조물의 강도와 내구성에 영향을 미칠 수 있으며, 이를 조기에 발견하지 못하면 심각한 인명 사고 및 경제적 손실이 발생할 수 있습니다.따라서 검사 기술을 통해 용접 부위의 결함을 탐지하고 조처하는 것은 필수적입니다. 파괴 검사와 비 파괴 검사 중 비 파괴 검사 기술은 더 안전하고 신뢰할 수 있는 제품 생산을 지원하며, 구조물 및 선박의 수명을 증가하는 역할을 합니다. 또한 경제적으로도 유리한 부분이 존재합니다.이러한 검사 방법 중 하나인 Radiography Film(이하 RT 필름) 촬영 기술은 우수한 결과를 제공하고 있으며, 안전한 제품 제조, 구조물 및 선박의 수명 연장에 기여하고 있습니다.<br>
&nbsp;RT 필름은 이러한 RT 필름 촬영 기술을 활용해, 결함을 판독하기 쉽도록 시각적으로 기록하는 매체입니다. X선 또는 감마선을 이용하여 물체 내부의 결함, 용접 부위의 불완전성, 금속 구조의 두께 등을 시각적으로 확인할 수 있게 기록합니다.이후, 검사원을 투입하여 결함을 판단하게 되는데 다수의 RT 필름을 판독하려면 많은 인적 자원이 필요하며 시간 또한 많이 소모되게 됩니다.<br>
&nbsp;이러한 배경 속 인적 자원과 시간 소모를 줄이기 위해 이러한 모델을 제작하게 되었습니다.

#### 1.2) 개요
   &nbsp;이미지가 실제 조선소에서 제공되는 만큼 이미지가 너무 불균일하여, 먼저 다양한 이미지 전처리 방법들을 시도하였고 최종적으로 Unet을 통한 Image Segmentation을 실시하였습니다.
   또한, 제공되는 이미지에 이상데이터가 거의 없어 CutPaste 기법을 통하여 이상데이터를 만들어준 뒤 이를 학습시켰습니다.

---
### 2. 팀소개

* 정다현, dahyun@pusan.ac.kr, 딥러닝 모델 개발 및 구현, 보고서 작성

* 지민철, flyl123@naver.com, 딥러닝 모델 개발 및 구현, 이미지 전처리

* 이현규, gusrb2776@pusan.ac.kr, 딥러닝 모델 개발 및 구현, 플랫폼 개발

---
### 3. 시스템 구성도
![설계도](https://github.com/pnucse-capstone/capstone-2023-1-28/assets/57137757/832e8fb5-b036-49bd-98dc-fb38fca9eaef)

 * 효과적인 학습을 위해 먼저 UNet을 통해 Image Segmentation을 실시하여 불량을 검출할 용접부위를 추출합니다.
 * 추출된 이미지를 배경으로 CutPaste를 사용하여 이상을 가진 이미지를 추가하여줍니다.
 * 위의 작업이 필요한 이유는 실제 조선소에서 데이터를 수급하는데, 그 데이터의 수가 현저히 적고 이상 데이터는 거의 없기때문입니다.
 * CutPaste를 사용하여 만들어진 이상을 가진 이미지를 ResNet을 통해 학습시킵니다.
 * 해당 모델을 Flask를 통하여 간단한 웹페이지로 클라이언트에게 제공합니다.
 * 실제 서비스가 이루어진다면, 불량 검출 담당자와 용접 담당자가 달라서, 불량 검출 담당자가 직접 파일을 업로드하고 결과를 보는 방식이 아닌, 용접 담당자가 이미지를 업로드하면 불량 검출이 되도록 만들었습니다.
 
---
### 4. 소개 및 시연 영상
![2023년 전기 졸업 졸업과제 28 이지정의 필름쇼](https://github.com/pnucse-capstone/capstone-2023-1-28/assets/82069570/7071083f-d92d-4017-81d9-aa1efc6d22ac.png)(https://youtu.be/clH3B_cBuLg?si=023pZLOXeYsrVO8p)



---
### 5. 설치 및 사용법

---
**로컬에서 실행시**
weight파일의 크기가 너무 커서 구글 드라이브를 통해 설치 후 checkpoint에 넣으시면 됩니다.<br>
'''[https://drive.google.com/drive/folders/1QdCFcnN_ZmeIpSOhmo2l0qMOW6KQlWje?usp=drive_link](https://drive.google.com/drive/folders/1QdCFcnN_ZmeIpSOhmo2l0qMOW6KQlWje?usp=sharing)'''<br>
아나콘다 가상환경 기준으로 설명드리겠습니다.<br>
```conda creat -n 파일명 python=3.10.14```<br>
가상환경 생성 후 requirements.txt를 이용해 필요 라이브러리를 다운받습니다.<br>
- ```pip install -r requirements.txt```<br>
그 뒤 app.py를 실행시킨 후 127.0.0.1:5000에 접속합니다.

