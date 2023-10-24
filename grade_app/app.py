# import os
# from flask import Flask, render_template, request, send_file, redirect
# import torch
# import numpy as np
# import cv2
# import eval
#
# print("hello")
#
# # Python 스크립트가 실행되기 전에 인코딩 설정 변경
# os.environ['PYTHONIOENCODING'] = 'utf-8'
#
# app = Flask(__name__)
#
# # 모델과 가중치 불러오기
# model = eval.UNet()  # UNet 모델 생성
# model.load_state_dict(torch.load('checkpoint/model_epochX.pth'))  # 학습된 모델 가중치 불러오기
# model.eval()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
#
# # 이미지 업로드를 위한 폴더 설정
# UPLOAD_FOLDER = 'uploads'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
#
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         # 업로드된 이미지 처리
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files['file']
#         if file.filename == '':
#             return redirect(request.url)
#         if file:
#             image_path = os.path.join(UPLOAD_FOLDER, file.filename)
#             file.save(image_path)
#
#             # 이미지 전처리 (Normalization, ToTensor)
#             image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 흑백 이미지로 읽기
#             image = image / 255.0
#             image = image.astype(np.float32)
#             image = np.expand_dims(image, axis=0)  # 배치 차원 추가
#             image = torch.from_numpy(image).to(device)
#
#             # 모델 예측
#             with torch.no_grad():
#                 output = model(image)
#
#             # 결과 이미지 저장
#             result_image = output[0, 0].cpu().numpy() * 255  # 0 또는 1 값으로 변환
#             result_image = result_image.astype(np.uint8)
#             cv2.imwrite('result.jpg', result_image)
#
#             return render_template('result.html', result_image='result.jpg')
#
#     return render_template('index.html')
#
# if __name__ == '__main__':
#     app.run()

from flask import Flask

app = Flask(__name__)

@app.route("/")
def spring():
    return "<h1>Flask Server<h1>"

if __name__ == '__main__':
    app.run(host="127.0.0.1")