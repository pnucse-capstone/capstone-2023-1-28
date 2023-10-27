import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


def draw_contour(size_url, input_dict, output_dict):
    # 여기도 중복처리 방지를 해야겠네,,, 아니면 한걸 또하네
    # 적용시켜줘야 하는 애들 이름을 넘겨줘서
    # 그안에 있으면 하는걸로 하자
    result_dir = './static/results/' + size_url + '/png/'
    lst_input = list(input_dict.keys())
    lst_output = list(output_dict.keys())

    for i in range(len(lst_input)):
        original_image = cv2.imread(os.path.join(result_dir, lst_input[i]), cv2.IMREAD_GRAYSCALE)

        # 마스크 이미지 열기
        mask_image = cv2.imread(os.path.join(result_dir, lst_output[i]))
        gray_mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

        # 테두리 추출
        blurred_image = cv2.GaussianBlur(gray_mask, (5, 5), 0)
        edges = cv2.Canny(blurred_image, threshold1=50, threshold2=100)
        edges = cv2.resize(edges, (mask_image.shape[1], mask_image.shape[0]))

        # 테두리 그리기
        original_image_color = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        border_color = (0, 255, 255)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(original_image_color, contours, -1, border_color, thickness=2)

        # 이미지를 그레이스케일로 저장 (cmap=gray)
        plt.imsave(os.path.join(result_dir,lst_output[i]), original_image_color, cmap='gray')



def resize_and_save_as_npy(size_url):
    upload_image_dir = 'static/uploads/' + size_url + '/'
    upload_output_dir = 'static/upload_numpy/' + size_url + '/'

    # upload 폴더에 있는 파일들 꺼내기
    upload_image_files = os.listdir(upload_image_dir)
    already_output_files = os.listdir(upload_output_dir)

    resize_size = (512, 512)

    if not os.path.exists(upload_output_dir):
        os.makedirs(upload_output_dir)

    for filename in upload_image_files:
        input_path = os.path.join(upload_image_dir, filename)

        # 중복처리 방지
        output_filename = 'input_' + os.path.splitext(filename)[0] + '.npy'
        if output_filename in already_output_files:
            continue

        image = cv2.imread(input_path, flags=cv2.IMREAD_UNCHANGED)

        if size_url != 'small':
            # 이미지를 그레이스케일로 변환
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        w = image.shape[0]
        h = image.shape[1]

        print(w, h)
        new_img = cv2.resize(image, resize_size, interpolation=cv2.INTER_AREA)

        # .npy로 저장
        output_filename = 'input_' + os.path.splitext(filename)[0] + '.npy'
        output_path = os.path.join(upload_output_dir, output_filename)
        np.save(output_path, new_img)

