import os
import shutil
import cv2
import numpy as np
from matplotlib import pyplot as plt


def draw_contour(size_url, input_dict, output_dict):
    # 여기도 중복처리 방지를 해야겠네,,, 아니면 한걸 또하네
    # 적용시켜줘야 하는 애들 이름을 넘겨줘서
    # 그안에 있으면 하는걸로 하자
    input_dir = './static/results/' + size_url + '/png/'
    result_dir = 'static/contour/' + size_url + '/png/'
    lst_input = list(input_dict.keys())
    lst_output = list(output_dict.keys())

    answer_dict = dict()

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for i in range(len(lst_input)):
        # 여기서 갖고올 때, results에 있는 input_ 어쩌고를 갖고옴 <-- 근데 내 results 폴더가 바꼈으니 이것도 바꿔야함
        original_image = cv2.imread(os.path.join(input_dir, lst_input[i]), cv2.IMREAD_GRAYSCALE)

        # 여기서도 results에 있는 output_ 어쩌고를 갖고와야함
        mask_image = cv2.imread(os.path.join(input_dir, lst_output[i]))
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
        plt.imsave(os.path.join(result_dir + lst_output[i]), original_image_color, cmap='gray')
        answer_dict[lst_output[i]] = result_dir

    answer_lst = []
    answer_lst.append(answer_dict)
    return answer_lst



def resize_and_save_as_npy(size_url):
    upload_image_dir = 'static/preprocessing/' + size_url + '/'
    upload_output_dir = 'static/prepro_numpy/' + size_url + '/'

    if not os.path.exists(upload_output_dir):
        os.makedirs(upload_output_dir)

    # upload 폴더에 있는 파일들 꺼내기
    upload_image_files = os.listdir(upload_image_dir)
    already_output_files = os.listdir(upload_output_dir)

    resize_size = (512, 512)


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

def preprocesse_image(size_url):
    input_dir = "static/uploads/" + size_url + '/'
    output_dir = "static/preprocessing/" + size_url + '/'
    temp_dir = "static/temp/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # if not os.path.exists(temp_dir):
    #     os.makedirs(temp_dir)

    input_files = os.listdir(input_dir)
    already_output_files = os.listdir(output_dir)   # 중복처리 방지용

    if size_url == 'small':
        for filename in input_files:
            src = input_dir + filename
            des = output_dir + filename
            shutil.move(src,des)
        return

    cut_width = 1000

    for filename in input_files:
        input_path = input_dir + filename
        image = cv2.imread(input_path)

        height, width, _ = image.shape

        for i in range(0, width, cut_width):
            start_x = i
            end_x = min(i + cut_width, width)
            cropped_image = image[:, start_x:end_x]

            output_filename = f"{os.path.splitext(filename)[0]}_{i}.png"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, cropped_image)
        # if size_url == 'big':                                 전처리 하는거 보여줄려고 이랬는데 굳이 하지말자
        #     shutil.move(input_dir + filename, temp_dir)
        # else:
        os.remove(input_dir+filename)


# 이미지 전처리 -- big만 가능하니까 그외에는 app.py에서 막기
def different(selectValue, threshValue = 801, kernelValue = 23):
    # thresh : 쓰레스홀드, erode : 이로딩, dilation : 딜레이션, final : 최종
    image_dir = "static/uploads/big"        # 그냥 업로드에서 갖고오자 <-- 새로운 이미지들
    output_dir = "static/different"  # 이미지를 저장할 디렉토리 경로
    answer = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(image_dir):
        output_filename = f"{os.path.splitext(filename)[0]}_" + selectValue + '_' +threshValue + '_' + kernelValue + ".png"
        output_path = os.path.join(output_dir, output_filename)

        # 중복방지
        if output_filename in os.listdir(output_dir):
            answer.append(output_path)
            continue

        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)

        # 그레이스케일 (대형관과 중형관만)
        imgBW = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 쓰레시홀드 <-- 이거랑
        ThreshRange = int(threshValue)

        imgThresh = cv2.adaptiveThreshold(imgBW, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                          ThreshRange, 0)

        # ThreshRange(thresh일때,) 랑 Knum 을 바꿔야해
        if selectValue == "thresh":
            cv2.imwrite(output_path,imgThresh)
            answer.append(output_path)
            continue


        # 커널 크기, 작을 수록 선이 많아지고 클 수록 작아짐
        Knum = int(kernelValue)

        kernel = np.ones((Knum, Knum), np.uint8)

        # 이로딩연산 <-- 이거
        imgEro = cv2.erode(imgThresh, kernel, iterations=1)

        if selectValue == 'erode':
            cv2.imwrite(output_path,imgThresh)
            answer.append(output_path)
            continue

        # 딜레이션 연산 <-- 이거
        imgDil = cv2.dilate(imgEro, kernel, iterations=1)

        if selectValue == 'dilation':
            cv2.imwrite(output_path,imgThresh)
            answer.append(output_path)
            continue

        contours, hierarchy = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 색 조정
        setColor = (0, 255, 0)

        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                cv2.drawContours(image, [cnt], -1, setColor, 3)

        # 이거 까지
        cv2.imwrite(output_path, image)
        answer.append(output_path)

    return answer


def check_for_new_files():
    input_dir = 'static/uploads'
    results_dir = 'static/results'

    subfolders = ['small', 'middle', 'big']
    new_files = {}

    for folder in subfolders:
        input_folder = os.path.join(input_dir, folder)
        result_folder = os.path.join(os.path.join(results_dir, folder), 'png')
        key_folder = 'uploads/' + folder + '/'

        new_files[key_folder] = []

        if not os.path.exists(input_folder):
            os.makedirs(input_folder)
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        input_files = os.listdir(input_folder)
        result_files = os.listdir(result_folder)

        new_files[key_folder] = [file for file in input_files if file not in result_files]
    return new_files