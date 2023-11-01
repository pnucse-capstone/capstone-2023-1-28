import logging
import os
import shutil
import cv2
import numpy as np
from matplotlib import pyplot as plt
import app
import torch
import myUnet

def preprocesse_image(size_url):
    input_dir = "static/uploads/" + size_url + '/'
    result_dir = "static/preprocessing/" + size_url + '/'
    temp_dir = "static/temp/"

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # if not os.path.exists(temp_dir):
    #     os.makedirs(temp_dir)

    input_files = os.listdir(input_dir)
    already_output_files = os.listdir(result_dir)   # 중복처리 방지용

    if size_url == 'small':
        for filename in input_files:
            src = input_dir + filename
            des = result_dir + filename
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
            output_path = os.path.join(result_dir, output_filename)
            cv2.imwrite(output_path, cropped_image)
        # if size_url == 'big':                                 전처리 하는거 보여줄려고 이랬는데 굳이 하지말자
        #     shutil.move(input_dir + filename, temp_dir)
        # else:
        # os.remove(input_dir+filename)             #1 수정


def unet_model_run(size_url):
    data_dir = 'static/prepro_numpy/' + size_url
    ckpt_dir = './checkpoint/' + size_url
    result_dir = 'static/results/' + size_url

    logging.info("모델 동작중")
    if not os.path.exists(result_dir):
        logging.info("폴더생성")
        os.makedirs(os.path.join(result_dir, 'png'))
        logging.info("폴더생성2")
        os.makedirs(os.path.join(result_dir, 'numpy'))

    # 기타 설정
    lr = 1e-3
    # batch_size는 모델 돌아가는 애들 길이만큼 그냥 한번에 들고오기로 결정.
    num_epoch = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # preprocessing에 있는거 자동으로 들고옴 <-- 이것도 DB에서 갖고오는걸로 해야하나..? 근데 파일이 아니라 경로 그자체를 들고오는거라 좀 애매하네
    transform = myUnet.transforms.Compose([myUnet.Normalization(mean=0.5, std=0.5), myUnet.ToTensor()])

    ## 이미 있는 경우 또 안하기 위해서
    lst_input = os.listdir(data_dir)
    filtered_input = []
    file_path = 'static/results/' + size_url + '/numpy/'
    for filename in lst_input:
        if not os.path.exists(file_path + filename):
            filtered_input.append(filename)
    # filtered_input에 이제 이름이 담기는데 이 이름으로 저장을 해야지 곂친거 또 안함
    # 만약에 같은 이름의 파일을 또 저장한다하면 uuid써야하는데 거기까진 구현하지말자.
    print("처음 모델 돌리는애들 : ", filtered_input)
    if len(filtered_input) ==0: return

    batch_size = len(filtered_input)
    dataset_test = myUnet.Dataset(data_dir=data_dir, transform=transform, lst_input=filtered_input)
    # 이거 원래 num_workers=8 (쓰레드 수)임
    loader_test = myUnet.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    net = myUnet.UNet().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)

    # tensor --> numpy
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    # 정규화 해제
    fn_denorm = lambda x, mean, std: (x * std) + mean
    # 아웃풋 이미지를 바이너리 클래스로 분류해주는거
    fn_class = lambda x: 1.0 * (x > 0.5)


    st_epoch = 0
    # 저장한 모델로드
    net, optim, st_epoch = myUnet.load(ckpt_dir=ckpt_dir, net=net, optim=optim)
    output_dict = dict()
    input_dict = dict()
    ## 모델 예측
    with torch.no_grad():
        net.eval()

        for batch, data in enumerate(loader_test, 1):
            # forward pass
            input = data['input'].to(device)

            output = net(input)

            # Tensorboard 저장하기
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            for j in range(input.shape[0]):
                input_np_name = filtered_input[j]
                output_np_name = filtered_input[j].replace('input_', 'output_')
                input_png_name = filtered_input[j].replace('.npy','.png')
                output_png_name = output_np_name.replace('.npy','.png')

                # 나머지도 바꿔야하면 바꾸는데 굳이? 싶긴해 --> 아니 바꿔야해... 그래야 안곂쳐....
                # UUID 처리해주는게 제일 좋긴함
                plt.imsave(os.path.join(result_dir, 'png', input_png_name), input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', output_png_name), output[j].squeeze(), cmap='gray')
                np.save(os.path.join(result_dir, 'numpy', input_np_name), input[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', output_np_name), output[j].squeeze())

                output_dict[output_png_name] = 'static/results/' + size_url + '/png/'
                input_dict[input_png_name] = 'static/results/' + size_url + '/png/'
    # 이거 내부에서 contour한거 저장하는게 필요해
    contour_lst = draw_contour(size_url, input_dict, output_dict)
    output_lst = []
    input_lst = []

    # output_lst.append(output_dict)
    input_lst.append(input_dict)

    # 수정필요 images, outputs <-- contour한거 , results <-- 결과...
    # insert_db(output_lst, size_url, 'outputs')
    app.insert_db(input_lst, size_url, 'images')
    app.insert_db(contour_lst, size_url, 'outputs')
    # insert_db(contour_lst, size_url, 'contours')

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


def draw_contour(size_url, input_dict, output_dict):
    # 여기도 중복처리 방지를 해야겠네,,, 아니면 한걸 또하네
    # 적용시켜줘야 하는 애들 이름을 넘겨줘서
    # 그안에 있으면 하는걸로 하자
    input_dir = './static/results/' + size_url + '/png/'
    result_dir = 'static/contour/' + size_url + '/png/'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    lst_input = list(input_dict.keys())
    lst_output = list(output_dict.keys())

    answer_dict = dict()

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



# 이미지 전처리 -- big만 가능하니까 그외에는 app.py에서 막기
def different(selectValue, threshValue = 801, kernelValue = 23):
    # thresh : 쓰레스홀드, erode : 이로딩, dilation : 딜레이션, final : 최종
    image_dir = "static/uploads/big"        # 새로운 이미지 올라온거
    result_dir = "static/different"  # 이미지를 저장할 디렉토리 경로
    answer = []

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for filename in os.listdir(image_dir):
        output_filename = f"{os.path.splitext(filename)[0]}_" + selectValue + '_' +threshValue + '_' + kernelValue + ".png"
        output_path = os.path.join(result_dir, output_filename)

        # 중복방지
        if output_filename in os.listdir(result_dir):
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


def expand_img(size_url):
    img_dir = './static/results/' + size_url + '/png'  # img
    npy_dir = './static/results/' + size_url + '/numpy'  # numpy
    result_dir = './static/cutpaste/datasets/' + size_url

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # A 폴더와 B 폴더의 파일 목록 가져오기
    png_files = [os.path.join(img_dir, filename) for filename in os.listdir(img_dir) if
                 filename.endswith('.png') and filename.startswith('output_')]
    npy_files = [os.path.join(npy_dir, filename) for filename in os.listdir(npy_dir) if
                 filename.endswith('.npy') and filename.startswith('output_')]

    # 파일 목록을 정렬하여 순서대로 처리
    png_files.sort()
    npy_files.sort()


    # A 폴더와 B 폴더의 파일 목록을 동시에 처리
    for png_file, npy_file in zip(png_files, npy_files):
        mask = np.load(npy_file)

        # 원본 이미지 로드 (PNG 형식)
        original_image = cv2.imread(png_file, cv2.IMREAD_COLOR)  # UNCHANGED 플래그를 사용하여 투명도 정보를 유지

        # 마스크와 원본 이미지의 형태 일치시키기
        if mask.shape[:2] != original_image.shape[:2]:
            mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 마스크를 사용하여 원본 이미지에서 용접 부위를 추출
        welding_area = original_image.copy()
        welding_area[mask == 1] = 0

        cv2.resize(welding_area, (512,512))

        # 결과 이미지를 투명한 PNG로 저장
        # cv2.imwrite('hello.png', welding_area)

        gray = cv2.cvtColor(welding_area, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        binary_mask = 255 - binary_mask

        dst = cv2.inpaint(welding_area, binary_mask, 3, cv2.INPAINT_TELEA)
        # cv2.imwrite('hello2.png', dst)

        welding_area = dst

        row_indices, col_indices = np.where(mask == 0)

        # 좌표를 사용하여 최소한의 직사각형을 구함
        min_row, max_row = min(row_indices), max(row_indices)
        min_col, max_col = min(col_indices), max(col_indices)

        # 최소한의 직사각형을 사용하여 용접 부위 잘라내기
        extracted_area = welding_area[min_row:max_row + 1, min_col:max_col + 1]
        extracted_area = cv2.resize(extracted_area, (512,512))

        # 결과 이미지 저장
        #cv2.imwrite('용접부위만_512x512.png', extracted_area)

        result_filename = os.path.basename(png_file)
        output_path = os.path.join(result_dir, result_filename)
        cv2.imwrite(output_path, extracted_area)