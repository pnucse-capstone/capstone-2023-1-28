import os

import numpy as np
import torch

from flask import Flask, render_template
from matplotlib import pyplot as plt

import func
from flaskext.mysql import MySQL
from flask import jsonify

import myUnet

mysql = MySQL()
app = Flask(__name__)

app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = '1004'
app.config['MYSQL_DATABASE_DB'] = 'my_grade'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
app.secret_key = "ABCDEFG"
mysql.init_app(app)


@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template('index.html')

@app.route('/predict/<size_url>', methods=['GET', 'POST'])
def predict(size_url):
    print("실행")
    print("이미지 npy로 변환중")
    func.resize_and_save_as_npy(size_url)
    print("모델 작업중")
    model_run(size_url)
    print("DB에서 경로 받아오는중")
    images = import_img_db(size_url)

    # JSON으로 전송
    return jsonify(images)


if __name__ == '__main__':
    app.run()



def model_run(size_url):
    # 기타 설정
    lr = 1e-3
    # 수정ㅁㅁ
    batch_size = 8
    num_epoch = 100

    data_dir = 'static/upload_numpy/' + size_url
    ckpt_dir = './checkpoint/' + size_url
    result_dir = 'static/results/' + size_url

    if not os.path.exists(result_dir):
        os.makedirs(os.path.join(result_dir, 'png'))
        os.makedirs(os.path.join(result_dir, 'numpy'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # uploads에 있는거 자동으로 들고옴 <-- 이것도 DB에서 갖고오는걸로 해야하나..? 근데 파일이 아니라 경로 그자체를 들고오는거라 좀 애매하네
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
    func.draw_contour(size_url, input_dict, output_dict)
    output_lst = []
    input_lst = []

    output_lst.append(output_dict)
    input_lst.append(input_dict)
    insert_db(output_lst, size_url, 'outputs')
    insert_db(input_lst, size_url, 'images')

def insert_db(lst, size_url, table):
    tableName = table
    # tableName = size_url + '_' + table
    # 수정 시작해봄
    conn = mysql.connect()
    cursor = conn.cursor()
    values_to_insert = []
    for output_dict in lst:
        for name in output_dict:
            # 이미 존재하는 image_name을 확인하기 위한 SQL 쿼리
            check_sql = f"SELECT image_name FROM {tableName} WHERE image_name = %s AND size = %s"
            cursor.execute(check_sql, (name, size_url))
            result = cursor.fetchone()

            if result is None:  # 이미 존재하지 않는 경우에만 추가
                values_to_insert.append((name, output_dict[name] + name, size_url))
                # 'size_url'을 'size' 열에 삽입할 값으로 추가

    insert_sql = f"INSERT INTO {tableName} (image_name, image_dir, size) VALUES (%s, %s, %s)"
    try:
        cursor.executemany(insert_sql, values_to_insert)
        conn.commit()
    except Exception as e:
        conn.rollback()
        print("에러:", str(e))
    finally:
        cursor.close()
        conn.close()

def import_img_db(size_url):
    conn = mysql.connect()
    cursor = conn.cursor()

    input_table_name = 'images'
    output_table_name = 'outputs'
    # input_table_name = size_url + '_images'
    # output_table_name = size_url + '_outputs'

    output_sql = f"SELECT image_dir FROM {output_table_name} WHERE size = %s"
    input_sql = f"SELECT image_dir FROM {input_table_name} WHERE size = %s"  # 추가된 조건

    cursor.execute(output_sql, (size_url,))
    output_dirs = cursor.fetchall()
    cursor.execute(input_sql, (size_url,))
    input_dirs = cursor.fetchall()

    images = []
    for input_item, output_item in zip(input_dirs, output_dirs):
        images.append(input_item[0])
        images.append(output_item[0])


    # for image_dir in output_dirs:
    #     images.append(image_dir[0])
    #
    # for image_dir in input_dirs:
    #     images.append(image_dir[0])

    cursor.close()
    conn.close()

    return images
