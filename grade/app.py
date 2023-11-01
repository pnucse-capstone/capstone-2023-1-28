import os

import numpy as np
import torch

from flask import Flask, render_template, request, jsonify
from matplotlib import pyplot as plt

import func
from flaskext.mysql import MySQL

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
    images = func.check_for_new_files()
    return render_template('index.html', images=images)

@app.route('/data_anal', methods=['GET'])
def data_anal():
    images = func.check_for_new_files()
    return render_template('data_anal.html', images=images)


@app.route('/detection', methods=['GET','POST'])
def detection():
    return render_template('detection.html')

@app.route('/prepro', methods=['POST'])
def prepro():
    selectValue = request.form.get('selectValue')
    threshValue = request.form.get('threshValue')
    kernelValue = request.form.get('kernelValue')

    images = func.different(selectValue, threshValue, kernelValue)
    return jsonify(images)


@app.route('/anomal', methods=['GET','POST'])
def anomal():
    return render_template('detection.html')


@app.route('/predict/<size_url>', methods=['GET', 'POST'])
def predict(size_url):
    print("이미지 전처리 중")
    func.preprocesse_image(size_url)
    print("이미지 npy로 변환중")
    func.resize_and_save_as_npy(size_url)
    print("모델 작업중")
    func.model_run(size_url)
    print("DB에서 경로 받아오는중")
    images = import_img_db(size_url)

    # JSON으로 전송
    return jsonify(images)


if __name__ == '__main__':
    app.run()





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

    print(images)
    # for image_dir in output_dirs:
    #     images.append(image_dir[0])
    #
    # for image_dir in input_dirs:
    #     images.append(image_dir[0])

    cursor.close()
    conn.close()

    return images
