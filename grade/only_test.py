from collections import defaultdict
import os
from PIL import Image


def combine_img(file_dir, file_groups):
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
        result_image = result_image.resize((512,512))
        result_image.save(os.path.join(file_dir, f"{group_name}.png"))


input_dir = "./static/cutpaste/datasets/big"

input_lst = []
for filename in os.listdir(input_dir):
    input_lst.append(filename)
anomal_lst = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
size_url = 'big'

# input_lst = [
#     "input_image-00549.png",
#     "input_image-00586.png",
#     "input_image-00587.png",
# ]
# anomal_lst = [1,1,1]
# size_url = 'small'


# 파일을 그룹화할 딕셔너리를 생성합니다.
result_dict = {}

if size_url != 'small':
    # 그룹화 하기 위한 선언
    anomal_groups = defaultdict(list)
    file_groups = defaultdict(list)

    for filename, anom in zip(input_lst, anomal_lst):
        print(filename)
        common_part = filename.rsplit('_', 1)[0]  # 숫자를 제외한 공통 부분을 찾습니다.
        anomal_groups[common_part].append(anom)
        file_groups[common_part].append(filename)

    for group, anomalies in anomal_groups.items():
        if 0 in anomalies:
            result_dict[group] = 0  # 그룹 내에 불량 파일이 하나라도 있으면 해당 그룹은 불량
        else:
            result_dict[group] = 1  # 그룹 내에 모두 정상 파일인 경우

    combine_img('static/cutpaste/datasets/big',file_groups)
    combine_img('static/cutpaste/results/big',file_groups)
    print("anomal_groups : ", anomal_groups)
    print("file_groups : ", file_groups)
else:
    for i in range(len(anomal_lst)):
        if anomal_lst[i]:
            result_dict[input_lst[i]] = 1
        else:
            result_dict[input_lst[i]] = 0
# 결과 출력

print("result : ", result_dict)