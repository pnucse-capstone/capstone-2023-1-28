import os

# a 폴더와 b 폴더 경로 설정
a_folder = 'a'
b_folder = 'b'

# a 폴더 내 파일 목록 가져오기
a_files =['a1_123_522.png','a2_123_6346.png','a3_123_754.png']

# b 폴더 내 파일 목록 가져오기
b_files = ['a1_123.png', 'a2_123.png', 'a3_123.png']

# a 폴더에 있는 파일의 이름을 추출
a_filenames = [os.path.splitext(file)[0] for file in a_files]

# b 폴더에서 a 폴더에 있는 파일의 이름과 일치하지 않는 파일을 필터링
filtered_b_files = [file for file in b_files if os.path.splitext(file)[0] not in a_filenames]

# 결과 출력
print("b 폴더에서 a 폴더에 있는 파일을 제외한 파일 목록:")
for file in filtered_b_files:
    print(file)