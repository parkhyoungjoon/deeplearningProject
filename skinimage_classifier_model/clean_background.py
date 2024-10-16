import os
from rembg import remove
from PIL import Image

# 배경 제거할 이미지 경로와 저장할 이미지 경로
input_dir = '../image/skinImage/'  # 배경 제거할 이미지들이 있는 폴더
output_dir = '../image/skinImageb/'  # 결과 이미지를 저장할 폴더

# 결과 저장 폴더가 존재하지 않으면 생성
os.makedirs(output_dir, exist_ok=True)

# 입력 디렉토리의 모든 이미지 파일 처리
for filename in os.listdir(input_dir):
    # 파일 확장자가 이미지 파일인지 확인
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        input_path = os.path.join(input_dir, filename)  # 입력 파일 경로
        output_path = os.path.join(output_dir, filename)  # 출력 파일 경로

        # 이미지 열기 및 배경 제거
        img = Image.open(input_path)
        out = remove(img)

        # JPEG는 알파 채널을 지원하지 않으므로 RGB로 변환
        if out.mode == 'RGBA':
            out = out.convert('RGB')

        # 결과 이미지 저장
        out.save(output_path)
        print(f"Processed and saved: {output_path}")