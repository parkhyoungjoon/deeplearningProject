import os
import pandas as pd
import json

TRAIN_PATH = './DATA/Validation/'
# 여러 폴더 경로를 리스트로 저장
folder_paths = os.listdir(TRAIN_PATH)

# 빈 데이터프레임 리스트 생성
dataframes = []

# 각 폴더 내의 JSON 파일을 읽어와 데이터프레임으로 변환
for folder_path in folder_paths:
    FOLDER_PATH = TRAIN_PATH+folder_path
    print(f"Processing folder: {folder_path}")
    
    # 폴더 내의 모든 JSON 파일 리스트
    json_files = [file for file in os.listdir(FOLDER_PATH) if file.endswith('.json')]

    for file in json_files:
        file_path = os.path.join(FOLDER_PATH, file)
        print(f"Loading file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 파일 내용 확인 및 데이터프레임으로 변환
                if data:
                    # Aspects만 추출
                    for review in data:
                        aspects = pd.json_normalize(review.get('Aspects'))
                        dataframes.append(aspects)
                else:
                    print(f"No data found in {file}")
                    
        except json.JSONDecodeError:
            print(f"Error loading {file}: Invalid JSON")

# 데이터프레임 결합
if dataframes:
    final_dataframe = pd.concat(dataframes, ignore_index=True)
    print(final_dataframe)
else:
    print("No valid dataframes to concatenate.")
final_dataframe.to_csv('./DATA/IT_reivew_test.csv')