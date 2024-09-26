from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from sklearn.model_selection import KFold
import numpy as np

def prepare_and_upload_data(file_path, n_splits=5):
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)

    # 데이터 로드
    with open(file_path, "r", encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # K-fold 교차 검증 설정
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    file_ids = []
    for fold, (train_index, val_index) in enumerate(kf.split(data)):
        train_data = [data[i] for i in train_index]
        val_data = [data[i] for i in val_index]
        
        # 훈련 데이터 저장
        train_file = f"prompt_guide_train_fold_{fold}.jsonl"
        with open(train_file, "w") as f:
            for item in train_data:
                json.dump(item, f)
                f.write("\n")
        
        # 훈련 파일 업로드
        response = client.files.create(
            file=open(train_file, "rb"),
            purpose="fine-tune"
        )
        
        file_ids.append(response.id)
        print(f"Uploaded training file ID for fold {fold}:", response.id)
        
        # 검증 데이터 저장
        val_file = f"val_fold_{fold}.jsonl"
        with open(val_file, "w") as f:
            for item in val_data:
                json.dump(item, f)
                f.write("\n")
        validation_files = [f"val_fold_{fold}.jsonl" for fold in range(n_splits)]
    return file_ids, validation_files
