import os
from openai import OpenAI
from dotenv import load_dotenv

def check_finetune_status(job_ids):
    """
    주어진 fine-tuning 작업 ID들의 상태를 확인하는 함수

    :param job_ids: 확인할 작업 ID 리스트
    :return: 각 작업 ID에 대한 상태 정보를 담은 딕셔너리
    """
    # 환경 변수에서 API 키 로드
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)

    job_statuses = {}

    for job_id in job_ids:
        try:
            # 작업 상태 조회
            job_status = client.fine_tuning.jobs.retrieve(job_id)
            
            # 필요한 정보 추출
            status_info = {
                "status": job_status.status,
                "fine_tuned_model": job_status.fine_tuned_model,
                "created_at": job_status.created_at,
                "finished_at": job_status.finished_at,
                "trained_tokens": job_status.trained_tokens
            }
            
            job_statuses[job_id] = status_info
        except Exception as e:
            job_statuses[job_id] = {"error": str(e)}

    return job_statuses

# 사용 예시
if __name__ == "__main__":
    job_ids = [
        "ftjob-ZtKvTLgJ0e5ceQ5SX4x93kh9",
        "ftjob-JlWtpNLJVvxM14jumO1ufjLM",
        "ftjob-s4beZbEjRzf4VFeNzUIGQkkB",
        "ftjob-JzrvxIggVbQKm0J27q0ZsOX5"
    ]
    
    statuses = check_finetune_status(job_ids)
    
    for job_id, status in statuses.items():
        print(f"Job ID: {job_id}")
        for key, value in status.items():
            print(f"  {key}: {value}")
        print()