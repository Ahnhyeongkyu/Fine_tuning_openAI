from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import logging

def train_models(file_ids, model_name="gpt-4o-mini-2024-07-18"):
    """
    주어진 파일 ID들을 사용하여 순차적으로 모델을 학습시키는 함수

    :param file_ids: 학습에 사용할 파일들의 ID 리스트
    :param model_name: 사용할 기본 모델 이름 (기본값: "gpt-4o-mini-2024-07-18")
    :return: 생성된 학습 작업들의 ID 리스트
    """

    # 환경 변수에서 API 키 로드
    load_dotenv()  # 환경 변수 로드
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

    job_ids = []
    for fold, file_id in enumerate(file_ids):
        print(f"폴드 {fold} 처리 시작")
        
        # 각 폴드마다 새로운 클라이언트 객체 생성
        client = OpenAI(api_key=api_key)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                job = client.fine_tuning.jobs.create(
                    training_file=file_id,
                    model=model_name
                )
                job_id = job.id
                job_ids.append(job_id)
                print(f"폴드 {fold} fine-tuning 작업 생성 완료. 작업 ID: {job_id}")
                
                # 작업 상태 확인
                while True:
                    job_status = client.fine_tuning.jobs.retrieve(job_id)
                    status = job_status.status
                    print(f"작업 {job_id} 상태: {status}")
                    
                    if status == "succeeded":
                        print(f"폴드 {fold} fine-tuning 작업 완료")
                        break
                    elif status in ["failed", "cancelled"]:
                        print(f"폴드 {fold} fine-tuning 작업 {status}")
                        break
                    
                    time.sleep(60)  # 1분 대기
                
                break  # 성공적으로 처리되면 재시도 루프 탈출
            
            except Exception as e:
                print(f"폴드 {fold} 처리 중 오류 발생 (시도 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    print("잠시 후 재시도합니다...")
                    time.sleep(60)  # 1분 대기 후 재시도
                else:
                    print(f"폴드 {fold} 처리 실패")
        
        print(f"폴드 {fold} 처리 완료")
        
    return job_ids  # 생성된 모든 작업 ID 반환

def get_trained_model_ids(job_ids):
    """
    완료된 fine-tuning 작업의 결과를 조회하여 성공한 모델의 ID와 실패한 작업 정보를 반환합니다.

    이 함수는 모든 fine-tuning 작업이 이미 완료된 상태에서 호출된다고 가정합니다.
    각 작업의 최종 상태를 확인하고, 성공한 모델의 ID와 실패한 작업의 정보를 수집합니다.

    :param job_ids: fine-tuning 작업 ID 리스트
    :return: (trained_model_ids, failed_jobs) 튜플
             trained_model_ids: 성공적으로 학습된 모델 ID 리스트
             failed_jobs: 실패한 작업 ID와 실패 사유를 담은 딕셔너리
    """
    # 환경 변수에서 OpenAI API 키 로드
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # OpenAI 클라이언트 초기화
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # 로깅 설정: 정보를 콘솔에 출력하고 필요시 파일에 저장할 수 있게 함
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    trained_model_ids = []  # 성공적으로 학습된 모델 ID를 저장할 리스트
    failed_jobs = {}        # 실패한 작업 정보를 저장할 딕셔너리

    # 각 작업 ID에 대해 상태 확인
    for job_id in job_ids:
        try:
            # OpenAI API를 통해 작업 상태 조회
            job_status = client.fine_tuning.jobs.retrieve(job_id)
            logger.info(f"작업 {job_id} 상태: {job_status.status}")

            # 작업 상태에 따른 처리
            if job_status.status == "succeeded":
                # 성공한 경우: 모델 ID를 리스트에 추가
                trained_model_ids.append(job_status.fine_tuned_model)
                logger.info(f"Fine-tuned 모델 ID: {job_status.fine_tuned_model}")
            elif job_status.status == "failed":
                # 실패한 경우: 실패 사유를 딕셔너리에 저장
                error_message = getattr(job_status, 'error', 'No error message available')
                failed_jobs[job_id] = error_message
                logger.error(f"작업 {job_id} 실패. 사유: {error_message}")
            elif job_status.status == "cancelled":
                # 취소된 경우: 취소 정보를 딕셔너리에 저장
                failed_jobs[job_id] = "작업이 취소됨"
                logger.warning(f"작업 {job_id}가 취소되었습니다.")
            else:
                # 예상치 못한 상태: 경고 로그 출력 및 정보 저장
                logger.warning(f"작업 {job_id}가 예상치 못한 상태입니다: {job_status.status}")
                failed_jobs[job_id] = f"Unexpected status: {job_status.status}"
        
        except Exception as e:
            # 상태 조회 중 예외 발생 시 처리
            logger.error(f"작업 {job_id} 상태 조회 중 오류 발생: {str(e)}")
            failed_jobs[job_id] = str(e)

    # 전체 결과 요약 로깅
    logger.info(f"총 {len(trained_model_ids)}개의 모델이 성공적으로 학습되었습니다.")
    if failed_jobs:
        logger.warning(f"{len(failed_jobs)}개의 작업이 실패했습니다.")

    return trained_model_ids, failed_jobs

if __name__ == "__main__":
    # 이전 단계에서 얻은 file_ids를 입력으로 사용
    file_ids = ["file-abc123", "file-def456", "file-ghi789"]  # 예시 ID들
    job_ids = [
                "ft:gpt-4o-mini-2024-07-18:personal::A2Hr9tnf", 
                "ft:gpt-4o-mini-2024-07-18:personal::A2I5cpCr", 
                "ft:gpt-4o-mini-2024-07-18:personal::A2IHcDoe",
                "ft:gpt-4o-mini-2024-07-18:personal::A2ITbvRE",
                "ft:gpt-4o-mini-2024-07-18:personal::A2IfArnp"
                ]
    get_trained_model_ids(job_ids)
    print("Created job IDs:", job_ids)