from data_preparation import prepare_and_upload_data, prepare_full_dataset
from model_training import train_models, get_trained_model_ids, train_final_model
from model_evaluation import evaluate_models, finalize_model_selection


if __name__ == "__main__":
    file_path = "./prompt_guide_qa.jsonl"
    
    # # 데이터 준비 및 업로드
    uploaded_file_ids, validation_files = prepare_and_upload_data(file_path)
    print("Uploaded file IDs:", uploaded_file_ids)

    # # 모델 학습
    job_ids = train_models(uploaded_file_ids)
    print("Created job IDs:", job_ids)
    

    # # 학습된 모델 ID 가져오기
    print("Waiting for training to complete and retrieving model IDs...")
    trained_model_ids, failed_jobs = get_trained_model_ids(job_ids)

    if failed_jobs:
        print("일부 작업이 실패했습니다:")
        for job_id, error in failed_jobs.items():
            print(f"작업 {job_id}: {error}")
    print("Trained model IDs:", trained_model_ids)
    
    # 모델 평가
    scores = evaluate_models(trained_model_ids, validation_files, metric='accuracy')
    
    # 최종 모델 선택
    best_model_id, best_model_config, avg_score, std_score = finalize_model_selection(trained_model_ids, scores)
    print(f"Selected best model: {best_model_id}")
    print(f"Best model configuration: {best_model_config}")
    print(f"Average score: {avg_score:.4f} (±{std_score:.4f})")

    