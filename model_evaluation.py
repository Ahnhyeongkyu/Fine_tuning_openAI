from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import json
from typing import Dict, Any
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_similarity(text1, text2):
    """
    두 텍스트 간의 의미적 유사도를 계산하는 함수
    
    :param text1: 첫 번째 텍스트
    :param text2: 두 번째 텍스트
    :return: 두 텍스트 간의 코사인 유사도 (0~1 사이의 값)
    """
    # 문장 임베딩 모델 로드
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    # 두 텍스트를 벡터로 변환
    embeddings = model.encode([text1, text2])
    # 코사인 유사도 계산 및 반환
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def evaluate_models(model_ids, validation_files, metric='similarity', similarity_threshold=0.8):
    """
    K-fold 교차 검증을 사용하여 fine-tuned 모델들의 성능을 평가하는 함수

    :param model_ids: fine-tuned 모델 ID 리스트
    :param validation_files: 검증 데이터 파일 경로 리스트
    :param metric: 평가 지표 (기본값: 'similarity')
    :param similarity_threshold: 유사도 임계값 (기본값: 0.8)
    :return: 각 폴드의 성능 결과와 평균 성능
    """
    # 환경 변수에서 OpenAI API 키 로드
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)

    results = {}  # 각 폴드의 결과를 저장할 딕셔너리

    # 각 폴드(모델)에 대해 평가 수행
    for fold, (model_id, val_file) in enumerate(zip(model_ids, validation_files)):
        logger.info(f"폴드 {fold}의 모델 평가 중 (모델 ID: {model_id})")
        
        # 검증 데이터 파일 읽기
        try:
            with open(val_file, 'r', encoding='utf-8') as f:
                validation_data = [json.loads(line) for line in f]
        except Exception as e:
            logger.error(f"검증 데이터 파일 {val_file} 읽기 실패: {str(e)}")
            continue

        similarities = []  # 각 예측의 유사도를 저장할 리스트

        # 각 검증 데이터에 대해 예측 수행
        for i, item in enumerate(validation_data):
            system_content = item['messages'][0]['content']  # 시스템 메시지
            prompt = item['messages'][1]['content']  # 사용자 질문
            true_answer = item['messages'][2]['content']  # 실제 정답

            try:
                # OpenAI API를 사용하여 예측 수행
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,  # 낮은 temperature로 일관된 출력 유도
                    max_tokens=1000  # 출력 토큰 수 제한
                )
                predicted_answer = response.choices[0].message.content

                # 예측과 실제 답변 간의 유사도 계산
                similarity = calculate_similarity(true_answer, predicted_answer)
                similarities.append(similarity)

                # 상세 로깅
                logger.info(f"샘플 {i}:")
                logger.info(f"시스템: {system_content}")
                logger.info(f"질문: {prompt}")
                logger.info(f"실제 답변: {true_answer[:100]}...")
                logger.info(f"예측 답변: {predicted_answer[:100]}...")
                logger.info(f"유사도: {similarity}")

            except Exception as e:
                logger.error(f"예측 중 오류 발생 (샘플 {i}): {str(e)}")

        # 유효한 예측 결과가 없는 경우 처리
        if not similarities:
            logger.warning(f"폴드 {fold}: 유효한 예측 결과 없음")
            results[f"fold_{fold}"] = 0.0
            continue

        # 평균 유사도 및 정확도 계산
        avg_similarity = sum(similarities) / len(similarities)
        accuracy = sum(1 for s in similarities if s >= similarity_threshold) / len(similarities)

        # 결과 저장
        results[f"fold_{fold}"] = {
            'average_similarity': avg_similarity,
            'accuracy': accuracy
        }
        logger.info(f"폴드 {fold} - 평균 유사도: {avg_similarity:.4f}, 정확도: {accuracy:.4f}")

    # 전체 평균 계산
    if results:
        avg_similarity = sum(r['average_similarity'] for r in results.values()) / len(results)
        avg_accuracy = sum(r['accuracy'] for r in results.values()) / len(results)
        logger.info(f"전체 평균 유사도: {avg_similarity:.4f}, 평균 정확도: {avg_accuracy:.4f}")
    else:
        logger.warning("유효한 결과가 없습니다.")
        avg_similarity = avg_accuracy = 0.0

    return results, (avg_similarity, avg_accuracy)

def get_model_config(client, model_id):
    try:
        model_info = client.fine_tuning.jobs.retrieve(model_id)
        config = {
            "base_model": model_info.model,
            "hyperparameters": {
                "n_epochs": model_info.hyperparameters.n_epochs,
                "batch_size": model_info.hyperparameters.batch_size,
                "learning_rate_multiplier": model_info.hyperparameters.learning_rate_multiplier
            }
        }
        return config
    except Exception as e:
        logger.error(f"모델 정보를 가져오는 중 오류 발생: {str(e)}")
        return None

def finalize_model_selection(trained_model_ids, scores, metric='similarity'):
    """
    평가 결과를 바탕으로 최고의 모델을 선택하는 함수

    :param trained_model_ids: 학습된 모델 ID 리스트
    :param scores: evaluate_models 함수에서 반환된 점수 딕셔너리
    :param metric: 사용할 평가 지표 ('similarity' 또는 'accuracy')
    :return: 최고 모델 ID, 최고 모델 설정, 평균 점수, 표준 편차
    """
    load_dotenv()
    OPENAI_API_KEY = os.getenv("PERSONAL_OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)

    if not scores or not isinstance(scores, dict):
        logger.warning("유효하지 않은 scores. 기본값 사용.")
        return trained_model_ids[0], {"model_id": trained_model_ids[0], "score": 0.0}, 0.0, 0.0

    metric_scores = [fold_score[metric] for fold_score in scores.values() if isinstance(fold_score, dict)]
    
    if not metric_scores:
        logger.warning(f"선택한 지표 {metric}에 대한 유효한 점수가 없습니다.")
        return trained_model_ids[0], {"model_id": trained_model_ids[0], "score": 0.0}, 0.0, 0.0

    best_score = max(metric_scores)
    best_fold = max(scores, key=lambda k: scores[k][metric])
    best_model_id = trained_model_ids[int(best_fold.split('_')[1])]

    avg_score = sum(metric_scores) / len(metric_scores)
    std_score = (sum((score - avg_score) ** 2 for score in metric_scores) / len(metric_scores)) ** 0.5

    # 최고 모델의 설정 가져오기
    best_model_config = get_model_config(client, best_model_id)
    if best_model_config:
        best_model_config["score"] = best_score
    else:
        best_model_config = {"model_id": best_model_id, "score": best_score}

    logger.info(f"선택된 최고 모델: {best_model_id}, 점수: {best_score}")
    logger.info(f"평균 점수: {avg_score:.4f}, 표준 편차: {std_score:.4f}")
    logger.info(f"최고 모델 설정: {best_model_config}")

    return best_model_id, best_model_config, avg_score, std_score


if __name__ == "__main__":
    # 이 부분은 모듈을 직접 실행할 때만 사용됩니다.
    model_ids = []  # 예시 ID
    validation_files = ["val_fold_0.jsonl", "val_fold_1.jsonl"]
    # avg_score = evaluate_all_folds(model_ids, validation_files, metric='accuracy')
    # print(f"Overall average accuracy: {avg_score}")