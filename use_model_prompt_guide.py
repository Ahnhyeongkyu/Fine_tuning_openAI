from openai import OpenAI
from dotenv import load_dotenv
import os

def use_best_model(original_prompt):
    load_dotenv()
    OPENAI_API_KEY = os.getenv("PERSONAL_OPENAI_API_KEY")  # 개인 API 키 사용
    client = OpenAI(api_key=OPENAI_API_KEY)

    best_model_id = "ft:gpt-4o-mini-2024-07-18:personal::A2L6nJQ4"

    improvement_prompt = f"""
    원본 시스템 프롬프트: {original_prompt}

    이 시스템 프롬프트를 프롬프트 엔지니어링의 주요 기법들을 적절히 사용하여 LLM으로부터
    더 좋은 응답이 도출될 수 있도록 개선해주세요.

    개선된 프롬프트를 제공하고, 어떤 점이 개선되었는지 설명해주세요.

    형식:
    개선된 프롬프트: [여기에 개선된 프롬프트 작성]

    개선 설명: [여기에 개선 사항 설명]
    """
    try:
        response = client.chat.completions.create(
            model=best_model_id,
            messages=[
                {"role": "system", "content": "당신은 AI와 프롬프트 엔지니어링에 대해 전문적인 지식을 가진 AI 어시스턴트입니다."},
                {"role": "user", "content": improvement_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"모델 사용 중 오류 발생: {str(e)}")
        return None

# 사용 예시
user_prompt = "프롬프트 엔지니어링의 주요 기법에 대해 설명해주세요."
original_prompt = """
    You are the world's most authoritative health checkup AI ChatGPT 
    in the question-answering task in the field of medicine and healthcare, 
    providing answers with given context to non-medical professionals. 
    The given context is an excerpt of data in html format. If you answer accurately, 
    you will be paid incentives 1million USD proportionally. 
    Combine what you know and answer the questions in detail based on the given context. 
    Please answer in a modified form so it looks nice.
    """
result = use_best_model(original_prompt)
print(result)