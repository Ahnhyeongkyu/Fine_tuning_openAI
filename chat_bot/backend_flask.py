from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from openai import OpenAI

app = Flask(__name__)
CORS(app)

load_dotenv()
OPENAI_API_KEY = os.getenv("PERSONAL_OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

FINETUNED_MODEL_ID = "ft:gpt-4o-mini-2024-07-18:personal::A2L6nJQ4"
SYSTEM_PROMPT = "당신은 AI와 프롬프트 엔지니어링에 대해 전문적인 지식을 가진 AI 어시스턴트입니다."

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    try:
        response = client.chat.completions.create(
            model=FINETUNED_MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )
        ai_response = response.choices[0].message.content
        return jsonify({"response": ai_response})
    except Exception as e:
        print(f"Error using the model: {str(e)}")
        return jsonify({"error": "Failed to get response from AI"}), 500

if __name__ == '__main__':
    app.run(debug=True)