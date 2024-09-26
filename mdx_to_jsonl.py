import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
# LLM 초기화
llm = ChatOpenAI(temperature=0.7)

# 질문 생성을 위한 프롬프트 템플릿
question_template = PromptTemplate(
    input_variables=["technique", "description"],
    template="다음은 '{technique}'라는 프롬프트 기법에 대한 설명입니다:\n\n{description}\n\n이 프롬프트 기법에 대한 관련성 있는 질문을 생성하세요. 질문:"
)

# 답변 생성을 위한 프롬프트 템플릿
answer_template = PromptTemplate(
    input_variables=["question", "technique", "description"],
    template="당신은 AI와 프롬프트 엔지니어링에 대해 전문적인 지식을 가진 AI 어시스턴트입니다. '{technique}' 프롬프트 기법에 대한 다음 질문에 답변하세요. 필요하다면 예시를 들어 설명하세요.\n\n기법 설명: {description}\n\n질문: {question}\n\n답변:"
)

# 질문 생성 체인
question_chain = question_template | llm | StrOutputParser()

# 답변 생성 체인
answer_chain = answer_template | llm | StrOutputParser()

def extract_technique_from_filename(filename):
    # 파일 확장자 제거
    name = Path(filename).stem
    
    # '.en' 제거 (있는 경우)
    name = name.replace('.en', '')
    
    # 대시를 공백으로 변경하고 각 단어의 첫 글자를 대문자로
    technique = ' '.join(word.capitalize() for word in name.split('-'))
    
    # 특별한 경우 처리
    special_cases = {
        'Cot': 'Chain of Thought',
        'Dsp': 'Directional Stimulus Prompting',
        'Pal': 'Program-Aided Language Models',
        'Rag': 'Retrieval Augmented Generation',
        'Tot': 'Tree of Thoughts'
    }
    
    for acronym, full_name in special_cases.items():
        technique = technique.replace(acronym, full_name)
    
    return technique

def extract_technique_from_mdx(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 파일 이름에서 기법 이름 추출
    technique_name = extract_technique_from_filename(file_path)
    
    # 간단한 처리: import 문과 JSX 컴포넌트를 제거
    lines = [line for line in content.split('\n') if not line.strip().startswith('import') and not line.strip().startswith('<')]
    description = ' '.join(lines)
    
    return technique_name, description

def generate_qa_pairs(technique, description, num_pairs=3):
    qa_pairs = []
    for _ in range(num_pairs):
        try:
            question = question_chain.invoke({"technique": technique, "description": description})
            answer = answer_chain.invoke({"question": question, "technique": technique, "description": description})
            
            qa_pair = {
                "messages": [
                    {"role": "system", "content": "당신은 AI와 프롬프트 엔지니어링에 대해 전문적인 지식을 가진 AI 어시스턴트입니다."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            }
            qa_pairs.append(qa_pair)
            print(f"Generated QA pair for '{technique}'")
        except Exception as e:
            print(f"Error generating QA pair for '{technique}': {str(e)}")
    
    return qa_pairs

def process_mdx_files(input_dir, output_file, num_pairs_per_technique=3):
    input_path = Path(input_dir)
    all_qa_pairs = []
    
    for mdx_file in input_path.glob('*.mdx'):
        print(f"Processing file: {mdx_file}")
        technique, description = extract_technique_from_mdx(mdx_file)
        print(f"Generating QA pairs for: {technique}")
        qa_pairs = generate_qa_pairs(technique, description, num_pairs_per_technique)
        all_qa_pairs.extend(qa_pairs)
    
    print(f"Total QA pairs generated: {len(all_qa_pairs)}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for qa_pair in all_qa_pairs:
            f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
    
    print(f"Output written to {output_file}")

# 사용 예:
process_mdx_files('./techniques', 'prompt_guide_qa.jsonl', num_pairs_per_technique=6)