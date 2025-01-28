import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from collections import deque

# GPT-2 모델 로드
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Bangboo의 기억 시스템
MEMORY_FILE = "bangboo_memory.json"
CONVERSATION_HISTORY_FILE = "conversation_history.json"  # 대화 기록 파일
short_term_memory = deque(maxlen=10)  # 단기 기억 (최대 10개의 최근 대화)

# 기억 초기화 및 관리
def load_memory():
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)

memory = load_memory()

# 대화 기록 불러오기
def load_conversation_history():
    try:
        with open(CONVERSATION_HISTORY_FILE, "r") as f:
            history = json.load(f)
            return deque(history, maxlen=10)
    except FileNotFoundError:
        return deque(maxlen=10)

def save_conversation_history(history):
    with open(CONVERSATION_HISTORY_FILE, "w") as f:
        json.dump(list(history), f, indent=4)

short_term_memory = load_conversation_history()  # 이전 대화 기록 불러오기

# 기억 저장 및 불러오기 함수
def save_user_memory(user_id, key, value):
    if user_id not in memory:
        memory[user_id] = {}
    memory[user_id][key] = value
    save_memory(memory)

def get_user_memory(user_id, key):
    return memory.get(user_id, {}).get(key)

# 웅나 생성 함수
def generate_woongna_response(user_input):
    # 대답의 길이에 따라 "웅나"의 길이를 조정
    response_length = len(user_input)
    woongna = "웅" + "나" * (response_length // 5 + 1)
    
    # "나"가 3개 이상일 경우 적절히 "웅나"로 바꾸기
    if len(woongna) > 6:  # "웅나나나"처럼 3개 이상인 경우
        woongna_list = ["웅나" for _ in range((len(woongna) - 1) // 2)]
        woongna = " ".join(woongna_list)

    # 대답 내용에 따라 기호 추가
    if "?" in user_input:
        symbol = "?!"
    elif "!" in user_input:
        symbol = "!!"
    elif "싫어" in user_input or "아니" in user_input:
        symbol = "..."
    else:
        symbol = "."
    
    # 최종 웅나 대답 생성
    final_response = f"{woongna} ({user_input}){symbol}"
    return final_response

# GPT-2를 사용한 Bangboo 대화 생성
def generate_bangboo_reply(context):
    input_ids = tokenizer.encode(context, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.9,
        temperature=0.7
    )
    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    return reply

# Bangboo의 대화 로직
def bangboo_reply(user_id, user_input):
    # 단기 기억에 대화 저장
    short_term_memory.append(f"User: {user_input}")

    # 사용자 이름 기억
    user_name = get_user_memory(user_id, "name")
    if user_name is None and "이름" in user_input:
        user_name = user_input.split("이름은 ")[-1].strip()
        save_user_memory(user_id, "name", user_name)
        return f"반가워요, {user_name}! 이제 이름을 기억할게요!"
    
    # GPT-2 기반 대화 생성
    context = "\n".join(short_term_memory) + f"\nBangboo: "
    raw_reply = generate_bangboo_reply(context)

    # Bangboo 스타일 적용
    bangboo_response = f"{raw_reply} 🐾 (Bangboo 스타일!)"

    # 웅나 스타일로 변환
    woongna_response = generate_woongna_response(user_input)

    # 단기 기억 추가
    short_term_memory.append(f"Bangboo: {bangboo_response}")

    # 최종 출력
    return woongna_response

# 대화 실행
def start_bangboo_conversation():
    user_id = "user123"  # 고정 사용자 ID
    print("Bangboo: 안녕하세요! 저는 Bangboo에요. 당신과 대화하고 싶어요!")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["끝", "종료", "bye"]:
            print("Bangboo: 대화해줘서 고마워요! 나중에 또 만나요! 🐾")
            save_conversation_history(short_term_memory)  # 종료 시 대화 기록 저장
            break
        reply = bangboo_reply(user_id, user_input)
        print(f"Bangboo: {reply}")

# 대화 시작
start_bangboo_conversation()