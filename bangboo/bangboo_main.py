import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from collections import deque

# 로컬 GPT-2 모델 로드
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_local")
model = GPT2LMHeadModel.from_pretrained("./gpt2_local")

# Bangboo의 기억 시스템
MEMORY_FILE = "bangboo_memory.json"
CONVERSATION_HISTORY_FILE = "conversation_history.json"
short_term_memory = deque(maxlen=200)  # 최근 대화 최대 200개 기억

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
            return deque(history, maxlen=200)
    except FileNotFoundError:
        return deque(maxlen=200)

def save_conversation_history(history):
    with open(CONVERSATION_HISTORY_FILE, "w") as f:
        json.dump(list(history), f, indent=4)

short_term_memory = load_conversation_history()  # 이전 대화 기록 불러오기

# Bangboo의 대답 생성 함수
def generate_bangboo_reply(context):
    input_ids = tokenizer.encode(context, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.9,
        temperature=0.8
    )
    reply = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return reply

# Bangboo의 웅나 대답 생성 함수
def generate_woongna_response(user_input):
    # GPT-2 대답 생성
    context = "\n".join(short_term_memory) + f"\nUser: {user_input}\nBangboo:"
    gpt2_reply = generate_bangboo_reply(context)
    
    # 대답 끝 기호 설정
    if "?" in gpt2_reply:
        symbol = "?!"
    elif "!" in gpt2_reply:
        symbol = "!!"
    else:
        symbol = "."

    # 웅나 반복 처리
    response_length = len(user_input)
    woongna = "웅" + "나" * (response_length // 3 + 2)  # 나의 반복 횟수 증가

    # "나"가 3개 이상이면 다시 "웅나"로 변환
    if woongna.count("나") > 3:
        woongna_list = ["웅나" for _ in range((response_length // 6) + 1)]
        woongna = " ".join(woongna_list)

    # 최종 대답 생성
    final_response = f"{woongna}{symbol} ({gpt2_reply})"
    return final_response

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
    
    # 웅나 스타일의 GPT-2 대답 생성
    woongna_response = generate_woongna_response(user_input)

    # 단기 기억에 Bangboo의 대답 저장
    short_term_memory.append(f"Bangboo: {woongna_response}")

    # 최종 출력
    return woongna_response

# 기억 저장 및 불러오기 함수
def save_user_memory(user_id, key, value):
    if user_id not in memory:
        memory[user_id] = {}
    memory[user_id][key] = value
    save_memory(memory)

def get_user_memory(user_id, key):
    return memory.get(user_id, {}).get(key)

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