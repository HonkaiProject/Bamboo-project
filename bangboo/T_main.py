import importlib
import os
import json
from collections import deque
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# GPT-2 모델 로드
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_local")
model = GPT2LMHeadModel.from_pretrained("./gpt2_local")

# Bangboo 설정
MEMORY_FILE = "bangboo_memory.json"
CONVERSATION_HISTORY_FILE = "conversation_history.json"
MODULES_DIR = "./modules"
short_term_memory = deque(maxlen=200)

# 모듈 관리 시스템
modules = {}

def load_modules():
    """modules 폴더에 있는 모든 모듈을 로드."""
    for filename in os.listdir(MODULES_DIR):
        if filename.endswith(".py"):
            module_name = filename[:-3]
            try:
                module = importlib.import_module(f"modules.{module_name}")
                modules[module_name] = module
                print(f"모듈 '{module_name}'이(가) 로드되었습니다.")
            except Exception as e:
                print(f"모듈 '{module_name}'을(를) 로드하는 중 오류 발생: {e}")

def execute_module(module_name, *args):
    """특정 모듈의 main() 함수를 호출."""
    if module_name in modules:
        try:
            return modules[module_name].main(*args)
        except Exception as e:
            return f"모듈 '{module_name}' 실행 중 오류 발생: {e}"
    else:
        return f"모듈 '{module_name}'이(가) 존재하지 않습니다."

def initialize_files():
    """초기 파일 생성"""
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "w") as f:
            json.dump({}, f)
    if not os.path.exists(CONVERSATION_HISTORY_FILE):
        with open(CONVERSATION_HISTORY_FILE, "w") as f:
            json.dump([], f)

initialize_files()
load_modules()

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

short_term_memory = load_conversation_history()

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

def generate_woongna_response(user_input):
    context = "\n".join(short_term_memory) + f"\nUser: {user_input}\nBangboo:"
    gpt2_reply = generate_bangboo_reply(context)
    
    if "?" in gpt2_reply:
        symbol = "?!"
    elif "!" in gpt2_reply:
        symbol = "!!"
    else:
        symbol = "."

    response_length = len(user_input)
    woongna = "웅" + "나" * (response_length // 3 + 2)

    if woongna.count("나") > 3:
        woongna_list = ["웅나" for _ in range((response_length // 6) + 1)]
        woongna = " ".join(woongna_list)

    final_response = f"{woongna}{symbol} ({gpt2_reply})"
    return final_response

def bangboo_reply(user_input):
    if user_input.startswith("모듈 실행"):
        module_name = user_input.split(" ")[2]
        return execute_module(module_name)

    short_term_memory.append(f"User: {user_input}")
    woongna_response = generate_woongna_response(user_input)
    short_term_memory.append(f"Bangboo: {woongna_response}")
    return woongna_response

def start_bangboo_conversation():
    print("Bangboo: 안녕하세요! 저는 Bangboo에요. 당신과 대화하고 싶어요!")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["끝", "종료", "bye"]:
            print("Bangboo: 대화해줘서 고마워요! 나중에 또 만나요!")
            save_conversation_history(short_term_memory)
            break

        reply = bangboo_reply(user_input)
        print(f"Bangboo: {reply}")

start_bangboo_conversation()