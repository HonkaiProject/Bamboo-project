import json
import cv2
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from collections import deque
import os

# 로컬 GPT-2 모델 로드
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_local")
model = GPT2LMHeadModel.from_pretrained("./gpt2_local")

# YOLOv5 모델 로드 (Pre-trained)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_yolo = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True).to(device)

# Bangboo의 기억 시스템
MEMORY_FILE = "bangboo_memory.json"
CONVERSATION_HISTORY_FILE = "conversation_history.json"
short_term_memory = deque(maxlen=200)  # 최근 대화 최대 200개 기억

# 파일 초기화 및 관리
def initialize_files():
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "w") as f:
            json.dump({}, f)
    if not os.path.exists(CONVERSATION_HISTORY_FILE):
        with open(CONVERSATION_HISTORY_FILE, "w") as f:
            json.dump([], f)

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
    context = "\n".join(short_term_memory) + f"\nUser: {user_input}\nBangboo:"
    gpt2_reply = generate_bangboo_reply(context)
    
    # 대답 끝 기호 설정
    if "?" in gpt2_reply:
        symbol = "?!"
    elif "!" in gpt2_reply:
        symbol = "!!"
    else:
        symbol = "."

    response_length = len(user_input)
    woongna = "웅" + "나" * (response_length // 3 + 2)

    # "나"가 3개 이상이면 다시 "웅나"로 변환
    if woongna.count("나") > 3:
        woongna_list = ["웅나" for _ in range((response_length // 6) + 1)]
        woongna = " ".join(woongna_list)

    final_response = f"{woongna}{symbol} ({gpt2_reply})"
    return final_response

# Bangboo의 대화 로직
def bangboo_reply(user_id, user_input):
    short_term_memory.append(f"User: {user_input}")

    user_name = memory.get("name", "친구")
    if user_name is None and "이름" in user_input:
        user_name = user_input.split("이름은 ")[-1].strip()
        memory["name"] = user_name
        save_memory(memory)
        return f"반가워요, {user_name}! 이제 이름을 기억할게요!"
    
    woongna_response = generate_woongna_response(user_input)
    short_term_memory.append(f"Bangboo: {woongna_response}")
    return woongna_response

# YOLO 객체 탐지
def describe_scene(frame):
    results = model_yolo(frame)
    detected_objects = results.pandas().xyxy[0]["name"].tolist()
    return detected_objects

def capture_camera():
    """카메라 실행 및 객체 탐지."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Bangboo: 카메라가 없습니다. 대화 기능만 활성화됩니다.")
        return False

    print("Bangboo: 카메라가 실행 중입니다. 'q'를 눌러 종료하세요.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Bangboo: 카메라에서 프레임을 읽을 수 없습니다.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_objects = describe_scene(frame_rgb)

        print(f"Bangboo: 주변 환경에서 {', '.join(detected_objects)}이(가) 탐지되었어요.")
        cv2.imshow("Bangboo Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return True

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

# 프로그램 실행
initialize_files()

# 카메라 먼저 실행
camera_status = capture_camera()

# 대화 기능 실행
start_bangboo_conversation()