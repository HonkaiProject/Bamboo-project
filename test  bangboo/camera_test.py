import cv2
import os
import face_recognition
import torch
import json
import threading

device = "cuda" if torch.cuda.is_available() else "cpu"
model_yolo = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True).to(device)

MEMORY_FILE = os.path.join(os.getcwd(), "bangboo_memory.json")

known_face_encodings = []
known_face_names = []

frame_count = 0

# 얼굴 및 객체 탐지 함수
def detect_faces_and_objects_periodic(frame, interval=5):
    global frame_count
    frame_count += 1

    # 얼굴 인식
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if not face_locations:
        cv2.imwrite("debug_frame.jpg", frame)
        print("얼굴이 감지되지 않았습니다. 'debug_frame.jpg' 파일을 확인하세요.")

    # 객체 탐지는 특정 주기로만 수행
    detected_objects = []
    if frame_count % interval == 0:
        results = model_yolo(frame)
        detected_objects = results.pandas().xyxy[0]["name"].tolist()

    return face_locations, face_encodings, detected_objects

# 얼굴 등록 함수
def register_face(frame, name):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    if not face_locations:
        return "Bangboo: 얼굴이 감지되지 않았어요. 다시 시도해 주세요."

    face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)
    return f"Bangboo: {name}님의 얼굴을 기억했어요!"

def save_memory(memory):
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(memory, f, indent=4)
        print(f"Bangboo: 메모리 파일 '{MEMORY_FILE}'에 저장되었습니다.")
    except Exception as e:
        print(f"Bangboo: 메모리 저장 중 오류 발생 - {e}")

def load_memory():
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

memory = load_memory()

# 기억 관련 함수
def save_user_memory(user_id, key, value):
    if user_id not in memory:
        memory[user_id] = {}
    memory[user_id][key] = value
    save_memory(memory)

def get_user_memory(user_id, key):
    return memory.get(user_id, {}).get(key)

def bangboo_reply(user_id, user_input, frame=None):
    # 얼굴 등록 요청 처리
    if "얼굴 등록" in user_input:
        name = user_input.split("이름은 ")[-1].strip()
        if frame is not None:
            return register_face(frame, name)

    # 탐지 결과 출력 요청
    if "앞에 뭐가 보이냐" in user_input or "주변에 뭐가 있어" in user_input:
        return "Bangboo: 지금 주변을 탐지하고 있어요!"

    # 사용자 이름 기억
    user_name = get_user_memory(user_id, "name")
    if user_name is None and "이름" in user_input:
        user_name = user_input.split("이름은 ")[-1].strip()
        save_user_memory(user_id, "name", user_name)
        return f"반가워요, {user_name}! 이제 이름을 기억할게요!"

def capture_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Bangboo: 카메라가 없습니다. 대화 기능만 활성화됩니다.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Bangboo: 카메라에서 프레임을 읽을 수 없습니다.")
            break

        # 얼굴 및 객체 탐지
        face_locations, face_encodings, detected_objects = detect_faces_and_objects_periodic(frame)

        # 얼굴 인식
        recognized_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = known_face_names[match_index]

            recognized_names.append(name)

        # 얼굴에 사각형 및 이름 표시
        for (top, right, bottom, left), name in zip(face_locations, recognized_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 화면에 프레임 출력
        cv2.imshow("Bangboo Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def start_bangboo_camera():
    user_id = "test_user"
    print("Bangboo: 반가워요! 카메라를 실행할게요.")

    cap = cv2.VideoCapture(0)
    while True:
        user_input = input("사용자: ")
        if user_input.lower() in ["끝", "종료", "bye", "바이", "exit", "stop"]:
            print("Bangboo: 카메라를 종료할게요.")
            break

        ret, frame = cap.read()
        if not ret:
            frame = None

        replay = bangboo_reply(user_id, user_input, frame)
        print(replay)

    cap.release()

camera_thread = threading.Thread(target=capture_camera)
camera_thread.start()
start_bangboo_camera()
