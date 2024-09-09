import subprocess
import time
import cv2
import numpy as np
import threading
import tensorflow as tf
import torch
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# TensorFlow Lite 인터프리터 초기화 (TFLite 모델 사용 예시)
interpreter = tf.lite.Interpreter(model_path='speed_sign_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# YOLOv5 모델 초기화 (YOLOv5 모델 사용 예시)
device = select_device('cpu')
model_yolo = DetectMultiBackend('128_128.pt', device=device)
model_yolo.warmup(imgsz=(1, 3, 128, 128))

# 스레드 간에 공유할 변수 및 락
frame_lock = threading.Lock()
shared_frame = None

# 이미지 전처리 함수 (TFLite 모델에 사용)
def preprocess_image(image):
    img = cv2.resize(image, (128, 128))
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 128, 128, 1)  # 배치 크기 포함하여 4차원으로 변환
    return img

# YOLOv5 탐지 후 TFLite 분류
def detect_and_predict_tflite(frame, model_yolo, device, imgsz=(128, 128), conf_thres=0.25, iou_thres=0.45):
    img = cv2.resize(frame, imgsz)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)

    pred = model_yolo(img)
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    max_conf = 0
    best_box = None

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                if conf > max_conf:
                    max_conf = conf
                    best_box = xyxy

    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box)
        if x2 > x1 and y2 > y1:
            cropped_img = frame[y1:y2, x1:x2]
            cropped_resized_img = cv2.resize(cropped_img, (128, 128))
            gray_img = cv2.cvtColor(cropped_resized_img, cv2.COLOR_RGB2GRAY)

            # TFLite 예측
            processed_image = preprocess_image(gray_img)
            interpreter.set_tensor(input_details[0]['index'], processed_image)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            class_labels = ['30', '40', '50', '60', '70', '80', '90']
            predicted_class_index = np.argmax(output_data, axis=1)[0]
            predicted_class_label = class_labels[predicted_class_index]

            return predicted_class_label
    return None

# 비디오 캡처 스레드
def video_capture_thread():
    global shared_frame

    # libcamera-vid 명령어로 실시간 비디오 스트림을 캡처
    capture_command = [
        "libcamera-vid", "-n", "--timeout", "0", "--width", "640", "--height", "480",
        "--framerate", "30", "--output", "-"
    ]
    
    process = subprocess.Popen(capture_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        while True:
            # H.264 스트림에서 한 프레임씩 읽기
            stdout_data = process.stdout.read(640 * 480 * 3)

            if len(stdout_data) == 0:
                print("No data captured from camera.")
                break

            # 프레임을 numpy 배열로 변환하고 OpenCV로 디코딩
            np_arr = np.frombuffer(stdout_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                with frame_lock:  # 스레드 안전하게 공유 프레임 업데이트
                    shared_frame = frame

    except KeyboardInterrupt:
        print("Stopping video capture...")
        process.terminate()

# 딥러닝 연산 스레드
def dl_inference_thread():
    global shared_frame

    try:
        while True:
            # 딥러닝 연산은 프레임이 업데이트되었을 때만 수행
            with frame_lock:
                if shared_frame is not None:
                    frame_to_process = shared_frame.copy()
                else:
                    frame_to_process = None

            if frame_to_process is not None:
                start_time = time.time()

                # YOLOv5 탐지 및 TFLite 분류 실행
                predicted_class_label = detect_and_predict_tflite(frame_to_process, model_yolo, device)

                if predicted_class_label:
                    elapsed_time = time.time() - start_time
                    fps = 1.0 / elapsed_time
                    print(f"FPS: {fps:.2f} | Predicted speed limit: {predicted_class_label}")

            time.sleep(0.05)  # 너무 빠른 루프 방지

    except KeyboardInterrupt:
        print("Stopping inference...")

# 메인 실행 함수
if __name__ == "__main__":
    # 비디오 캡처 스레드 생성 및 시작
    video_thread = threading.Thread(target=video_capture_thread)
    dl_thread = threading.Thread(target=dl_inference_thread)

    video_thread.start()
    dl_thread.start()

    video_thread.join()
    dl_thread.join()
