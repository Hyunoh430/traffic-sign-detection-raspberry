import torch
import cv2
import time
import numpy as np
import tensorflow as tf
import subprocess
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# TensorFlow Lite 인터프리터 초기화
interpreter = tf.lite.Interpreter(model_path='speed_sign_model.tflite')
interpreter.allocate_tensors()

# 입력 및 출력 텐서 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 이미지 전처리 함수
def preprocess_image(image):
    img = cv2.resize(image, (128, 128))  # 이미지를 128x128 크기로 리사이즈
    img = img.astype('float32') / 255.0  # 모델에 맞춰 스케일링
    img = img.reshape(1, 128, 128, 1)  # 배치 크기 포함하여 4차원으로 변환
    return img

# YOLOv5 탐지 후 TFLite 분류
def detect_and_predict_tflite(frame, model_yolo, device, imgsz=(128, 128), conf_thres=0.25, iou_thres=0.45):
    # YOLOv5 이미지 전처리
    img = cv2.resize(frame, imgsz)  # 이미지를 128x128 크기로 리사이즈
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
    img = img.transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)로 변환
    img = torch.from_numpy(img).to(device).float() / 255.0  # 정규화
    img = img.unsqueeze(0)  # 배치 차원 추가

    # YOLOv5 추론
    pred = model_yolo(img)
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    max_conf = 0
    best_box = None

    # confidence가 가장 높은 박스 찾기
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                if conf > max_conf:
                    max_conf = conf
                    best_box = xyxy

    # 박스가 유효한 경우 크롭 및 TFLite 모델로 예측
    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box)
        if x2 > x1 and y2 > y1:  # 유효한 박스인지 확인
            cropped_img = frame[y1:y2, x1:x2]  # 박스 영역 크롭
            cropped_resized_img = cv2.resize(cropped_img, (128, 128))  # 128x128로 리사이즈

            # Grayscale 변환
            gray_img = cv2.cvtColor(cropped_resized_img, cv2.COLOR_RGB2GRAY)

            # TFLite 모델을 위한 이미지 전처리
            processed_image = preprocess_image(gray_img)

            # TFLite 모델에 입력 설정
            interpreter.set_tensor(input_details[0]['index'], processed_image)

            # TFLite 모델 예측
            interpreter.invoke()

            # 예측 결과 가져오기
            output_data = interpreter.get_tensor(output_details[0]['index'])
            class_labels = ['30', '40', '50', '60', '70', '80', '90']
            predicted_class_index = np.argmax(output_data, axis=1)[0]
            predicted_class_label = class_labels[predicted_class_index]

            # 분류 결과 출력
            return predicted_class_label
    return None

# 실시간 이미지 캡처 및 YOLOv5 -> TFLite 예측
def capture_and_predict(weights='128_128.pt', imgsz=(128, 128)):
    # YOLOv5 모델 초기화
    device = select_device('cpu')
    model_yolo = DetectMultiBackend(weights, device=device)
    model_yolo.warmup(imgsz=(1, 3, *imgsz))  # YOLOv5 모델 워밍업

    try:
        while True:
            start_time = time.time()  # 시작 시간 기록

            # 실시간 이미지 캡처: libcamera로 이미지를 캡처하여 메모리에 저장
            capture_command = [
                "libcamera-still", "-n", "--timeout", "1", "--width", "640", "--height", "480", "--output", "-"
            ]
            # 표준 출력(stdout)과 표준 에러(stderr)를 무시
            capture = subprocess.Popen(capture_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            stdout, _ = capture.communicate()

            # 이미지를 메모리에서 읽고 numpy 배열로 변환
            np_arr = np.frombuffer(stdout, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                # YOLOv5 탐지 및 TFLite 예측
                predicted_class_label = detect_and_predict_tflite(frame, model_yolo, device, imgsz=imgsz)

                if predicted_class_label:
                    end_time = time.time()  # 종료 시간 기록
                    elapsed_time = end_time - start_time  # 전체 처리 시간 계산
                    fps = 1.0 / elapsed_time

                    # FPS와 예측 결과 출력
                    print(f"Predicted speed limit: {predicted_class_label} | FPS: {fps:.2f}")

            # 반복 주기 속도 조절
            #time.sleep(0.1)

    except KeyboardInterrupt:
        print("Real-time prediction stopped.")

# 실시간 예측 실행
capture_and_predict(weights='128_128.pt', imgsz=(128, 128))
