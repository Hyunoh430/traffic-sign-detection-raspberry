import YB_Pcb_Car  # Yahboom car library import
import torch
import cv2
import time
import numpy as np
import tensorflow as tf
import subprocess
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
import curses  # 키보드 입력 감지용 curses 모듈

# Yahboom car 초기화
car = YB_Pcb_Car.YB_Pcb_Car()

# TensorFlow Lite 인터프리터 초기화
interpreter = tf.lite.Interpreter(model_path='speed_sign_model.tflite')
interpreter.allocate_tensors()

# 입력 및 출력 텐서 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 이미지 전처리 함수
def preprocess_image(image):
    img = cv2.resize(image, (128, 128))
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 128, 128, 1)
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
            processed_image = preprocess_image(gray_img)
            interpreter.set_tensor(input_details[0]['index'], processed_image)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            class_labels = ['30', '40', '50', '60', '70', '80', '90']
            predicted_class_index = np.argmax(output_data, axis=1)[0]
            predicted_class_label = class_labels[predicted_class_index]
            return predicted_class_label
    return None

# curses를 사용하여 키보드 입력 감지
def capture_and_predict(stdscr, weights='128_128.pt', imgsz=(128, 128)):
    device = select_device('cpu')
    model_yolo = DetectMultiBackend(weights, device=device)
    model_yolo.warmup(imgsz=(1, 3, *imgsz))

    speed = 30  # 차량 속도 초기화
    max_speed = 0  # 차량 속도의 최댓값
    stdscr.nodelay(True)  # 키 입력을 기다리지 않고 즉시 반환
    stdscr.timeout(10)  # 100ms마다 화면을 갱신

    try:
        while True:
            start_time = time.time()

            # 실시간 이미지 캡처
            capture_command = [
                "libcamera-still", "-n", "--timeout", "1", "--width", "640", "--height", "480", "--output", "-"
            ]
            capture = subprocess.Popen(capture_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            stdout, _ = capture.communicate()

            np_arr = np.frombuffer(stdout, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                predicted_class_label = detect_and_predict_tflite(frame, model_yolo, device, imgsz=imgsz)

                if predicted_class_label:
                    max_speed = int(predicted_class_label)
                    stdscr.addstr(0, 0, f"Predicted speed limit: {predicted_class_label}")

                # 키보드 입력을 감지하여 속도를 변경 ('w'로 속도 증가, 's'로 속도 감소)
                key = stdscr.getch()
                if key == ord('w'):  # 'w' 키를 누르면 속도 증가
                    if speed < max_speed:
                        speed += 1
                elif key == ord('s'):  # 's' 키를 누르면 속도 감소
                    if speed > 0:
                        speed -= 1

                # 차량의 속도에 맞춰서 움직임 설정
                car.Car_Run(speed, speed)

                # FPS 출력
                end_time = time.time()
                elapsed_time = end_time - start_time
                fps = 1.0 / elapsed_time
                stdscr.addstr(1, 0, f"Speed: {speed} | FPS: {fps:.2f}")
                stdscr.refresh()

    except KeyboardInterrupt:
        stdscr.addstr(2, 0, "Stopping car...")
        stdscr.refresh()
        car.Car_Stop()  # Ctrl+C로 프로그램 중지 시 차량 정지

# curses 메인 루프 실행
curses.wrapper(capture_and_predict)
