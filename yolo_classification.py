import torch
import cv2
import time  # 시간 측정용 및 TensorFlow Lite용
import numpy as np
import tensorflow as tf
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from pathlib import Path

# TensorFlow Lite 인터프리터 초기화
interpreter = tf.lite.Interpreter(model_path='speed_sign_model.tflite')
interpreter.allocate_tensors()

# 입력 및 출력 텐서 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 이미지 전처리 함수
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0  # 모델에 맞춰 스케일링
    img = img.reshape(1, 128, 128, 1)  # 배치 크기 포함하여 4차원으로 변환
    return img

# detect_image 함수: 객체 탐지 후 가장 높은 confidence 박스를 cropped 후 grayscale 변환 및 저장
def detect_and_predict_tflite(weights, img_path, device='cpu', imgsz=(128, 128), conf_thres=0.25, iou_thres=0.45, crop_save_path='cropped_result.jpg'):
    # Load model
    device = select_device(device)
    model_yolo = DetectMultiBackend(weights, device=device)
    model_yolo.warmup(imgsz=(1, 3, *imgsz))  # 128x128 입력 크기로 워밍업

    # Load and preprocess image
    img0 = cv2.imread(img_path)  # (H, W, C) 형태로 이미지 로드
    img = cv2.resize(img0, imgsz)  # 이미지를 128x128 크기로 리사이즈 (H, W, C)
    
    # BGR에서 RGB로 변환 (OpenCV는 기본적으로 BGR로 이미지를 읽기 때문에)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # (H, W, C) -> (C, H, W)로 변환 (PyTorch가 요구하는 형태)
    img = img.transpose((2, 0, 1))  # 채널 축 변경
    
    # 텐서로 변환 및 정규화
    img = torch.from_numpy(img).to(device).float() / 255.0  # 0~255 범위를 0~1로 정규화
    img = img.unsqueeze(0)  # 배치 차원 추가

    # Inference (YOLOv5)
    start_time = time.time()  # 추론 시작 시간 기록
    pred = model_yolo(img)
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    end_time = time.time()  # 추론 종료 시간 기록

    # 추론에 걸린 시간 계산
    inference_time = end_time - start_time
    print(f"Inference Time: {inference_time:.2f} seconds")

    max_conf = 0
    best_box = None

    # Draw boxes and labels, and find the box with the highest confidence
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                # Check for the highest confidence
                if conf > max_conf:
                    max_conf = conf
                    best_box = xyxy

    # Crop, resize, and convert to grayscale if we found a valid box
    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box)  # Convert to integer
        cropped_img = img0[y1:y2, x1:x2]  # Crop the image using the best box coordinates
        cropped_resized_img = cv2.resize(cropped_img, (128, 128))  # Resize the cropped image to 128x128
        
        # Grayscale 변환
        gray_img = cv2.cvtColor(cropped_resized_img, cv2.COLOR_RGB2GRAY)  # RGB 이미지를 그레이스케일로 변환
        
        # Save the cropped, resized, and grayscale image
        cv2.imwrite(crop_save_path, gray_img)
        print(f"Cropped, resized, and grayscale result saved to {crop_save_path}")

        # 이미지 전처리 (TFLite 입력 준비)
        processed_image = preprocess_image(crop_save_path)

        # TFLite 모델에 입력 데이터 설정
        interpreter.set_tensor(input_details[0]['index'], processed_image)

        # 예측 시작 시간 기록
        start_time = time.time()

        # 예측 수행 (TFLite 모델)
        interpreter.invoke()

        # 예측 종료 시간 기록
        end_time = time.time()

        # 예측 결과 가져오기
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # 클래스 라벨
        class_labels = ['30', '40', '50', '60', '70', '80', '90']

        # 예측 결과 해석
        predicted_class_index = np.argmax(output_data, axis=1)[0]
        predicted_class_label = class_labels[predicted_class_index]

        # 걸린 시간 계산
        elapsed_time = end_time - start_time

        print(f"The predicted speed limit is: {predicted_class_label}")
        print(f"Time taken for prediction: {elapsed_time:.4f} seconds")
    else:
        print("No valid bounding box found.")

# Example usage
detect_and_predict_tflite(weights='128_128.pt', img_path='image1.png', imgsz=(128, 128), crop_save_path='cropped_result.jpg')

