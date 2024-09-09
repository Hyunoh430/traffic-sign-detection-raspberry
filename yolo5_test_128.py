import torch
import cv2
import time  # 시간 측정용
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from pathlib import Path

def detect_image(weights, img_path, device='cpu', imgsz=(128, 128), conf_thres=0.25, iou_thres=0.45, save_path='result.jpg', crop_save_path='cropped_result.jpg'):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    model.warmup(imgsz=(1, 3, *imgsz))  # 128x128 입력 크기로 워밍업

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

    # Inference
    start_time = time.time()  # 추론 시작 시간 기록
    pred = model(img)
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
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

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

    # Save the original image with bounding boxes
    cv2.imwrite(save_path, img0)
    print(f"Result saved to {save_path}")

    return img0

# Example usage
result_img = detect_image(weights='128_128.pt', img_path='image1.jpg', imgsz=(128, 128), save_path='result_image.jpg', crop_save_path='cropped_result.jpg')
cv2.imshow('Detection', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
