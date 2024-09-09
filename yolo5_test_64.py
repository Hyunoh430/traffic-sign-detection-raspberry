import torch
import cv2
import time  # 시간 측정용
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from pathlib import Path

def detect_image(weights, img_path, device='cpu', imgsz=(64, 64), conf_thres=0.25, iou_thres=0.45, save_path='result.jpg'):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    model.warmup(imgsz=(1, 3, *imgsz))  # 64x64 입력 크기로 워밍업

    # Load and preprocess image
    img0 = cv2.imread(img_path)  # (H, W, C) 형태로 이미지 로드
    img = cv2.resize(img0, imgsz)  # 이미지를 64x64 크기로 리사이즈 (H, W, C)
    
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

    # Draw boxes and labels
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the result image
    cv2.imwrite(save_path, img0)
    print(f"Result saved to {save_path}")

    return img0

# Example usage
result_img = detect_image(weights='yolov5s.pt', img_path='image1.jpg', imgsz=(64, 64), save_path='result_image.jpg')
cv2.imshow('Detection', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
