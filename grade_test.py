import os
import cv2
from glob import glob

from run import detect_and_predict_tflite  # run.py에 정의된 함수 import

# GTSDB → CNN 클래스 매핑
gtsdb_to_cnn_class = {
    1: 0,  # 30
    2: 1,  # 50
    3: 2,  # 60
    4: 3,  # 70
    5: 4,  # 80
    6: 5,  # 100
    7: 6   # 120
}

def select_main_label(label_lines):
    # 중심에 가까운 객체 선택 (또는 면적이 가장 큰 객체로 변경 가능)
    centers = []
    for line in label_lines:
        parts = line.strip().split()
        cls = int(parts[0])
        x, y = float(parts[1]), float(parts[2])
        distance = (x - 0.5)**2 + (y - 0.5)**2
        centers.append((distance, cls))
    centers.sort()
    return centers[0][1]  # 가장 중심에 가까운 클래스 반환

# 평가 디렉토리
image_dir = "/Users/hyunoh/Documents/vscode/embedded/traffic_sign_recognition/dataset/new_yolo_dataset/images_yolo"
label_dir = "/Users/hyunoh/Documents/vscode/embedded/traffic_sign_recognition/dataset/new_yolo_dataset/labels_yolo"

image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
correct = 0
total = 0
correct_log = []
incorrect_log = []


for img_path in image_paths:
    base = os.path.basename(img_path)
    label_path = os.path.join(label_dir, base + ".txt")

    if not os.path.exists(label_path):
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()
        if not lines:
            continue
        gt_class_raw = select_main_label(lines)

    if gt_class_raw not in gtsdb_to_cnn_class:
        continue  # CNN이 처리할 수 없는 클래스

    gt_class = gtsdb_to_cnn_class[gt_class_raw]

    pred_label, bbox = detect_and_predict_tflite(
        weights='128_128.pt',
        img_path=img_path,
        crop_save_path='temp_crop.jpg'
    )

    if pred_label is None or bbox is None:
        continue

    pred_class = int(pred_label) // 10 - 3
    is_correct = (pred_class == gt_class)

    # 시각화용 저장
    boxed_img = cv2.imread(img_path)
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(boxed_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 저장 경로 생성
    boxed_dir = "boxed_results"
    os.makedirs(boxed_dir, exist_ok=True)
    save_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(boxed_dir, f"boxed_{save_name}"), boxed_img)

    log_entry = f"{os.path.basename(img_path)}\tGT: {gt_class}\tPred: {pred_class}\tResult: {'O' if is_correct else 'X'}"

    if is_correct:
        correct_log.append(log_entry)
        correct += 1
    else:
        incorrect_log.append(log_entry)

    total += 1


with open("correct.txt", "w") as f:
    f.write("\n".join(correct_log))

with open("incorrect.txt", "w") as f:
    f.write("\n".join(incorrect_log))



print(f"\n총 평가 개수: {total}")
print(f"정답 수: {correct}")
print(f"정확도: {correct / total:.4f}")
