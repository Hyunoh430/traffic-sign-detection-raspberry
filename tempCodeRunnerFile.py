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

    pred_label = detect_and_predict_tflite(
        weights='128_128.pt',
        img_path=img_path,
        crop_save_path='temp_crop.jpg'
    )

    if pred_label is None:
        continue

    pred_class = int(pred_label) // 10 - 3

    if pred_class == gt_class:
        correct += 1
    total += 1

print(f"\n총 평가 개수: {total}")
print(f"정답 수: {correct}")
print(f"정확도: {correct / total:.4f}")
