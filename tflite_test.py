import numpy as np
import cv2
import tensorflow as tf
import time  # 시간 측정을 위한 모듈

# TensorFlow Lite 인터프리터 사용
interpreter = tf.lite.Interpreter(model_path='speed_sign_model.tflite')

# 인터프리터 초기화
interpreter.allocate_tensors()

# 입력 및 출력 텐서 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 예측할 이미지 경로
image_path = 'cropped_0.jpg'  # 예측할 이미지 파일의 경로를 입력

# 이미지 전처리 함수
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0  # 모델에 맞춰 스케일링
    img = img.reshape(1, 128, 128, 1)  # 배치 크기 포함하여 4차원으로 변환
    return img

# 이미지 전처리
processed_image = preprocess_image(image_path)

# TFLite 모델에 입력 데이터 설정
interpreter.set_tensor(input_details[0]['index'], processed_image)

# 예측 시작 시간 기록
start_time = time.time()

# 예측 수행
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
