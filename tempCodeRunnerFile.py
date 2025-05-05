import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import os

# 모델 정의
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

# 현재 작업 디렉토리 출력
print("현재 작업 디렉토리:", os.getcwd())

# 모델 구조 시각화 및 저장
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

print("모델 구조가 'model_plot.png' 파일로 저장되었습니다.")

# 모델 요약 정보 출력
model.summary()