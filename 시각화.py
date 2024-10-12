import tensorflow as tf
from tensorflow.keras import layers, models
import visualkeras
from PIL import ImageFont

# CNN 모델 생성
model = models.Sequential()

# Input layer (128, 128, 1) 크기의 입력
# 시각화를 위해 명시적으로 Input 레이어 추가
model.add(layers.Input(shape=(128, 128, 1), name="Input"))

# Conv Layer 1: 3x3 커널, 32개의 필터
model.add(layers.Conv2D(32, (3, 3), activation='relu', name="Conv1"))
# MaxPooling Layer 1: 2x2 풀링
model.add(layers.MaxPooling2D(pool_size=(2, 2), name="MaxPool1"))

# Conv Layer 2: 3x3 커널, 64개의 필터
model.add(layers.Conv2D(64, (3, 3), activation='relu', name="Conv2"))
# MaxPooling Layer 2: 2x2 풀링
model.add(layers.MaxPooling2D(pool_size=(2, 2), name="MaxPool2"))

# Conv Layer 3: 3x3 커널, 128개의 필터
model.add(layers.Conv2D(128, (3, 3), activation='relu', name="Conv3"))
# MaxPooling Layer 3: 2x2 풀링
model.add(layers.MaxPooling2D(pool_size=(2, 2), name="MaxPool3"))

# Conv Layer 4: 3x3 커널, 256개의 필터
model.add(layers.Conv2D(256, (3, 3), activation='relu', name="Conv4"))
# MaxPooling Layer 4: 2x2 풀링
model.add(layers.MaxPooling2D(pool_size=(2, 2), name="MaxPool4"))

# Flatten layer
model.add(layers.Flatten(name="Flatten"))

# Dense Layer 1: 2048 노드
model.add(layers.Dense(2048, activation='relu', name="Dense1"))

# Dense Layer 2: 512 노드
model.add(layers.Dense(512, activation='relu', name="Dense2"))

# Dense Layer 3: 128 노드
model.add(layers.Dense(128, activation='relu', name="Dense3"))

# Output Layer: 7개의 클래스
model.add(layers.Dense(7, activation='softmax', name="Output"))

# 시각화 폰트 설정
font = ImageFont.load_default()

# Input 레이어 포함한 모델 시각화 및 이미지 저장
visualkeras.layered_view(model, to_file='model_visualization_with_input.png', 
                         legend=True, 
                         font=font).show()
