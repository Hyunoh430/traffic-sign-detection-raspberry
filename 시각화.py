from tensorflow import keras
from tensorflow.keras import layers
import visualkeras
from PIL import ImageFont

#font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)  # 또는 40

# Define the input layer
input_layer = keras.Input(shape=(128, 128, 1), name='input')

# Encoder
x = layers.Conv2D(32, 3, activation='relu')(input_layer)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(128, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(256, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Flatten and Dense layers
x = layers.Flatten()(x)
x = layers.Dense(2048, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(7, activation='softmax')(x)

# Create the model
autoencoder = keras.Model(input_layer, output, name='autoencoder')

# Visualize the model
visualkeras.layered_view(autoencoder, legend=True,  to_file='autoencoder.png')

# If you want to see a summary of the model
autoencoder.summary()