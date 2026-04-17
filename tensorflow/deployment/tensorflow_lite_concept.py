import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def build_model():
    return keras.Sequential([
        layers.Input(shape=(4,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])


def main():
    x = np.random.rand(200, 4).astype("float32")
    y = (x.mean(axis=1) > 0.5).astype("float32")

    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(x, y, epochs=2, batch_size=16, verbose=0)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open("model.tflite", "wb") as file:
        file.write(tflite_model)

    print("TensorFlow Lite model saved to model.tflite")
    print("Model size in bytes:", len(tflite_model))


if __name__ == "__main__":
    main()
