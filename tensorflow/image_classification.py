import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_image_classifier(input_shape=(32, 32, 3), num_classes=10):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model


if __name__ == "__main__":
    model = build_image_classifier()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
