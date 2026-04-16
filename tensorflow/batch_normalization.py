import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_bn_model():
    model = keras.Sequential([
        layers.Input(shape=(20,)),
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dense(32),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    return model


if __name__ == "__main__":
    model = build_bn_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
