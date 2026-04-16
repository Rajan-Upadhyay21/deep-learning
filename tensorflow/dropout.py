import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_dropout_model():
    model = keras.Sequential([
        layers.Input(shape=(50,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])
    return model


if __name__ == "__main__":
    model = build_dropout_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
