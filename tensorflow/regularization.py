import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_regularized_model():
    model = keras.Sequential([
        layers.Input(shape=(30,)),
        layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(0.01)),
        layers.Dense(32, activation="relu",
                     kernel_regularizer=regularizers.l1(0.01)),
        layers.Dense(1, activation="sigmoid")
    ])
    return model


if __name__ == "__main__":
    model = build_regularized_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
