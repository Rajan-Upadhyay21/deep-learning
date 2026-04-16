import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_text_classifier(vocab_size=10000, sequence_length=100):
    model = keras.Sequential([
        layers.Input(shape=(sequence_length,)),
        layers.Embedding(vocab_size, 128),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    return model


if __name__ == "__main__":
    model = build_text_classifier()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
