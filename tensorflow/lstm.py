import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_model(vocab_size=5000, embedding_dim=64, max_length=100):
    model = keras.Sequential([
        layers.Input(shape=(max_length,)),
        layers.Embedding(vocab_size, embedding_dim),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    return model


if __name__ == "__main__":
    model = build_lstm_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
