import tensorflow as tf
from tensorflow import keras
import numpy as np

from sentiment_transformer import build_sentiment_model


def create_dummy_dataset(samples=500, max_length=100, vocab_size=10000):
    x = np.random.randint(1, vocab_size, size=(samples, max_length))
    y = np.random.randint(0, 2, size=(samples, 1))
    return x, y


def main():
    max_length = 100
    vocab_size = 10000

    x_train, y_train = create_dummy_dataset(
        samples=1000,
        max_length=max_length,
        vocab_size=vocab_size,
    )

    model = build_sentiment_model(vocab_size=vocab_size, max_length=max_length)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        x_train,
        y_train,
        epochs=3,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
    )

    model.save("sentiment_transformer_model.keras")
    print("Model saved successfully.")


if __name__ == "__main__":
    main()
