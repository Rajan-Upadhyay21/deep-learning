import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_model(vocab_size=5000, sequence_length=100, num_classes=3):
    return keras.Sequential([
        layers.Input(shape=(sequence_length,)),
        layers.Embedding(vocab_size, 128),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])


def create_dummy_data(samples=400, sequence_length=100, vocab_size=5000, num_classes=3):
    x = np.random.randint(1, vocab_size, size=(samples, sequence_length))
    y = np.random.randint(0, num_classes, size=(samples,))
    return x, y


def main():
    x, y = create_dummy_data()
    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x, y, epochs=3, batch_size=32, validation_split=0.2, verbose=1)


if __name__ == "__main__":
    main()
