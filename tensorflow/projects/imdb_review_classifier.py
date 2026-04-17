from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences


def build_model(vocab_size=10000, sequence_length=200):
    return keras.Sequential([
        layers.Input(shape=(sequence_length,)),
        layers.Embedding(vocab_size, 128),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])


def main():
    vocab_size = 10000
    sequence_length = 200

    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)
    x_train = pad_sequences(x_train, maxlen=sequence_length)
    x_test = pad_sequences(x_test, maxlen=sequence_length)

    model = build_model(vocab_size=vocab_size, sequence_length=sequence_length)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x_train[:10000], y_train[:10000], epochs=2, batch_size=64, validation_split=0.2, verbose=1)

    loss, acc = model.evaluate(x_test[:2000], y_test[:2000], verbose=0)
    print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
