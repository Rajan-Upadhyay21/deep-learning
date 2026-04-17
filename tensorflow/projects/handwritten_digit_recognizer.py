import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model():
    return keras.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Rescaling(1.0 / 255),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax"),
    ])


def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.1, verbose=1)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
