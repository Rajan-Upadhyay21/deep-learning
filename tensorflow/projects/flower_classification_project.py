import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_model(num_classes=5):
    return keras.Sequential([
        layers.Input(shape=(96, 96, 3)),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])


def create_dummy_data(samples=180, num_classes=5):
    x = np.random.randint(0, 256, size=(samples, 96, 96, 3)).astype("float32")
    y = np.random.randint(0, num_classes, size=(samples,))
    return x, y


def main():
    x, y = create_dummy_data()
    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x, y, epochs=3, batch_size=16, validation_split=0.2, verbose=1)


if __name__ == "__main__":
    main()
