import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_model():
    return keras.Sequential([
        layers.Input(shape=(5,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])


def main():
    x_train = np.random.rand(300, 5).astype("float32")
    y_train = (x_train.sum(axis=1) > 2.5).astype("float32")

    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(x_train, y_train, epochs=2, batch_size=32, verbose=0)

    batch_inputs = np.random.rand(10, 5).astype("float32")
    batch_predictions = model.predict(batch_inputs, verbose=0)

    print("Batch input shape:", batch_inputs.shape)
    print("Batch prediction shape:", batch_predictions.shape)
    print("Predictions:\n", batch_predictions.flatten())


if __name__ == "__main__":
    main()
