import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_model():
    return keras.Sequential([
        layers.Input(shape=(5,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])


def main():
    x_train = np.random.rand(300, 5).astype("float32")
    y_train = (x_train.sum(axis=1) > 2.5).astype("float32")

    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=0)
    model.save("inference_model.keras")

    loaded_model = keras.models.load_model("inference_model.keras")
    new_inputs = np.random.rand(4, 5).astype("float32")
    predictions = loaded_model.predict(new_inputs, verbose=0)

    print("New inputs:\n", new_inputs)
    print("Predictions:\n", predictions.flatten())


if __name__ == "__main__":
    main()
