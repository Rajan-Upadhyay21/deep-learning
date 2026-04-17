import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_model():
    return keras.Sequential([
        layers.Input(shape=(4,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])


def main():
    x_train = np.random.rand(200, 4).astype("float32")
    y_train = (x_train.mean(axis=1) > 0.5).astype("float32")

    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=0)

    sample = np.array([[0.7, 0.6, 0.8, 0.9]], dtype="float32")
    prediction = model.predict(sample, verbose=0)[0][0]
    label = "Positive" if prediction >= 0.5 else "Negative"

    print("Sample:", sample)
    print(f"Prediction: {prediction:.4f}")
    print("Label:", label)


if __name__ == "__main__":
    main()
