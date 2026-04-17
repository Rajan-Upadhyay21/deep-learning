import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model():
    return keras.Sequential([
        layers.Input(shape=(6,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])


def preprocess(raw_inputs):
    return np.array(raw_inputs, dtype="float32")


def postprocess(predictions, threshold=0.5):
    labels = ["Positive" if value >= threshold else "Negative" for value in predictions.flatten()]
    return labels


def main():
    x = np.random.rand(200, 6).astype("float32")
    y = (x.mean(axis=1) > 0.5).astype("float32")

    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(x, y, epochs=2, batch_size=32, verbose=0)

    raw_samples = [
        [0.8, 0.7, 0.9, 0.6, 0.8, 0.7],
        [0.1, 0.3, 0.2, 0.4, 0.2, 0.1],
    ]

    processed = preprocess(raw_samples)
    predictions = model.predict(processed, verbose=0)
    labels = postprocess(predictions)

    for idx, (score, label) in enumerate(zip(predictions.flatten(), labels), start=1):
        print(f"Sample {idx}: {label} ({score:.4f})")


if __name__ == "__main__":
    main()
