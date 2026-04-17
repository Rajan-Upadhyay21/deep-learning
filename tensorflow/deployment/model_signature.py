import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def build_model():
    return keras.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])


def main():
    x = np.random.rand(100, 3).astype("float32")
    y = (x.sum(axis=1) > 1.5).astype("float32")

    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(x, y, epochs=1, verbose=0)

    export_path = "signature_model"
    model.export(export_path)

    loaded = tf.saved_model.load(export_path)
    print("Available signatures:", list(loaded.signatures.keys()))

    serve_fn = loaded.signatures["serve"]
    sample = tf.constant([[0.2, 0.7, 0.8]], dtype=tf.float32)
    result = serve_fn(sample)
    print("Signature output keys:", list(result.keys()))


if __name__ == "__main__":
    main()
