import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_encoder(embedding_dim=64):
    return keras.Sequential([
        layers.Input(shape=(64, 64, 3)),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(embedding_dim),
    ])


def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.sum(a * b, axis=1)


def main():
    encoder = build_encoder()
    sample_a = np.random.randint(0, 256, size=(4, 64, 64, 3)).astype("float32")
    sample_b = np.random.randint(0, 256, size=(4, 64, 64, 3)).astype("float32")

    emb_a = encoder.predict(sample_a, verbose=0)
    emb_b = encoder.predict(sample_b, verbose=0)
    scores = cosine_similarity(emb_a, emb_b)

    print("Similarity scores:", scores)


if __name__ == "__main__":
    main()
