import numpy as np
from variational_autoencoder import build_variational_autoencoder


def interpolate_vectors(start, end, steps=10):
    alphas = np.linspace(0.0, 1.0, steps)
    return np.array([(1 - alpha) * start + alpha * end for alpha in alphas], dtype="float32")


def main():
    _, _, decoder = build_variational_autoencoder(latent_dim=16)

    start = np.random.normal(size=(16,))
    end = np.random.normal(size=(16,))
    interpolated = interpolate_vectors(start, end, steps=8)

    outputs = decoder.predict(interpolated, verbose=0)
    print("Interpolated output shape:", outputs.shape)


if __name__ == "__main__":
    main()
