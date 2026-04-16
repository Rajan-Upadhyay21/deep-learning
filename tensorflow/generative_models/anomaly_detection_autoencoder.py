import numpy as np
from autoencoder import build_autoencoder


def compute_reconstruction_error(original, reconstructed):
    return np.mean((original - reconstructed) ** 2, axis=1)


def main():
    autoencoder, _ = build_autoencoder()
    samples = np.random.rand(5, 784).astype("float32")

    reconstructed = autoencoder.predict(samples, verbose=0)
    errors = compute_reconstruction_error(samples, reconstructed)

    for index, error in enumerate(errors, start=1):
        print(f"Sample {index} reconstruction error: {error:.6f}")


if __name__ == "__main__":
    main()
