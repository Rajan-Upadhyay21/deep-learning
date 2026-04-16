import numpy as np
from autoencoder import build_autoencoder


def main():
    _, encoder = build_autoencoder(input_dim=784, latent_dim=32)
    data = np.random.rand(8, 784).astype("float32")
    compressed = encoder.predict(data, verbose=0)

    print("Original shape:", data.shape)
    print("Compressed shape:", compressed.shape)


if __name__ == "__main__":
    main()
