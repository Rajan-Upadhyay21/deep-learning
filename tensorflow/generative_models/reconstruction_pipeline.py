import numpy as np
from autoencoder import build_autoencoder


def prepare_dummy_data(samples=10, input_dim=784):
    return np.random.rand(samples, input_dim).astype("float32")


def main():
    model, encoder = build_autoencoder()
    data = prepare_dummy_data(samples=6)

    latent_vectors = encoder.predict(data, verbose=0)
    reconstructions = model.predict(data, verbose=0)

    print("Input shape:", data.shape)
    print("Latent shape:", latent_vectors.shape)
    print("Reconstructed shape:", reconstructions.shape)


if __name__ == "__main__":
    main()
