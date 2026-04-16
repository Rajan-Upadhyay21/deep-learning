import tensorflow as tf
import numpy as np

from variational_autoencoder import build_variational_autoencoder


def sample_latent_vectors(num_samples: int = 5, latent_dim: int = 16):
    return np.random.normal(size=(num_samples, latent_dim)).astype("float32")


def main():
    _, _, decoder = build_variational_autoencoder(latent_dim=16)
    latent_samples = sample_latent_vectors(num_samples=5, latent_dim=16)
    generated = decoder.predict(latent_samples, verbose=0)
    print("Generated samples shape:", generated.shape)


if __name__ == "__main__":
    main()
