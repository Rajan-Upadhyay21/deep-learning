import numpy as np


def mean_squared_reconstruction_error(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)


def mean_absolute_reconstruction_error(original, reconstructed):
    return np.mean(np.abs(original - reconstructed))


if __name__ == "__main__":
    original = np.random.rand(4, 784).astype("float32")
    reconstructed = np.random.rand(4, 784).astype("float32")

    print("MSE:", mean_squared_reconstruction_error(original, reconstructed))
    print("MAE:", mean_absolute_reconstruction_error(original, reconstructed))
