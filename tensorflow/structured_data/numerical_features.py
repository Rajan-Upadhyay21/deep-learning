import numpy as np
from tensorflow.keras.layers import Normalization


def create_numerical_data(samples=10, features=3):
    return np.random.rand(samples, features).astype("float32") * 100.0


def main():
    x = create_numerical_data()
    normalizer = Normalization()
    normalizer.adapt(x)

    normalized = normalizer(x)
    print("Original:\n", x[:3])
    print("Normalized:\n", normalized.numpy()[:3])


if __name__ == "__main__":
    main()
