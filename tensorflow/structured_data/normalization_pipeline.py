import numpy as np


def min_max_scale(x):
    x = np.array(x, dtype="float32")
    x_min = x.min(axis=0, keepdims=True)
    x_max = x.max(axis=0, keepdims=True)
    return (x - x_min) / (x_max - x_min + 1e-8)


def standardize(x):
    x = np.array(x, dtype="float32")
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-8
    return (x - mean) / std


def main():
    data = np.random.rand(5, 4).astype("float32") * 50.0
    scaled = min_max_scale(data)
    standardized = standardize(data)

    print("Min-max scaled:\n", scaled)
    print("Standardized:\n", standardized)


if __name__ == "__main__":
    main()
