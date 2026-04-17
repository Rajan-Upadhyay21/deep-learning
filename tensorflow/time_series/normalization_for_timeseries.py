import numpy as np


def standardize(series):
    series = np.array(series, dtype="float32")
    mean = np.mean(series)
    std = np.std(series) + 1e-8
    normalized = (series - mean) / std
    return normalized, mean, std


def min_max_scale(series):
    series = np.array(series, dtype="float32")
    min_value = np.min(series)
    max_value = np.max(series)
    scaled = (series - min_value) / (max_value - min_value + 1e-8)
    return scaled, min_value, max_value


def main():
    series = np.array([10, 15, 20, 18, 25, 30], dtype="float32")
    standardized, mean, std = standardize(series)
    scaled, min_value, max_value = min_max_scale(series)

    print("Standardized:", standardized)
    print("Mean:", mean, "Std:", std)
    print("Scaled:", scaled)
    print("Min:", min_value, "Max:", max_value)


if __name__ == "__main__":
    main()
