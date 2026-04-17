import numpy as np


def min_max_normalize(data: np.ndarray) -> np.ndarray:
    data_min = data.min(axis=0, keepdims=True)
    data_max = data.max(axis=0, keepdims=True)
    return (data - data_min) / (data_max - data_min + 1e-8)


def preprocess_input(raw_samples):
    array = np.array(raw_samples, dtype="float32")
    normalized = min_max_normalize(array)
    return normalized


def main():
    raw_samples = [
        [10, 200, 0.5],
        [15, 180, 0.8],
        [8, 220, 0.3],
    ]
    processed = preprocess_input(raw_samples)
    print("Processed inputs:\n", processed)


if __name__ == "__main__":
    main()
