import numpy as np


def fill_missing_with_mean(data):
    data = np.array(data, dtype="float32")
    col_means = np.nanmean(data, axis=0)
    indices = np.where(np.isnan(data))
    data[indices] = np.take(col_means, indices[1])
    return data


def main():
    raw = np.array([
        [1.0, 2.0, np.nan],
        [2.0, np.nan, 5.0],
        [3.0, 4.0, 6.0],
    ], dtype="float32")

    filled = fill_missing_with_mean(raw)
    print("Filled data:\n", filled)


if __name__ == "__main__":
    main()
