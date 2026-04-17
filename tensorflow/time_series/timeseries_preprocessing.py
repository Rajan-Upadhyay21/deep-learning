import numpy as np


def fill_missing_values(series):
    series = np.array(series, dtype="float32")
    mean_value = np.nanmean(series)
    return np.where(np.isnan(series), mean_value, series)


def smooth_series(series, window_size=3):
    series = np.array(series, dtype="float32")
    smoothed = np.convolve(series, np.ones(window_size) / window_size, mode="valid")
    return smoothed


def main():
    raw_series = [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0]
    filled = fill_missing_values(raw_series)
    smoothed = smooth_series(filled, window_size=3)

    print("Filled series:", filled)
    print("Smoothed series:", smoothed)


if __name__ == "__main__":
    main()
