import numpy as np


def create_trend(time, slope=0.05):
    return slope * time


def create_seasonality(time, period=12, amplitude=1.0):
    return amplitude * np.sin(2 * np.pi * time / period)


def create_series(length=60):
    time = np.arange(length, dtype="float32")
    trend = create_trend(time, slope=0.03)
    seasonality = create_seasonality(time, period=12, amplitude=2.0)
    noise = 0.2 * np.random.randn(length).astype("float32")
    return time, trend + seasonality + noise


def main():
    time, series = create_series()
    print("Time shape:", time.shape)
    print("Series shape:", series.shape)
    print("First 10 values:", series[:10])


if __name__ == "__main__":
    main()
