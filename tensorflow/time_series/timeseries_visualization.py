import numpy as np
import matplotlib.pyplot as plt


def create_series(length=100):
    time = np.arange(length, dtype="float32")
    series = np.sin(0.1 * time) + 0.2 * np.random.randn(length).astype("float32")
    return time, series


def plot_series(time, series, title="Time Series"):
    plt.figure(figsize=(8, 4))
    plt.plot(time, series)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()


def main():
    time, series = create_series()
    plot_series(time, series, title="Synthetic Time Series")


if __name__ == "__main__":
    main()
