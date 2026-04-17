import numpy as np


class SequenceWindowGenerator:
    def __init__(self, window_size=12, horizon=1):
        self.window_size = window_size
        self.horizon = horizon

    def transform(self, series):
        series = np.array(series, dtype="float32")
        x, y = [], []

        for i in range(len(series) - self.window_size - self.horizon + 1):
            x.append(series[i:i + self.window_size])
            y.append(series[i + self.window_size:i + self.window_size + self.horizon])

        return np.array(x, dtype="float32"), np.array(y, dtype="float32")


def main():
    generator = SequenceWindowGenerator(window_size=5, horizon=2)
    series = np.arange(20, dtype="float32")
    x, y = generator.transform(series)

    print("Input windows shape:", x.shape)
    print("Targets shape:", y.shape)


if __name__ == "__main__":
    main()
