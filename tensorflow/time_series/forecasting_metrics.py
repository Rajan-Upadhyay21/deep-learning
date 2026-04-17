import numpy as np


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def mean_squared_error(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    y_true = np.array([10.0, 12.0, 14.0, 16.0])
    y_pred = np.array([9.5, 12.5, 13.0, 15.8])

    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("MSE:", mean_squared_error(y_true, y_pred))
    print("RMSE:", root_mean_squared_error(y_true, y_pred))


if __name__ == "__main__":
    main()
