import tensorflow as tf
from tensorflow.keras import losses


def demonstrate_losses():
    y_true_binary = tf.constant([[1.0], [0.0], [1.0]])
    y_pred_binary = tf.constant([[0.9], [0.2], [0.7]])

    y_true_multi = tf.constant([0, 2])
    y_pred_multi = tf.constant([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])

    bce = losses.BinaryCrossentropy()
    cce = losses.SparseCategoricalCrossentropy()
    mse = losses.MeanSquaredError()

    print("Binary Crossentropy:", bce(y_true_binary, y_pred_binary).numpy())
    print("Sparse Categorical Crossentropy:", cce(y_true_multi, y_pred_multi).numpy())
    print("Mean Squared Error:", mse(y_true_binary, y_pred_binary).numpy())


if __name__ == "__main__":
    demonstrate_losses()
