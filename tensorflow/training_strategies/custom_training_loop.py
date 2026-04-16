import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_dataset(samples=512, features=20):
    x = np.random.rand(samples, features).astype("float32")
    y = (x.sum(axis=1) > features / 2).astype("float32")
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(32)


def build_model(input_dim=20):
    return keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])


def main():
    dataset = create_dataset()
    model = build_model()

    loss_fn = keras.losses.BinaryCrossentropy()
    optimizer = keras.optimizers.Adam(1e-3)
    train_acc = keras.metrics.BinaryAccuracy()

    epochs = 3
    for epoch in range(epochs):
        train_acc.reset_state()
        epoch_loss = 0.0
        batches = 0

        for x_batch, y_batch in dataset:
            with tf.GradientTape() as tape:
                preds = model(x_batch, training=True)
                loss = loss_fn(y_batch, preds)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_acc.update_state(y_batch, preds)
            epoch_loss += float(loss)
            batches += 1

        print(
            f"Epoch {epoch + 1}: "
            f"loss={epoch_loss / batches:.4f}, "
            f"accuracy={train_acc.result().numpy():.4f}"
        )


if __name__ == "__main__":
    main()
