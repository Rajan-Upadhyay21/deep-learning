import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class AccumulatingTrainer:
    def __init__(self, model, optimizer, loss_fn, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.accumulation_steps = accumulation_steps
        self.step_counter = 0
        self.accumulated_grads = [
            tf.Variable(tf.zeros_like(v), trainable=False)
            for v in model.trainable_variables
        ]

    @tf.function
    def train_step(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            preds = self.model(x_batch, training=True)
            loss = self.loss_fn(y_batch, preds) / self.accumulation_steps

        grads = tape.gradient(loss, self.model.trainable_variables)

        for acc_grad, grad in zip(self.accumulated_grads, grads):
            acc_grad.assign_add(grad)

        self.step_counter += 1

        if self.step_counter % self.accumulation_steps == 0:
            self.optimizer.apply_gradients(
                zip(self.accumulated_grads, self.model.trainable_variables)
            )
            for acc_grad in self.accumulated_grads:
                acc_grad.assign(tf.zeros_like(acc_grad))

        return loss


def build_model():
    return keras.Sequential([
        layers.Input(shape=(10,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])


def main():
    x = np.random.rand(128, 10).astype("float32")
    y = (x.sum(axis=1) > 5).astype("float32")
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(8)

    model = build_model()
    _ = model(tf.zeros((1, 10)))
    trainer = AccumulatingTrainer(
        model=model,
        optimizer=keras.optimizers.Adam(1e-3),
        loss_fn=keras.losses.BinaryCrossentropy(),
        accumulation_steps=4
    )

    for epoch in range(2):
        for x_batch, y_batch in dataset:
            loss = trainer.train_step(x_batch, y_batch)
        print(f"Epoch {epoch + 1}: loss={float(loss):.4f}")


if __name__ == "__main__":
    main()
