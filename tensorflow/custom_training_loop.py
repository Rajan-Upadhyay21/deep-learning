import tensorflow as tf
import numpy as np

X = np.random.randn(128, 4).astype(np.float32)
y = np.random.randint(0, 3, size=(128,))

dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(32)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(3)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

for epoch in range(5):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_y in dataset:
        with tf.GradientTape() as tape:
            logits = model(batch_x, training=True)
            loss = loss_fn(batch_y, logits)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        predictions = tf.argmax(logits, axis=1)
        total_correct += tf.reduce_sum(tf.cast(predictions == batch_y, tf.int32)).numpy()
        total_samples += batch_y.shape[0]
        total_loss += loss.numpy()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(list(dataset)):.4f}, Accuracy: {total_correct / total_samples:.4f}")
