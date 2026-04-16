import tensorflow as tf
import numpy as np

X = np.random.randn(100, 5).astype(np.float32)
y = np.random.randint(0, 2, size=(100,))

dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(buffer_size=100).batch(16).prefetch(tf.data.AUTOTUNE)

for batch_x, batch_y in dataset.take(1):
    print("Batch Features Shape:", batch_x.shape)
    print("Batch Labels Shape:", batch_y.shape)
