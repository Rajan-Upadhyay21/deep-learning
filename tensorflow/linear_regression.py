import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0, 10, 100).reshape(-1, 1).astype(np.float32)
y = 2 * X + 1 + np.random.randn(100, 1).astype(np.float32) * 0.8

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss="mse")
model.fit(X, y, epochs=100, verbose=0)

predicted = model.predict(X, verbose=0)

print("Learned Weight and Bias:")
weights, bias = model.layers[0].get_weights()
print("Weight:", weights[0][0])
print("Bias:", bias[0])

plt.scatter(X, y, label="Data")
plt.plot(X, predicted, label="Fitted Line")
plt.legend()
plt.title("Linear Regression with TensorFlow")
plt.show()
