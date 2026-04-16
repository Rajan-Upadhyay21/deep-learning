import tensorflow as tf

x = tf.Variable(2.0)

with tf.GradientTape() as tape:
    y = x ** 2 + 3 * x + 1

gradient = tape.gradient(y, x)

print("X:")
print(x.numpy())

print("\nY:")
print(y.numpy())

print("\nGradient of Y with respect to X:")
print(gradient.numpy())
