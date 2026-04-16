import tensorflow as tf

x = tf.constant(5)
y = tf.constant(3)

z = x + y

print("TensorFlow Version:", tf.__version__)
print("Eager Execution Enabled:", tf.executing_eagerly())

print("\nX:")
print(x)

print("\nY:")
print(y)

print("\nZ = X + Y:")
print(z)
print("Value of Z:", z.numpy())
