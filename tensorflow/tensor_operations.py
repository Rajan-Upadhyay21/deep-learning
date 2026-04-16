import tensorflow as tf

a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

print("Tensor A:")
print(a)

print("\nTensor B:")
print(b)

print("\nAddition:")
print(a + b)

print("\nSubtraction:")
print(a - b)

print("\nElement-wise Multiplication:")
print(a * b)

print("\nElement-wise Division:")
print(a / b)

print("\nMatrix Multiplication:")
print(tf.matmul(a, b))

print("\nBroadcasting Example:")
c = tf.constant([1.0, 2.0])
print(a + c)
