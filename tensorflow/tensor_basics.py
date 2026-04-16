
Code files:

## `tensor_basics.py`

```python
import tensorflow as tf

tensor_1d = tf.constant([1, 2, 3, 4])
tensor_2d = tf.constant([[1, 2], [3, 4]])
tensor_zeros = tf.zeros((2, 3))
tensor_ones = tf.ones((3, 2))
tensor_random = tf.random.uniform((2, 2))

print("1D Tensor:")
print(tensor_1d)

print("\n2D Tensor:")
print(tensor_2d)

print("\nZeros Tensor:")
print(tensor_zeros)

print("\nOnes Tensor:")
print(tensor_ones)

print("\nRandom Tensor:")
print(tensor_random)

print("\nShape of 2D Tensor:")
print(tensor_2d.shape)

print("\nFirst element of 1D Tensor:")
print(tensor_1d[0])

print("\nFirst row of 2D Tensor:")
print(tensor_2d[0])

reshaped = tf.reshape(tensor_1d, (2, 2))
print("\nReshaped Tensor:")
print(reshaped)
