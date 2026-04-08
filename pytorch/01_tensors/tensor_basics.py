import torch

tensor_1d = torch.tensor([1, 2, 3, 4])
tensor_2d = torch.tensor([[1, 2], [3, 4]])
tensor_zeros = torch.zeros((2, 3))
tensor_ones = torch.ones((3, 2))
tensor_random = torch.rand((2, 2))

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

reshaped = tensor_1d.reshape(2, 2)
print("\nReshaped Tensor:")
print(reshaped)
