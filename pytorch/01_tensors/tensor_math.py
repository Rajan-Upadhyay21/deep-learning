import torch

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

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
print(torch.matmul(a, b))

print("\nBroadcasting Example:")
c = torch.tensor([1.0, 2.0])
print(a + c)
