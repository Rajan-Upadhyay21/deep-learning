import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:")
print(device)

tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print("\nTensor on CPU:")
print(tensor)

tensor = tensor.to(device)
print("\nTensor moved to selected device:")
print(tensor)

if torch.cuda.is_available():
    print("\nCUDA is available.")
else:
    print("\nCUDA is not available.")
