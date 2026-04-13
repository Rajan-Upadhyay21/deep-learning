import torch
import torch.nn as nn

x = torch.randn(4, 6)

batch_norm = nn.BatchNorm1d(6)
output = batch_norm(x)

print("Input:")
print(x)

print("\nAfter Batch Normalization:")
print(output)
