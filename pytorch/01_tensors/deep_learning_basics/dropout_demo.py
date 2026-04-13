import torch
import torch.nn as nn

x = torch.randn(3, 5)

dropout = nn.Dropout(p=0.5)
dropout.train()

output = dropout(x)

print("Input:")
print(x)

print("\nAfter Dropout:")
print(output)
