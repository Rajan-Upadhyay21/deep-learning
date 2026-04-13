import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 3)
)

x = torch.randn(5, 4)
output = model(x)

print("Input Shape:", x.shape)
print("Output Shape:", output.shape)
print(output)
