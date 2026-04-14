import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)

optimizer = optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3}
])

print("Learning rates for optimizer parameter groups:")
for idx, group in enumerate(optimizer.param_groups):
    print(f"Group {idx + 1}: lr = {group['lr']}")
