from torchvision import models
import torch.nn as nn

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 3)

for param in model.fc.parameters():
    param.requires_grad = True

frozen_count = 0
trainable_count = 0

for param in model.parameters():
    if param.requires_grad:
        trainable_count += param.numel()
    else:
        frozen_count += param.numel()

print("Frozen parameters:", frozen_count)
print("Trainable parameters:", trainable_count)
