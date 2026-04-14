import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)

optimizer = optim.Adam(model.parameters(), lr=0.001)

checkpoint = {
    "epoch": 1,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "description": "Fine-tuning checkpoint for ResNet18"
}

torch.save(checkpoint, "fine_tuning_checkpoint.pth")

print("Checkpoint saved as fine_tuning_checkpoint.pth")
