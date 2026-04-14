import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 3)
model = model.to(device)
model.eval()

x = torch.randn(6, 3, 224, 224).to(device)
y = torch.randint(0, 3, (6,)).to(device)

with torch.no_grad():
    outputs = model(x)
    predictions = outputs.argmax(dim=1)
    accuracy = (predictions == y).float().mean().item()

print("Predictions:", predictions.cpu())
print("Targets:", y.cpu())
print("Accuracy:", accuracy)
