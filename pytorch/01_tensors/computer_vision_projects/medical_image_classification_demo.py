import torch
import torch.nn as nn

class MedicalImageClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = MedicalImageClassifier(num_classes=2)
x = torch.randn(4, 1, 64, 64)
output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print(output)
