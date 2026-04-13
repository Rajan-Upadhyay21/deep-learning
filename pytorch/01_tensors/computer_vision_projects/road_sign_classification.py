import torch
import torch.nn as nn

class RoadSignClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
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

model = RoadSignClassifier(num_classes=6)
x = torch.randn(5, 3, 64, 64)
output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print(output)
