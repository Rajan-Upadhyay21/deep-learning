import torch
import torch.nn as nn

class FaceMaskDetector(nn.Module):
    def __init__(self):
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
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)

model = FaceMaskDetector()
x = torch.randn(3, 3, 64, 64)
output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print(output)
