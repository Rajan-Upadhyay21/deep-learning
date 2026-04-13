
Code files:

## `image_classification_project.py`

```python
import torch
import torch.nn as nn

class SimpleImageClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = SimpleImageClassifier(num_classes=5)
x = torch.randn(4, 3, 64, 64)
output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print(output)
