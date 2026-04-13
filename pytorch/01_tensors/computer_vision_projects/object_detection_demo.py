import torch
import torch.nn as nn

class SimpleObjectDetector(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.flatten = nn.Flatten()
        self.class_head = nn.Linear(32, num_classes)
        self.box_head = nn.Linear(32, 4)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        class_logits = self.class_head(x)
        box_coords = self.box_head(x)
        return class_logits, box_coords

model = SimpleObjectDetector(num_classes=3)
x = torch.randn(4, 3, 128, 128)
class_logits, box_coords = model(x)

print("Input shape:", x.shape)
print("Class logits shape:", class_logits.shape)
print("Bounding boxes shape:", box_coords.shape)
