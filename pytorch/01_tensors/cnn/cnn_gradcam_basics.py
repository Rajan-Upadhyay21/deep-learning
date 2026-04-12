import torch
import torch.nn as nn

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 3)

    def forward(self, x):
        feat = self.features(x)
        pooled = self.pool(feat).view(x.size(0), -1)
        out = self.fc(pooled)
        return out, feat

model = TinyCNN()
x = torch.randn(1, 3, 32, 32, requires_grad=True)

output, feature_maps = model(x)
target_class = output.argmax(dim=1)

print("Predicted Class:", target_class.item())
print("Feature Maps Shape:", feature_maps.shape)
print("This is a basic Grad-CAM style feature extraction setup.")
