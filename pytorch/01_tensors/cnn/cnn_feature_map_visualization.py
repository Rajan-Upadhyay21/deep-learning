import torch
import torch.nn as nn

conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
x = torch.randn(1, 3, 32, 32)

feature_maps = conv(x)

print("Input Shape:", x.shape)
print("Feature Maps Shape:", feature_maps.shape)

for i in range(feature_maps.shape[1]):
    print(f"Feature map {i + 1} shape:", feature_maps[0, i].shape)
