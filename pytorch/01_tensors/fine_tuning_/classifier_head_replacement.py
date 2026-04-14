from torchvision import models
import torch.nn as nn

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

print("Original classifier:")
print(model.classifier)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 4)

print("\nUpdated classifier:")
print(model.classifier)
