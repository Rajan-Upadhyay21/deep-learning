
Code files:

## `fine_tuning_overview.py`

```python
from torchvision import models
import torch.nn as nn

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

print("Original final layer:")
print(model.fc)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)

print("\nUpdated final layer:")
print(model.fc)
