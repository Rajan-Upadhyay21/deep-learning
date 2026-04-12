import torch
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 10)
        )

    def forward(self, x):
        return self.model(x)

model = SmallCNN()
model.eval()

single_image = torch.randn(1, 3, 32, 32)

with torch.no_grad():
    output = model(single_image)
    prediction = output.argmax(dim=1)

print("Model output:", output)
print("Predicted class:", prediction.item())
