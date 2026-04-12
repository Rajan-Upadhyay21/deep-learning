import torch
import torch.nn as nn
import torch.optim as optim

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(8 * 16 * 16, 10)
        )

    def forward(self, x):
        return self.model(x)

model = SmallCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

checkpoint = {
    "epoch": 1,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict()
}

torch.save(checkpoint, "cnn_checkpoint.pth")
print("Checkpoint saved as cnn_checkpoint.pth")
