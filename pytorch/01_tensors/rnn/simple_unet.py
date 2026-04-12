import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 16)
        self.pool = nn.MaxPool2d(2)
        self.down2 = DoubleConv(16, 32)

        self.up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.pool(x1)
        x3 = self.down2(x2)

        x4 = self.up(x3)
        x5 = torch.cat([x4, x1], dim=1)
        out = self.final_conv(x5)
        return out

model = SimpleUNet()
x = torch.randn(2, 3, 128, 128)
output = model(x)

print("Input Shape:", x.shape)
print("Output Shape:", output.shape)
