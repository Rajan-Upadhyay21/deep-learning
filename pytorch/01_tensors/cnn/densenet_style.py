import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        new_features = self.layer(x)
        return torch.cat([x, new_features], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        layers = []
        current_channels = in_channels

        for _ in range(num_layers):
            layers.append(DenseLayer(current_channels, growth_rate))
            current_channels += growth_rate

        self.block = nn.Sequential(*layers)
        self.out_channels = current_channels

    def forward(self, x):
        return self.block(x)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)


class DenseNetStyle(nn.Module):
    def __init__(self, num_classes=10, growth_rate=16):
        super().__init__()

        self.stem = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)

        self.block1 = DenseBlock(32, growth_rate, num_layers=3)
        self.trans1 = TransitionLayer(self.block1.out_channels, 64)

        self.block2 = DenseBlock(64, growth_rate, num_layers=3)
        self.trans2 = TransitionLayer(self.block2.out_channels, 96)

        self.block3 = DenseBlock(96, growth_rate, num_layers=3)

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(self.block3.out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.block3.out_channels, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


model = DenseNetStyle(num_classes=10)
x = torch.randn(2, 3, 32, 32)
output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print(model)
