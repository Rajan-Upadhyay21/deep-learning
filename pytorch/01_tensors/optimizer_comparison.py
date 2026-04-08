import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

X = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = X.pow(2) + 0.2 * torch.rand(X.size())

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_model(optimizer_class, name, lr=0.01):
    model = SimpleNet()
    criterion = nn.MSELoss()

    if name == "SGD":
        optimizer = optimizer_class(model.parameters(), lr=lr)
    else:
        optimizer = optimizer_class(model.parameters(), lr=lr)

    losses = []

    for _ in range(200):
        predictions = model(X)
        loss = criterion(predictions, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses

sgd_losses = train_model(optim.SGD, "SGD")
momentum_losses = train_model(lambda params, lr: optim.SGD(params, lr=lr, momentum=0.9), "Momentum")
adam_losses = train_model(optim.Adam, "Adam")
adamw_losses = train_model(optim.AdamW, "AdamW")

plt.figure(figsize=(8, 5))
plt.plot(sgd_losses, label="SGD")
plt.plot(momentum_losses, label="Momentum")
plt.plot(adam_losses, label="Adam")
plt.plot(adamw_losses, label="AdamW")
plt.title("Optimizer Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
