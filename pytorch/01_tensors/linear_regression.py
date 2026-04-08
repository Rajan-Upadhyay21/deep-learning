import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

X = torch.unsqueeze(torch.linspace(0, 10, 100), dim=1)
y = 2 * X + 1 + torch.randn(X.size()) * 0.8

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 500

for epoch in range(epochs):
    predictions = model(X)
    loss = criterion(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

with torch.no_grad():
    predicted = model(X)

print("\nLearned Weight and Bias:")
print("Weight:", model.weight.item())
print("Bias:", model.bias.item())

plt.scatter(X.numpy(), y.numpy(), label="Data")
plt.plot(X.numpy(), predicted.numpy(), label="Fitted Line")
plt.legend()
plt.title("Linear Regression with PyTorch")
plt.show()
