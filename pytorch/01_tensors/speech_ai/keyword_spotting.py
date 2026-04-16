import torch
import torch.nn as nn
import torch.optim as optim

X = torch.randn(100, 20)
y = torch.randint(0, 2, (100,))

model = nn.Sequential(
    nn.Linear(20, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(15):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        predictions = outputs.argmax(dim=1)
        accuracy = (predictions == y).float().mean().item()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
