import torch
import torch.nn as nn
import torch.optim as optim

X = torch.randn(200, 3, 32, 32)
y = torch.randint(0, 10, (200,))

X_train, X_val = X[:160], X[160:]
y_train, y_val = y[:160], y[160:]

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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_loss = float("inf")
patience = 3
counter = 0

for epoch in range(20):
    model.train()
    train_outputs = model(X_train)
    train_loss = criterion(train_outputs, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

    print(f"Epoch {epoch + 1}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered.")
        break
