import torch
import torch.nn as nn
import torch.optim as optim

x = torch.randn(200, 5)
y = torch.sum(x, dim=1, keepdim=True) + 0.1 * torch.randn(200, 1)

x_train, x_val = x[:160], x[160:]
y_train, y_val = y[:160], y[160:]

model = nn.Sequential(
    nn.Linear(5, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

best_val_loss = float("inf")
patience = 5
counter = 0

for epoch in range(100):
    model.train()
    train_pred = model(x_train)
    train_loss = criterion(train_pred, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(x_val)
        val_loss = criterion(val_pred, y_val)

    print(f"Epoch {epoch+1}: train_loss={train_loss.item():.4f}, val_loss={val_loss.item():.4f}")

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered.")
        break
