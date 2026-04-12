import torch
import torch.nn as nn
import torch.optim as optim

x = torch.randn(100, 4)
y = torch.sum(x, dim=1, keepdim=True)

model = nn.Linear(4, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

for epoch in range(15):
    pred = model(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, LR: {current_lr}")
