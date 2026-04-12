import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(128, 20).to(device)
y = torch.randint(0, 4, (128,)).to(device)

model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 4)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

for epoch in range(3):
    optimizer.zero_grad()

    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        outputs = model(x)
        loss = criterion(outputs, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
