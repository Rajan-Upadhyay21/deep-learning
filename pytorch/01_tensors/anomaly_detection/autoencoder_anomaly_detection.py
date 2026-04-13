import torch
import torch.nn as nn
import torch.optim as optim

normal_data = torch.randn(200, 10)
anomaly_data = torch.randn(10, 10) * 4 + 6

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 6),
            nn.ReLU(),
            nn.Linear(6, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 6),
            nn.ReLU(),
            nn.Linear(6, 10)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    reconstructed = model(normal_data)
    loss = criterion(reconstructed, normal_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    reconstruction_normal = model(normal_data)
    reconstruction_anomaly = model(anomaly_data)

    normal_error = torch.mean((normal_data - reconstruction_normal) ** 2, dim=1)
    anomaly_error = torch.mean((anomaly_data - reconstruction_anomaly) ** 2, dim=1)

print("Average Normal Reconstruction Error:", normal_error.mean().item())
print("Average Anomaly Reconstruction Error:", anomaly_error.mean().item())
