import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 10
data_dim = 2

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, data_dim)
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
g_optimizer = optim.Adam(G.parameters(), lr=0.001)
d_optimizer = optim.Adam(D.parameters(), lr=0.001)

for epoch in range(200):
    real_data = torch.randn(32, data_dim).to(device)
    real_labels = torch.ones(32, 1).to(device)
    fake_labels = torch.zeros(32, 1).to(device)

    z = torch.randn(32, latent_dim).to(device)
    fake_data = G(z)

    d_real = D(real_data)
    d_fake = D(fake_data.detach())

    d_loss_real = criterion(d_real, real_labels)
    d_loss_fake = criterion(d_fake, fake_labels)
    d_loss = d_loss_real + d_loss_fake

    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    z = torch.randn(32, latent_dim).to(device)
    generated_data = G(z)
    g_loss = criterion(D(generated_data), real_labels)

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
