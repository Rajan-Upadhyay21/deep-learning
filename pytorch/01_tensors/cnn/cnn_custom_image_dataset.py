import torch
from torch.utils.data import Dataset, DataLoader

class CustomImageDataset(Dataset):
    def __init__(self, num_samples=20):
        self.images = torch.randn(num_samples, 3, 32, 32)
        self.labels = torch.randint(0, 5, (num_samples,))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

dataset = CustomImageDataset()
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for images, labels in loader:
    print("Batch image shape:", images.shape)
    print("Batch labels:", labels)
    break
