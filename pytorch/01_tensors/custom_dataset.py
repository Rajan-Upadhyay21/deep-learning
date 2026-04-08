import torch
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self):
        self.features = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        self.labels = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

dataset = SimpleDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch_features, batch_labels in loader:
    print("Batch Features:")
    print(batch_features)
    print("Batch Labels:")
    print(batch_labels)
    print("-" * 30)
