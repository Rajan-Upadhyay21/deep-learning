import torch
from torch.utils.data import Dataset, DataLoader

class SimpleAudioDataset(Dataset):
    def __init__(self, num_samples=20, signal_length=16000, num_classes=3):
        self.audio = torch.randn(num_samples, signal_length)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        return self.audio[idx], self.labels[idx]

dataset = SimpleAudioDataset()
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch_audio, batch_labels in loader:
    print("Batch Audio Shape:", batch_audio.shape)
    print("Batch Labels:", batch_labels)
    break
