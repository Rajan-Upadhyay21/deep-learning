import torch

outputs = torch.tensor([
    [2.1, 0.5, 0.3],
    [0.2, 1.8, 0.6],
    [0.1, 0.3, 2.4],
    [1.7, 0.4, 0.2]
])

labels = torch.tensor([0, 1, 2, 0])

predictions = outputs.argmax(dim=1)
accuracy = (predictions == labels).float().mean()

print("Predictions:", predictions)
print("Labels:", labels)
print("Accuracy:", accuracy.item())
