import torch

logits = torch.tensor([
    [2.1, 0.4, 0.2],
    [0.3, 1.8, 0.1],
    [0.2, 0.5, 2.2],
    [1.5, 0.7, 0.2]
])

targets = torch.tensor([0, 1, 2, 0])

preds = torch.argmax(logits, dim=1)

accuracy = (preds == targets).float().mean()

print("Predictions:", preds)
print("Targets:", targets)
print("Accuracy:", accuracy.item())
