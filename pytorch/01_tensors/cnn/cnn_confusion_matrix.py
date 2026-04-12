import torch
from sklearn.metrics import confusion_matrix

y_true = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0])
y_pred = torch.tensor([0, 1, 2, 0, 0, 2, 1, 1])

cm = confusion_matrix(y_true.numpy(), y_pred.numpy())

print("True Labels:", y_true.tolist())
print("Predicted Labels:", y_pred.tolist())
print("\nConfusion Matrix:")
print(cm)
