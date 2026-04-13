import torch
import torch.nn as nn

mse = nn.MSELoss()
bce = nn.BCELoss()
cross_entropy = nn.CrossEntropyLoss()

pred_reg = torch.tensor([[2.5], [0.0], [2.1]])
target_reg = torch.tensor([[3.0], [-0.5], [2.0]])

pred_bce = torch.tensor([[0.9], [0.2], [0.8]])
target_bce = torch.tensor([[1.0], [0.0], [1.0]])

pred_ce = torch.tensor([[2.0, 1.0, 0.1],
                        [0.5, 2.5, 0.3]])
target_ce = torch.tensor([0, 1])

print("MSE Loss:", mse(pred_reg, target_reg).item())
print("BCE Loss:", bce(pred_bce, target_bce).item())
print("Cross Entropy Loss:", cross_entropy(pred_ce, target_ce).item())
