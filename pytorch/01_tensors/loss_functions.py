import torch
import torch.nn as nn

mse = nn.MSELoss()
bce = nn.BCELoss()
cross_entropy = nn.CrossEntropyLoss()
huber = nn.HuberLoss()
kl_div = nn.KLDivLoss(reduction="batchmean")

pred_reg = torch.tensor([[2.5], [0.0], [2.1]])
target_reg = torch.tensor([[3.0], [-0.5], [2.0]])

pred_bce = torch.tensor([[0.9], [0.2], [0.8]])
target_bce = torch.tensor([[1.0], [0.0], [1.0]])

pred_ce = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
target_ce = torch.tensor([0, 1])

pred_kl = torch.log(torch.tensor([[0.6, 0.4], [0.7, 0.3]]))
target_kl = torch.tensor([[0.5, 0.5], [0.6, 0.4]])

print("MSE Loss:")
print(mse(pred_reg, target_reg).item())

print("\nBCE Loss:")
print(bce(pred_bce, target_bce).item())

print("\nCross Entropy Loss:")
print(cross_entropy(pred_ce, target_ce).item())

print("\nHuber Loss:")
print(huber(pred_reg, target_reg).item())

print("\nKL Divergence Loss:")
print(kl_div(pred_kl, target_kl).item())
