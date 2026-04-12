import torch

y_true = torch.tensor([3.0, -0.5, 2.0, 7.0])
y_pred = torch.tensor([2.5, 0.0, 2.1, 7.8])

mse = torch.mean((y_true - y_pred) ** 2)
mae = torch.mean(torch.abs(y_true - y_pred))
rmse = torch.sqrt(mse)

print("MSE:", mse.item())
print("MAE:", mae.item())
print("RMSE:", rmse.item())
