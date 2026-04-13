import torch

x = torch.tensor([2.0], requires_grad=True)
y_true = torch.tensor([5.0])

w = torch.tensor([1.5], requires_grad=True)
b = torch.tensor([0.5], requires_grad=True)

y_pred = w * x + b
loss = (y_pred - y_true) ** 2

loss.backward()

print("Prediction:", y_pred.item())
print("Loss:", loss.item())
print("Gradient w.r.t w:", w.grad.item())
print("Gradient w.r.t b:", b.grad.item())
