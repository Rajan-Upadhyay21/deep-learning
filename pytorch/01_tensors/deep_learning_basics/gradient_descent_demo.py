import torch

x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = 2 * x + 1

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

learning_rate = 0.01

for epoch in range(500):
    y_pred = w * x + b
    loss = ((y_pred - y) ** 2).mean()

    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    w.grad.zero_()
    b.grad.zero_()

print("Learned weight:", w.item())
print("Learned bias:", b.item())
print("Final loss:", loss.item())
