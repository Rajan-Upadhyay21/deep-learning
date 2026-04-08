import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1

print("X:")
print(x)

print("\nY:")
print(y)

y.backward()

print("\nGradient of x:")
print(x.grad)

x.grad.zero_()

z = x * 5
with torch.no_grad():
    no_grad_result = z * 2

print("\nResult with no_grad:")
print(no_grad_result)
