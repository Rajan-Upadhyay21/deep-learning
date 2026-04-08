import torch
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 200)

relu = torch.relu(x)
sigmoid = torch.sigmoid(x)
tanh = torch.tanh(x)
gelu = torch.nn.functional.gelu(x)
silu = torch.nn.functional.silu(x)

plt.figure(figsize=(8, 5))
plt.plot(x.numpy(), relu.numpy(), label="ReLU")
plt.plot(x.numpy(), sigmoid.numpy(), label="Sigmoid")
plt.plot(x.numpy(), tanh.numpy(), label="Tanh")
plt.plot(x.numpy(), gelu.numpy(), label="GELU")
plt.plot(x.numpy(), silu.numpy(), label="SiLU")
plt.title("Activation Functions")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()
