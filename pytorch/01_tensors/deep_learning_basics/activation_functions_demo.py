import torch
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 200)

relu = torch.relu(x)
sigmoid = torch.sigmoid(x)
tanh = torch.tanh(x)
leaky_relu = torch.nn.functional.leaky_relu(x)

plt.figure(figsize=(8, 5))
plt.plot(x.numpy(), relu.numpy(), label="ReLU")
plt.plot(x.numpy(), sigmoid.numpy(), label="Sigmoid")
plt.plot(x.numpy(), tanh.numpy(), label="Tanh")
plt.plot(x.numpy(), leaky_relu.numpy(), label="Leaky ReLU")
plt.title("Activation Functions")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()
