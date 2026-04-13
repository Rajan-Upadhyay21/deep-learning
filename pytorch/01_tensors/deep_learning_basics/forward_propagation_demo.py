import torch

x = torch.tensor([1.0, 2.0])

W1 = torch.tensor([[0.5, -0.4],
                   [0.8, 0.2]])
b1 = torch.tensor([0.1, 0.2])

W2 = torch.tensor([[0.7],
                   [-1.2]])
b2 = torch.tensor([0.3])

hidden = torch.matmul(x, W1) + b1
hidden_activated = torch.relu(hidden)

output = torch.matmul(hidden_activated, W2) + b2
final_output = torch.sigmoid(output)

print("Hidden Layer Output:")
print(hidden_activated)

print("\nFinal Output:")
print(final_output)
