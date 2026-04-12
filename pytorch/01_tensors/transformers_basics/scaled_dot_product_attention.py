import torch
import torch.nn.functional as F

torch.manual_seed(42)

Q = torch.randn(1, 4, 8)
K = torch.randn(1, 4, 8)
V = torch.randn(1, 4, 8)

d_k = Q.size(-1)

scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
weights = F.softmax(scores, dim=-1)
output = torch.matmul(weights, V)

print("Scores:")
print(scores)

print("\nAttention Weights:")
print(weights)

print("\nOutput:")
print(output)
