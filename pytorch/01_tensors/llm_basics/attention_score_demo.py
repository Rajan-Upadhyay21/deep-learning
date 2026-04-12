import torch
import torch.nn.functional as F

torch.manual_seed(42)

query = torch.randn(1, 4)
keys = torch.randn(3, 4)
values = torch.randn(3, 4)

scores = torch.matmul(keys, query.T).squeeze() / (query.size(-1) ** 0.5)
weights = F.softmax(scores, dim=0)
context = torch.sum(weights.unsqueeze(1) * values, dim=0)

print("Attention Scores:")
print(scores)

print("\nAttention Weights:")
print(weights)

print("\nContext Vector:")
print(context)
