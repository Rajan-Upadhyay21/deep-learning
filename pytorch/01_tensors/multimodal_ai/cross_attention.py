import torch
import torch.nn.functional as F

torch.manual_seed(42)

text_queries = torch.randn(1, 3, 4)
image_keys = torch.randn(1, 5, 4)
image_values = torch.randn(1, 5, 4)

scores = torch.matmul(text_queries, image_keys.transpose(-2, -1)) / (4 ** 0.5)
weights = F.softmax(scores, dim=-1)
output = torch.matmul(weights, image_values)

print("Cross-Attention Scores Shape:")
print(scores.shape)

print("\nAttention Weights Shape:")
print(weights.shape)

print("\nOutput Shape:")
print(output.shape)
