import torch
import torch.nn.functional as F

torch.manual_seed(42)

batch_size = 2
seq_len = 4
d_model = 8

x = torch.randn(batch_size, seq_len, d_model)

W_q = torch.randn(d_model, d_model)
W_k = torch.randn(d_model, d_model)
W_v = torch.randn(d_model, d_model)

Q = x @ W_q
K = x @ W_k
V = x @ W_v

scores = Q @ K.transpose(-2, -1) / (d_model ** 0.5)
attention_weights = F.softmax(scores, dim=-1)
output = attention_weights @ V

print("Input Shape:", x.shape)
print("Attention Weights Shape:", attention_weights.shape)
print("Output Shape:", output.shape)
