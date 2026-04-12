import torch
import torch.nn as nn

torch.manual_seed(42)

batch_size = 2
seq_len = 5
d_model = 16
num_heads = 4

x = torch.randn(batch_size, seq_len, d_model)

multihead_attn = nn.MultiheadAttention(
    embed_dim=d_model,
    num_heads=num_heads,
    batch_first=True
)

output, attn_weights = multihead_attn(x, x, x)

print("Input Shape:", x.shape)
print("Output Shape:", output.shape)
print("Attention Weights Shape:", attn_weights.shape)
