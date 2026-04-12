import torch

seq_len = 5
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

print("Causal Mask Shape:")
print(mask.shape)

print("\nCausal Mask:")
print(mask)
