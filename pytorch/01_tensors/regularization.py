import torch
import torch.nn as nn

x = torch.randn(5, 10)

dropout = nn.Dropout(p=0.5)
batch_norm = nn.BatchNorm1d(10)
layer_norm = nn.LayerNorm(10)

dropout.train()
dropout_output = dropout(x)

batch_norm_output = batch_norm(x)
layer_norm_output = layer_norm(x)

print("Original Input:")
print(x)

print("\nAfter Dropout:")
print(dropout_output)

print("\nAfter BatchNorm:")
print(batch_norm_output)

print("\nAfter LayerNorm:")
print(layer_norm_output)

linear = nn.Linear(10, 1)
l2_penalty = sum((param ** 2).sum() for param in linear.parameters())

print("\nL2 Penalty:")
print(l2_penalty.item())
