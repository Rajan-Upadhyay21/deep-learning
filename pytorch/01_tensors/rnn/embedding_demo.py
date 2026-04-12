import torch
import torch.nn as nn

vocab_size = 10
embedding_dim = 6

embedding = nn.Embedding(vocab_size, embedding_dim)

tokens = torch.tensor([1, 3, 5, 7])
embedded = embedding(tokens)

print("Tokens:")
print(tokens)

print("\nEmbedded Shape:")
print(embedded.shape)

print("\nEmbeddings:")
print(embedded)
