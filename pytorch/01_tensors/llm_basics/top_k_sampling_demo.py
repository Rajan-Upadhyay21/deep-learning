import numpy as np

vocab = ["chat", "model", "data", "token", "prompt", "answer"]
probabilities = np.array([0.30, 0.25, 0.10, 0.15, 0.12, 0.08])

k = 3
top_k_indices = np.argsort(probabilities)[-k:]
top_k_probs = probabilities[top_k_indices]
top_k_probs = top_k_probs / top_k_probs.sum()

top_k_vocab = [vocab[i] for i in top_k_indices]
sampled_token = np.random.choice(top_k_vocab, p=top_k_probs)

print("Top-k Vocabulary:", top_k_vocab)
print("Top-k Probabilities:", top_k_probs)
print("\nSampled Token:")
print(sampled_token)
