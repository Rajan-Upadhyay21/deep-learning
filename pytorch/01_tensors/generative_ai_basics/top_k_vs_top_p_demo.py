import numpy as np

vocab = ["chat", "assistant", "search", "retrieval", "token", "context"]
probs = np.array([0.28, 0.24, 0.18, 0.12, 0.10, 0.08])

k = 3
top_k_indices = np.argsort(probs)[-k:]
top_k_vocab = [vocab[i] for i in top_k_indices]
top_k_probs = probs[top_k_indices]
top_k_probs = top_k_probs / top_k_probs.sum()

print("Top-k Vocabulary:")
print(top_k_vocab)
print("Top-k Probabilities:")
print(top_k_probs)

sorted_indices = np.argsort(probs)[::-1]
sorted_vocab = [vocab[i] for i in sorted_indices]
sorted_probs = probs[sorted_indices]

p_threshold = 0.75
selected_vocab = []
selected_probs = []
cumulative_prob = 0.0

for token, prob in zip(sorted_vocab, sorted_probs):
    selected_vocab.append(token)
    selected_probs.append(prob)
    cumulative_prob += prob
    if cumulative_prob >= p_threshold:
        break

selected_probs = np.array(selected_probs)
selected_probs = selected_probs / selected_probs.sum()

print("\nTop-p Vocabulary:")
print(selected_vocab)
print("Top-p Probabilities:")
print(selected_probs)
