import numpy as np

vocab = ["chat", "model", "data", "token", "prompt", "answer"]
probabilities = np.array([0.30, 0.25, 0.15, 0.12, 0.10, 0.08])

sorted_indices = np.argsort(probabilities)[::-1]
sorted_probs = probabilities[sorted_indices]
sorted_vocab = [vocab[i] for i in sorted_indices]

p = 0.75
cumulative = 0.0
selected_vocab = []
selected_probs = []

for word, prob in zip(sorted_vocab, sorted_probs):
    selected_vocab.append(word)
    selected_probs.append(prob)
    cumulative += prob
    if cumulative >= p:
        break

selected_probs = np.array(selected_probs)
selected_probs = selected_probs / selected_probs.sum()

sampled_token = np.random.choice(selected_vocab, p=selected_probs)

print("Top-p Vocabulary:", selected_vocab)
print("Top-p Probabilities:", selected_probs)
print("\nSampled Token:")
print(sampled_token)
