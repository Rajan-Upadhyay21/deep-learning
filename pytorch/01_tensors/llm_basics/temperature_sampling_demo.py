import numpy as np

vocab = ["AI", "models", "learn", "patterns", "fast"]
logits = np.array([2.0, 1.2, 0.8, 1.5, 0.5])

temperature = 0.7
scaled_logits = logits / temperature
exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
probs = exp_logits / exp_logits.sum()

sampled_token = np.random.choice(vocab, p=probs)

print("Vocabulary:", vocab)
print("Temperature:", temperature)
print("Sampling Probabilities:", probs)
print("\nSampled Token:")
print(sampled_token)
