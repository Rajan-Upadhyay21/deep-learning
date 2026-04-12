import numpy as np

vocab = ["model", "prompt", "token", "response", "generation"]
logits = np.array([2.5, 1.8, 1.1, 0.9, 0.6])

for temperature in [0.5, 1.0, 1.5]:
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    probs = exp_logits / exp_logits.sum()

    sampled_token = np.random.choice(vocab, p=probs)

    print(f"Temperature: {temperature}")
    print("Probabilities:", probs)
    print("Sampled Token:", sampled_token)
    print("-" * 40)
