
Code files:

## `text_generation_demo.py`

```python
import numpy as np

vocab = ["AI", "can", "generate", "useful", "content", "for", "users"]
probabilities = np.array([0.10, 0.20, 0.15, 0.18, 0.14, 0.12, 0.11])

generated_tokens = []

for _ in range(5):
    token = np.random.choice(vocab, p=probabilities)
    generated_tokens.append(token)

generated_text = " ".join(generated_tokens)

print("Generated Tokens:")
print(generated_tokens)

print("\nGenerated Text:")
print(generated_text)
