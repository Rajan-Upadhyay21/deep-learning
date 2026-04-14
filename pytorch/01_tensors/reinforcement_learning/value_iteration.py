import numpy as np

rewards = np.array([0, 1, 5, 10])
gamma = 0.9
values = np.zeros_like(rewards, dtype=float)

for iteration in range(5):
    new_values = rewards + gamma * values
    values = new_values
    print(f"Iteration {iteration + 1}: {values}")
