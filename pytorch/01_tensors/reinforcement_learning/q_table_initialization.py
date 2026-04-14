import numpy as np

states = 5
actions = 4

q_table = np.zeros((states, actions))

print("Q-table Shape:", q_table.shape)
print("\nInitialized Q-table:")
print(q_table)
