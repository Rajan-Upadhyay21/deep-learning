import numpy as np
import random

q_values = np.array([1.2, 2.5, 0.8, 1.7])
actions = ["A", "B", "C", "D"]
epsilon = 0.2

if random.random() < epsilon:
    selected_action = random.choice(actions)
    decision_type = "exploration"
else:
    selected_action = actions[np.argmax(q_values)]
    decision_type = "exploitation"

print("Q-values:", q_values)
print("Epsilon:", epsilon)
print("Decision Type:", decision_type)
print("Selected Action:", selected_action)
