import random

arms = {
    "Arm_1": 1.0,
    "Arm_2": 2.0,
    "Arm_3": 0.5
}

selected_arm = random.choice(list(arms.keys()))
reward = arms[selected_arm] + random.uniform(-0.2, 0.2)

print("Selected Arm:", selected_arm)
print("Observed Reward:", round(reward, 3))
