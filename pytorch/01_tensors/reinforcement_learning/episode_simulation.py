states = ["S1", "S2", "S3", "Goal"]
actions = ["right", "right", "right"]
rewards = [0, 1, 2, 10]

print("Episode Simulation:")

for step, (state, action, reward) in enumerate(zip(states[:-1], actions, rewards[:-1]), start=1):
    next_state = states[step]
    print(f"Step {step}")
    print(f"State: {state}")
    print(f"Action: {action}")
    print(f"Reward: {reward}")
    print(f"Next State: {next_state}")
    print("-" * 30)

print("Final Goal Reached:", states[-1])
print("Final Reward:", rewards[-1])
