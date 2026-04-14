alpha = 0.1
gamma = 0.9

current_q = 2.0
reward = 5.0
max_future_q = 3.5

updated_q = current_q + alpha * (reward + gamma * max_future_q - current_q)

print("Current Q-value:", current_q)
print("Reward:", reward)
print("Max Future Q-value:", max_future_q)
print("Updated Q-value:", updated_q)
