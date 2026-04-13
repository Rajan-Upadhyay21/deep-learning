import numpy as np

R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [0, 0, 5, 4],
], dtype=float)

num_users, num_items = R.shape
k = 2
alpha = 0.01
lambda_reg = 0.01
epochs = 500

np.random.seed(42)
P = np.random.rand(num_users, k)
Q = np.random.rand(num_items, k)

for _ in range(epochs):
    for i in range(num_users):
        for j in range(num_items):
            if R[i, j] > 0:
                error = R[i, j] - np.dot(P[i, :], Q[j, :].T)
                P[i, :] += alpha * (error * Q[j, :] - lambda_reg * P[i, :])
                Q[j, :] += alpha * (error * P[i, :] - lambda_reg * Q[j, :])

predicted_R = np.dot(P, Q.T)

print("Original Rating Matrix:")
print(R)

print("\nPredicted Rating Matrix:")
print(predicted_R.round(2))
