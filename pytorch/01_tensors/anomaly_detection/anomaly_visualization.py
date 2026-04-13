import numpy as np
import matplotlib.pyplot as plt

normal_data = np.random.normal(loc=0, scale=1, size=(100, 2))
anomalies = np.array([[4, 4], [5, 5], [4.5, 5.2]])

plt.figure(figsize=(8, 5))
plt.scatter(normal_data[:, 0], normal_data[:, 1], label="Normal Data")
plt.scatter(anomalies[:, 0], anomalies[:, 1], label="Anomalies", marker="x", s=100)
plt.title("Anomaly Visualization")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
