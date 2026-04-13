import numpy as np

data = np.array([10, 12, 11, 13, 12, 14, 11, 13, 12, 50])

q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

anomalies = data[(data < lower_bound) | (data > upper_bound)]

print("Data:", data)
print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
print("Anomalies:", anomalies)
