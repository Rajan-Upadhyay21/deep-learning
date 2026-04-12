import numpy as np

training_data = np.array([10, 12, 11, 13, 12, 14, 11, 13])
production_data = np.array([18, 20, 19, 21, 22, 20, 19, 23])

train_mean = training_data.mean()
prod_mean = production_data.mean()

drift_value = abs(prod_mean - train_mean)

print("Training Mean:", train_mean)
print("Production Mean:", prod_mean)
print("Mean Drift:", drift_value)

if drift_value > 5:
    print("Warning: Possible data drift detected.")
else:
    print("No major drift detected.")
