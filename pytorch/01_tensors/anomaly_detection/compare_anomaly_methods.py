import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

normal_data = np.random.normal(loc=0, scale=1, size=(100, 2))
anomalies = np.array([[4, 4], [5, 5], [4.5, 5.2]])
X = np.vstack([normal_data, anomalies])

iso_labels = IsolationForest(contamination=0.03, random_state=42).fit_predict(X)
svm_labels = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit_predict(X)
lof_labels = LocalOutlierFactor(n_neighbors=20, contamination=0.03).fit_predict(X)

print("Isolation Forest detected anomalies:", np.sum(iso_labels == -1))
print("One-Class SVM detected anomalies:", np.sum(svm_labels == -1))
print("Local Outlier Factor detected anomalies:", np.sum(lof_labels == -1))
