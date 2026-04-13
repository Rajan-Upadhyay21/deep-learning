
Code files:

## `z_score_anomaly_detection.py`

```python
import numpy as np
from scipy.stats import zscore

data = np.array([10, 12, 11, 13, 12, 14, 11, 13, 12, 50])

z_scores = zscore(data)
threshold = 2.0

anomalies = data[np.abs(z_scores) > threshold]

print("Data:", data)
print("Z-Scores:", z_scores)
print("Anomalies:", anomalies)
