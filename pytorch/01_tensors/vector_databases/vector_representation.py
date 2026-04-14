
Code files:

## `vector_representation.py`

```python
import numpy as np

text_vector = np.array([0.82, 0.14, 0.33, 0.55])
image_vector = np.array([0.79, 0.18, 0.29, 0.51])
query_vector = np.array([0.80, 0.15, 0.31, 0.53])

print("Text Vector:")
print(text_vector)

print("\nImage Vector:")
print(image_vector)

print("\nQuery Vector:")
print(query_vector)

print("\nVector Dimension:")
print(len(query_vector))
