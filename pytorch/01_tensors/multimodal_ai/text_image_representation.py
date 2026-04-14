
Code files:

## `text_image_representation.py`

```python
import numpy as np

text_features = np.array([0.8, 0.2, 0.1, 0.4])
image_features = np.array([0.75, 0.25, 0.15, 0.35])

print("Text Representation:")
print(text_features)

print("\nImage Representation:")
print(image_features)

print("\nCombined Representation Shape:")
combined = np.concatenate([text_features, image_features])
print(combined.shape)

print("\nCombined Representation:")
print(combined)
