import numpy as np

text_features = np.array([0.7, 0.2, 0.5])
image_features = np.array([0.6, 0.4, 0.8])

concat_fusion = np.concatenate([text_features, image_features])
average_fusion = (text_features + image_features) / 2

print("Text Features:")
print(text_features)

print("\nImage Features:")
print(image_features)

print("\nConcatenation Fusion:")
print(concat_fusion)

print("\nAverage Fusion:")
print(average_fusion)
