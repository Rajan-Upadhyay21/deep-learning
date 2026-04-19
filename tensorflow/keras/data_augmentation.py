from tensorflow import keras

augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1)
])

print("Keras data augmentation pipeline created:")
print(augmentation)
