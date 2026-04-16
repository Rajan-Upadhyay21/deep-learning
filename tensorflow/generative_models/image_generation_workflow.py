import tensorflow as tf
import matplotlib.pyplot as plt

from generator_network import build_generator


def generate_images(generator, num_images=4, latent_dim=100):
    noise = tf.random.normal((num_images, latent_dim))
    generated = generator(noise, training=False)
    return generated


def display_generated_images(images):
    images = (images + 1.0) / 2.0

    for index in range(images.shape[0]):
        plt.figure(figsize=(2, 2))
        plt.imshow(images[index, :, :, 0], cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    generator = build_generator()
    images = generate_images(generator, num_images=3)
    print("Generated batch shape:", images.shape)
    display_generated_images(images)
