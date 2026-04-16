import tensorflow as tf
from tensorflow import keras
import numpy as np

from generator_network import build_generator
from discriminator_network import build_discriminator


class GANTrainer:
    def __init__(self, latent_dim: int = 100):
        self.latent_dim = latent_dim
        self.generator = build_generator(latent_dim)
        self.discriminator = build_discriminator()

        self.loss_fn = keras.losses.BinaryCrossentropy()
        self.gen_optimizer = keras.optimizers.Adam(learning_rate=1e-4)
        self.disc_optimizer = keras.optimizers.Adam(learning_rate=1e-4)

    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_noise = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.generator(random_noise, training=True)

            real_preds = self.discriminator(real_images, training=True)
            fake_preds = self.discriminator(fake_images, training=True)

            gen_loss = self.loss_fn(tf.ones_like(fake_preds), fake_preds)
            real_loss = self.loss_fn(tf.ones_like(real_preds), real_preds)
            fake_loss = self.loss_fn(tf.zeros_like(fake_preds), fake_preds)
            disc_loss = real_loss + fake_loss

        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))

        return gen_loss, disc_loss


def create_dummy_real_images(batch_size=16):
    return np.random.normal(size=(batch_size, 28, 28, 1)).astype("float32")


if __name__ == "__main__":
    trainer = GANTrainer(latent_dim=100)
    real_images = create_dummy_real_images(batch_size=8)
    gen_loss, disc_loss = trainer.train_step(real_images)
    print("Generator loss:", float(gen_loss))
    print("Discriminator loss:", float(disc_loss))
