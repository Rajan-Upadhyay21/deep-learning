import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_variational_autoencoder(input_dim: int = 784, latent_dim: int = 16):
    encoder_inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(encoder_inputs)
    x = layers.Dense(128, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="vae_encoder")

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(128, activation="relu")(latent_inputs)
    x = layers.Dense(256, activation="relu")(x)
    decoder_outputs = layers.Dense(input_dim, activation="sigmoid")(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="vae_decoder")

    class VAE(keras.Model):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def call(self, inputs):
            z_mean, z_log_var, z = self.encoder(inputs)
            reconstructed = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(inputs, reconstructed)
            ) * input_dim
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            self.add_loss(reconstruction_loss + kl_loss)
            return reconstructed

    vae = VAE(encoder, decoder)
    return vae, encoder, decoder


if __name__ == "__main__":
    vae, encoder, decoder = build_variational_autoencoder()
    vae.compile(optimizer="adam")
    dummy = tf.random.uniform((4, 784))
    output = vae(dummy)
    print("Output shape:", output.shape)
