#!/usr/bin/env python3
""" Variational Autoencoder Implementation """

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder
    Args:
        input_dims: dimensions of the model input
        hidden_layers: list of nodes for each hidden layer in the encoder
        latent_dims: dimensions of the latent space representation
    Returns: encoder, decoder, auto
    """
    # --- ENCODER ---
    X_input = keras.Input(shape=(input_dims,))

    # Loop through hidden layers for encoder
    Y_prev = X_input
    for nodes in hidden_layers:
        Y_prev = keras.layers.Dense(units=nodes, activation='relu')(Y_prev)

    # Latent space: mean and log variance
    z_mean = keras.layers.Dense(units=latent_dims, activation=None)(Y_prev)
    z_log_sigma = keras.layers.Dense(units=latent_dims,
                                     activation=None)(Y_prev)

    # Reparameterization trick (Sampling)
    def sampling(args):
        """ Sampling similar points in latent space """
        mu, log_sig = args
        batch = keras.backend.shape(mu)[0]
        dim = keras.backend.int_shape(mu)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return mu + keras.backend.exp(log_sig / 2) * epsilon

    z = keras.layers.Lambda(sampling,
                            output_shape=(latent_dims,))([z_mean, z_log_sigma])

    # The encoder outputs the sample z, the mean, and the log variance
    encoder = keras.Model(X_input, [z, z_mean, z_log_sigma])

    # --- DECODER ---
    X_decode = keras.Input(shape=(latent_dims,))

    # Reverse hidden layers for decoder
    Y_prev_dec = X_decode
    for nodes in reversed(hidden_layers):
        Y_prev_dec = keras.layers.Dense(units=nodes,
                                        activation='relu')(Y_prev_dec)

    # Final layer uses sigmoid activation
    output_dec = keras.layers.Dense(units=input_dims,
                                    activation='sigmoid')(Y_prev_dec)
    decoder = keras.Model(X_decode, output_dec)

    # --- FULL AUTOENCODER ---
    # The input to the decoder is the sampled z from the encoder
    sampled_z = encoder(X_input)[0]
    auto_output = decoder(sampled_z)
    auto = keras.Model(X_input, auto_output)

    # Custom VAE Loss (Reconstruction + KL Divergence)
    def vae_loss(x, x_reconstructed):
        """ Reconstruction loss (BCE) + KL Divergence """
        # Reconstruction loss
        recon_loss = keras.backend.binary_crossentropy(x, x_reconstructed)
        recon_loss = keras.backend.sum(recon_loss, axis=1)

        # KL Divergence loss
        kl_loss = -0.5 * keras.backend.sum(
            1 + z_log_sigma - keras.backend.square(z_mean) -
            keras.backend.exp(z_log_sigma),
            axis=1
        )
        return keras.backend.mean(recon_loss + kl_loss)

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
