"""Generator 3 of 4: Variational Autoencoder.

A VAE learns a smooth low-dimensional latent space from which we can sample. It
occupies the same slot in the taxonomy as the professor's "Bayes" family: it is
an explicit probabilistic latent-variable model trained by maximising a lower
bound on the likelihood, rather than by playing an adversarial game.

Why we include it: it is the natural counterpart to the GAN. Both learn a
non-linear generator ``z -> x`` from the same data in the same space, so the
comparison isolates the effect of *how* they are trained. And unlike a GAN it
converges monotonically, which makes the "show me a loss curve that has
converged" requirement of the statement easy to satisfy honestly.

Loss: reconstruction MSE plus ``beta`` times the KL divergence to a standard
normal. ``beta`` is deliberately small. With 1380 output dimensions and only
~570 training windows, a KL term at full strength collapses the posterior onto
the prior and the decoder ends up emitting the dataset mean for every ``z`` --
posterior collapse, which would make the generator useless while still showing
a beautifully decreasing loss curve.
"""

from __future__ import annotations

import numpy as np

from .. import config
from ..keras_setup import keras, set_seeds
from .base import LatentSpaceGenerator


@keras.saving.register_keras_serializable(package="b5t1")
class Sampling(keras.layers.Layer):
    """Reparameterisation trick, and the layer that owns the KL term.

    ``z = mu + sigma * eps`` keeps the sampling differentiable with respect to
    ``mu`` and ``log_var``, which is the whole trick that makes a VAE trainable
    by backpropagation.
    """

    def __init__(self, beta: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def call(self, inputs):
        mu, log_var = inputs
        eps = keras.random.normal(keras.ops.shape(mu))
        kl = -0.5 * keras.ops.sum(
            1.0 + log_var - keras.ops.square(mu) - keras.ops.exp(log_var), axis=-1
        )
        self.add_loss(self.beta * keras.ops.mean(kl))
        return mu + keras.ops.exp(0.5 * log_var) * eps

    def get_config(self):
        return {**super().get_config(), "beta": self.beta}


class VAEGenerator(LatentSpaceGenerator):
    name = "vae"
    label = "VAE"

    def __init__(
        self,
        seed: int = config.SEED,
        latent_dim: int = config.VAE_LATENT,
        hidden: tuple[int, ...] = (256, 128),
        beta: float = config.VAE_BETA,
        epochs: int = config.VAE_EPOCHS,
        batch_size: int = 32,
        validation_split: float = 0.15,
        learning_rate: float = 3e-4,
        patience: int = 40,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.beta = beta
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.patience = patience
        self.encoder: keras.Model | None = None
        self.decoder: keras.Model | None = None
        self.obs_sigma_: np.ndarray | None = None

    def _build(self, dim: int) -> keras.Model:
        set_seeds(self.seed)

        # -- encoder ------------------------------------------------------
        enc_in = keras.layers.Input(shape=(dim,), name="x")
        h = enc_in
        for units in self.hidden:
            h = keras.layers.Dense(units, activation="relu")(h)
        mu = keras.layers.Dense(self.latent_dim, name="mu")(h)
        log_var = keras.layers.Dense(self.latent_dim, name="log_var")(h)
        z = Sampling(beta=self.beta, name="z")([mu, log_var])
        self.encoder = keras.Model(enc_in, [mu, log_var, z], name="encoder")

        # -- decoder ------------------------------------------------------
        dec_in = keras.layers.Input(shape=(self.latent_dim,), name="z_in")
        h = dec_in
        for units in reversed(self.hidden):
            h = keras.layers.Dense(units, activation="relu")(h)
        # Linear output: our data is standardised, so it is positive and
        # negative on an unbounded range. The professor makes the same point at
        # [01:56:06] when he removes the sigmoid from his generator.
        dec_out = keras.layers.Dense(dim, activation="linear")(h)
        self.decoder = keras.Model(dec_in, dec_out, name="decoder")

        vae = keras.Model(enc_in, self.decoder(z), name="vae")
        # clipnorm is not cosmetic here: with 1380 output dimensions and ~570
        # training windows the first epochs produce very large gradients and an
        # unclipped Adam run diverges to NaN.
        vae.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate, clipnorm=1.0),
            loss="mse",
        )
        return vae

    def _fit_latent(self, Z: np.ndarray) -> None:
        vae = self._build(Z.shape[1])
        stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True
        )
        hist = vae.fit(
            Z,
            Z,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[stop],
            shuffle=True,
            verbose=0,
        )
        self.history = {k: [float(x) for x in v] for k, v in hist.history.items()}

        # Observation noise of the decoder likelihood, estimated per dimension.
        #
        # This step is easy to forget and it matters. Training with an MSE
        # reconstruction loss means the decoder models the *mean* of
        # p(x | z) = N(decoder(z), sigma^2 I). Returning decoder(z) as a "sample"
        # therefore returns a conditional mean, not a draw from the model, and
        # systematically understates the variance: our first version produced
        # windows six times calmer than the real ones. Adding the residual
        # sigma back is what makes the VAE an actual generative model.
        recon = vae.predict(Z, batch_size=512, verbose=0)
        self.obs_sigma_ = (Z - recon).std(axis=0)

    def _sample_latent(self, n: int) -> np.ndarray:
        assert self.decoder is not None and self.obs_sigma_ is not None
        z = self.rng.standard_normal((n, self.latent_dim)).astype("float32")
        mean = self.decoder.predict(z, batch_size=512, verbose=0)
        eps = self.rng.standard_normal(mean.shape)
        return mean + self.obs_sigma_ * eps
