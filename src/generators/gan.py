"""Generator 4 of 4: GAN, ported from the professor's live-coded notebook.

Structure and training loop follow ``Taller_GANs.ipynb`` closely, including the
details he had to debug on screen, because reproducing his solution and then
saying *why* each piece is there is worth more than inventing a different one:

* **Dense generator ``150 -> 256 -> 512 -> 1024 -> dim``.** Our latent size is
  ``GAN_LATENT = 138``, which is his own heuristic from [01:50:00]: about 10% of
  the data dimensionality, because the 1380 window values are heavily correlated
  and do not really occupy 1380 independent directions.
* **Linear output activation.** At [01:56:06] he replaces the ``tanh`` inherited
  from image GANs with a linear unit, since returns are not bounded to
  ``[0, 1]``. We keep it linear for the same reason.
* **No batch normalisation.** He drops it at [01:53:49] as an image-specific
  ingredient.
* **Two separate Adam optimisers.** At [02:11:49] he hits the error that forces
  this: an optimiser keeps per-variable state and cannot be shared between the
  discriminator and the combined model.
* **The adaptive batch-ratio trick** from [02:07:16]. This is the interesting
  part, described below.

The balancing trick
-------------------
If the discriminator learns much faster than the generator, its gradient
saturates and the generator stops improving. Instead of tuning learning rates,
the professor rebalances *how much data each side sees* every epoch:

    ratio      = (d_loss + 1) / (g_loss + 1)
    batch_disc = round(ratio * batch)      # strong discriminator -> fewer samples
    batch_gen  = round(batch / ratio)      # ... and more for the generator

The ``+1`` in both terms is the fix he applied live at [02:13:53] after the ratio
diverged to infinity when one loss reached zero. We additionally clip the ratio
to a sane interval, because at [02:14:57] he observes his own minimum batch of 3
was already too small to train on.
"""

from __future__ import annotations

import numpy as np

from .. import config
from ..keras_setup import keras, set_seeds
from ..layers import MagnitudeFeatures
from .base import LatentSpaceGenerator


class GANGenerator(LatentSpaceGenerator):
    name = "gan"
    label = "GAN"

    def __init__(
        self,
        seed: int = config.SEED,
        latent_dim: int = config.GAN_LATENT,
        epochs: int = config.GAN_EPOCHS,
        batch: int = config.GAN_BATCH,
        ratio_clip: tuple[float, float] = (0.25, 4.0),
        learning_rate: float = 1e-4,
        beta_1: float = 0.5,
        label_smoothing: float = 0.9,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch = batch
        self.ratio_clip = ratio_clip
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.label_smoothing = label_smoothing
        self.generator: keras.Model | None = None
        self.discriminator: keras.Model | None = None
        self.combined: keras.Model | None = None

    # -- networks ---------------------------------------------------------
    def _build(self, dim: int, output_scale: float) -> None:
        set_seeds(self.seed)
        # These are the settings the professor left commented out on the very
        # first line of his GAN cell, `Adam(learning_rate=0.0001, beta_1=0.5)`.
        # With the Keras defaults (lr=1e-3, beta_1=0.9) our run diverged: the
        # discriminator loss climbed past 6 and the generator produced windows
        # six times more volatile than the real ones.
        opt_gen = keras.optimizers.Adam(self.learning_rate, beta_1=self.beta_1)
        opt_disc = keras.optimizers.Adam(self.learning_rate, beta_1=self.beta_1)

        self.generator = keras.Sequential(
            [
                keras.layers.Input(shape=(self.latent_dim,)),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dense(512, activation="relu"),
                keras.layers.Dense(1024, activation="relu"),
                # Bounded output, scaled to the data range.
                #
                # The professor debates exactly this at [01:58:22] and settles on
                # tanh "que igual ayuda", because returns will never saturate the
                # bound. His data was raw returns living inside +-1, ours is
                # standardised, so we keep his tanh and multiply by the observed
                # range. With a plain linear output the generator's activations
                # blew up in the first epochs, the discriminator saturated and
                # the samples came out seven times more volatile than the real
                # windows.
                keras.layers.Dense(dim, activation="tanh"),
                keras.layers.Rescaling(output_scale),
            ],
            name="generator",
        )

        self.discriminator = keras.Sequential(
            [
                keras.layers.Input(shape=(dim,)),
                # Without this the discriminator could not see the single most
                # obvious defect of the fakes. A ReLU network cannot compute a
                # variance, so it was unable to notice that the generated
                # windows were five times more volatile than the real ones, and
                # the generator diverged unopposed: discriminator loss climbed to
                # 8 while generator loss fell to 0.007. See src/layers.py.
                MagnitudeFeatures(),
                keras.layers.Dense(256),
                keras.layers.LeakyReLU(0.2),
                keras.layers.Dense(128),
                keras.layers.LeakyReLU(0.2),
                keras.layers.Dense(1, activation="sigmoid"),
            ],
            name="discriminator",
        )
        self.discriminator.compile(
            loss="binary_crossentropy", optimizer=opt_disc, metrics=["accuracy"]
        )

        # Freeze the discriminator inside the combined model: when we train the
        # generator we want the gradient to flow through the critic without
        # updating it. This is the "congelo el discriminador" step at [02:06:01].
        self.discriminator.trainable = False
        self.combined = keras.Sequential([self.generator, self.discriminator], name="gan")
        self.combined.compile(loss="binary_crossentropy", optimizer=opt_gen)
        self.discriminator.trainable = True

    # -- training ---------------------------------------------------------
    def _fit_latent(self, Z: np.ndarray) -> None:
        Z = np.asarray(Z, dtype="float32")
        # Robust range rather than the maximum, so a single outlier window does
        # not set the scale for the whole generator.
        output_scale = float(np.quantile(np.abs(Z), 0.999)) or 1.0
        self._build(Z.shape[1], output_scale)
        assert self.generator and self.discriminator and self.combined

        d_losses, g_losses, d_steps_log, g_steps_log = [], [], [], []
        ratio = 1.0
        lo, hi = self.ratio_clip

        for _ in range(self.epochs):
            # The professor's ratio, applied to the number of gradient *steps*
            # instead of the batch size.
            #
            # His version scales ``batch_disc`` and ``batch_gen``, but
            # ``train_on_batch`` performs exactly one update whatever the batch
            # size, so a bigger batch only reduces the variance of the gradient:
            # it does not let the lagging side catch up. Converting the same
            # ratio into a step count keeps his idea (rebalance by loss ratio,
            # with the ``+1`` he added live at [02:13:53] to stop it diverging)
            # and makes it actually control who learns faster.
            n_steps_d = int(np.clip(round(ratio), 1, 5))
            n_steps_g = int(np.clip(round(1.0 / ratio), 1, 5))

            # ---- discriminator ------------------------------------------
            self.discriminator.trainable = True
            d_loss = 0.0
            for _ in range(n_steps_d):
                idx = self.rng.integers(0, len(Z), size=self.batch)
                real = Z[idx]
                noise = self.rng.standard_normal(
                    (self.batch, self.latent_dim)
                ).astype("float32")
                fake = self.generator.predict(noise, verbose=0)

                x = np.concatenate([real, fake])
                # One-sided label smoothing: real -> 0.9 instead of 1.0, which
                # keeps the discriminator from becoming over-confident. Same
                # failure mode the ratio trick is fighting.
                y = np.concatenate(
                    [
                        np.full((self.batch, 1), self.label_smoothing),
                        np.zeros((self.batch, 1)),
                    ]
                ).astype("float32")
                out = self.discriminator.train_on_batch(x, y)
                d_loss += float(out[0] if isinstance(out, (list, tuple)) else out)
            d_loss /= n_steps_d

            # ---- generator ----------------------------------------------
            self.discriminator.trainable = False
            g_loss = 0.0
            for _ in range(n_steps_g):
                noise = self.rng.standard_normal(
                    (2 * self.batch, self.latent_dim)
                ).astype("float32")
                mislabelled = np.ones((2 * self.batch, 1), dtype="float32")
                out = self.combined.train_on_batch(noise, mislabelled)
                g_loss += float(out[0] if isinstance(out, (list, tuple)) else out)
            g_loss /= n_steps_g

            d_losses.append(d_loss)
            g_losses.append(g_loss)
            d_steps_log.append(float(n_steps_d))
            g_steps_log.append(float(n_steps_g))

            ratio = float(np.clip((d_loss + 1.0) / (g_loss + 1.0), lo, hi))

        self.history = {
            "d_loss": d_losses,
            "g_loss": g_losses,
            "pasos_disc": d_steps_log,
            "pasos_gen": g_steps_log,
        }

    def _sample_latent(self, n: int) -> np.ndarray:
        assert self.generator is not None
        noise = self.rng.standard_normal((n, self.latent_dim)).astype("float32")
        return self.generator.predict(noise, batch_size=512, verbose=0)
