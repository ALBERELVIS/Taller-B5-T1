"""Generator 5: autoregressive model in the native time domain.

The professor mentions autoregressive models as one of the class families and
calls them "muy sencillo" at [00:13:25]. They are also, for *this* problem, the
family with the right inductive bias, and that is the point we want to make with
them.

What it models
--------------
Instead of treating the window as one 1380-dim vector, it factorises the joint
density the way a time series actually decomposes:

    p(r_1..r_T) = prod_t p(r_t | r_{t-k..t-1})

A small network reads the last ``AR_CONTEXT`` days and predicts, for each of the
23 assets, a conditional mean and a conditional log volatility. Training
minimises the Gaussian negative log-likelihood, so the network is explicitly
learning **volatility clustering**: turbulent days are followed by wide
predicted distributions. Since our label is "will the next month be turbulent",
this is precisely the structure that has to survive into the synthetic data --
and the structure that the PCA/covariance-based generators destroy.

Cross-sectional correlation
---------------------------
A diagonal conditional distribution would generate 23 independent assets, and
averaging them into a portfolio would diversify the volatility away. So the
innovations are drawn from a fixed correlation matrix estimated from the
standardised training residuals: the network supplies the time-varying scale,
the residual correlation supplies the co-movement.

Why it does not live in the PCA space
-------------------------------------
Principal components of a flattened window are not ordered in time, so there is
nothing to be autoregressive about. This generator therefore works on raw
returns. It still consumes and produces the same ``(n, 60, 23)`` windows as
every other generator, so the downstream comparison is unaffected.

Avoiding leakage of real days into the samples
----------------------------------------------
Rolling a sequence forward needs a seed context. We seed with a real ``k``-day
prefix but then roll ``AR_BURN_IN`` extra steps and keep only the **last** 60,
so not a single real observation survives in the returned window.
"""

from __future__ import annotations

import numpy as np

from .. import config
from ..keras_setup import keras, set_seeds
from .base import Generator

_LOG_SIGMA_MIN, _LOG_SIGMA_MAX = -8.0, 2.0


class AutoregressiveGenerator(Generator):
    name = "autoregresivo"
    label = "Autorregresivo (NLL gaussiana)"

    def __init__(
        self,
        seed: int = config.SEED,
        context: int = config.AR_CONTEXT,
        hidden: int = 128,
        epochs: int = config.AR_EPOCHS,
        batch_size: int = 256,
        burn_in: int = config.AR_BURN_IN,
        validation_split: float = 0.15,
        learning_rate: float = 3e-4,
        patience: int = 25,
        clip_sigmas: float = 25.0,
    ):
        super().__init__(seed=seed)
        self.context = context
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.burn_in = burn_in
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.patience = patience
        self.clip_sigmas = clip_sigmas
        self.model: keras.Model | None = None
        self.scale_: np.ndarray | None = None
        self.chol_resid_: np.ndarray | None = None
        self.prefixes_: np.ndarray | None = None

    # -- network ----------------------------------------------------------
    def _build(self, n_assets: int) -> keras.Model:
        set_seeds(self.seed)
        inputs = keras.layers.Input(shape=(self.context, n_assets))
        h = keras.layers.Flatten()(inputs)
        # |r| is given explicitly for the same reason as in the classifier: a
        # ReLU network struggles to build a second moment out of signed inputs,
        # and the conditional scale is a function of recent magnitudes.
        mag = keras.layers.Flatten()(keras.layers.Lambda(keras.ops.abs)(inputs))
        h = keras.layers.Concatenate()([h, mag])
        h = keras.layers.Dense(self.hidden, activation="relu")(h)
        h = keras.layers.Dense(self.hidden, activation="relu")(h)
        mean = keras.layers.Dense(n_assets, name="mean")(h)
        log_sigma = keras.layers.Dense(n_assets, name="log_sigma")(h)
        out = keras.layers.Concatenate(name="params")([mean, log_sigma])

        model = keras.Model(inputs, out, name="ar_step")
        model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate, clipnorm=1.0),
            loss=_gaussian_nll,
        )
        return model

    # -- fitting ----------------------------------------------------------
    def _fit(self, X: np.ndarray) -> None:
        n, T, A = X.shape
        # Standardise per asset so the NLL is not dominated by the noisiest
        # names; undone at sampling time.
        self.scale_ = X.std(axis=(0, 1), keepdims=True)
        Xs = X / self.scale_

        # Validation is split by *window*, not by teacher-forcing pair. Pairs
        # taken from the same window overlap heavily, so a random pair-level
        # split would put near-duplicates on both sides and make early stopping
        # far too optimistic. Same reasoning as the embargo in the main split,
        # applied inside the generator.
        n_val = max(int(round(self.validation_split * n)), 1)
        perm = np.random.default_rng(self.seed).permutation(n)
        val_w, train_w = Xs[perm[:n_val]], Xs[perm[n_val:]]

        ctx, target = _teacher_forcing_pairs(train_w, self.context)
        ctx_val, target_val = _teacher_forcing_pairs(val_w, self.context)
        self.model = self._build(A)
        # Early stopping is essential rather than cosmetic: an overfitted scale
        # head predicts absurd volatilities on the states it meets during the
        # rollout, and because the rollout feeds its own output back in, the
        # sequence explodes. Restoring the best validation weights keeps the
        # conditional variance calibrated.
        stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True
        )
        hist = self.model.fit(
            ctx,
            target,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(ctx_val, target_val),
            callbacks=[stop],
            shuffle=True,
            verbose=0,
        )
        self.history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}

        # Residual correlation, so generated assets co-move like real ones.
        params = self.model.predict(ctx, batch_size=1024, verbose=0)
        mean, log_sigma = params[:, :A], np.clip(params[:, A:], _LOG_SIGMA_MIN, _LOG_SIGMA_MAX)
        resid = (target - mean) / np.exp(log_sigma)
        corr = np.corrcoef(resid.T)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        self.chol_resid_ = _safe_cholesky(corr)

        self.prefixes_ = Xs[:, :self.context, :].copy()

    # -- sampling ---------------------------------------------------------
    def _sample(self, n: int) -> np.ndarray:
        assert self.model is not None and self.prefixes_ is not None
        assert self.scale_ is not None and self.chol_resid_ is not None
        T, A = self.n_steps, self.n_assets
        total = T + self.burn_in

        idx = self.rng.integers(0, len(self.prefixes_), size=n)
        path = np.zeros((n, self.context + total, A), dtype="float32")
        path[:, :self.context] = self.prefixes_[idx]

        # Hard bound on each generated day. Because the data is standardised per
        # asset, one unit is one training standard deviation, so the bound reads
        # directly in sigmas. This is the "design your own layer to impose
        # constraints" idea the professor raises at [02:20:15]: the rollout feeds
        # its own output back in, so a single absurd day would contaminate
        # everything after it. The default is deliberately loose (the real
        # training windows reach about 21 sigma) so that genuine fat tails
        # survive and only runaway divergence is caught.
        for t in range(self.context, self.context + total):
            params = self.model.predict(
                path[:, t - self.context:t], batch_size=max(n, 1), verbose=0
            )
            mean = params[:, :A]
            sigma = np.exp(np.clip(params[:, A:], _LOG_SIGMA_MIN, _LOG_SIGMA_MAX))
            eps = self.rng.standard_normal((n, A)) @ self.chol_resid_.T
            path[:, t] = np.clip(mean + sigma * eps, -self.clip_sigmas, self.clip_sigmas)

        # Keep only the tail: the real prefix and the burn-in are discarded.
        return path[:, -T:, :] * self.scale_


def _teacher_forcing_pairs(Xs: np.ndarray, context: int) -> tuple[np.ndarray, np.ndarray]:
    """Every (context window, next day) pair inside every training window.

    567 stress windows of 60 days give ~28k transitions, so the step model has
    far more training signal than a model that treats a whole window as one
    sample. That is a genuine advantage of the autoregressive factorisation in
    the small-data regime.
    """
    n, T, A = Xs.shape
    ctx, tgt = [], []
    for t in range(context, T):
        ctx.append(Xs[:, t - context:t, :])
        tgt.append(Xs[:, t, :])
    return (
        np.concatenate(ctx).astype("float32"),
        np.concatenate(tgt).astype("float32"),
    )


def _gaussian_nll(y_true, y_pred):
    """Negative log-likelihood of a diagonal Gaussian with predicted mean and scale."""
    n_assets = keras.ops.shape(y_true)[-1]
    mean = y_pred[:, :n_assets]
    log_sigma = keras.ops.clip(y_pred[:, n_assets:], _LOG_SIGMA_MIN, _LOG_SIGMA_MAX)
    z = (y_true - mean) * keras.ops.exp(-log_sigma)
    return keras.ops.mean(log_sigma + 0.5 * keras.ops.square(z), axis=-1)


def _safe_cholesky(corr: np.ndarray) -> np.ndarray:
    from scipy import linalg

    jitter = 0.0
    for _ in range(8):
        try:
            return linalg.cholesky(corr + jitter * np.eye(len(corr)), lower=True)
        except linalg.LinAlgError:
            jitter = max(jitter * 10, 1e-8)
    vals, vecs = np.linalg.eigh(corr)
    return vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None)))
