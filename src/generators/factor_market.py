"""Generator: one-factor market model, motivated by the finding in section 8.3.

The main analysis of this project concludes with a single testable rule: a
generator helps downstream if it reproduces **the statistic the label is
computed from** -- the volatility of the equal-weight portfolio -- and does not
help if it only reproduces "realism in general". Everything else the generators
compete on is secondary.

This generator is the smallest one we could think of that satisfies that rule
**by construction**. It splits every real return into an equal-weight market
factor and an idiosyncratic residual (``src.factors``), models each with the
tool that fits it, and puts them back together at sampling time:

    r_{i,t} = beta_i * f_t + eps_{i,t}

* ``f_t`` is a 1D time series with strong volatility clustering, so it gets a
  tiny probabilistic autoregressor with |r| features and a Gaussian NLL loss,
  the same architectural trick used in ``AutoregressiveGenerator`` and in the
  downstream classifier, and for the same reason: a ReLU network cannot build a
  conditional variance without magnitudes in its input.
* ``eps_{i,t}`` is a 23-dimensional vector with no natural temporal structure
  once the market has been removed. We model it as a multivariate normal fitted
  with Ledoit-Wolf shrinkage. In 23 dimensions with ~34k samples the shrinkage
  intensity is basically zero, so unlike the fully-Gaussian generator this one
  does not have to trade correlation strength for numerical stability.

The point of the design is not that this is the strongest generator on paper --
a VAE has strictly more capacity -- but that it targets, on purpose, the
specific weakness diagnosed in notebook 03. If the rule from section 8.3 is
right, this one should place exactly where its fidelity in portfolio
volatility predicts, whether that is the top of the ranking or somewhere in the
middle.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from sklearn.covariance import LedoitWolf

from .. import config
from .. import factors
from ..keras_setup import keras, set_seeds
from .base import Generator

_LOG_SIGMA_MIN, _LOG_SIGMA_MAX = -8.0, 2.0


class FactorMarketGenerator(Generator):
    """One-factor market + Gaussian idiosyncratic residual generator."""

    name = "factor_mercado"
    label = "Factor de mercado + idio"

    def __init__(
        self,
        seed: int = config.SEED,
        context: int = 10,
        hidden: int = 32,
        epochs: int = 250,
        batch_size: int = 512,
        validation_split: float = 0.15,
        learning_rate: float = 3e-4,
        patience: int = 30,
        burn_in: int = 60,
        clip_sigmas: float = 25.0,
    ):
        super().__init__(seed=seed)
        self.context = context
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.patience = patience
        self.burn_in = burn_in
        self.clip_sigmas = clip_sigmas

        self.model: keras.Model | None = None
        self.betas_: np.ndarray | None = None
        self.market_scale_: float | None = None
        self.market_prefixes_: np.ndarray | None = None
        self.idio_mean_: np.ndarray | None = None
        self.idio_chol_: np.ndarray | None = None
        self.idio_shrinkage_: float | None = None

    # -- network for the 1D market-factor path ---------------------------
    def _build(self) -> keras.Model:
        set_seeds(self.seed)
        inputs = keras.layers.Input(shape=(self.context, 1))
        h = keras.layers.Flatten()(inputs)
        # |r| for the same reason as in the classifier and in the plain
        # autoregressive: the conditional scale of the market is a function of
        # recent magnitudes, and a ReLU stack cannot build it from signed inputs.
        mag = keras.layers.Flatten()(keras.layers.Lambda(keras.ops.abs)(inputs))
        h = keras.layers.Concatenate()([h, mag])
        h = keras.layers.Dense(self.hidden, activation="relu")(h)
        h = keras.layers.Dense(self.hidden, activation="relu")(h)
        mean = keras.layers.Dense(1, name="mean")(h)
        log_sigma = keras.layers.Dense(1, name="log_sigma")(h)
        out = keras.layers.Concatenate(name="params")([mean, log_sigma])
        model = keras.Model(inputs, out, name="market_factor_ar")
        model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate, clipnorm=1.0),
            loss=_gaussian_nll_1d,
        )
        return model

    # -- fitting ----------------------------------------------------------
    def _fit(self, X: np.ndarray) -> None:
        decomposition = factors.decompose_windows(X)
        self.betas_ = decomposition.betas.astype(np.float64)
        market = decomposition.market                       # (n, T)
        idio = decomposition.idio                           # (n, T, A)

        # 1D probabilistic autoregressor on the market factor.
        self.market_scale_ = float(market.std()) or 1.0
        market_std = (market / self.market_scale_).astype("float32")

        n, T = market_std.shape
        n_val = max(int(round(self.validation_split * n)), 1)
        perm = np.random.default_rng(self.seed).permutation(n)
        val_w = market_std[perm[:n_val]]
        train_w = market_std[perm[n_val:]]

        ctx, target = _teacher_forcing_pairs_1d(train_w, self.context)
        ctx_val, target_val = _teacher_forcing_pairs_1d(val_w, self.context)

        self.model = self._build()
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
        self.market_prefixes_ = market_std[:, : self.context].copy()

        # Multivariate Gaussian for the idiosyncratic residual. Flatten across
        # (window, day) because once the market is removed there is no
        # meaningful temporal structure to preserve at this scale (checked by
        # the low serial autocorrelation of |eps| in notebook 03).
        idio_flat = idio.reshape(-1, X.shape[2]).astype(np.float64)
        self.idio_mean_ = idio_flat.mean(axis=0)
        lw = LedoitWolf(assume_centered=False).fit(idio_flat)
        self.idio_shrinkage_ = float(lw.shrinkage_)
        # In 23 dimensions with ~34k samples the sample covariance is very well
        # conditioned, so lw.shrinkage_ tends to be tiny and the resulting
        # covariance stays essentially unshrunk -- exactly what we want.
        self.idio_chol_ = _safe_cholesky(lw.covariance_)

    # -- sampling ---------------------------------------------------------
    def _sample(self, n: int) -> np.ndarray:
        assert self.model is not None and self.market_prefixes_ is not None
        assert self.betas_ is not None and self.market_scale_ is not None
        assert self.idio_mean_ is not None and self.idio_chol_ is not None
        T, A = self.n_steps, self.n_assets
        total = T + self.burn_in

        # (1) roll the market factor forward, burning in extra steps so no real
        #     day survives in the returned window.
        idx = self.rng.integers(0, len(self.market_prefixes_), size=n)
        path = np.zeros((n, self.context + total), dtype="float32")
        path[:, : self.context] = self.market_prefixes_[idx]
        for t in range(self.context, self.context + total):
            ctx = path[:, t - self.context : t, None]
            params = self.model.predict(ctx, batch_size=max(n, 1), verbose=0)
            mean = params[:, 0]
            sigma = np.exp(np.clip(params[:, 1], _LOG_SIGMA_MIN, _LOG_SIGMA_MAX))
            eps = self.rng.standard_normal(n)
            path[:, t] = np.clip(
                mean + sigma * eps, -self.clip_sigmas, self.clip_sigmas
            )
        market_synth = path[:, -T:] * self.market_scale_       # (n, T)

        # (2) draw independent idiosyncratic days from the fitted Gaussian.
        d = A
        eps_flat = (
            self.rng.standard_normal((n * T, d)) @ self.idio_chol_.T
            + self.idio_mean_
        )
        idio_synth = eps_flat.reshape(n, T, A).astype(np.float32)

        # (3) recompose r_{i,t} = beta_i * f_t + eps_{i,t}.
        market_synth = market_synth.astype(np.float32)
        betas = self.betas_.astype(np.float32)
        return market_synth[:, :, None] * betas[None, None, :] + idio_synth


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _teacher_forcing_pairs_1d(
    Xs: np.ndarray, context: int
) -> tuple[np.ndarray, np.ndarray]:
    """Every ``(context window, next value)`` pair inside every 1D window.

    ``Xs`` is ``(n, T)``; we build ``(n * (T - context), context, 1)`` for the
    context and ``(n * (T - context),)`` for the target.
    """
    n, T = Xs.shape
    ctx, tgt = [], []
    for t in range(context, T):
        ctx.append(Xs[:, t - context : t])
        tgt.append(Xs[:, t])
    ctx_arr = np.concatenate(ctx).astype("float32")[..., None]
    tgt_arr = np.concatenate(tgt).astype("float32")
    return ctx_arr, tgt_arr


def _gaussian_nll_1d(y_true, y_pred):
    """NLL of a 1D Gaussian with predicted mean and log-sigma.

    ``y_true`` is ``(batch,)``; ``y_pred`` is ``(batch, 2)`` = ``[mean, log_sigma]``.
    """
    mean = y_pred[:, 0]
    log_sigma = keras.ops.clip(y_pred[:, 1], _LOG_SIGMA_MIN, _LOG_SIGMA_MAX)
    z = (y_true - mean) * keras.ops.exp(-log_sigma)
    return keras.ops.mean(log_sigma + 0.5 * keras.ops.square(z))


def _safe_cholesky(cov: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor with the smallest jitter that keeps it PD."""
    jitter = 0.0
    scale = float(np.mean(np.diag(cov))) or 1.0
    for _ in range(8):
        try:
            return linalg.cholesky(cov + jitter * np.eye(len(cov)), lower=True)
        except linalg.LinAlgError:
            jitter = max(jitter * 10, 1e-10 * scale)
    vals, vecs = np.linalg.eigh(cov)
    return vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None)))
