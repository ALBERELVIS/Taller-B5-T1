"""Market factor / idiosyncratic decomposition of return windows.

The professor's argument in class is that a small equity universe has essentially
one dominant driver: the market itself. Everything else -- sector rotations,
name-specific news -- is second order. Notebook 03 quantifies this from the
other direction: almost every generator we train fails the **volatility of the
equal-weight portfolio** (and, one column to the right, the average pairwise
correlation), which is exactly what would happen if the market factor were
missing from the samples. And the label of this project is computed from that
same portfolio volatility, so losing the factor is losing the point.

This module exposes the smallest useful decomposition consistent with that
picture:

    r_{i,t} = beta_i * f_t + eps_{i,t}

with the market factor ``f_t`` estimated as the cross-sectional mean of the 23
assets (the equal-weight portfolio return) and ``beta_i`` fitted by univariate
OLS on the whole training block. It is the same one-factor idea that underlies
the CAPM baseline: crude, transparent and enough to isolate the piece the
generators keep breaking.

Kept deliberately small: no leave-one-out, no orthogonalisation, no sector
factor. Anything more would drift towards a model of the market rather than a
piece of infrastructure, and the whole point of the ``factor_mercado`` generator
is to see what happens when a generator *just* respects this one structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MarketDecomposition:
    """Fitted one-factor decomposition of a stack of return windows."""

    betas: np.ndarray          # shape (n_assets,)
    idio: np.ndarray           # shape (n_windows, n_steps, n_assets)
    market: np.ndarray         # shape (n_windows, n_steps)
    r_squared: np.ndarray      # per-asset R^2 across all (window, day) pairs


def market_factor(X: np.ndarray) -> np.ndarray:
    """Equal-weight portfolio return per window and day.

    ``X`` is ``(n, T, A)``; the return follows the same aggregation as the
    label (equal-weight cartera), so a generator that reproduces ``market``
    reproduces by construction the statistic on which stress is measured.
    """
    if X.ndim != 3:
        raise ValueError(f"expected (n, T, A), got {X.shape}")
    return X.mean(axis=2)


def market_betas(X: np.ndarray, market: np.ndarray | None = None) -> np.ndarray:
    """Univariate OLS ``r_{i,t} ~ beta_i * f_t`` pooled across ``(window, day)``.

    Runs without an intercept, matching the standard CAPM setup on demeaned
    returns. With 34k+ points per asset the estimate is very stable, so no
    ridge / shrinkage is needed at this stage.
    """
    if market is None:
        market = market_factor(X)
    x = market.reshape(-1)
    denom = float(x @ x)
    if denom <= 0.0:
        return np.zeros(X.shape[2], dtype=np.float64)
    R = X.reshape(-1, X.shape[2])
    return (R.T @ x) / denom


def idio_residuals(
    X: np.ndarray,
    betas: np.ndarray,
    market: np.ndarray | None = None,
) -> np.ndarray:
    """``r_{i,t} - beta_i * f_t`` in the original (n, T, A) shape."""
    if market is None:
        market = market_factor(X)
    return X - market[..., None] * betas[None, None, :]


def decompose_windows(X: np.ndarray) -> MarketDecomposition:
    """Fit the one-factor model on a stack of windows and return every piece."""
    market = market_factor(X)
    betas = market_betas(X, market=market)
    idio = idio_residuals(X, betas, market=market)

    # Per-asset R^2 pooled across all (window, day) pairs. Useful for reporting
    # in the notebook: how much of each asset's variance is really the market.
    R = X.reshape(-1, X.shape[2])
    total_ss = ((R - R.mean(axis=0)) ** 2).sum(axis=0)
    resid_ss = (idio.reshape(-1, X.shape[2]) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(total_ss > 0, 1.0 - resid_ss / total_ss, 0.0)

    return MarketDecomposition(
        betas=betas.astype(np.float64),
        idio=idio.astype(np.float32),
        market=market.astype(np.float32),
        r_squared=np.asarray(r2, dtype=np.float64),
    )
