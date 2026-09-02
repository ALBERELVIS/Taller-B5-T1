"""Data pipeline: prices -> log returns -> sliding windows -> rare-event label -> splits.

The financial problem
---------------------
Standing at the close of day ``t`` we look at the previous ``WINDOW_X`` days of
log returns for the 23 surviving S&P 500 constituents and ask a binary question
about the next ``HORIZON`` trading days.

Two candidate definitions of "rare stress event" are implemented, and choosing
between them was an experimental result rather than a preference:

``"vol"`` (default)
    Will realised volatility of the equal-weight portfolio over the next
    ``HORIZON`` days land in the top ``1 - VOL_QUANTILE`` of its distribution?
    This is the standard risk-management formulation (it drives VaR, position
    sizing and margin) and volatility is strongly persistent, so the question is
    genuinely answerable.

``"drawdown"``
    Will the portfolio fall at least ``|DD_THRESHOLD|`` below today's level at
    some point in the next ``HORIZON`` days? We measured this one first and it
    turns out to carry essentially **no** out-of-sample signal (test PR-AUC lift
    ~1.1, ROC ~0.5), which is what weak-form market efficiency predicts: future
    returns are not forecastable from past returns, while future *volatility*
    is. We keep it in the code base and report it in notebook 01 as a
    documented negative result.

Leakage note
------------
The ``"vol"`` threshold is a quantile of the target distribution, so it must be
estimated on the **training block only**. Computing it over the whole sample
would let the test period influence its own labels. That is why labelling
happens after the chronological split, not before.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import config


# --------------------------------------------------------------------------
# 1. Raw prices and returns
# --------------------------------------------------------------------------
def load_prices(force_download: bool = False) -> pd.DataFrame:
    """Adjusted close prices for the tickers that survive the whole period.

    Cached to parquet: the Yahoo download takes ~60 s and several notebooks need
    it, so re-downloading would only add noise and waiting time.
    """
    if config.PRICES_CACHE.exists() and not force_download:
        return pd.read_parquet(config.PRICES_CACHE)

    import yfinance as yf

    tickers = list(pd.read_csv(config.TICKERS_URL))
    prices = yf.download(
        tickers, start=config.START_DATE, auto_adjust=True, progress=False
    )["Close"]
    # Keeping only tickers without missing values gives the longest common
    # history instead of a ragged panel.
    prices = prices.dropna(axis=1)
    prices.to_parquet(config.PRICES_CACHE)
    return prices


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff().dropna()


# --------------------------------------------------------------------------
# 2. Forward-looking quantities (the strictly future part of each window)
# --------------------------------------------------------------------------
def _forward_slot(n: int, horizon: int) -> np.ndarray:
    """Indices ``i`` for which days ``i+1 .. i+horizon`` are fully inside the data."""
    return np.arange(0, max(n - horizon, 0))


def forward_drawdown(market_returns: np.ndarray, horizon: int) -> np.ndarray:
    """Worst cumulative simple return over days ``i+1 .. i+horizon``."""
    n = len(market_returns)
    out = np.full(n, np.nan)
    if n <= horizon:
        return out
    level = np.exp(np.cumsum(market_returns))
    window_min = sliding_window_view(level, horizon).min(axis=1)
    i = _forward_slot(n, horizon)
    out[i] = window_min[i + 1] / level[i] - 1.0
    return out


def forward_volatility(market_returns: np.ndarray, horizon: int) -> np.ndarray:
    """Annualised realised volatility over days ``i+1 .. i+horizon``."""
    n = len(market_returns)
    out = np.full(n, np.nan)
    if n <= horizon:
        return out
    window_std = sliding_window_view(market_returns, horizon).std(axis=1)
    i = _forward_slot(n, horizon)
    out[i] = window_std[i + 1] * np.sqrt(252.0)
    return out


# --------------------------------------------------------------------------
# 3. Windowing
# --------------------------------------------------------------------------
def build_windows(
    returns: pd.DataFrame,
    window_x: int = config.WINDOW_X,
    horizon: int = config.HORIZON,
) -> tuple[np.ndarray, dict[str, np.ndarray], pd.DatetimeIndex]:
    """Build ``X (N, window_x, n_assets)`` plus the continuous forward targets.

    Window ``k`` uses returns ``[i - window_x, i)`` as features, where ``i`` is
    the decision index, and the targets describe returns ``[i, i + horizon)``.
    There is no overlap between what the model sees and what it must predict.
    """
    r = returns.values
    n = r.shape[0]
    market = r.mean(axis=1)                       # equal-weight portfolio

    dd = forward_drawdown(market, horizon)
    vol = forward_volatility(market, horizon)

    idx = np.arange(window_x, n - horizon + 1)
    decision = idx - 1                            # last observed day of the window

    X = np.stack([r[i - window_x:i] for i in idx]).astype(np.float32)
    targets = {"drawdown": dd[decision], "volatility": vol[decision]}

    valid = ~np.isnan(targets["volatility"]) & ~np.isnan(targets["drawdown"])
    X = X[valid]
    targets = {k: v[valid] for k, v in targets.items()}
    dates = returns.index[decision[valid]]
    return X, targets, dates


# --------------------------------------------------------------------------
# 4. Chronological split with embargo
# --------------------------------------------------------------------------
def split_slices(
    n: int,
    train_frac: float = config.TRAIN_FRAC,
    val_frac: float = config.VAL_FRAC,
    embargo: int = config.EMBARGO,
) -> tuple[slice, slice, slice]:
    """Time-ordered slices with ``embargo`` windows removed at each boundary.

    Why the embargo: consecutive windows share 59 of their 60 days and every
    label peeks ``HORIZON`` days ahead. Cutting the series at a single point
    would let the last training labels be decided by days that belong to the
    validation period. Dropping ``WINDOW_X + HORIZON`` windows around each
    boundary removes that overlap entirely.
    """
    i_a = int(round(train_frac * n))
    i_b = int(round((train_frac + val_frac) * n))
    return slice(0, i_a), slice(i_a + embargo, i_b), slice(i_b + embargo, n)


@dataclass
class Split:
    """A chronological train/val/test partition of the windowed dataset."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    dates_train: pd.DatetimeIndex
    dates_val: pd.DatetimeIndex
    dates_test: pd.DatetimeIndex
    target: str
    threshold: float
    embargo: int
    target_value_test: np.ndarray

    @property
    def input_shape(self) -> tuple[int, int]:
        return self.X_train.shape[1], self.X_train.shape[2]

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, y, d in (
            ("train", self.y_train, self.dates_train),
            ("val", self.y_val, self.dates_val),
            ("test", self.y_test, self.dates_test),
        ):
            rows.append(
                {
                    "bloque": name,
                    "n_ventanas": len(y),
                    "desde": d[0].date(),
                    "hasta": d[-1].date(),
                    "n_positivos": int(y.sum()),
                    "tasa_positivos": round(float(y.mean()), 4),
                    "rachas": count_episodes(y),
                    "episodios_independientes": count_episodes(y, min_gap=126),
                }
            )
        return pd.DataFrame(rows)


def episode_bounds(y: np.ndarray, min_gap: int = 1) -> list[tuple[int, int]]:
    """Contiguous runs of positive labels, merging runs closer than ``min_gap``.

    Consecutive stress windows overlap almost completely, so they are not
    independent observations. ``min_gap`` merges the short interruptions inside a
    single market episode so we can count episodes instead of runs.
    """
    y = np.asarray(y).astype(int)
    edges = np.diff(np.concatenate([[0], y, [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    if len(starts) == 0:
        return []

    merged = [[int(starts[0]), int(ends[0])]]
    for s, e in zip(starts[1:], ends[1:]):
        if s - merged[-1][1] <= min_gap:
            merged[-1][1] = int(e)
        else:
            merged.append([int(s), int(e)])
    return [tuple(m) for m in merged]


def count_episodes(y: np.ndarray, min_gap: int = 1) -> int:
    """Number of independent stress episodes.

    The honest measure of how much the minority class really contains: several
    hundred positive windows drawn from a handful of episodes are not several
    hundred independent observations.
    """
    return len(episode_bounds(y, min_gap=min_gap))


# --------------------------------------------------------------------------
# 5. Labelling (after the split, so no threshold is estimated on future data)
# --------------------------------------------------------------------------
def label_from_targets(
    target_value: np.ndarray,
    train_slice: slice,
    target: str = config.TARGET,
    vol_quantile: float = config.VOL_QUANTILE,
    dd_threshold: float = config.DD_THRESHOLD,
) -> tuple[np.ndarray, float]:
    """Turn the continuous forward target into a binary label.

    For ``"vol"`` the cut is the ``vol_quantile`` quantile **of the training
    block only**; for ``"drawdown"`` it is an absolute, economically meaningful
    level so no estimation is involved.
    """
    if target == "vol":
        threshold = float(np.quantile(target_value[train_slice], vol_quantile))
        return (target_value >= threshold).astype(np.int64), threshold
    if target == "drawdown":
        return (target_value <= dd_threshold).astype(np.int64), float(dd_threshold)
    raise ValueError(f"unknown target: {target!r}")


# --------------------------------------------------------------------------
# 6. One-call entry point
# --------------------------------------------------------------------------
def load_problem(
    target: str = config.TARGET,
    force_download: bool = False,
) -> tuple[Split, pd.DataFrame, dict]:
    """Returns ``(split, returns, extras)`` ready to use.

    ``extras`` carries the continuous targets, the dates and the label for the
    full sample, which the notebooks need for the timeline plots and the
    backtest.
    """
    prices = load_prices(force_download=force_download)
    returns = log_returns(prices)
    X, targets, dates = build_windows(returns)

    key = "volatility" if target == "vol" else "drawdown"
    target_value = targets[key]

    tr, va, te = split_slices(len(target_value))
    y, threshold = label_from_targets(target_value, tr, target=target)

    split = Split(
        X_train=X[tr], y_train=y[tr],
        X_val=X[va], y_val=y[va],
        X_test=X[te], y_test=y[te],
        dates_train=dates[tr], dates_val=dates[va], dates_test=dates[te],
        target=target, threshold=threshold, embargo=config.EMBARGO,
        target_value_test=target_value[te],
    )
    extras = {
        "X": X, "y": y, "dates": dates, "targets": targets,
        "target_value": target_value, "threshold": threshold,
        "slices": (tr, va, te),
    }
    return split, returns, extras
