"""Economic evaluation: does the statistical gain turn into financial value?

PR-AUC tells us whether the model ranks turbulent periods above calm ones. It
does not tell us whether acting on that ranking would have been profitable, and
those two questions genuinely come apart: a model can improve its ranking of
mid-volatility days while being useless about the days that actually matter.

So we run the simplest possible de-risking rule, the one a risk desk would
actually recognise:

    at the close of each test day, if the model's stress probability is above the
    threshold that was tuned on validation, hold cash tomorrow; otherwise hold
    the equal-weight portfolio.

and compare it with buy and hold on the same days. A model that detects
volatility regimes should cut volatility and drawdown; whether it also improves
the return is not guaranteed, since avoiding turbulence also means missing the
sharp rebounds that follow it. We report both.

Transaction costs are charged on every switch, because a rule that flips daily
would otherwise look better than it is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def next_day_market_returns(
    returns: pd.DataFrame, dates: pd.DatetimeIndex
) -> np.ndarray:
    """Equal-weight portfolio return on the day *after* each decision date.

    ``dates[k]`` is the last day the model was allowed to see for window ``k``,
    so the first return it can act on is the following trading day. Getting this
    off by one would be look-ahead bias of the most damaging kind: the strategy
    would be trading on the very day whose move it is supposed to predict.
    """
    market = returns.mean(axis=1)
    pos = market.index.get_indexer(dates)
    nxt = pos + 1
    out = np.full(len(dates), np.nan)
    valid = (pos >= 0) & (nxt < len(market))
    out[valid] = market.values[nxt[valid]]
    return out


@dataclass
class BacktestResult:
    label: str
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float
    time_in_market: float
    n_switches: int
    equity: np.ndarray

    def as_dict(self) -> dict:
        return {
            "estrategia": self.label,
            "retorno_anual": self.annual_return,
            "volatilidad_anual": self.annual_vol,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "tiempo_invertido": self.time_in_market,
            "n_cambios": self.n_switches,
        }


def _summarise(
    label: str,
    strategy_returns: np.ndarray,
    exposure: np.ndarray,
    n_switches: int,
) -> BacktestResult:
    equity = np.exp(np.cumsum(strategy_returns))
    peak = np.maximum.accumulate(equity)
    mdd = float((equity / peak - 1.0).min())
    vol = float(strategy_returns.std() * np.sqrt(TRADING_DAYS))
    ann = float(strategy_returns.mean() * TRADING_DAYS)
    return BacktestResult(
        label=label,
        annual_return=ann,
        annual_vol=vol,
        sharpe=float(ann / vol) if vol > 0 else float("nan"),
        max_drawdown=mdd,
        time_in_market=float(exposure.mean()),
        n_switches=int(n_switches),
        equity=equity,
    )


def derisking_backtest(
    scores: np.ndarray,
    threshold: float,
    market_next: np.ndarray,
    cost_bps: float = 1.0,
    label: str = "de-risking",
) -> tuple[BacktestResult, BacktestResult]:
    """Run the rule and buy-and-hold on exactly the same days.

    Returns ``(strategy, buy_and_hold)``.
    """
    valid = ~np.isnan(market_next)
    r = market_next[valid]
    s = scores[valid]

    # Exposure for day t+1 is decided with information up to day t only.
    exposure = (s < threshold).astype(float)
    switches = int(np.sum(np.abs(np.diff(np.concatenate([[1.0], exposure])))))
    cost = np.abs(np.diff(np.concatenate([[1.0], exposure]))) * (cost_bps / 10_000.0)

    strategy = _summarise(label, exposure * r - cost, exposure, switches)
    benchmark = _summarise("buy & hold", r, np.ones_like(r), 0)
    return strategy, benchmark


def compare_backtests(results: list[BacktestResult]) -> pd.DataFrame:
    df = pd.DataFrame([r.as_dict() for r in results])
    return df.set_index("estrategia")


# --------------------------------------------------------------------------
# Statistical fingerprints of a financial series
# --------------------------------------------------------------------------
# These three tests are the ones a risk desk would actually run on synthetic
# data before trusting it: is there volatility clustering (ARCH), how heavy are
# the tails, and does the series look stationary. They complement the pointwise
# statistics of notebook 03 without duplicating them.
def _flatten_portfolio_returns(X: np.ndarray) -> np.ndarray:
    """Concatenate the equal-weight portfolio returns across all windows.

    The tests below assume a 1D time series; taking the portfolio return
    keeps them anchored to the same magnitude the target is computed from and
    makes them directly comparable across generators.
    """
    if X.ndim != 3:
        raise ValueError(f"expected (n, T, A), got {X.shape}")
    return X.mean(axis=2).reshape(-1)


def ljung_box_squared(X: np.ndarray, lags: int = 10) -> float:
    """Ljung-Box p-value on the squared portfolio returns (ARCH effect proxy).

    A small p-value means there is significant serial correlation in the
    squared returns, i.e. volatility clustering. Real financial series almost
    always score very small values; a generator that erases the clustering
    (e.g. an iid Gaussian) scores close to 1.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    r = _flatten_portfolio_returns(np.asarray(X))
    r2 = r ** 2 - float(np.mean(r ** 2))
    result = acorr_ljungbox(r2, lags=[lags], return_df=True)
    return float(result["lb_pvalue"].iloc[-1])


def hill_index(X: np.ndarray, tail_fraction: float = 0.05) -> float:
    """Hill estimator of the tail index of ``|r_portfolio|``.

    Larger values mean thinner tails; a Gaussian has an infinite Hill index in
    the limit, while empirical daily equity returns typically land in the
    3-5 range. Reported so that the ARCH / kurtosis / Hill triple gives a full
    fingerprint of the second-moment structure.
    """
    r = np.abs(_flatten_portfolio_returns(np.asarray(X)))
    if r.size == 0:
        return float("nan")
    k = max(int(tail_fraction * r.size), 10)
    sorted_r = np.sort(r)[-k:]
    threshold = sorted_r[0]
    if threshold <= 0.0:
        return float("nan")
    logs = np.log(sorted_r / threshold)
    mean_log = float(np.mean(logs))
    return float(1.0 / mean_log) if mean_log > 0 else float("nan")


def adf_pvalue(X: np.ndarray) -> float:
    """Augmented Dickey-Fuller p-value on the portfolio return series.

    A p-value close to zero rejects the unit root, i.e. the series looks
    stationary. Real log returns are strongly stationary, so any generator that
    drifts (e.g. an autoregressive model without proper burn-in) shows up
    immediately here.
    """
    from statsmodels.tsa.stattools import adfuller

    r = _flatten_portfolio_returns(np.asarray(X))
    # ADF on ~30-100k points is fast but memory-hungry: subsample to 20k with a
    # deterministic slice so the p-value stays reproducible and cheap.
    if r.size > 20_000:
        r = r[:: max(1, r.size // 20_000)][:20_000]
    result = adfuller(r, autolag="AIC")
    return float(result[1])


# --------------------------------------------------------------------------
# Train on synthetic, test on real (TSTR)
# --------------------------------------------------------------------------
# TSTR is the standard sanity check in the synthetic-time-series literature:
# train a downstream model with only synthetic data and evaluate on the real
# validation and test sets. If a generator captures what matters for the task,
# the TSTR score should approach the "train on real" ceiling; if it does not,
# TSTR falls off a cliff. We do it here because our sweep already gives us the
# synthetic pools for free, so the extra cost is just one classifier fit per
# generator and seed.
def tstr_score(
    X_synth_pool: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    n_synth: int | None = None,
    n_negatives_synth: int | None = None,
) -> dict:
    """Train the frozen CNN on synthetic minority + a light real-free negative
    pool, evaluate on real val/test.

    ``n_negatives_synth`` controls how many negatives to build for the training
    set. TSTR should not use real windows even for the majority class, so we
    build negatives by scaling the synthetic minority windows down: multiply
    each synthetic stress window by a small factor so that its realised
    volatility falls well below the training threshold. This keeps the training
    input distribution roughly in the same space and makes the CNN see
    something to contrast against.
    """
    from . import downstream

    n_synth = int(n_synth or len(X_synth_pool))
    if n_synth > len(X_synth_pool):
        raise ValueError(
            f"synthetic pool too small: {len(X_synth_pool)} < {n_synth}"
        )
    X_pos = X_synth_pool[:n_synth]

    if n_negatives_synth is None:
        n_negatives_synth = n_synth
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_synth, size=n_negatives_synth)
    scale = rng.uniform(0.2, 0.4, size=(n_negatives_synth, 1, 1)).astype("float32")
    X_neg = X_pos[idx] * scale

    X_train = np.concatenate([X_pos, X_neg], axis=0).astype("float32")
    y_train = np.concatenate(
        [np.ones(len(X_pos)), np.zeros(len(X_neg))]
    ).astype("int64")
    perm = rng.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    model, _ = downstream.train_classifier(
        X_train, y_train, X_val, y_val, seed=seed
    )
    m_val, m_test, _ = downstream.evaluate_model(model, X_val, y_val, X_test, y_test)
    return {
        "seed": seed,
        "n_synth": n_synth,
        "val_pr_auc": m_val.pr_auc,
        "val_lift": m_val.lift,
        "test_pr_auc": m_test.pr_auc,
        "test_lift": m_test.lift,
    }
