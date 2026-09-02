"""Expensive experiments, computed once and cached to ``results/``.

The notebooks are meant to be re-runnable end to end, but a few steps train
neural networks and take minutes. Each function here computes its result, writes
it to disk and returns the cached copy on subsequent calls, so re-running a
notebook to fix a typo in a markdown cell does not retrain anything.

Pass ``force=True`` to recompute.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import config, downstream, evaluate
from . import generators as gens
from .data import Split
from .representation import WindowRepresentation

MODEL_SELECTION_CSV = config.RESULTS_DIR / "seleccion_arquitectura.csv"
GENERATOR_STATS_CSV = config.RESULTS_DIR / "calidad_generadores.csv"
BACKTEST_NPZ = config.RESULTS_DIR / "backtest_scores.npz"
NEGATIVE_RESULT_CSV = config.RESULTS_DIR / "target_drawdown_negativo.csv"
TSTR_CSV = config.RESULTS_DIR / "tstr.csv"
PURGED_KFOLD_CSV = config.RESULTS_DIR / "robustez_purged_kfold.csv"


# --------------------------------------------------------------------------
# Notebook 02: choosing the downstream architecture
# --------------------------------------------------------------------------
def model_selection_table(
    split: Split,
    variants: tuple[str, ...] = ("cnn_prof", "cnn_raw", "cnn_mag", "cnn_mag_bn", "cnn_mag_gap"),
    seeds: tuple[int, ...] = (0, 1),
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Compare reference models and CNN variants on 100% real data.

    Returns ``(table, histories)``. The reference models are not there to be
    beaten easily: realised volatility is a genuinely strong predictor of future
    volatility, so a network that cannot beat it is not learning anything a risk
    manager did not already know.
    """
    hist_path = config.RESULTS_DIR / "seleccion_arquitectura_historias.json"
    if MODEL_SELECTION_CSV.exists() and hist_path.exists() and not force:
        return (
            pd.read_csv(MODEL_SELECTION_CSV, index_col=0),
            json.loads(hist_path.read_text()),
        )

    rows, histories = {}, {}

    def add(label: str, s_val: np.ndarray, s_test: np.ndarray) -> None:
        thr = downstream.best_threshold(split.y_val, s_val)
        m_val = downstream.evaluate_scores(split.y_val, s_val, thr)
        m_test = downstream.evaluate_scores(split.y_test, s_test, thr)
        rows[label] = {
            **{f"val_{k}": v for k, v in m_val.as_dict().items()},
            **{f"test_{k}": v for k, v in m_test.as_dict().items()},
        }

    add("Baseline: vol. realizada",
        downstream.volatility_score(split.X_val), downstream.volatility_score(split.X_test))
    add("Baseline: drawdown actual",
        downstream.drawdown_score(split.X_val), downstream.drawdown_score(split.X_test))
    s_val, s_test = downstream.logistic_baseline(
        split.X_train, split.y_train, split.X_val, split.X_test
    )
    add("Regresión logística", s_val, s_test)

    for variant in variants:
        val_scores, test_scores, elapsed = [], [], 0.0
        for seed in seeds:
            t0 = time.time()
            model, history = downstream.train_classifier(
                split.X_train, split.y_train, split.X_val, split.y_val,
                seed=seed, variant=variant,
            )
            elapsed += time.time() - t0
            val_scores.append(downstream.predict_scores(model, split.X_val))
            test_scores.append(downstream.predict_scores(model, split.X_test))
            if seed == seeds[0]:
                histories[variant] = history
        # Average the scores over seeds: a single run of a small network on 567
        # positives is noisy, and we are choosing an architecture, not a seed.
        add(f"CNN {variant}", np.mean(val_scores, axis=0), np.mean(test_scores, axis=0))
        rows[f"CNN {variant}"]["segundos"] = round(elapsed / len(seeds), 1)

    table = pd.DataFrame(rows).T
    table.to_csv(MODEL_SELECTION_CSV)
    hist_path.write_text(json.dumps(histories))
    return table, histories


# --------------------------------------------------------------------------
# Notebook 01: the documented negative result
# --------------------------------------------------------------------------
def drawdown_negative_result(force: bool = False) -> pd.DataFrame:
    """Show that the drawdown target carries no out-of-sample signal.

    This is why the main task is the volatility regime and not the drawdown. We
    keep the evidence rather than quietly dropping the idea: it is a textbook
    illustration of weak-form efficiency, and it is the reason the project has a
    measurable effect to study at all.
    """
    if NEGATIVE_RESULT_CSV.exists() and not force:
        return pd.read_csv(NEGATIVE_RESULT_CSV, index_col=0)

    from .data import load_problem

    rows = {}
    for target, label in (("vol", "Régimen de volatilidad"), ("drawdown", "Caída >= 8%")):
        split, _, _ = load_problem(target=target)
        for score_fn, sname in (
            (downstream.volatility_score, "vol. realizada"),
            (downstream.drawdown_score, "drawdown actual"),
        ):
            s_val, s_test = score_fn(split.X_val), score_fn(split.X_test)
            thr = downstream.best_threshold(split.y_val, s_val)
            m = downstream.evaluate_scores(split.y_test, s_test, thr)
            rows[f"{label} / {sname}"] = {
                "tasa_base_test": m.base_rate,
                "test_pr_auc": m.pr_auc,
                "lift": m.lift,
                "test_roc_auc": m.roc_auc,
            }
    table = pd.DataFrame(rows).T
    table.to_csv(NEGATIVE_RESULT_CSV)
    return table


# --------------------------------------------------------------------------
# Notebook 03: quality of the synthetic data
# --------------------------------------------------------------------------
def generator_quality(
    split: Split,
    seed: int = 0,
    n_samples: int = 2000,
    names: list[str] | None = None,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, dict]]:
    """Fit every generator once and measure how well it reproduces the real data.

    The three statistics reported are chosen because they are the ones our label
    depends on, not because they are the usual suspects:

    ``std``        overall scale of daily returns;
    ``vol_cartera`` volatility of the equal-weight portfolio, which is literally
                   what the target is computed from;
    ``corr_media``  average pairwise correlation, the market factor. A generator
                   can match the first statistic perfectly and still fail the
                   other two by producing 23 independent assets.
    ``kurtosis``    fat tails, the professor's point at [01:17:16].
    """
    names = names or list(gens.REGISTRY)
    X_min = split.X_train[split.y_train == 1]
    representation = WindowRepresentation(
        n_components=config.GENERATOR_SPACE, seed=seed
    ).fit(split.X_train)

    synth: dict[str, np.ndarray] = {}
    histories: dict[str, dict] = {}
    rows = {"REAL": _quality_stats(X_min)}

    for name in names:
        cls = gens.REGISTRY[name]
        kwargs = {"seed": seed}
        if issubclass(cls, gens.LatentSpaceGenerator):
            kwargs["representation"] = representation
        t0 = time.time()
        gen = cls(**kwargs).fit(X_min)
        synth[name] = gen.sample(n_samples)
        histories[name] = gen.history
        gen.save_history()
        rows[name] = _quality_stats(synth[name])
        rows[name]["segundos_ajuste"] = round(time.time() - t0, 1)
        print(f"  {name:15s} ajustado en {time.time() - t0:5.1f}s", flush=True)

    table = pd.DataFrame(rows).T
    table.to_csv(GENERATOR_STATS_CSV)
    return table, synth, histories


def _quality_stats(X: np.ndarray) -> dict:
    market = X.mean(axis=2)
    corrs = []
    for w in X[:400]:
        c = np.corrcoef(w.T)
        corrs.append(np.nanmean(c[np.triu_indices_from(c, k=1)]))
    flat = X.ravel()
    # The three tests below add the statistical fingerprint that a risk desk
    # would check before trusting synthetic financial data: ARCH-Ljung-Box for
    # volatility clustering, Hill for tail thickness, ADF for stationarity.
    # Any generator that erases the clustering or the stationarity shows up
    # here, not in the pointwise moments.
    try:
        arch_p = evaluate.ljung_box_squared(X)
    except Exception:
        arch_p = float("nan")
    try:
        hill = evaluate.hill_index(X)
    except Exception:
        hill = float("nan")
    try:
        adf_p = evaluate.adf_pvalue(X)
    except Exception:
        adf_p = float("nan")
    return {
        "std": float(X.std()),
        "vol_cartera": float(market.std(axis=1).mean()),
        "corr_media": float(np.nanmean(corrs)),
        "kurtosis": float(((flat - flat.mean()) ** 4).mean() / flat.var() ** 2),
        "max_abs": float(np.abs(X).max()),
        "arch_lb_p": float(arch_p),
        "hill_index": float(hill),
        "adf_p": float(adf_p),
    }


# --------------------------------------------------------------------------
# Notebook 05: model for the backtest
# --------------------------------------------------------------------------
def backtest_scores(
    split: Split,
    configs: dict[str, tuple[str, int | None, int]],
    seeds: tuple[int, ...] = (0, 1, 2),
    force: bool = False,
) -> dict[str, dict]:
    """Train the configurations we want to backtest and cache their test scores.

    ``configs`` maps a label to ``(generator_name, n_real, n_synth)``.
    Scores are averaged over seeds, so the backtest does not depend on one lucky
    initialisation.
    """
    if BACKTEST_NPZ.exists() and not force:
        data = np.load(BACKTEST_NPZ, allow_pickle=True)
        return {k: v.item() for k, v in data.items()}

    from .sweep import build_training_set, fit_generators, subsample_real

    out: dict[str, dict] = {}
    needed = {c[0] for c in configs.values() if c[0] != "sin_sinteticos"}

    pools_by_seed = {}
    for seed in seeds:
        pools_by_seed[seed], _ = fit_generators(
            split, seed, names=sorted(needed), save_histories=False
        ) if needed else ({}, {})

    for label, (gen_name, n_real, n_synth) in configs.items():
        val_s, test_s = [], []
        for seed in seeds:
            rng = np.random.default_rng(1000 * seed + (n_real or 0))
            X_real, y_real = subsample_real(split.X_train, split.y_train, n_real, rng)
            X_tr, y_tr = build_training_set(
                X_real, y_real, pools_by_seed[seed].get(gen_name), n_synth
            )
            model, _ = downstream.train_classifier(
                X_tr, y_tr, split.X_val, split.y_val, seed=seed
            )
            val_s.append(downstream.predict_scores(model, split.X_val))
            test_s.append(downstream.predict_scores(model, split.X_test))
        s_val = np.mean(val_s, axis=0)
        s_test = np.mean(test_s, axis=0)
        out[label] = {
            "val": s_val,
            "test": s_test,
            "threshold": downstream.best_threshold(split.y_val, s_val),
        }
        print(f"  backtest listo: {label}", flush=True)

    np.savez(BACKTEST_NPZ, **{k: np.array(v, dtype=object) for k, v in out.items()})
    return out


# --------------------------------------------------------------------------
# Notebook 04 / 05: TSTR (train on synthetic, test on real)
# --------------------------------------------------------------------------
def tstr_table(
    split: Split,
    seeds: tuple[int, ...] = (0, 1, 2),
    n_synth: int = 2000,
    names: list[str] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Train the frozen CNN with **only synthetic** minorities per generator,
    evaluate on real val/test, average over seeds.

    Reuses the synthetic pools already fitted for the sweep. Nothing else in
    the project has to change: this is the canonical benchmark of the
    synthetic-time-series literature applied to our own setting.
    """
    if TSTR_CSV.exists() and not force:
        return pd.read_csv(TSTR_CSV)

    from .sweep import fit_generators

    names = names or list(gens.REGISTRY)
    rows = []
    for seed in seeds:
        pools, _ = fit_generators(split, seed, names=names, save_histories=False)
        for name in names:
            pool = pools.get(name)
            if pool is None or len(pool) < n_synth:
                continue
            metrics = evaluate.tstr_score(
                pool,
                split.X_val, split.y_val,
                split.X_test, split.y_test,
                seed=seed, n_synth=n_synth,
            )
            metrics.update({"generator": name})
            rows.append(metrics)
            print(
                f"  [seed {seed}] TSTR {name:16s} "
                f"val_pr={metrics['val_pr_auc']:.3f} "
                f"test_pr={metrics['test_pr_auc']:.3f}",
                flush=True,
            )

    table = pd.DataFrame(rows)
    table.to_csv(TSTR_CSV, index=False)
    return table


# --------------------------------------------------------------------------
# Notebook 05: robustez, purged K-Fold temporal
# --------------------------------------------------------------------------
def _chronological_folds(
    n_samples: int, n_folds: int, purge: int
) -> list[tuple[slice, slice]]:
    """K non-overlapping test blocks over ``[0, n_samples)`` with embargo.

    Each fold's test slice is a contiguous chronological block. The training
    slice is everything outside a ``purge``-window neighbourhood of the test
    slice, taken from **both sides**. This is the standard purged K-Fold used
    in financial ML (López de Prado): it stops labels that peek into the future
    from leaking through overlapping windows.
    """
    fold_size = n_samples // n_folds
    folds: list[tuple[slice, slice]] = []
    for k in range(n_folds):
        start = k * fold_size
        end = n_samples if k == n_folds - 1 else start + fold_size
        test_slice = slice(start, end)
        # Training indices: outside [start - purge, end + purge).
        left = np.arange(0, max(0, start - purge))
        right = np.arange(min(n_samples, end + purge), n_samples)
        train_idx = np.concatenate([left, right])
        folds.append((train_idx, test_slice))
    return folds


def purged_kfold_gain(
    split: Split,
    winners: dict[int | None, tuple[str, int]],
    n_folds: int = 5,
    purge: int = config.WINDOW_X + config.HORIZON,
    seeds: tuple[int, ...] = (0, 1, 2),
    force: bool = False,
) -> pd.DataFrame:
    """Repeat the sweep-winning configurations on ``n_folds`` chronological
    folds cut inside the train+val block, with embargo, and report mean +/- SEM
    of the relative gain over "no synthetics" per generator and per level of
    real data.

    ``winners`` maps ``n_real -> (generator, n_synth)`` -- typically read from
    the aggregated results of the main sweep. The test set of the main
    experiment is *not* touched here; the point is to check that the shape of
    the curves survives being re-run over different chronological cuts.
    """
    if PURGED_KFOLD_CSV.exists() and not force:
        return pd.read_csv(PURGED_KFOLD_CSV)

    from .sweep import build_training_set, fit_generators, subsample_real

    # Pool train + val into a single chronological block so we have five
    # non-trivial folds; test stays untouched.
    X_pool = np.concatenate([split.X_train, split.X_val], axis=0)
    y_pool = np.concatenate([split.y_train, split.y_val], axis=0)
    n_pool = len(y_pool)

    rows = []
    for seed in seeds:
        # Fit generators once per seed on the *original* training minority so
        # the synthetic pools we reuse per fold are the same distribution as in
        # the main experiment. Refitting inside every fold would confound "the
        # generator was retrained" with "the classifier was retrained".
        gen_names = sorted({w[0] for w in winners.values() if w[0] != "sin_sinteticos"})
        pools, _ = fit_generators(split, seed, names=gen_names, save_histories=False)

        for fold_idx, (train_idx, test_slice) in enumerate(
            _chronological_folds(n_pool, n_folds, purge)
        ):
            X_train = X_pool[train_idx]
            y_train = y_pool[train_idx]
            X_test = X_pool[test_slice]
            y_test = y_pool[test_slice]
            if int(np.sum(y_test == 1)) < 5 or int(np.sum(y_train == 1)) < 20:
                continue

            for n_real, (gen_name, n_synth) in winners.items():
                rng = np.random.default_rng(1000 * seed + (n_real or 0) + fold_idx)
                X_real, y_real = subsample_real(X_train, y_train, n_real, rng)

                # (a) baseline: no synthetics.
                model, _ = downstream.train_classifier(
                    X_real, y_real, X_test, y_test, seed=seed
                )
                s = downstream.predict_scores(model, X_test)
                thr = downstream.best_threshold(y_test, s)
                pr_base = downstream.evaluate_scores(y_test, s, thr).pr_auc

                # (b) winning config: add n_synth synthetics.
                X_tr, y_tr = build_training_set(
                    X_real, y_real, pools.get(gen_name), n_synth
                )
                model, _ = downstream.train_classifier(
                    X_tr, y_tr, X_test, y_test, seed=seed
                )
                s = downstream.predict_scores(model, X_test)
                thr = downstream.best_threshold(y_test, s)
                pr_win = downstream.evaluate_scores(y_test, s, thr).pr_auc

                rows.append(
                    {
                        "seed": seed,
                        "fold": fold_idx,
                        "n_real": -1 if n_real is None else int(n_real),
                        "generator": gen_name,
                        "n_synth": int(n_synth),
                        "pr_base": pr_base,
                        "pr_win": pr_win,
                        "gain": pr_win / pr_base - 1.0 if pr_base > 0 else float("nan"),
                    }
                )
                print(
                    f"  [seed {seed} fold {fold_idx}] r={n_real} "
                    f"{gen_name:16s} gain={rows[-1]['gain']:+.3f}",
                    flush=True,
                )

    table = pd.DataFrame(rows)
    table.to_csv(PURGED_KFOLD_CSV, index=False)
    return table


def purged_kfold_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- SEM of the relative gain per generator and level of real data."""
    grouped = table.groupby(["n_real", "generator", "n_synth"])["gain"]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    return out
