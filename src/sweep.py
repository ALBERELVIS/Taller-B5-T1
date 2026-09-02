"""The experiment: a double loop over real and synthetic sample counts.

This reproduces the analysis the professor spent half his session insisting on
([00:42:23] onwards): for each generator, plot the downstream error against the
number of real samples, with one curve per amount of synthetic data. His
expected shape is

* large gains from synthetic data when real data is scarce;
* no gain, or a loss, once there is plenty of real data;
* and a trivial noise generator that is a surprisingly hard baseline to beat.

Structure of one experiment cell
--------------------------------
``(seed, generator, n_real, n_synth)`` ->

1. subsample ``n_real`` real training windows (keeping the natural class
   balance);
2. append ``n_synth`` synthetic **minority** windows, all labelled 1;
3. train the frozen architecture from scratch;
4. tune the decision threshold on the real validation set, evaluate on the real
   test set.

Invariants that make the comparison honest
------------------------------------------
* the architecture never changes;
* validation and test are 100% real in every cell;
* generators are fitted once per seed on the minority windows of the **training
  block only**, then sampled -- never refitted per cell, which also means the
  synthetic pool is identical across ``n_real`` values;
* ``n_synth = 0`` does not depend on the generator, so it is computed once per
  ``(seed, n_real)`` and shared, saving a fifth of the runtime;
* every run appends its row to ``results.csv`` immediately and writes its loss
  curve to disk, so the sweep is resumable and every reported number has a
  convergence plot behind it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, downstream
from . import generators as gens
from .data import Split
from .representation import WindowRepresentation

RESULTS_CSV = config.RESULTS_DIR / "results.csv"

_ROW_KEYS = ["seed", "generator", "n_real", "n_synth"]


# --------------------------------------------------------------------------
# Building one training set
# --------------------------------------------------------------------------
def subsample_real(
    X: np.ndarray, y: np.ndarray, n_real: int | None, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Take ``n_real`` training windows, preserving the class balance.

    Why a stratified random subsample and not simply the most recent ``n_real``
    windows: taking the tail would change the historical period at the same time
    as the sample size, so a drop in error could be scarcity or could be a
    different market regime. Sampling across the whole training block isolates
    the effect we want to measure. The class balance is preserved so that the
    minority rate does not itself become a moving part.
    """
    if n_real is None or n_real >= len(y):
        return X, y

    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_pos = max(int(round(n_real * len(pos) / len(y))), 1)
    n_neg = n_real - n_pos

    take = np.concatenate(
        [rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
    )
    rng.shuffle(take)
    return X[take], y[take]


def build_training_set(
    X_real: np.ndarray,
    y_real: np.ndarray,
    X_synth_pool: np.ndarray | None,
    n_synth: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Real windows plus ``n_synth`` synthetic minority windows.

    All synthetic windows carry label 1. That is the conditional-generation
    strategy the professor recommends for imbalanced problems at [00:28:15]:
    generate only the class you are short of. Note the side effect, which we
    discuss in the report: for small ``n_real`` and large ``n_synth`` the class
    prior of the training set is badly distorted. Ranking metrics such as PR-AUC
    are robust to that, which is one more reason they are the right choice here.
    """
    if n_synth <= 0 or X_synth_pool is None:
        return X_real, y_real
    X_syn = X_synth_pool[:n_synth]
    if len(X_syn) < n_synth:
        raise ValueError(f"synthetic pool too small: {len(X_syn)} < {n_synth}")
    return (
        np.concatenate([X_real, X_syn]),
        np.concatenate([y_real, np.ones(len(X_syn), dtype=y_real.dtype)]),
    )


# --------------------------------------------------------------------------
# Generator fitting (once per seed)
# --------------------------------------------------------------------------
def fit_generators(
    split: Split,
    seed: int,
    names: list[str] | None = None,
    pool_size: int | None = None,
    save_histories: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Fit every generator on the minority training windows and draw one pool.

    Returns ``(pools, histories)``. Drawing a single pool per generator and
    slicing prefixes of it -- exactly what the professor does with
    ``X_train_aux_synth[0:1000]`` -- keeps the smaller ``n_synth`` settings
    nested inside the larger ones, so the curves are not contaminated by
    sampling noise between grid points.
    """
    names = names or list(gens.REGISTRY)
    pool_size = pool_size or max(config.N_SYNTH_GRID)

    X_min = split.X_train[split.y_train == 1]
    # Shared representation fitted on the whole training block: same space for
    # every latent-space generator, and estimated from 11k windows instead of
    # the ~570 minority ones.
    representation = WindowRepresentation(
        n_components=config.GENERATOR_SPACE, seed=seed
    ).fit(split.X_train)

    pools: dict[str, np.ndarray] = {}
    histories: dict[str, dict] = {}
    for name in names:
        cls = gens.REGISTRY[name]
        kwargs = {"seed": seed}
        if issubclass(cls, gens.LatentSpaceGenerator):
            kwargs["representation"] = representation

        t0 = time.time()
        gen = cls(**kwargs).fit(X_min)
        pools[name] = gen.sample(pool_size)
        histories[name] = gen.history
        if save_histories:
            gen.save_history()
        print(
            f"  [seed {seed}] {name:15s} ajustado en {time.time() - t0:5.1f}s "
            f"-> pool {pools[name].shape}",
            flush=True,
        )
    return pools, histories


# --------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------
def load_results() -> pd.DataFrame:
    if RESULTS_CSV.exists():
        return pd.read_csv(RESULTS_CSV)
    return pd.DataFrame()


def _append_row(row: dict) -> None:
    df = pd.DataFrame([row])
    df.to_csv(RESULTS_CSV, mode="a", header=not RESULTS_CSV.exists(), index=False)


def _already_done(done: pd.DataFrame, row: dict) -> bool:
    if done.empty:
        return False
    mask = np.ones(len(done), dtype=bool)
    for k in _ROW_KEYS:
        col = done[k]
        want = row[k]
        mask &= col.isna() if want is None else (col == want)
    return bool(mask.any())


def run_sweep(
    split: Split,
    seeds: list[int] | None = None,
    n_real_grid: list[int | None] | None = None,
    n_synth_grid: list[int] | None = None,
    generator_names: list[str] | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Run the full grid, appending one row per cell to ``results.csv``."""
    seeds = seeds or config.SEEDS
    n_real_grid = n_real_grid or config.N_REAL_GRID
    n_synth_grid = n_synth_grid or config.N_SYNTH_GRID
    generator_names = generator_names or list(gens.REGISTRY)

    done = load_results() if resume else pd.DataFrame()
    if resume and not done.empty:
        print(f"Reanudando: {len(done)} celdas ya calculadas en {RESULTS_CSV.name}")

    total_start = time.time()
    for seed in seeds:
        pools, _ = fit_generators(split, seed, names=generator_names)

        for n_real in n_real_grid:
            rng = np.random.default_rng(1000 * seed + (n_real or 0))
            X_real, y_real = subsample_real(
                split.X_train, split.y_train, n_real, rng
            )

            # n_synth = 0 is the same experiment for every generator.
            jobs: list[tuple[str, int]] = [("sin_sinteticos", 0)]
            jobs += [
                (name, n_synth)
                for name in generator_names
                for n_synth in n_synth_grid
                if n_synth > 0
            ]

            for name, n_synth in jobs:
                row_key = {
                    "seed": seed,
                    "generator": name,
                    "n_real": n_real,
                    "n_synth": n_synth,
                }
                if _already_done(done, row_key):
                    continue

                X_tr, y_tr = build_training_set(
                    X_real, y_real, pools.get(name), n_synth
                )
                t0 = time.time()
                model, history = downstream.train_classifier(
                    X_tr, y_tr, split.X_val, split.y_val, seed=seed
                )
                m_val, m_test, _ = downstream.evaluate_model(
                    model, split.X_val, split.y_val, split.X_test, split.y_test
                )
                elapsed = time.time() - t0

                tag = f"s{seed}_{name}_r{n_real or 'all'}_g{n_synth}"
                (config.HISTORIES_DIR / f"down_{tag}.json").write_text(
                    json.dumps(history)
                )

                row = {
                    **row_key,
                    "n_real_efectivo": int(len(y_real)),
                    "n_train": int(len(y_tr)),
                    "tasa_positivos_train": float(y_tr.mean()),
                    "epocas": len(history["loss"]),
                    "segundos": round(elapsed, 1),
                    **{f"val_{k}": v for k, v in m_val.as_dict().items()},
                    **{f"test_{k}": v for k, v in m_test.as_dict().items()},
                }
                _append_row(row)
                print(
                    f"  [seed {seed}] r={str(n_real or 'all'):>5s} g={n_synth:>4d} "
                    f"{name:16s} val_pr={m_val.pr_auc:.3f} test_pr={m_test.pr_auc:.3f} "
                    f"({elapsed:4.1f}s)",
                    flush=True,
                )

    print(f"\nBarrido terminado en {(time.time() - total_start) / 60:.1f} min")
    return load_results()


def main() -> None:
    from .data import load_problem

    split, _, _ = load_problem()
    print(split.summary().to_string(index=False))
    run_sweep(split)


if __name__ == "__main__":
    main()
