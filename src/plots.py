"""Every figure reported in the README and the slides is produced here.

The statement is explicit that "el codigo debe generar todas las graficas y
tablas reportadas", so no figure is drawn by hand inside a notebook: the
notebooks call these functions and the functions save to ``results/figures``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402

plt.style.use("ggplot")

#: consistent colour and order for the generators across every figure
GEN_ORDER = [
    "sin_sinteticos", "ruido", "gaussiano", "student_t",
    "factor_mercado", "vae", "gan", "autoregresivo",
]
GEN_LABELS = {
    "sin_sinteticos": "Sin sintéticos",
    "ruido": "Ruido (trivial)",
    "gaussiano": "Gaussiano",
    "student_t": "t-Student",
    "factor_mercado": "Factor de mercado",
    "vae": "VAE",
    "gan": "GAN",
    "autoregresivo": "Autorregresivo",
    "gan_cuantica": "GAN híbrida cuántica",
    "gan_clasica_equiparada": "GAN clásica equiparada",
}
GEN_COLORS = {
    "sin_sinteticos": "#444444",
    "ruido": "#1f77b4",
    "gaussiano": "#ff7f0e",
    "student_t": "#d62728",
    "factor_mercado": "#e377c2",
    "vae": "#2ca02c",
    "gan": "#9467bd",
    "autoregresivo": "#8c564b",
    "gan_cuantica": "#17becf",
    "gan_clasica_equiparada": "#bcbd22",
}


def _save(fig: plt.Figure, name: str) -> Path:
    path = config.FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Notebook 01: the problem
# --------------------------------------------------------------------------
def plot_target_timeline(
    dates: pd.DatetimeIndex,
    target_value: np.ndarray,
    y: np.ndarray,
    threshold: float,
    slices: tuple[slice, slice, slice],
    name: str = "01_timeline_target",
) -> Path:
    """Forward volatility through time, with the stress windows shaded.

    The point of this figure is the *clustering*: the positives are not
    scattered, they come in a handful of blocks. That is the whole argument for
    why the minority class is far smaller than its raw count suggests.
    """
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.plot(dates, target_value, lw=0.6, color="#333333", label="Volatilidad futura (20d, anualizada)")
    ax.axhline(threshold, color="#d62728", ls="--", lw=1.2,
               label=f"Umbral (percentil 95 de train) = {threshold:.2f}")
    ax.fill_between(dates, 0, target_value.max(), where=y == 1,
                    color="#d62728", alpha=0.18, label="Ventanas de estrés")

    for sl, txt, color in zip(slices, ("TRAIN", "VAL", "TEST"), ("#1f77b4", "#ff7f0e", "#2ca02c")):
        ax.axvspan(dates[sl][0], dates[sl][-1], color=color, alpha=0.07)
        ax.text(dates[sl][len(dates[sl]) // 2], target_value.max() * 0.95, txt,
                ha="center", va="top", fontsize=9, color=color, fontweight="bold")

    ax.set_ylabel("Volatilidad anualizada")
    ax.set_xlabel("Fecha de decisión")
    ax.set_title("Target: régimen de alta volatilidad futura. Los eventos raros vienen en bloques")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, name)


def plot_variance_curve(cum_variance: np.ndarray, name: str = "01_pca_varianza") -> Path:
    """Cumulative explained variance: the figure that killed the PCA plan."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    k = np.arange(1, len(cum_variance) + 1)
    ax.plot(k, cum_variance, lw=2, color="#1f77b4")
    ax.plot(k, k / 1380.0, ls=":", color="#888888", label="Ruido blanco (referencia lineal)")
    for target in (0.5, 0.9):
        idx = int(np.searchsorted(cum_variance, target))
        if idx < len(cum_variance):
            ax.axhline(target, color="#d62728", ls="--", lw=0.8)
            ax.annotate(f"{target:.0%} -> {idx + 1} componentes",
                        xy=(idx + 1, target), xytext=(idx + 20, target - 0.08),
                        fontsize=8, color="#d62728")
    ax.set_xlabel("Número de componentes principales")
    ax.set_ylabel("Varianza explicada acumulada")
    ax.set_title("Los retornos diarios no se comprimen: la curva es casi una recta")
    ax.legend(fontsize=8)
    return _save(fig, name)


# --------------------------------------------------------------------------
# Notebook 02: downstream model selection
# --------------------------------------------------------------------------
def plot_model_comparison(
    table: pd.DataFrame,
    metric: str = "test_pr_auc",
    baseline_rate: float | None = None,
    name: str = "02_comparativa_modelos",
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = ["#8c8c8c" if "baseline" in str(i).lower() or "vol" in str(i).lower()
              else "#1f77b4" for i in table.index]
    bars = ax.bar(range(len(table)), table[metric].values, color=colors)
    if baseline_rate is not None:
        ax.axhline(baseline_rate, color="#d62728", ls="--", lw=1.2,
                   label=f"Azar (tasa base = {baseline_rate:.3f})")
        ax.legend(fontsize=8)
    for b, v in zip(bars, table[metric].values):
        ax.annotate(f"{v:.3f}", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels(table.index, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("PR-AUC en test")
    ax.set_title("Selección del modelo downstream sobre datos 100% reales")
    return _save(fig, name)


def plot_loss_curves(
    histories: dict[str, dict],
    name: str,
    title: str,
    ylabel: str = "loss",
    logy: bool = False,
) -> Path:
    """Grid of loss curves, one panel per model.

    The statement asks for a convergence curve for every training, so this is
    used both for the generators and for the downstream runs.
    """
    n = len(histories)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows), squeeze=False)

    for ax, (key, hist) in zip(axes.ravel(), histories.items()):
        for series, values in hist.items():
            if series.startswith(("pasos_", "batch_")) or not len(values):
                continue
            ax.plot(values, lw=1.2, label=series)
        ax.set_title(GEN_LABELS.get(key, key), fontsize=10)
        ax.set_xlabel("época")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12, y=1.01)
    return _save(fig, name)


# --------------------------------------------------------------------------
# Notebook 03: quality of the synthetic data
# --------------------------------------------------------------------------
def plot_real_vs_synth_windows(
    X_real: np.ndarray,
    synth: dict[str, np.ndarray],
    n_examples: int = 3,
    name: str = "03_ventanas_real_vs_sintetico",
) -> Path:
    """Individual windows, the professor's own visual check.

    Each panel shows one sample: 23 asset paths over 60 days. What to look for
    is what he points out at [01:20:02]: real assets move *together* in packs,
    and a generator that misses that co-movement produces windows that look
    plausible asset by asset while being far too diversified as a portfolio.
    """
    keys = list(synth)
    fig, axes = plt.subplots(
        len(keys) + 1, n_examples,
        figsize=(3.3 * n_examples, 2.3 * (len(keys) + 1)),
        squeeze=False, sharey=True,
    )
    for j in range(n_examples):
        axes[0, j].plot(np.cumsum(X_real[j * 7], axis=0), lw=0.7)
        axes[0, j].set_title(f"REAL #{j}", fontsize=9)
    for i, key in enumerate(keys, start=1):
        for j in range(n_examples):
            axes[i, j].plot(np.cumsum(synth[key][j * 7], axis=0), lw=0.7)
            axes[i, j].set_title(f"{GEN_LABELS.get(key, key)} #{j}", fontsize=9)
    for ax in axes.ravel():
        ax.tick_params(labelsize=7)
    fig.suptitle("Retorno acumulado de los 23 activos en una ventana (real vs sintético)",
                 fontsize=12, y=1.005)
    return _save(fig, name)


def plot_distribution_diagnostics(
    X_real: np.ndarray,
    synth: dict[str, np.ndarray],
    name: str = "03_diagnostico_distribuciones",
) -> Path:
    """Three diagnostics that decide whether a generator is usable here."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    def market_vol(X):
        return X.mean(axis=2).std(axis=1)

    # (a) distribution of daily returns, log scale to expose the tails
    bins = np.linspace(-0.12, 0.12, 120)
    axes[0].hist(X_real.ravel(), bins=bins, density=True, histtype="step",
                 lw=2.2, color="black", label="Real")
    for key, Xs in synth.items():
        axes[0].hist(Xs.ravel(), bins=bins, density=True, histtype="step",
                     lw=1.2, color=GEN_COLORS.get(key), label=GEN_LABELS.get(key, key))
    axes[0].set_yscale("log")
    axes[0].set_title("(a) Retornos diarios: colas")
    axes[0].set_xlabel("retorno diario")
    axes[0].legend(fontsize=7)

    def mean_corr(X):
        out = []
        for w in X[:400]:
            c = np.corrcoef(w.T)
            out.append(np.nanmean(c[np.triu_indices_from(c, k=1)]))
        return np.array(out)

    # (b) and (c) are box plots rather than histograms: seven overlaid densities
    # are unreadable, and what we need to compare here is the location and
    # spread of each generator against the real one, not the exact shape.
    for ax, fn, title, xlabel in (
        (axes[1], market_vol, "(b) Volatilidad de la cartera equiponderada",
         "desviación típica diaria de la cartera"),
        (axes[2], mean_corr, "(c) Correlación media entre activos",
         "correlación media por ventana"),
    ):
        keys = ["REAL"] + list(synth)
        values = [fn(X_real)] + [fn(Xs) for Xs in synth.values()]
        bp = ax.boxplot(values, vert=False, showfliers=False, patch_artist=True,
                        widths=0.6, medianprops={"color": "black"})
        for patch, key in zip(bp["boxes"], keys):
            patch.set_facecolor("#000000" if key == "REAL" else GEN_COLORS.get(key, "#888888"))
            patch.set_alpha(0.45 if key == "REAL" else 0.75)
        # Reference line at the real median, so the gap is readable at a glance.
        ax.axvline(np.median(values[0]), color="black", ls="--", lw=1.2)
        ax.set_yticklabels(["REAL"] + [GEN_LABELS.get(k, k) for k in synth], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel(xlabel)

    fig.suptitle("¿Reproducen los generadores lo que define el estrés de mercado?",
                 fontsize=12, y=1.02)
    return _save(fig, name)


# --------------------------------------------------------------------------
# Notebook 04/05: the sweep, i.e. the key figure of the workshop
# --------------------------------------------------------------------------
def aggregate(results: pd.DataFrame, metric: str = "test_pr_auc") -> pd.DataFrame:
    """Mean and standard error over seeds for each experiment cell."""
    df = results.copy()
    df["n_real"] = df["n_real"].fillna(-1)
    grouped = df.groupby(["generator", "n_real", "n_synth"])[metric]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    return out


def plot_sweep_per_generator(
    results: pd.DataFrame,
    metric: str = "test_pr_auc",
    name: str = "04_barrido_por_generador",
) -> Path:
    """One panel per generator: metric vs number of real samples, curve per n_synth.

    This is the shape the professor asked for at [00:42:23]. The "Sin
    sintéticos" curve is repeated in every panel as the reference to beat.
    """
    agg = aggregate(results, metric)
    generators = [g for g in GEN_ORDER if g in set(agg["generator"]) and g != "sin_sinteticos"]
    baseline = agg[agg["generator"] == "sin_sinteticos"].sort_values("n_real")

    ncols = 3
    nrows = int(np.ceil(len(generators) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows),
                             squeeze=False, sharey=True)
    synth_levels = sorted(v for v in agg["n_synth"].unique() if v > 0)
    cmap = plt.get_cmap("viridis")

    for ax, gen in zip(axes.ravel(), generators):
        sub = agg[agg["generator"] == gen]
        ax.errorbar(_xpos(baseline["n_real"]), baseline["mean"], yerr=baseline["sem"],
                    color="black", lw=2.2, marker="o", ms=4, label="0 sintéticos", zorder=5)
        for i, lvl in enumerate(synth_levels):
            s = sub[sub["n_synth"] == lvl].sort_values("n_real")
            if s.empty:
                continue
            ax.errorbar(_xpos(s["n_real"]), s["mean"], yerr=s["sem"],
                        color=cmap(i / max(len(synth_levels) - 1, 1)),
                        lw=1.5, marker="s", ms=3.5, label=f"{lvl} sintéticos")
        ax.set_title(GEN_LABELS.get(gen, gen), fontsize=11)
        ax.set_xscale("log")
        ax.set_xlabel("nº de datos reales de entrenamiento")
        ax.set_ylabel(_metric_label(metric))
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(generators):]:
        ax.axis("off")
    fig.suptitle(
        f"{_metric_label(metric)} frente a datos reales, por cantidad de datos sintéticos",
        fontsize=13, y=1.01,
    )
    return _save(fig, name)


def plot_sweep_generator_comparison(
    results: pd.DataFrame,
    metric: str = "test_pr_auc",
    name: str = "04_comparativa_generadores",
) -> Path:
    """One panel per n_real: every generator's best synthetic amount side by side."""
    agg = aggregate(results, metric)
    n_reals = sorted(agg["n_real"].unique())
    ncols = min(3, len(n_reals))
    nrows = int(np.ceil(len(n_reals) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows), squeeze=False)

    for ax, nr in zip(axes.ravel(), n_reals):
        sub = agg[agg["n_real"] == nr]
        base = sub[sub["generator"] == "sin_sinteticos"]["mean"]
        base = float(base.iloc[0]) if len(base) else np.nan
        ax.axhline(base, color="black", ls="--", lw=1.5, label="Sin sintéticos")
        for gen in GEN_ORDER:
            if gen == "sin_sinteticos":
                continue
            s = sub[sub["generator"] == gen].sort_values("n_synth")
            if s.empty:
                continue
            ax.errorbar(s["n_synth"], s["mean"], yerr=s["sem"], marker="o", ms=4,
                        lw=1.5, color=GEN_COLORS.get(gen), label=GEN_LABELS.get(gen, gen))
        label = "todos" if nr < 0 else f"{int(nr)}"
        ax.set_title(f"{label} datos reales", fontsize=11)
        ax.set_xlabel("nº de sintéticos añadidos")
        ax.set_ylabel(_metric_label(metric))
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(n_reals):]:
        ax.axis("off")
    fig.suptitle("Comparación entre modelos generativos", fontsize=13, y=1.01)
    return _save(fig, name)


def plot_gain_heatmap(
    results: pd.DataFrame,
    metric: str = "test_pr_auc",
    name: str = "05_heatmap_ganancia",
) -> Path:
    """Relative gain over the no-synthetic baseline, generator x n_real.

    Best amount of synthetic data per cell. Red means the synthetic data helped.
    """
    agg = aggregate(results, metric)
    base = (
        agg[agg["generator"] == "sin_sinteticos"]
        .set_index("n_real")["mean"]
        .to_dict()
    )
    gens_ = [g for g in GEN_ORDER if g in set(agg["generator"]) and g != "sin_sinteticos"]
    n_reals = sorted(agg["n_real"].unique())

    M = np.full((len(gens_), len(n_reals)), np.nan)
    for i, g in enumerate(gens_):
        for j, nr in enumerate(n_reals):
            s = agg[(agg["generator"] == g) & (agg["n_real"] == nr) & (agg["n_synth"] > 0)]
            if s.empty or nr not in base:
                continue
            M[i, j] = s["mean"].max() / base[nr] - 1.0

    fig, ax = plt.subplots(figsize=(1.4 * len(n_reals) + 4, 0.7 * len(gens_) + 3))
    lim = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(n_reals)))
    ax.set_xticklabels(["todos" if v < 0 else int(v) for v in n_reals])
    ax.set_yticks(range(len(gens_)))
    ax.set_yticklabels([GEN_LABELS.get(g, g) for g in gens_])
    for i in range(len(gens_)):
        for j in range(len(n_reals)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.1%}", ha="center", va="center", fontsize=9)
    ax.set_xlabel("nº de datos reales")
    ax.set_title(f"Mejor ganancia relativa en {_metric_label(metric)}\nfrente a no usar sintéticos")
    fig.colorbar(im, ax=ax, label="ganancia relativa")
    return _save(fig, name)


def fidelity_vs_gain(results: pd.DataFrame, quality: pd.DataFrame,
                     metric: str = "test_pr_auc") -> pd.DataFrame:
    """Join each generator's mean downstream gain with its fidelity statistics.

    This is the table behind the main analytical claim of the project: what
    predicts whether a generator helps is not how "realistic" its samples look
    in general, but how well it reproduces **the specific statistic the label is
    computed from** -- the volatility of the equal-weight portfolio.
    """
    agg = aggregate(results, metric)
    piv = agg.pivot_table(index=["generator", "n_synth"], columns="n_real", values="mean")
    base = piv.loc[("sin_sinteticos", 0)]
    real = quality.loc["REAL"]

    rows = []
    for gen in GEN_ORDER:
        if gen == "sin_sinteticos" or gen not in piv.index.get_level_values(0):
            continue
        if gen not in quality.index:
            continue
        best = piv.loc[gen].max()
        rows.append(
            {
                "generador": GEN_LABELS.get(gen, gen),
                "clave": gen,
                "ganancia_media": float(np.mean(best / base - 1.0)),
                "fidelidad_vol_cartera": quality.loc[gen, "vol_cartera"] / real["vol_cartera"],
                "fidelidad_correlacion": quality.loc[gen, "corr_media"] / real["corr_media"],
                "fidelidad_curtosis": quality.loc[gen, "kurtosis"] / real["kurtosis"],
            }
        )
    return pd.DataFrame(rows)


def plot_fidelity_vs_gain(
    table: pd.DataFrame,
    name: str = "05_fidelidad_vs_ganancia",
) -> Path:
    """Downstream gain against fidelity, for the statistic that matters and one that does not."""
    from scipy.stats import spearmanr

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    panels = [
        ("fidelidad_vol_cartera", "Fidelidad de la volatilidad de cartera\n(sintético / real)",
         "(a) La magnitud de la que depende la etiqueta"),
        ("fidelidad_curtosis", "Fidelidad de la curtosis\n(sintético / real)",
         "(b) Una magnitud que no la determina"),
    ]
    for ax, (col, xlabel, title) in zip(axes, panels):
        rho, pval = spearmanr(table[col], table["ganancia_media"])
        for _, r in table.iterrows():
            ax.scatter(r[col], r["ganancia_media"], s=140, zorder=3,
                       color=GEN_COLORS.get(r["clave"], "#888888"),
                       edgecolor="black", linewidth=0.8)
            ax.annotate(r["generador"], xy=(r[col], r["ganancia_media"]),
                        xytext=(6, 6), textcoords="offset points", fontsize=8)
        ax.axhline(0, color="black", ls="--", lw=1.0)
        ax.axvline(1.0, color="#888888", ls=":", lw=1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Ganancia media en PR-AUC frente a no usar sintéticos")
        ax.set_title(f"{title}\nSpearman ρ = {rho:+.2f} (p = {pval:.3f})", fontsize=10)
    fig.suptitle(
        "Un generador ayuda si reproduce la estadística concreta que define el target",
        fontsize=12, y=1.03,
    )
    return _save(fig, name)


def plot_backtest(
    equities: dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
    name: str = "05_backtest",
) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4.6))
    for label, eq in equities.items():
        style = {"color": "black", "lw": 2.2} if "hold" in label else {"lw": 1.6}
        ax.plot(dates[: len(eq)], eq, label=label, **style)
    ax.set_yscale("log")
    ax.set_ylabel("Valor de 1 EUR invertido (escala log)")
    ax.set_xlabel("Fecha")
    ax.set_title("Backtest en test: regla de de-risking frente a buy & hold")
    ax.legend(fontsize=8)
    return _save(fig, name)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _xpos(values) -> np.ndarray:
    """Map n_real to a plottable x, with the "all data" marker placed last."""
    v = np.asarray(values, dtype=float)
    return np.where(v < 0, 11336.0, v)


def _metric_label(metric: str) -> str:
    return {
        "test_pr_auc": "PR-AUC (test)",
        "val_pr_auc": "PR-AUC (val)",
        "test_roc_auc": "ROC-AUC (test)",
        "test_f1": "F1 (test)",
        "test_recall": "Recall (test)",
    }.get(metric, metric)


def load_downstream_histories(pattern: str = "down_*.json") -> dict[str, dict]:
    return {
        p.stem.replace("down_", ""): json.loads(p.read_text())
        for p in sorted(config.HISTORIES_DIR.glob(pattern))
    }


def load_generator_histories(seed: int = 0) -> dict[str, dict]:
    out = {}
    for p in sorted(config.HISTORIES_DIR.glob(f"gen_*_seed{seed}.json")):
        key = p.stem.replace("gen_", "").replace(f"_seed{seed}", "")
        out[key] = json.loads(p.read_text())
    return out
