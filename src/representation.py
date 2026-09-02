"""Shared representation in which the generative models live.

The problem we had to solve
---------------------------
A raw window is ``60 days x 23 assets = 1380`` numbers, but the training block
only contains ~490 stress windows. Estimating a density in 1380 dimensions from
490 samples is ill-posed: the empirical covariance has rank <= 489 and is
singular, so ``multivariate_normal`` cannot even sample from it.

Our first plan was to fix this with a PCA keeping 90% of the variance, expecting
the professor's rule of thumb (~10% of the dimensions). We measured it instead
of assuming it, and the answer was 986 of 1380 components. The explained
variance curve is almost a straight line: 60 components capture 31%, 400
capture 61%. Daily returns are close to white noise across time, so **there is
no low-dimensional linear structure to exploit**. Worse, truncating the PCA
removes variance, and the reconstructed windows are systematically too calm --
exactly the wrong bias for a model whose job is to recognise turbulence.

What we do instead
------------------
* **Native space (default).** Generators work on the standardised 1380-dim
  window. No information is thrown away. The singular-covariance problem is
  solved where it belongs, inside the Gaussian generator, with Ledoit-Wolf
  shrinkage, which is designed for the ``n < p`` regime.
* **PCA space (opt-in).** Still available with an explicit ``n_components``. We
  use it for the ablation that documents the decision above, and for the quantum
  bonus, where a small space is what makes simulation possible at all.

Both the scaler and the PCA are fitted on the **training block only**. Fitting
them on the whole dataset would leak test-period statistics into training, the
subtlest and most common form of look-ahead bias.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import config


def flatten_windows(X: np.ndarray) -> np.ndarray:
    """``(N, T, A) -> (N, T*A)``, the same round-trip the professor uses."""
    return X.reshape(X.shape[0], -1)


def unflatten_windows(X_flat: np.ndarray, n_steps: int, n_assets: int) -> np.ndarray:
    """``(N, T*A) -> (N, T, A)``."""
    return X_flat.reshape(X_flat.shape[0], n_steps, n_assets)


class WindowRepresentation:
    """Standardise, and optionally project, with an exact inverse back to windows.

    Parameters
    ----------
    n_components
        ``None`` keeps the native standardised space (1380 dims, no loss).
        An ``int`` projects onto that many principal components.
        A ``float`` in (0, 1) targets that cumulative explained variance.

    Generators only ever see ``transform`` output and only ever produce vectors
    that go back through ``inverse_transform``, so they are completely agnostic
    to which of the two spaces is in use.
    """

    def __init__(self, n_components: int | float | None = None, seed: int = config.SEED):
        self.n_components = n_components
        self.seed = seed
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.n_steps: int | None = None
        self.n_assets: int | None = None

    def fit(self, X_train: np.ndarray) -> "WindowRepresentation":
        self.n_steps, self.n_assets = X_train.shape[1], X_train.shape[2]
        scaled = self.scaler.fit_transform(flatten_windows(X_train))
        if self.n_components is not None:
            solver = "full" if isinstance(self.n_components, float) else "randomized"
            self.pca = PCA(
                n_components=self.n_components, svd_solver=solver, random_state=self.seed
            ).fit(scaled)
        return self

    @property
    def dim(self) -> int:
        """Dimensionality of the space the generators work in."""
        if self.pca is not None:
            return int(self.pca.n_components_)
        return int(self.n_steps * self.n_assets)

    @property
    def name(self) -> str:
        return "nativo" if self.pca is None else f"pca{self.dim}"

    def transform(self, X: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(flatten_windows(X))
        if self.pca is not None:
            scaled = self.pca.transform(scaled)
        return scaled.astype(np.float32)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float64)
        if self.pca is not None:
            Z = self.pca.inverse_transform(Z)
        flat = self.scaler.inverse_transform(Z)
        return unflatten_windows(flat, self.n_steps, self.n_assets).astype(np.float32)

    def reconstruction_error(self, X: np.ndarray) -> float:
        """Mean absolute error of the round-trip, in return units.

        Exactly zero (up to float precision) in the native space; reported for
        the PCA ablation so the reader can see how much is lost before any
        generator is trained.
        """
        return float(np.mean(np.abs(X - self.inverse_transform(self.transform(X)))))

    def explained_variance_curve(self) -> np.ndarray | None:
        if self.pca is None:
            return None
        return np.cumsum(self.pca.explained_variance_ratio_)


def variance_curve(X_train: np.ndarray, max_components: int = 400) -> np.ndarray:
    """Cumulative explained variance, used for the figure that justifies the choice."""
    scaled = StandardScaler().fit_transform(flatten_windows(X_train))
    pca = PCA(n_components=max_components, svd_solver="randomized",
              random_state=config.SEED).fit(scaled)
    return np.cumsum(pca.explained_variance_ratio_)
