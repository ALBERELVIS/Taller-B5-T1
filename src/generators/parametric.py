"""Generator 2 of 4: the classical parametric family (Gaussian and Student-t).

This is the professor's "no tan tonto" model, described at [00:11:02] and coded
live in ``Taller_Gaussian_solution.ipynb``: assume the flattened window follows a
known distribution, estimate its mean and covariance from the training data, and
sample from the fitted law.

Two things we add on top of his version, both of which he explicitly suggested
during the session:

**Student-t instead of Gaussian.** At [01:36:29] he says it would be interesting
to swap the Gaussian for a Student-t "porque los datos de finanzas, muchas veces
se distribuyen como una T-student", and at [01:17:16] he shows the consequence of
not doing it: the Gaussian synthetic windows never produce the extreme moves that
the real ones do, because a Gaussian has no fat tails. Since our minority class
*is* the turbulent regime, losing the tails is losing the point.

**Ledoit-Wolf shrinkage.** The window has 1380 dimensions and there are only
~570 stress windows in training, so the sample covariance has rank <= 566 and is
singular: ``multivariate_normal`` would refuse to sample from it (or silently
return garbage). Ledoit-Wolf shrinks the sample covariance towards a scaled
identity with an analytically optimal intensity, which is exactly the estimator
designed for this ``n < p`` regime and is standard practice in portfolio
construction. This is what let us keep the native 1380-dim space instead of
truncating with PCA.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from sklearn.covariance import LedoitWolf

from .. import config
from .base import LatentSpaceGenerator


class GaussianGenerator(LatentSpaceGenerator):
    """Multivariate normal fitted with Ledoit-Wolf shrinkage."""

    name = "gaussiano"
    label = "Gaussiano multivariante"

    def __init__(self, seed: int = config.SEED, shrinkage: bool = True, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.shrinkage = shrinkage
        self.mean_: np.ndarray | None = None
        self.chol_: np.ndarray | None = None
        self.shrinkage_coef_: float | None = None

    def _estimate(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = Z.mean(axis=0)
        if self.shrinkage:
            lw = LedoitWolf(assume_centered=False).fit(Z)
            cov = lw.covariance_
            self.shrinkage_coef_ = float(lw.shrinkage_)
        else:
            cov = np.cov(Z.T)
            self.shrinkage_coef_ = 0.0
        return mean, cov

    def _fit_latent(self, Z: np.ndarray) -> None:
        Z = np.asarray(Z, dtype=np.float64)
        self.mean_, cov = self._estimate(Z)
        # Sampling via a Cholesky factor is ~100x faster than
        # rng.multivariate_normal in 1380 dimensions, and the jitter guarantees
        # positive definiteness even if shrinkage is disabled.
        self.chol_ = _safe_cholesky(cov)

    def _sample_latent(self, n: int) -> np.ndarray:
        assert self.mean_ is not None and self.chol_ is not None
        z = self.rng.standard_normal((n, len(self.mean_)))
        return self.mean_ + z @ self.chol_.T


class StudentTGenerator(GaussianGenerator):
    """Multivariate Student-t: same first two moments, but with fat tails.

    Sampled through the standard Gaussian-scale-mixture representation

        x = mu + sqrt(nu / w) * L z ,   w ~ chi2(nu),  z ~ N(0, I)

    with the covariance rescaled by ``(nu - 2) / nu`` so that the resulting
    distribution keeps the *same* covariance as the Gaussian generator. Without
    that correction the Student-t would also be more volatile overall, and we
    could not tell whether any improvement came from the fat tails or simply
    from the larger variance.
    """

    name = "student_t"
    label = "t-Student multivariante"

    def __init__(self, seed: int = config.SEED, df: float = 4.0, **kwargs):
        super().__init__(seed=seed, **kwargs)
        if df <= 2:
            raise ValueError("df must be > 2 for the covariance to exist")
        self.df = df

    def _fit_latent(self, Z: np.ndarray) -> None:
        Z = np.asarray(Z, dtype=np.float64)
        self.mean_, cov = self._estimate(Z)
        self.chol_ = _safe_cholesky(cov * (self.df - 2.0) / self.df)

    def _sample_latent(self, n: int) -> np.ndarray:
        assert self.mean_ is not None and self.chol_ is not None
        z = self.rng.standard_normal((n, len(self.mean_)))
        w = self.rng.chisquare(self.df, size=(n, 1))
        return self.mean_ + np.sqrt(self.df / w) * (z @ self.chol_.T)


def _safe_cholesky(cov: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor, adding the smallest jitter that makes it work."""
    jitter = 0.0
    scale = float(np.mean(np.diag(cov))) or 1.0
    for _ in range(8):
        try:
            return linalg.cholesky(cov + jitter * np.eye(len(cov)), lower=True)
        except linalg.LinAlgError:
            jitter = max(jitter * 10, 1e-10 * scale)
    # Last resort: symmetric eigendecomposition with negative eigenvalues clipped.
    vals, vecs = np.linalg.eigh(cov)
    return vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None)))
