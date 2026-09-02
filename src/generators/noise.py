"""Generator 1 of 4: the deliberately naive baseline required by the statement.

    "un cuarto modelo simple, por ejemplo que coja datos originales y les anada
     ruido"  --  enunciado, task 2

Take a real stress window, add a small Gaussian perturbation, keep the label.
That is the whole model. There is no training and nothing is learned, so it has
no loss curve.

Why it matters: it is the control experiment. Any sophisticated generator that
cannot beat "jitter the real data" has not earned its complexity. In the
professor's own live session this baseline was competitive with the Gaussian
generator, so we expect it to be a serious contender rather than a straw man.

Two details taken straight from the professor's remarks:

* ``sigma`` is kept small (0.1 of the per-asset training standard deviation) so
  the perturbation does not change the nature of the sample. At [01:43:29] he
  answers exactly this question for classification: perturb the input only and
  never touch the class label.
* Samples are drawn with replacement, so asking for more synthetic windows than
  there are real ones simply reuses the originals with fresh noise.
"""

from __future__ import annotations

import numpy as np

from .. import config
from .base import Generator


class NoiseGenerator(Generator):
    name = "ruido"
    label = "Ruido (baseline trivial)"

    def __init__(self, seed: int = config.SEED, sigma_ratio: float = 0.10):
        super().__init__(seed=seed)
        self.sigma_ratio = sigma_ratio
        self.X_real: np.ndarray | None = None
        self.sigma: np.ndarray | None = None

    def _fit(self, X: np.ndarray) -> None:
        self.X_real = X.copy()
        # One sigma per asset: perturbation scaled to each asset's own
        # volatility, otherwise a single global sigma would swamp the quiet
        # assets and barely touch the noisy ones.
        self.sigma = self.sigma_ratio * X.std(axis=(0, 1), keepdims=True)

    def _sample(self, n: int) -> np.ndarray:
        assert self.X_real is not None and self.sigma is not None
        idx = self.rng.integers(0, len(self.X_real), size=n)
        base = self.X_real[idx]
        return base + self.rng.normal(0.0, 1.0, size=base.shape) * self.sigma
