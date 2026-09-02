"""Common interface for every generative model.

Every generator, from "add a bit of noise" to the quantum circuit, exposes the
same three methods:

    fit(X)      # X: real minority windows, shape (n, T, A)
    sample(n)   # -> synthetic minority windows, shape (n, T, A)
    history     # dict of loss curves, empty for the closed-form ones

That uniformity is what makes the comparison fair and the sweep trivial: the
sweep code never knows which generator it is using. It is also what the
professor pointed out at the end of his session -- once you have a sampler, the
evaluation machinery is identical for all of them.

Design note: generators receive and return **raw return windows**, not latent
vectors. Whether a model internally works in the native standardised space, in
a PCA space or in the time domain is its own business, so each one can use the
representation that suits its inductive bias while still being plugged into the
exact same downstream experiment.
"""

from __future__ import annotations

import abc
import json
from pathlib import Path

import numpy as np

from .. import config
from ..representation import WindowRepresentation


class Generator(abc.ABC):
    """Base class: name, seed, loss history and the fit/sample contract."""

    #: short label used in filenames, tables and plot legends
    name: str = "generator"
    #: human-readable label for the report
    label: str = "Generator"

    def __init__(self, seed: int = config.SEED):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.history: dict[str, list[float]] = {}
        self.n_steps: int | None = None
        self.n_assets: int | None = None
        self._fitted = False

    # -- contract ---------------------------------------------------------
    @abc.abstractmethod
    def _fit(self, X: np.ndarray) -> None:
        ...

    @abc.abstractmethod
    def _sample(self, n: int) -> np.ndarray:
        ...

    def fit(self, X: np.ndarray) -> "Generator":
        """Fit on real minority windows ``(n, T, A)``.

        ``X`` must come from the training block only. Fitting a generator on
        validation or test windows would inject future information into the
        synthetic data and silently invalidate every result downstream.
        """
        if X.ndim != 3:
            raise ValueError(f"expected (n, T, A), got {X.shape}")
        self.n_steps, self.n_assets = X.shape[1], X.shape[2]
        self._fit(X)
        self._fitted = True
        return self

    def sample(self, n: int) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: call fit() before sample()")
        if n <= 0:
            return np.empty((0, self.n_steps, self.n_assets), dtype=np.float32)
        out = self._sample(n)
        expected = (n, self.n_steps, self.n_assets)
        if out.shape != expected:
            raise ValueError(f"{self.name}: sample returned {out.shape}, expected {expected}")
        return out.astype(np.float32)

    # -- bookkeeping ------------------------------------------------------
    def save_history(self, directory: Path | None = None) -> Path | None:
        """Persist the loss curves so the report can prove convergence.

        The statement requires a loss curve for *every* training, so histories
        are written to disk rather than only shown inside a notebook.
        """
        if not self.history:
            return None
        directory = directory or config.HISTORIES_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"gen_{self.name}_seed{self.seed}.json"
        path.write_text(json.dumps(self.history, indent=2))
        return path

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, seed={self.seed})"


class LatentSpaceGenerator(Generator):
    """Generators that model the flattened window in a fixed vector space.

    Handles the representation round-trip (standardise, optionally PCA, and back)
    so subclasses only deal with plain ``(n, d)`` matrices. This is the same
    flatten/reshape trick the professor uses to turn a ``(60+1, 23)`` block into
    a single 1403-dim vector, generalised so the space is configurable.
    """

    def __init__(
        self,
        seed: int = config.SEED,
        n_components: int | float | None = config.GENERATOR_SPACE,
        representation: WindowRepresentation | None = None,
    ):
        super().__init__(seed=seed)
        self.n_components = n_components
        # A representation fitted on the *whole* training block can be injected
        # so that every generator shares the same space (fairer comparison and
        # less noisy scaling than estimating it from ~500 minority windows).
        self.representation = representation

    def _fit(self, X: np.ndarray) -> None:
        if self.representation is None:
            self.representation = WindowRepresentation(
                n_components=self.n_components, seed=self.seed
            ).fit(X)
        self._fit_latent(self.representation.transform(X))

    def _sample(self, n: int) -> np.ndarray:
        assert self.representation is not None
        return self.representation.inverse_transform(self._sample_latent(n))

    @property
    def dim(self) -> int:
        assert self.representation is not None, "call fit() first"
        return self.representation.dim

    @abc.abstractmethod
    def _fit_latent(self, Z: np.ndarray) -> None:
        ...

    @abc.abstractmethod
    def _sample_latent(self, n: int) -> np.ndarray:
        ...
