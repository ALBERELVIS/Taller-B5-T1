"""Single place where the Keras backend is chosen.

TensorFlow is installed but its DLLs are blocked by an application-control
policy on this machine, so we run Keras 3 on the PyTorch backend. That is also
what the professor does in ``Taller_con_Datos_SP500_promedio.ipynb`` and it
gives us the GPU.

Every other module imports Keras from here so the environment variable is
guaranteed to be set before Keras is first imported.
"""

from __future__ import annotations

import os
import random

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seeds(seed: int) -> None:
    """Make a single training run reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    keras.utils.set_random_seed(seed)


__all__ = ["keras", "torch", "DEVICE", "set_seeds"]
