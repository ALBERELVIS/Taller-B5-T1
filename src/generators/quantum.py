"""Bonus: hybrid quantum-classical GAN, and its parameter-matched classical twin.

Not required by the statement. It is here as an extra generative model, clearly
labelled as such, because the three mandatory ones must come from the families
seen in class.

What is actually quantum
------------------------
The generator's core is a **variational quantum circuit** simulated on
PennyLane's ``default.qubit``. Latent noise is angle-encoded into
``n_qubits`` rotations, entangled by ``StronglyEntanglingLayers``, and read out
as the Pauli-Z expectation value of each qubit. Those expectations feed a tiny
linear decoder. The circuit parameters are trained by gradient descent through
the simulator, together with the decoder and against a classical
discriminator, so the quantum part is genuinely *trainable* rather than a fixed
feature map. No quantum hardware is involved.

Making the comparison fair
--------------------------
Comparing an 8-qubit circuit against a 1.4-million-parameter dense GAN would say
nothing. So:

* both generators live in the **same 16-dimensional PCA space**. A quantum
  circuit reading out 8 expectation values cannot address 1380 output
  dimensions without a large classical decoder that would do all the real work
  and make the "quantum" label meaningless;
* the classical twin is an MLP whose parameter count is matched **exactly** to
  the quantum generator's;
* identical discriminator, identical optimiser, identical latent dimension,
  identical number of epochs, identical seeds.

Whatever the outcome, it is then attributable to the generator family and not to
capacity. We do not assume the quantum model wins: with 8 qubits on a simulator
there is no reason it should, and the honest expected result is parity at best.

Everything runs in PyTorch rather than Keras because PennyLane's ``TorchLayer``
plugs straight into ``torch.nn``, and Torch is already the Keras backend here.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import numpy as np

from .. import config
from ..keras_setup import torch
from .base import LatentSpaceGenerator


# --------------------------------------------------------------------------
# Dependency handling: plug and play
# --------------------------------------------------------------------------
def ensure_pennylane(auto_install: bool = True):
    """Import PennyLane, installing it on first use if necessary.

    The rest of the project has no quantum dependency, so PennyLane is imported
    lazily and only here. That way ``pip install -r requirements.txt`` stays
    light and nothing else breaks if the install fails.
    """
    try:
        return importlib.import_module("pennylane")
    except ImportError:
        if not auto_install:
            raise
        print("PennyLane no encontrado. Instalando (solo la primera vez)...", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "pennylane"]
        )
        importlib.invalidate_caches()
        return importlib.import_module("pennylane")


# --------------------------------------------------------------------------
# Generators
# --------------------------------------------------------------------------
def build_quantum_generator(
    n_qubits: int, n_layers: int, out_dim: int, seed: int
) -> torch.nn.Module:
    """Variational circuit + linear decoder."""
    qml = ensure_pennylane()
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # Angle encoding of the latent noise: each latent component becomes a
        # rotation angle, so the "random seed" of the generator is the initial
        # quantum state rather than an input to a linear layer.
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    torch.manual_seed(seed)
    qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
    return torch.nn.Sequential(qlayer, torch.nn.Linear(n_qubits, out_dim))


def build_matched_classical_generator(
    n_qubits: int, out_dim: int, target_params: int, seed: int
) -> torch.nn.Module:
    """MLP ``n_qubits -> h -> out_dim`` with ``h`` chosen to match the parameter count."""
    # params(h) = (n_qubits*h + h) + (h*out_dim + out_dim)
    best_h, best_gap = 1, None
    for h in range(1, 256):
        total = (n_qubits * h + h) + (h * out_dim + out_dim)
        gap = abs(total - target_params)
        if best_gap is None or gap < best_gap:
            best_h, best_gap = h, gap
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(n_qubits, best_h),
        torch.nn.Tanh(),
        torch.nn.Linear(best_h, out_dim),
    )


def build_discriminator(in_dim: int, seed: int) -> torch.nn.Module:
    """Small critic, identical for both generators.

    ``|x|`` is concatenated for the same reason as everywhere else in this
    project: a ReLU network cannot compute a second moment, and without it the
    critic cannot see that the fakes have the wrong variance. See src/layers.py.
    """
    torch.manual_seed(seed + 777)

    class Critic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(2 * in_dim, 32),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Linear(32, 16),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Linear(16, 1),
            )

        def forward(self, x):
            return self.net(torch.cat([x, torch.abs(x)], dim=-1))

    return Critic()


# --------------------------------------------------------------------------
# The shared GAN harness
# --------------------------------------------------------------------------
class _TorchGANBase(LatentSpaceGenerator):
    """GAN trained in PyTorch; subclasses only choose the generator network."""

    def __init__(
        self,
        seed: int = config.SEED,
        n_qubits: int = config.QUANTUM_QUBITS,
        n_layers: int = config.QUANTUM_LAYERS,
        epochs: int = config.QUANTUM_EPOCHS,
        batch: int = 32,
        learning_rate: float = 5e-3,
        label_smoothing: float = 0.9,
        n_components: int | float | None = config.QUANTUM_COMPONENTS,
        **kwargs,
    ):
        kwargs.pop("n_components", None)
        super().__init__(seed=seed, n_components=n_components, **kwargs)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch = batch
        self.learning_rate = learning_rate
        self.label_smoothing = label_smoothing
        self.generator: torch.nn.Module | None = None
        self.n_params_: int | None = None
        self.output_scale_: float = 1.0

    def _make_generator(self, out_dim: int) -> torch.nn.Module:
        raise NotImplementedError

    def _fit_latent(self, Z: np.ndarray) -> None:
        Zt = torch.tensor(np.asarray(Z, dtype="float32"))
        dim = Zt.shape[1]
        # Bounded output for the same reason as the classical GAN: tanh scaled to
        # the robust range of the data keeps the generator from diverging.
        self.output_scale_ = float(np.quantile(np.abs(Z), 0.999)) or 1.0

        self.generator = self._make_generator(dim)
        disc = build_discriminator(dim, self.seed)
        self.n_params_ = sum(p.numel() for p in self.generator.parameters())

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        opt_d = torch.optim.Adam(disc.parameters(), lr=self.learning_rate)
        bce = torch.nn.BCEWithLogitsLoss()

        g_losses, d_losses = [], []
        for _ in range(self.epochs):
            # ---- discriminator ------------------------------------------
            idx = self.rng.integers(0, len(Zt), size=self.batch)
            real = Zt[idx]
            with torch.no_grad():
                fake = self._forward(self.batch)
            opt_d.zero_grad()
            d_real = bce(disc(real), torch.full((self.batch, 1), self.label_smoothing))
            d_fake = bce(disc(fake), torch.zeros(self.batch, 1))
            d_loss = d_real + d_fake
            d_loss.backward()
            opt_d.step()

            # ---- generator ----------------------------------------------
            opt_g.zero_grad()
            g_loss = bce(disc(self._forward(self.batch)), torch.ones(self.batch, 1))
            g_loss.backward()
            opt_g.step()

            d_losses.append(float(d_loss.item()))
            g_losses.append(float(g_loss.item()))

        self.history = {"d_loss": d_losses, "g_loss": g_losses}

    def _forward(self, n: int) -> torch.Tensor:
        assert self.generator is not None
        # Latent noise bounded to [-pi, pi] so it is a valid set of rotation
        # angles for the quantum circuit. The classical twin gets exactly the
        # same latent distribution, otherwise the comparison would be unfair.
        z = torch.tensor(
            self.rng.uniform(-np.pi, np.pi, size=(n, self.n_qubits)).astype("float32")
        )
        return torch.tanh(self.generator(z)) * self.output_scale_

    def _sample_latent(self, n: int) -> np.ndarray:
        assert self.generator is not None
        out = []
        with torch.no_grad():
            for start in range(0, n, 256):
                out.append(self._forward(min(256, n - start)).numpy())
        return np.concatenate(out)


class QuantumGANGenerator(_TorchGANBase):
    name = "gan_cuantica"
    label = "GAN híbrida cuántica (VQC)"

    def _make_generator(self, out_dim: int) -> torch.nn.Module:
        return build_quantum_generator(self.n_qubits, self.n_layers, out_dim, self.seed)


class MatchedClassicalGANGenerator(_TorchGANBase):
    """Classical control with the *same* parameter budget as the circuit."""

    name = "gan_clasica_equiparada"
    label = "GAN clásica equiparada en parámetros"

    def _make_generator(self, out_dim: int) -> torch.nn.Module:
        # Parameter count of the quantum generator, computed analytically so we
        # do not need to build the circuit just to count it:
        #   StronglyEntanglingLayers -> n_layers * n_qubits * 3
        #   linear decoder           -> n_qubits * out_dim + out_dim
        target = self.n_layers * self.n_qubits * 3 + self.n_qubits * out_dim + out_dim
        return build_matched_classical_generator(
            self.n_qubits, out_dim, target, self.seed
        )
