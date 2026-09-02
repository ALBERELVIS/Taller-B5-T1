"""Central configuration for the B5-T1 workshop.

Every magic number of the project lives here so that the notebooks, the sweep
and the report always speak about exactly the same experiment.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
HISTORIES_DIR = RESULTS_DIR / "histories"

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, HISTORIES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Raw data
# --------------------------------------------------------------------------
# Same ticker list and start date used by the professor's notebooks, so that
# our universe is directly comparable with his examples.
TICKERS_URL = (
    "https://raw.githubusercontent.com/alfonso-santos/"
    "microcredencial-carteras-python-2023/main/Tema_5_APT/data/sp500_tickers.csv"
)
START_DATE = "1945-01-01"
PRICES_CACHE = DATA_DIR / "prices_close.parquet"

# --------------------------------------------------------------------------
# Problem definition
# --------------------------------------------------------------------------
WINDOW_X = 60          # trading days of history fed to the model
HORIZON = 20           # trading days ahead we look at for the stress event

# Main target: "vol" (forward realised volatility in its top tail) or
# "drawdown". See the docstring of data.py for the measurements that led us to
# make "vol" the main task and keep "drawdown" as a documented negative result.
TARGET = "vol"
VOL_QUANTILE = 0.95    # a window is "stress" if forward vol is above this quantile
DD_THRESHOLD = -0.08   # alternative label: drawdown reaching this level

# Gap (in trading days) inserted between the chronological blocks. It must be at
# least WINDOW_X + HORIZON so that no training target can overlap the feature
# window of the next block.
EMBARGO = WINDOW_X + HORIZON

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test fraction is the remainder

# --------------------------------------------------------------------------
# Representation
# --------------------------------------------------------------------------
# None -> native standardised space (1380 dims, no information loss). See the
# docstring of representation.py for why we did not truncate with PCA.
GENERATOR_SPACE: int | None = None
PCA_ABLATION_COMPONENTS = 128   # used only for the documented ablation
QUANTUM_COMPONENTS = 16         # small space where the quantum circuit is simulable

# --------------------------------------------------------------------------
# Downstream classifier
# --------------------------------------------------------------------------
# Architecture frozen in notebook 02 and reused unchanged in every sweep run.
DOWNSTREAM_VARIANT = "cnn_mag"
DOWNSTREAM_EPOCHS = 120
DOWNSTREAM_BATCH = 128
DOWNSTREAM_PATIENCE = 20

# --------------------------------------------------------------------------
# Generative models
# --------------------------------------------------------------------------
NOISE_SIGMA_RATIO = 0.10   # perturbation size of the trivial generator
STUDENT_T_DF = 4.0         # degrees of freedom, typical for daily equity returns

VAE_LATENT = 32
VAE_BETA = 1e-3            # small on purpose: avoids posterior collapse
VAE_EPOCHS = 400

GAN_LATENT = 138           # ~10% of 1380, the professor's rule of thumb
GAN_EPOCHS = 4000
GAN_BATCH = 32

AR_CONTEXT = 10            # days of context for the autoregressive model
AR_EPOCHS = 250
AR_BURN_IN = 60            # discarded steps so no real day survives in a sample

# Quantum bonus (simulator only, no hardware)
QUANTUM_QUBITS = 8
QUANTUM_LAYERS = 3
QUANTUM_EPOCHS = 600

# --------------------------------------------------------------------------
# Sweep grid
# --------------------------------------------------------------------------
N_REAL_GRID = [500, 1000, 2000, 4000, None]      # None -> use every real window
N_SYNTH_GRID = [0, 250, 500, 1000, 2000]         # synthetic MINORITY windows added
SEEDS = [0, 1, 2]

SEED = 42
