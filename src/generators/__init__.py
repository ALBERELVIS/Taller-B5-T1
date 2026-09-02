"""Generative models, all behind the same fit/sample interface."""

from __future__ import annotations

from .autoregressive import AutoregressiveGenerator
from .base import Generator, LatentSpaceGenerator
from .factor_market import FactorMarketGenerator
from .gan import GANGenerator
from .noise import NoiseGenerator
from .parametric import GaussianGenerator, StudentTGenerator
from .vae import VAEGenerator

#: Registry used by the sweep, the notebooks and the report. Order is the order
#: in which they appear in every table and legend.
REGISTRY: dict[str, type[Generator]] = {
    "ruido": NoiseGenerator,
    "gaussiano": GaussianGenerator,
    "student_t": StudentTGenerator,
    "vae": VAEGenerator,
    "gan": GANGenerator,
    "autoregresivo": AutoregressiveGenerator,
    "factor_mercado": FactorMarketGenerator,
}

__all__ = [
    "Generator",
    "LatentSpaceGenerator",
    "NoiseGenerator",
    "GaussianGenerator",
    "StudentTGenerator",
    "VAEGenerator",
    "GANGenerator",
    "AutoregressiveGenerator",
    "FactorMarketGenerator",
    "REGISTRY",
]
