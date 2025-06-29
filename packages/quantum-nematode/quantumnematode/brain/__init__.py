"""Module for quantum brain."""

from .arch.mlp import MLPBrain
from .arch.modular import ModularBrain

__all__ = [
    "MLPBrain",
    "ModularBrain",
]
