"""Module for quantum brain architectures."""

from ._brain import Brain, BrainParams
from .mlp import MLPBrain
from .modular import ModularBrain

__all__ = [
    "Brain",
    "BrainParams",
    "MLPBrain",
    "ModularBrain",
]
