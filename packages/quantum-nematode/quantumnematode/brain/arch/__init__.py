"""Module for quantum brain architectures."""

from ._brain import Brain, BrainParams, ClassicalBrain, QuantumBrain
from .mlp import MLPBrain
from .modular import ModularBrain

__all__ = [
    "Brain",
    "BrainParams",
    "ClassicalBrain",
    "MLPBrain",
    "ModularBrain",
    "QuantumBrain",
]
