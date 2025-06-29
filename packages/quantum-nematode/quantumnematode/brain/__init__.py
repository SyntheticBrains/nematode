"""Module for quantum brain."""

from .arch.complex import ComplexBrain
from .arch.dynamic import DynamicBrain
from .arch.memory import MemoryBrain
from .arch.mlp import MLPBrain
from .arch.modular import ModularBrain
from .arch.reduced import ReducedBrain
from .arch.simple import SimpleBrain

__all__ = [
    "ComplexBrain",
    "DynamicBrain",
    "MLPBrain",
    "MemoryBrain",
    "ModularBrain",
    "ReducedBrain",
    "SimpleBrain",
]
