"""Module for quantum brain architectures."""

from ._brain import Brain, BrainParams
from .complex import ComplexBrain
from .dynamic import DynamicBrain
from .memory import MemoryBrain
from .modular import ModularBrain
from .reduced import ReducedBrain
from .simple import SimpleBrain

__all__ = [
    "Brain",
    "BrainParams",
    "ComplexBrain",
    "DynamicBrain",
    "MemoryBrain",
    "ModularBrain",
    "ReducedBrain",
    "SimpleBrain",
]
