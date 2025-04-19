"""Module for quantum brain architectures."""

from .complex import ComplexBrain
from .memory import MemoryBrain
from .reduced import ReducedBrain
from .simple import SimpleBrain

__all__ = ["ComplexBrain", "MemoryBrain", "ReducedBrain", "SimpleBrain"]
