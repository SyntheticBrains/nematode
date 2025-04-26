"""Module for quantum brain architectures."""

from .complex import ComplexBrain
from .dynamic import DynamicBrain
from .memory import MemoryBrain
from .reduced import ReducedBrain
from .simple import SimpleBrain

__all__ = ["ComplexBrain", "DynamicBrain", "MemoryBrain", "ReducedBrain", "SimpleBrain"]
