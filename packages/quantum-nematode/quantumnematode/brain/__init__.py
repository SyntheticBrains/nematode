"""Module for quantum brain."""

from .arch.mlp import MLPBrain
from .arch.modular import ModularBrain
from .arch.spiking import SpikingBrain
from .modules import ModuleName

__all__ = [
    "MLPBrain",
    "ModularBrain",
    "ModuleName",
    "SpikingBrain",
]
