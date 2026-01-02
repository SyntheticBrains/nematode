"""Module for quantum brain."""

from .arch.mlp import MLPBrain
from .arch.modular import ModularBrain
from .arch.spiking import SpikingBrain
from .modules import (
    SENSORY_MODULES,
    ModuleName,
    extract_classical_features,
    get_classical_feature_dimension,
)

__all__ = [
    "SENSORY_MODULES",
    "MLPBrain",
    "ModularBrain",
    "ModuleName",
    "SpikingBrain",
    "extract_classical_features",
    "get_classical_feature_dimension",
]
