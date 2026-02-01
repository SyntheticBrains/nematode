"""Module for quantum brain."""

from .arch.mlpreinforce import MLPReinforceBrain
from .arch.qvarcircuit import QVarCircuitBrain
from .arch.spikingreinforce import SpikingReinforceBrain
from .modules import (
    SENSORY_MODULES,
    ModuleName,
    extract_classical_features,
    get_classical_feature_dimension,
)

__all__ = [
    "SENSORY_MODULES",
    "MLPReinforceBrain",
    "ModuleName",
    "QVarCircuitBrain",
    "SpikingReinforceBrain",
    "extract_classical_features",
    "get_classical_feature_dimension",
]
