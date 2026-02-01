"""Module for quantum brain."""

from .arch.mlpreinforce import MLPBrain, MLPReinforceBrain
from .arch.qvarcircuit import ModularBrain, QVarCircuitBrain
from .arch.spikingreinforce import SpikingBrain, SpikingReinforceBrain
from .modules import (
    SENSORY_MODULES,
    ModuleName,
    extract_classical_features,
    get_classical_feature_dimension,
)

__all__ = [
    "SENSORY_MODULES",
    "MLPBrain",
    "MLPReinforceBrain",
    "ModularBrain",
    "ModuleName",
    "QVarCircuitBrain",
    "SpikingBrain",
    "SpikingReinforceBrain",
    "extract_classical_features",
    "get_classical_feature_dimension",
]
