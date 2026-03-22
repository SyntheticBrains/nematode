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
from .weights import (
    WeightComponent,
    WeightPersistence,
    load_weights,
    save_weights,
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
    "WeightComponent",
    "WeightPersistence",
    "extract_classical_features",
    "get_classical_feature_dimension",
    "load_weights",
    "save_weights",
]
