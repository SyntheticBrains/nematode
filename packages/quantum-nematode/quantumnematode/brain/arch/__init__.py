"""Module for quantum brain architectures."""

from ._brain import Brain, BrainData, BrainParams, ClassicalBrain, QuantumBrain
from .mlp import MLPBrain, MLPBrainConfig
from .modular import ModularBrain, ModularBrainConfig
from .qmlp import QMLPBrain, QMLPBrainConfig
from .qmodular import QModularBrain, QModularBrainConfig
from .spiking import SpikingBrain, SpikingBrainConfig

__all__ = [
    "Brain",
    "BrainData",
    "BrainParams",
    "ClassicalBrain",
    "MLPBrain",
    "MLPBrainConfig",
    "ModularBrain",
    "ModularBrainConfig",
    "QMLPBrain",
    "QMLPBrainConfig",
    "QModularBrain",
    "QModularBrainConfig",
    "QuantumBrain",
    "SpikingBrain",
    "SpikingBrainConfig",
]
