"""Module for quantum brain architectures."""

from ._brain import Brain, BrainData, BrainParams, ClassicalBrain, QuantumBrain
from .mlp import MLPBrain, MLPBrainConfig, MLPReinforceBrain, MLPReinforceBrainConfig
from .modular import ModularBrain, ModularBrainConfig, QVarCircuitBrain, QVarCircuitBrainConfig
from .ppo import MLPPPOBrain, MLPPPOBrainConfig, PPOBrain, PPOBrainConfig
from .qmlp import MLPDQNBrain, MLPDQNBrainConfig, QMLPBrain, QMLPBrainConfig
from .qmodular import QModularBrain, QModularBrainConfig, QQLearningBrain, QQLearningBrainConfig
from .spiking import (
    SpikingBrain,
    SpikingBrainConfig,
    SpikingReinforceBrain,
    SpikingReinforceBrainConfig,
)

__all__ = [
    "Brain",
    "BrainData",
    "BrainParams",
    "ClassicalBrain",
    "MLPBrain",
    "MLPBrainConfig",
    "MLPDQNBrain",
    "MLPDQNBrainConfig",
    "MLPPPOBrain",
    "MLPPPOBrainConfig",
    "MLPReinforceBrain",
    "MLPReinforceBrainConfig",
    "ModularBrain",
    "ModularBrainConfig",
    "PPOBrain",
    "PPOBrainConfig",
    "QMLPBrain",
    "QMLPBrainConfig",
    "QModularBrain",
    "QModularBrainConfig",
    "QQLearningBrain",
    "QQLearningBrainConfig",
    "QVarCircuitBrain",
    "QVarCircuitBrainConfig",
    "QuantumBrain",
    "SpikingBrain",
    "SpikingBrainConfig",
    "SpikingReinforceBrain",
    "SpikingReinforceBrainConfig",
]
