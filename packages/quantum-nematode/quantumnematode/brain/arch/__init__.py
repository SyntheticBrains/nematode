"""Module for quantum brain architectures."""

from ._brain import Brain, BrainData, BrainParams, ClassicalBrain, QuantumBrain
from .hybridclassical import HybridClassicalBrain, HybridClassicalBrainConfig
from .hybridquantum import HybridQuantumBrain, HybridQuantumBrainConfig
from .mlpdqn import MLPDQNBrain, MLPDQNBrainConfig, QMLPBrain, QMLPBrainConfig
from .mlpppo import MLPPPOBrain, MLPPPOBrainConfig, PPOBrain, PPOBrainConfig
from .mlpreinforce import MLPBrain, MLPBrainConfig, MLPReinforceBrain, MLPReinforceBrainConfig
from .qqlearning import QModularBrain, QModularBrainConfig, QQLearningBrain, QQLearningBrainConfig
from .qrc import QRCBrain, QRCBrainConfig
from .qsnnppo import QSNNPPOBrain, QSNNPPOBrainConfig
from .qsnnreinforce import QSNNReinforceBrain, QSNNReinforceBrainConfig
from .qvarcircuit import ModularBrain, ModularBrainConfig, QVarCircuitBrain, QVarCircuitBrainConfig
from .spikingreinforce import (
    SpikingBrain,
    SpikingBrainConfig,
    SpikingReinforceBrain,
    SpikingReinforceBrainConfig,
)

# Backward compatibility aliases
QSNNBrain = QSNNReinforceBrain
QSNNBrainConfig = QSNNReinforceBrainConfig

__all__ = [
    "Brain",
    "BrainData",
    "BrainParams",
    "ClassicalBrain",
    "HybridClassicalBrain",
    "HybridClassicalBrainConfig",
    "HybridQuantumBrain",
    "HybridQuantumBrainConfig",
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
    "QRCBrain",
    "QRCBrainConfig",
    "QSNNBrain",
    "QSNNBrainConfig",
    "QSNNPPOBrain",
    "QSNNPPOBrainConfig",
    "QSNNReinforceBrain",
    "QSNNReinforceBrainConfig",
    "QVarCircuitBrain",
    "QVarCircuitBrainConfig",
    "QuantumBrain",
    "SpikingBrain",
    "SpikingBrainConfig",
    "SpikingReinforceBrain",
    "SpikingReinforceBrainConfig",
]
