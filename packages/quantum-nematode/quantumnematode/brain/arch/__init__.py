"""Module for quantum brain architectures."""

from ._brain import Brain, BrainData, BrainParams, ClassicalBrain, QuantumBrain
from ._reservoir_hybrid_base import ReservoirHybridBase, ReservoirHybridBaseConfig
from ._reservoir_lstm_base import ReservoirLSTMBase, ReservoirLSTMBaseConfig
from .crh import CRHBrain, CRHBrainConfig
from .crhqlstm import CRHQLSTMBrain, CRHQLSTMBrainConfig
from .hybridclassical import HybridClassicalBrain, HybridClassicalBrainConfig
from .hybridquantum import HybridQuantumBrain, HybridQuantumBrainConfig
from .hybridquantumcortex import HybridQuantumCortexBrain, HybridQuantumCortexBrainConfig
from .lstmppo import LSTMPPOBrain, LSTMPPOBrainConfig
from .mlpdqn import MLPDQNBrain, MLPDQNBrainConfig, QMLPBrain, QMLPBrainConfig
from .mlpppo import MLPPPOBrain, MLPPPOBrainConfig, PPOBrain, PPOBrainConfig
from .mlpreinforce import MLPBrain, MLPBrainConfig, MLPReinforceBrain, MLPReinforceBrainConfig
from .qef import QEFBrain, QEFBrainConfig
from .qliflstm import QLIFLSTMBrain, QLIFLSTMBrainConfig
from .qqlearning import QModularBrain, QModularBrainConfig, QQLearningBrain, QQLearningBrainConfig
from .qrc import QRCBrain, QRCBrainConfig
from .qrh import QRHBrain, QRHBrainConfig
from .qrhqlstm import QRHQLSTMBrain, QRHQLSTMBrainConfig
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
    "CRHBrain",
    "CRHBrainConfig",
    "CRHQLSTMBrain",
    "CRHQLSTMBrainConfig",
    "ClassicalBrain",
    "HybridClassicalBrain",
    "HybridClassicalBrainConfig",
    "HybridQuantumBrain",
    "HybridQuantumBrainConfig",
    "HybridQuantumCortexBrain",
    "HybridQuantumCortexBrainConfig",
    "LSTMPPOBrain",
    "LSTMPPOBrainConfig",
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
    "QEFBrain",
    "QEFBrainConfig",
    "QLIFLSTMBrain",
    "QLIFLSTMBrainConfig",
    "QMLPBrain",
    "QMLPBrainConfig",
    "QModularBrain",
    "QModularBrainConfig",
    "QQLearningBrain",
    "QQLearningBrainConfig",
    "QRCBrain",
    "QRCBrainConfig",
    "QRHBrain",
    "QRHBrainConfig",
    "QRHQLSTMBrain",
    "QRHQLSTMBrainConfig",
    "QSNNBrain",
    "QSNNBrainConfig",
    "QSNNPPOBrain",
    "QSNNPPOBrainConfig",
    "QSNNReinforceBrain",
    "QSNNReinforceBrainConfig",
    "QVarCircuitBrain",
    "QVarCircuitBrainConfig",
    "QuantumBrain",
    "ReservoirHybridBase",
    "ReservoirHybridBaseConfig",
    "ReservoirLSTMBase",
    "ReservoirLSTMBaseConfig",
    "SpikingBrain",
    "SpikingBrainConfig",
    "SpikingReinforceBrain",
    "SpikingReinforceBrainConfig",
]
