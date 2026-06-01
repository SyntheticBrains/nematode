"""Module for quantum nematode brain architectures.

Every architecture self-registers via a ``@register_brain`` decorator on
its Brain class. The dispatcher in ``utils/brain_factory.py`` and the YAML
loader in ``utils/config_loader.py`` both consume the registry rather than
hand-maintained tables.

To add a new architecture, see ``docs/architecture/plugin-developer-guide.md``
for the walkthrough and the files-touched budget.
"""

from ._brain import Brain, BrainData, BrainParams, ClassicalBrain, QuantumBrain
from ._registry import (
    Registration,
    assert_registry_matches_enum,
    get_all_registrations,
    get_registration,
    instantiate_brain,
    list_registered_brains,
    register_brain,
)
from ._reservoir_hybrid_base import ReservoirHybridBase, ReservoirHybridBaseConfig
from ._reservoir_lstm_base import ReservoirLSTMBase, ReservoirLSTMBaseConfig
from ._rule import LearningRule, RuleStepReport
from ._topology import BrainTopology
from .cfc_ppo import CfCBrainConfig, CfCPPOBrain
from .connectome_ppo import ConnectomePPOBrain, ConnectomePPOBrainConfig
from .crh import CRHBrain, CRHBrainConfig
from .crhqlstm import CRHQLSTMBrain, CRHQLSTMBrainConfig
from .equivariant_quantum import EquivariantQuantumPPOBrain, EquivariantQuantumPPOBrainConfig
from .feedforward_ga import FeedforwardGABrain, FeedforwardGABrainConfig
from .hybridclassical import HybridClassicalBrain, HybridClassicalBrainConfig
from .hybridquantum import HybridQuantumBrain, HybridQuantumBrainConfig
from .hybridquantumcortex import HybridQuantumCortexBrain, HybridQuantumCortexBrainConfig
from .lstmppo import LSTMPPOBrain, LSTMPPOBrainConfig
from .mlpdqn import MLPDQNBrain, MLPDQNBrainConfig
from .mlpppo import MLPPPOBrain, MLPPPOBrainConfig
from .mlpreinforce import MLPReinforceBrain, MLPReinforceBrainConfig
from .qef import QEFBrain, QEFBrainConfig
from .qliflstm import QLIFLSTMBrain, QLIFLSTMBrainConfig
from .qqlearning import QQLearningBrain, QQLearningBrainConfig
from .qrc import QRCBrain, QRCBrainConfig
from .qrh import QRHBrain, QRHBrainConfig
from .qrhqlstm import QRHQLSTMBrain, QRHQLSTMBrainConfig
from .qsnnppo import QSNNPPOBrain, QSNNPPOBrainConfig
from .qsnnreinforce import QSNNReinforceBrain, QSNNReinforceBrainConfig
from .qvarcircuit import QVarCircuitBrain, QVarCircuitBrainConfig
from .spiking_ppo import SpikingPPOBrain, SpikingPPOBrainConfig
from .spikingreinforce import SpikingReinforceBrain, SpikingReinforceBrainConfig

# Every architecture module has now been imported and has self-registered
# via its ``@register_brain`` decorator. Verify the registry and the
# ``BrainType`` enum agree; mismatch fails loudly at import time so any
# accidental drift is caught immediately rather than at first dispatch.
assert_registry_matches_enum()

__all__ = [
    "Brain",
    "BrainData",
    "BrainParams",
    "BrainTopology",
    "CRHBrain",
    "CRHBrainConfig",
    "CRHQLSTMBrain",
    "CRHQLSTMBrainConfig",
    "CfCBrainConfig",
    "CfCPPOBrain",
    "ClassicalBrain",
    "ConnectomePPOBrain",
    "ConnectomePPOBrainConfig",
    "EquivariantQuantumPPOBrain",
    "EquivariantQuantumPPOBrainConfig",
    "FeedforwardGABrain",
    "FeedforwardGABrainConfig",
    "HybridClassicalBrain",
    "HybridClassicalBrainConfig",
    "HybridQuantumBrain",
    "HybridQuantumBrainConfig",
    "HybridQuantumCortexBrain",
    "HybridQuantumCortexBrainConfig",
    "LSTMPPOBrain",
    "LSTMPPOBrainConfig",
    "LearningRule",
    "MLPDQNBrain",
    "MLPDQNBrainConfig",
    "MLPPPOBrain",
    "MLPPPOBrainConfig",
    "MLPReinforceBrain",
    "MLPReinforceBrainConfig",
    "QEFBrain",
    "QEFBrainConfig",
    "QLIFLSTMBrain",
    "QLIFLSTMBrainConfig",
    "QQLearningBrain",
    "QQLearningBrainConfig",
    "QRCBrain",
    "QRCBrainConfig",
    "QRHBrain",
    "QRHBrainConfig",
    "QRHQLSTMBrain",
    "QRHQLSTMBrainConfig",
    "QSNNPPOBrain",
    "QSNNPPOBrainConfig",
    "QSNNReinforceBrain",
    "QSNNReinforceBrainConfig",
    "QVarCircuitBrain",
    "QVarCircuitBrainConfig",
    "QuantumBrain",
    "Registration",
    "ReservoirHybridBase",
    "ReservoirHybridBaseConfig",
    "ReservoirLSTMBase",
    "ReservoirLSTMBaseConfig",
    "RuleStepReport",
    "SpikingPPOBrain",
    "SpikingPPOBrainConfig",
    "SpikingReinforceBrain",
    "SpikingReinforceBrainConfig",
    "get_all_registrations",
    "get_registration",
    "instantiate_brain",
    "list_registered_brains",
    "register_brain",
]
