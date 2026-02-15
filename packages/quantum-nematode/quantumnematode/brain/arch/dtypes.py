"""Define the types of brains used in the quantum nematode project."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel


class BrainType(Enum):
    """Different types of brains.

    Naming convention: {Paradigm}{Architecture}_{Algorithm}
    - Q prefix = quantum
    - MLP prefix = classical multi-layer perceptron
    - Spiking prefix = spiking neural network
    """

    # New canonical names
    QVARCIRCUIT = "qvarcircuit"
    QQLEARNING = "qqlearning"
    QRC = "qrc"
    QSNN_REINFORCE = "qsnnreinforce"
    MLP_REINFORCE = "mlpreinforce"
    MLP_DQN = "mlpdqn"
    MLP_PPO = "mlpppo"
    SPIKING_REINFORCE = "spikingreinforce"

    # Deprecated aliases (kept for backward compatibility)
    QSNN = "qsnn"
    MODULAR = "modular"
    QMODULAR = "qmodular"
    MLP = "mlp"
    QMLP = "qmlp"
    PPO = "ppo"
    SPIKING = "spiking"


class DeviceType(Enum):
    """
    Different types of devices for running processing for brains.

    For quantum brains, choosing a device other than 'qpu'
    will result in the brain being run on a classical simulator.

    - CPU: Central Processing Unit
    - GPU: Graphics Processing Unit
    - QPU: Quantum Processing Unit
    """

    CPU = "cpu"
    GPU = "gpu"
    QPU = "qpu"


class BrainConfig(BaseModel):
    """Configuration for the brain architecture."""

    seed: int | None = None  # Random seed for reproducibility


BRAIN_TYPES = Literal[
    BrainType.QVARCIRCUIT,
    BrainType.QQLEARNING,
    BrainType.QRC,
    BrainType.QSNN_REINFORCE,
    BrainType.MLP_REINFORCE,
    BrainType.MLP_DQN,
    BrainType.MLP_PPO,
    BrainType.SPIKING_REINFORCE,
    # Deprecated aliases
    BrainType.QSNN,
    BrainType.MODULAR,
    BrainType.QMODULAR,
    BrainType.MLP,
    BrainType.QMLP,
    BrainType.PPO,
    BrainType.SPIKING,
]
QUANTUM_BRAIN_TYPES: set[BrainType] = {
    BrainType.QVARCIRCUIT,
    BrainType.QQLEARNING,
    BrainType.QSNN_REINFORCE,
    BrainType.QSNN,
    BrainType.MODULAR,
    BrainType.QMODULAR,
}
CLASSICAL_BRAIN_TYPES: set[BrainType] = {
    BrainType.QRC,
    BrainType.MLP_REINFORCE,
    BrainType.MLP_DQN,
    BrainType.MLP_PPO,
    BrainType.MLP,
    BrainType.QMLP,
    BrainType.PPO,
}
SPIKING_BRAIN_TYPES: set[BrainType] = {
    BrainType.SPIKING_REINFORCE,
    BrainType.SPIKING,
}

# Map deprecated names to canonical names
BRAIN_NAME_ALIASES: dict[str, str] = {
    "qsnn": "qsnnreinforce",
    "modular": "qvarcircuit",
    "qmodular": "qqlearning",
    "mlp": "mlpreinforce",
    "qmlp": "mlpdqn",
    "ppo": "mlpppo",
    "spiking": "spikingreinforce",
}

# Defaults
DEFAULT_BRAIN_TYPE = BrainType.QVARCIRCUIT
DEFAULT_QUBITS = 2
DEFAULT_SHOTS = 1024
