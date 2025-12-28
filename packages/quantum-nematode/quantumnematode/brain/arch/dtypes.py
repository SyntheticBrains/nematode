"""Define the types of brains used in the quantum nematode project."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel


class BrainType(Enum):
    """Different types of brains."""

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
    BrainType.MODULAR,
    BrainType.QMODULAR,
    BrainType.MLP,
    BrainType.QMLP,
    BrainType.PPO,
    BrainType.SPIKING,
]
QUANTUM_BRAIN_TYPES: set[BrainType] = {BrainType.MODULAR, BrainType.QMODULAR}
CLASSICAL_BRAIN_TYPES: set[BrainType] = {
    BrainType.MLP,
    BrainType.QMLP,
    BrainType.PPO,
    BrainType.SPIKING,
}

# Defaults
DEFAULT_BRAIN_TYPE = BrainType.MODULAR
DEFAULT_QUBITS = 2
DEFAULT_SHOTS = 1024
