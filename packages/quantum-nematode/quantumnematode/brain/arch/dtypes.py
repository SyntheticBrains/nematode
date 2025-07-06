"""Define the types of brains used in the quantum nematode project."""

from enum import Enum
from typing import Literal


class BrainType(Enum):
    """Different types of brains."""

    MODULAR = "modular"
    MLP = "mlp"


class DeviceType(Enum):
    """Different types of devices for running classical processing for brains."""

    CPU = "cpu"
    GPU = "gpu"


BRAIN_TYPES = Literal[BrainType.MODULAR, BrainType.MLP]
QUANTUM_BRAIN_TYPES = Literal[BrainType.MODULAR]
CLASSICAL_BRAIN_TYPES = Literal[BrainType.MLP]

# Defaults
DEFAULT_BRAIN_TYPE = BrainType.MODULAR
DEFAULT_QUBITS = 2
DEFAULT_SHOTS = 1024
