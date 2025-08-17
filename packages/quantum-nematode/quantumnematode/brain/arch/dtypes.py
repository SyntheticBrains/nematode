"""Define the types of brains used in the quantum nematode project."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel


class BrainType(Enum):
    """Different types of brains."""

    MODULAR = "modular"
    MLP = "mlp"
    QMLP = "qmlp"


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


BRAIN_TYPES = Literal[BrainType.MODULAR, BrainType.MLP, BrainType.QMLP]
QUANTUM_BRAIN_TYPES = Literal[BrainType.MODULAR]
CLASSICAL_BRAIN_TYPES = Literal[BrainType.MLP, BrainType.QMLP]

# Defaults
DEFAULT_BRAIN_TYPE = BrainType.MODULAR
DEFAULT_QUBITS = 2
DEFAULT_SHOTS = 1024
