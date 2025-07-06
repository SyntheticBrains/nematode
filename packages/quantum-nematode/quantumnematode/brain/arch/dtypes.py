"""Define the types of brains used in the quantum nematode project."""

from enum import Enum
from typing import Literal


class BrainType(Enum):
    """Enum for different types of brains."""

    MODULAR = "modular"
    MLP = "mlp"


BRAIN_TYPES = Literal[BrainType.MODULAR, BrainType.MLP]
QUANTUM_BRAIN_TYPES = Literal[BrainType.MODULAR]
CLASSICAL_BRAIN_TYPES = Literal[BrainType.MLP]
