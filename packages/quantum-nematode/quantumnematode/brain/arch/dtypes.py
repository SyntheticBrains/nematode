from typing import Literal
from enum import Enum


class BrainType(Enum):
    MODULAR = "modular"
    MLP = "mlp"

BRAIN_TYPES = Literal[BrainType.MODULAR, BrainType.MLP]
QUANTUM_BRAIN_TYPES = Literal[BrainType.MODULAR]
CLASSICAL_BRAIN_TYPES = Literal[BrainType.MLP]
