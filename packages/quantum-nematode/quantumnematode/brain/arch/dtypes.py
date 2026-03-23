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
    QRH = "qrh"
    QSNN_REINFORCE = "qsnnreinforce"
    QSNN_PPO = "qsnnppo"
    HYBRID_QUANTUM = "hybridquantum"
    HYBRID_CLASSICAL = "hybridclassical"
    HYBRID_QUANTUM_CORTEX = "hybridquantumcortex"
    MLP_REINFORCE = "mlpreinforce"
    MLP_DQN = "mlpdqn"
    MLP_PPO = "mlpppo"
    SPIKING_REINFORCE = "spikingreinforce"
    CRH = "crh"
    QLIF_LSTM = "qliflstm"
    QEF = "qef"
    QRH_QLSTM = "qrhqlstm"
    CRH_QLSTM = "crhqlstm"
    LSTM_PPO = "lstmppo"

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

    def to_torch_device_str(self) -> str:
        """Return a string accepted by ``torch.device()``.

        PyTorch only recognises ``"cpu"`` and ``"cuda"``.  ``GPU`` maps to
        ``"cuda"``; ``QPU`` maps to ``"cpu"`` because quantum circuits run on
        Qiskit backends, not PyTorch.
        """
        if self is DeviceType.GPU:
            return "cuda"
        return "cpu"


class BrainConfig(BaseModel):
    """Configuration for the brain architecture."""

    seed: int | None = None  # Random seed for reproducibility
    weights_path: str | None = None  # Path to load pre-trained weights from


BRAIN_TYPES = Literal[
    BrainType.QVARCIRCUIT,
    BrainType.QQLEARNING,
    BrainType.QRC,
    BrainType.QRH,
    BrainType.QSNN_REINFORCE,
    BrainType.QSNN_PPO,
    BrainType.HYBRID_QUANTUM,
    BrainType.HYBRID_CLASSICAL,
    BrainType.HYBRID_QUANTUM_CORTEX,
    BrainType.MLP_REINFORCE,
    BrainType.MLP_DQN,
    BrainType.MLP_PPO,
    BrainType.SPIKING_REINFORCE,
    BrainType.CRH,
    BrainType.QEF,
    BrainType.QLIF_LSTM,
    BrainType.QRH_QLSTM,
    BrainType.CRH_QLSTM,
    BrainType.LSTM_PPO,
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
    BrainType.QRH,
    BrainType.QSNN_REINFORCE,
    BrainType.QSNN_PPO,
    BrainType.HYBRID_QUANTUM,
    BrainType.HYBRID_QUANTUM_CORTEX,
    BrainType.QEF,
    BrainType.QLIF_LSTM,
    BrainType.QRH_QLSTM,
    BrainType.QSNN,
    BrainType.MODULAR,
    BrainType.QMODULAR,
}
CLASSICAL_BRAIN_TYPES: set[BrainType] = {
    BrainType.QRC,
    BrainType.HYBRID_CLASSICAL,
    BrainType.MLP_REINFORCE,
    BrainType.MLP_DQN,
    BrainType.MLP_PPO,
    BrainType.MLP,
    BrainType.QMLP,
    BrainType.PPO,
    BrainType.CRH,
    BrainType.CRH_QLSTM,
    BrainType.LSTM_PPO,
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
