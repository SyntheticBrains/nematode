"""Define the types of brains used in the quantum nematode project."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel


class BrainType(StrEnum):
    """Different types of brains.

    Naming convention: {Paradigm}{Architecture}_{Algorithm}
    - Q prefix = quantum
    - MLP prefix = classical multi-layer perceptron
    - Spiking prefix = spiking neural network

    StrEnum semantics: ``BrainType.MLP_PPO == "mlpppo"`` is True; the enum
    members ARE strings and can be compared to YAML brain-name keys
    without ``.value`` extraction.
    """

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
    CONNECTOMEPPO = "connectomeppo"
    FEEDFORWARDGA = "feedforwardga"
    CFC_PPO = "cfcppo"
    SPIKING_PPO = "spikingppo"
    EQUIVARIANT_QUANTUM_PPO = "equivariantquantum"
    TRANSFORMER_PPO = "transformerppo"


class DeviceType(StrEnum):
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
    # Action-policy mode: ``discrete`` (categorical over the action set, default,
    # for the grid substrate) or ``continuous`` (tanh-squashed Gaussian over a
    # normalized ``(speed, turn)`` vector, for the continuous-2D substrate). Brains
    # that implement a continuous head read this; the physical rescale lives in the
    # environment, so the emitted action is substrate-independent.
    action_mode: Literal["discrete", "continuous"] = "discrete"


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
    BrainType.CONNECTOMEPPO,
    BrainType.FEEDFORWARDGA,
    BrainType.CFC_PPO,
    BrainType.SPIKING_PPO,
    BrainType.EQUIVARIANT_QUANTUM_PPO,
    BrainType.TRANSFORMER_PPO,
]


def _family_set(family: str) -> set[BrainType]:
    """Return the set of BrainType members whose registration carries ``family``.

    Imports the registry lazily to avoid a circular import at module-load
    time (the registry imports BrainType from this module).
    """
    from quantumnematode.brain.arch._registry import family_members

    return family_members(family)


# These set-valued aliases used to be hand-maintained. They are now derived
# from the plugin-registry family tags carried by each architecture's
# ``@register_brain(...)`` decorator. Read via ``__getattr__`` so the lookup
# defers until the registry has been populated (which happens at import time
# of ``brain.arch.__init__``, AFTER this module finishes loading).
#
# Static-checker-visible declarations: under ``TYPE_CHECKING`` we name the
# attributes with their real ``set[BrainType]`` type so pyright / mypy
# see the proper type at import sites (otherwise the ``__getattr__`` return
# type ``object`` would propagate and break ``X in CLASSICAL_BRAIN_TYPES``).
if TYPE_CHECKING:
    QUANTUM_BRAIN_TYPES: set[BrainType]
    CLASSICAL_BRAIN_TYPES: set[BrainType]
    SPIKING_BRAIN_TYPES: set[BrainType]


def __getattr__(name: str) -> object:
    """Module-level lazy attribute lookup for the family sets.

    ``QUANTUM_BRAIN_TYPES`` / ``CLASSICAL_BRAIN_TYPES`` / ``SPIKING_BRAIN_TYPES``
    are computed on demand from the registry's family tags. Falling back via
    ``__getattr__`` (PEP 562) defers the registry query until after every
    architecture module has self-registered.
    """
    if name == "QUANTUM_BRAIN_TYPES":
        return _family_set("quantum")
    if name == "CLASSICAL_BRAIN_TYPES":
        return _family_set("classical")
    if name == "SPIKING_BRAIN_TYPES":
        return _family_set("spiking")
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


# Defaults
DEFAULT_BRAIN_TYPE = BrainType.QVARCIRCUIT
DEFAULT_QUBITS = 2
DEFAULT_SHOTS = 1024
