"""
Quantum Reservoir Hybrid (QRH) Brain Architecture.

This architecture implements a fixed quantum reservoir with configurable topology
(structured or random) and a PPO-trained classical actor-critic readout. The reservoir
generates rich feature representations via X/Y/Z Pauli expectations and pairwise ZZ
correlations, while avoiding barren plateaus entirely (no quantum parameters trained).

Evaluation showed random topology vastly outperforms C. elegans-inspired structured
topology (86.8% vs 0-0.25% on foraging). The structured topology is retained for
MI analysis and ablation studies but is not recommended for training.

Key Features:
- **Configurable Quantum Reservoir**: Random topology (default for training) or
  C. elegans connectome topology (for MI analysis / ablation). Both use CZ gates
  and controlled rotations (CRY/CRZ) with equivalent gate density.
- **Per-Qubit Input Encoding**: Sensory qubits receive direct input via RY/RZ gates;
  remaining qubits receive signal exclusively through the topology.
- **X/Y/Z + ZZ Feature Extraction**: Per-qubit Pauli expectations <X_i>, <Y_i>, <Z_i>
  + pairwise ZZ-correlations <Z_i Z_j> (75 features for 10 qubits) via exact
  statevector simulation. X/Y expectations capture phase information lost by Z-only
  measurement.
- **PPO Actor-Critic Readout**: Actor and critic MLPs trained with combined PPO loss
  (clipped surrogate + value loss + entropy bonus) via single optimizer
- **No Barren Plateaus**: Reservoir is fixed (no quantum parameters trained)

Architecture:
    Sensory Input -> Per-Qubit Encoding -> Quantum Reservoir (FIXED)
    -> X/Y/Z + ZZ Features -> PPO Readout

The reservoir avoids barren plateaus entirely because no quantum parameters are
trained. Only the classical actor-critic readout (~10K params) is optimized.

References
----------
- Cook et al. (2019) "Whole-animal connectomes of both C. elegans sexes"
- Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field, field_validator, model_validator
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantumnematode.brain.arch._reservoir_hybrid_base import (
    ReservoirHybridBase,
    ReservoirHybridBaseConfig,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.logging_config import logger

if TYPE_CHECKING:
    from quantumnematode.brain.actions import Action

# =============================================================================
# Default Hyperparameters (quantum-specific)
# =============================================================================

DEFAULT_NUM_RESERVOIR_QUBITS = 8
DEFAULT_RESERVOIR_DEPTH = 3
DEFAULT_RESERVOIR_SEED = 42
DEFAULT_SHOTS = 1024  # Retained for future QPU compatibility

# Validation constants
MIN_RESERVOIR_QUBITS = 2
MIN_RESERVOIR_DEPTH = 1
MIN_SHOTS = 100

# =============================================================================
# C. elegans Topology Constants
# =============================================================================
# 8-qubit mapping of the sensory-interneuron navigation subnetwork:
#   Qubits 0-1: ASEL/ASER (amphid sensory neurons -- salt/food chemotaxis)
#   Qubits 2-3: AIYL/AIYR (first-layer interneurons -- sensory integration)
#   Qubits 4-5: AIAL/AIAR (second-layer interneurons -- navigation command)
#   Qubits 6-7: AVAL/AVAR (command interneurons -- forward/reverse decision)
#
# Per-qubit input encoding:
#   Only sensory qubits (0-1) receive direct feature encoding, matching the
#   biological circuit where only ASE neurons sense the chemical gradient.
#   Interneuron and command qubits receive signal only through the topology.
#   ASEL (qubit 0) encodes gradient_strength via RY (on-response).
#   ASER (qubit 1) encodes relative_angle via RZ (off-response / phase).

# Sensory qubit indices -- only these receive direct input encoding
SENSORY_QUBITS: list[int] = [0, 1]

# Gap junction CZ pairs (bidirectional electrical coupling):
# - Bilateral pairs: left-right connections within each neuron class
# - Feedforward: sensory -> interneuron connections
GAP_JUNCTION_CZ_PAIRS: list[tuple[int, int]] = [
    # Bilateral pairs (symmetric left-right coupling)
    (0, 1),  # ASEL <-> ASER
    (2, 3),  # AIYL <-> AIYR
    (4, 5),  # AIAL <-> AIAR
    (6, 7),  # AVAL <-> AVAR
    # Feedforward excitation (sensory integration)
    (2, 4),  # AIYL -> AIAL
    (3, 5),  # AIYR -> AIAR
]

# Chemical synapse CRY/CRZ angles (directed, conditional signaling):
# Implemented as controlled rotations: the rotation on the target qubit is
# conditioned on the control qubit's state, modeling how postsynaptic response
# depends on presynaptic activity.
# Angles normalized from published synaptic weight ratios (Cook et al. 2019).
# Each tuple: (control_qubit, target_qubit, ry_angle, rz_angle)
CHEMICAL_SYNAPSE_ROTATIONS: list[tuple[int, int, float, float]] = [
    # ASE -> AIY (sensory -> first interneuron, strong excitatory)
    (0, 2, 0.8, 0.3),  # ASEL -> AIYL
    (1, 3, 0.8, 0.3),  # ASER -> AIYR
    # AIY -> AIA (first -> second interneuron, moderate excitatory)
    (2, 4, 0.5, 0.4),  # AIYL -> AIAL
    (3, 5, 0.5, 0.4),  # AIYR -> AIAR
    # AIA -> AVA (second interneuron -> command, strong excitatory)
    (4, 6, 0.7, 0.2),  # AIAL -> AVAL
    (5, 7, 0.7, 0.2),  # AIAR -> AVAR
    # Cross-lateral connections (weak inhibitory)
    (0, 3, 0.2, -0.3),  # ASEL -> AIYR (cross-lateral)
    (1, 2, 0.2, -0.3),  # ASER -> AIYL (cross-lateral)
]


def _compute_feature_dim(num_qubits: int) -> int:
    """Compute QRH feature dimension: 3N (X/Y/Z expectations) + N(N-1)/2 (ZZ correlations)."""
    return 3 * num_qubits + num_qubits * (num_qubits - 1) // 2


# =============================================================================
# Configuration
# =============================================================================


class QRHBrainConfig(ReservoirHybridBaseConfig):
    """Configuration for the QRHBrain architecture.

    Inherits PPO readout, LR scheduling, and sensory module config from
    ReservoirHybridBaseConfig. Adds quantum-specific fields.
    """

    # Reservoir parameters
    num_reservoir_qubits: int = Field(
        default=DEFAULT_NUM_RESERVOIR_QUBITS,
        description="Number of qubits in the quantum reservoir.",
    )
    reservoir_depth: int = Field(
        default=DEFAULT_RESERVOIR_DEPTH,
        description="Number of entanglement layers in the reservoir circuit.",
    )
    reservoir_seed: int = Field(
        default=DEFAULT_RESERVOIR_SEED,
        description="Seed for deterministic reservoir construction.",
    )
    shots: int = Field(
        default=DEFAULT_SHOTS,
        description="Measurement shots (retained for future QPU compatibility).",
    )

    # MI comparison toggle
    use_random_topology: bool = Field(
        default=False,
        description="Use random topology for MI comparison (default: structured).",
    )

    # Configurable sensory qubit count
    num_sensory_qubits: int | None = Field(
        default=None,
        description=(
            "Number of sensory qubits for input encoding. "
            "None = auto-compute as min(input_dim, num_reservoir_qubits)."
        ),
    )

    @field_validator("num_reservoir_qubits")
    @classmethod
    def validate_num_reservoir_qubits(cls, v: int) -> int:
        """Validate num_reservoir_qubits >= 2."""
        if v < MIN_RESERVOIR_QUBITS:
            msg = f"num_reservoir_qubits must be >= {MIN_RESERVOIR_QUBITS}, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("reservoir_depth")
    @classmethod
    def validate_reservoir_depth(cls, v: int) -> int:
        """Validate reservoir_depth >= 1."""
        if v < MIN_RESERVOIR_DEPTH:
            msg = f"reservoir_depth must be >= {MIN_RESERVOIR_DEPTH}, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("shots")
    @classmethod
    def validate_shots(cls, v: int) -> int:
        """Validate shots >= 100."""
        if v < MIN_SHOTS:
            msg = f"shots must be >= {MIN_SHOTS}, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("readout_hidden_dim")
    @classmethod
    def validate_readout_hidden_dim(cls, v: int) -> int:
        """Validate readout_hidden_dim >= 1."""
        from quantumnematode.brain.arch._reservoir_hybrid_base import MIN_READOUT_HIDDEN_DIM

        if v < MIN_READOUT_HIDDEN_DIM:
            msg = f"readout_hidden_dim must be >= {MIN_READOUT_HIDDEN_DIM}, got {v}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_sensory_qubit_count(self) -> QRHBrainConfig:
        """Validate num_sensory_qubits bounds when explicitly set."""
        if self.num_sensory_qubits is not None:
            if self.num_sensory_qubits < 1:
                msg = f"num_sensory_qubits must be >= 1, got {self.num_sensory_qubits}"
                raise ValueError(msg)
            if self.num_sensory_qubits > self.num_reservoir_qubits:
                msg = (
                    f"num_sensory_qubits ({self.num_sensory_qubits}) must be "
                    f"<= num_reservoir_qubits ({self.num_reservoir_qubits})"
                )
                raise ValueError(msg)
        return self


# =============================================================================
# QRH Brain
# =============================================================================


class QRHBrain(ReservoirHybridBase):
    """Quantum Reservoir Hybrid brain architecture.

    Uses a fixed quantum reservoir (random or structured topology) to generate
    X/Y/Z+ZZ feature representations, with a PPO-trained classical actor-critic
    readout. Random topology is recommended for training; structured topology
    (C. elegans-inspired) is available for MI analysis and ablation studies.
    """

    _brain_name = "QRH"

    def __init__(
        self,
        config: QRHBrainConfig,
        num_actions: int = 4,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] | None = None,
    ) -> None:
        # Reservoir configuration (must be set before super().__init__ computes feature_dim)
        self.num_qubits = config.num_reservoir_qubits
        self.reservoir_depth = config.reservoir_depth
        self.reservoir_seed = config.reservoir_seed
        self.use_random_topology = config.use_random_topology

        # Compute feature dimension before calling base init (Decision 7)
        feature_dim = self._compute_feature_dim()

        # Base class handles: seeding, sensory modules, input_dim, actor/critic,
        # LayerNorm, optimizer, LR scheduling, PPO params, buffer, state tracking
        super().__init__(config, feature_dim, num_actions, device, action_set)

        # Compute sensory qubit indices for input encoding
        if config.num_sensory_qubits is not None:
            num_sensory = config.num_sensory_qubits
        else:
            num_sensory = min(self.input_dim, self.num_qubits)
        self.sensory_qubits: list[int] = list(range(num_sensory))

        if self.input_dim > len(self.sensory_qubits):
            logger.warning(
                f"QRH: input_dim={self.input_dim} > num_sensory_qubits="
                f"{len(self.sensory_qubits)}. Features will wrap onto shared qubits. "
                f"Consider increasing num_sensory_qubits or num_reservoir_qubits.",
            )

        # Pre-compute random topology if needed
        self._random_cz_pairs: list[tuple[int, int]] | None = None
        self._random_rotations: list[tuple[int, int, float, float]] | None = None
        if self.use_random_topology:
            self._random_cz_pairs, self._random_rotations = self._generate_random_topology()

        # Pre-compute index arrays for vectorized feature extraction
        num_states = 2**self.num_qubits
        bits = np.arange(num_states, dtype=np.int64)
        self._signs = np.array(
            [1.0 - 2.0 * ((bits >> q) & 1) for q in range(self.num_qubits)],
        )
        self._low_indices: list[np.ndarray] = []
        self._high_indices: list[np.ndarray] = []
        for q in range(self.num_qubits):
            mask = 1 << q
            low = np.where((bits & mask) == 0)[0]
            self._low_indices.append(low)
            self._high_indices.append(low | mask)

        logger.info(
            f"QRHBrain initialized: {self.num_qubits} qubits "
            f"({len(self.sensory_qubits)} sensory), "
            f"{self.reservoir_depth} layers, "
            f"{'random' if self.use_random_topology else 'structured'} topology, "
            f"feature_dim={self.feature_dim}, input_dim={self.input_dim}, "
            f"actor/critic ({config.readout_hidden_dim}x{config.readout_num_layers})",
        )

    # =========================================================================
    # Abstract method implementations
    # =========================================================================

    def _compute_feature_dim(self) -> int:
        """Compute QRH feature dimension: 3N + N(N-1)/2."""
        return _compute_feature_dim(self.num_qubits)

    def _get_reservoir_features(self, sensory_features: np.ndarray) -> np.ndarray:
        """Run sensory features through reservoir and extract X/Y/Z + ZZ features."""
        statevector = self._encode_and_run(sensory_features)
        return self._extract_features(statevector)

    def _create_copy_instance(
        self,
        config: ReservoirHybridBaseConfig,
    ) -> QRHBrain:
        """Construct a fresh QRHBrain for the copy() method."""
        return QRHBrain(
            config=config,  # type: ignore[arg-type]
            num_actions=self.num_actions,
            device=self._device_type,
            action_set=self._action_set,
        )

    # =========================================================================
    # Reservoir Circuit Construction
    # =========================================================================

    def _build_structured_reservoir(self, qc: QuantumCircuit) -> None:
        """Apply structured C. elegans topology gates to the circuit.

        Applies gap junction CZ gates and chemical synapse controlled rotations
        (CRY/CRZ) based on the connectome mapping. Controlled rotations model
        directed synaptic transmission where the postsynaptic response depends
        on the presynaptic neuron's state.
        """
        # Gap junctions -> CZ gates
        for q_a, q_b in GAP_JUNCTION_CZ_PAIRS:
            if q_a < self.num_qubits and q_b < self.num_qubits:
                qc.cz(q_a, q_b)

        # Chemical synapses -> controlled RY/RZ rotations (CRY/CRZ)
        for ctrl, target, ry_angle, rz_angle in CHEMICAL_SYNAPSE_ROTATIONS:
            if ctrl < self.num_qubits and target < self.num_qubits:
                qc.cry(ry_angle, ctrl, target)
                qc.crz(rz_angle, ctrl, target)

    def _generate_random_topology(
        self,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int, float, float]]]:
        """Generate random topology with same density as structured."""
        rng = np.random.default_rng(self.reservoir_seed)

        # Same number of CZ pairs as structured
        num_cz = len(GAP_JUNCTION_CZ_PAIRS)
        random_cz: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        while len(random_cz) < num_cz:
            a, b = sorted(rng.choice(self.num_qubits, size=2, replace=False))
            pair = (int(a), int(b))
            if pair not in seen:
                seen.add(pair)
                random_cz.append(pair)

        # Same number of rotation pairs as structured (ctrl != target for CRY/CRZ)
        num_rot = len(CHEMICAL_SYNAPSE_ROTATIONS)
        random_rot: list[tuple[int, int, float, float]] = []
        for _ in range(num_rot):
            pair = rng.choice(self.num_qubits, size=2, replace=False)
            ctrl = int(pair[0])
            target = int(pair[1])
            ry = float(rng.uniform(-1.0, 1.0))
            rz = float(rng.uniform(-1.0, 1.0))
            random_rot.append((ctrl, target, ry, rz))

        return random_cz, random_rot

    def _build_random_reservoir(self, qc: QuantumCircuit) -> None:
        """Apply random topology gates (for MI comparison).

        Uses controlled rotations (CRY/CRZ) matching the structured reservoir,
        ensuring a fair comparison of topology structure vs random connectivity.
        """
        if self._random_cz_pairs is None or self._random_rotations is None:
            self._random_cz_pairs, self._random_rotations = self._generate_random_topology()

        for q_a, q_b in self._random_cz_pairs:
            qc.cz(q_a, q_b)

        for ctrl, target, ry_angle, rz_angle in self._random_rotations:
            qc.cry(ry_angle, ctrl, target)
            qc.crz(rz_angle, ctrl, target)

    def _encode_and_run(self, features: np.ndarray) -> np.ndarray:
        """Build and execute the reservoir circuit, returning the statevector.

        Constructs: H layer -> [per-qubit input encoding -> reservoir topology] x depth
        Uses Statevector simulation for exact feature computation.

        Per-qubit encoding (matching C. elegans biology):
        - Only sensory qubits (ASEL=0, ASER=1) receive direct input encoding
        - ASEL gets feature 0 (gradient_strength) via RY -- on-response
        - ASER gets feature 1 (relative_angle) via RZ -- off-response / phase
        - Interneuron and command qubits receive signal only through topology
        - Additional features (if any) are distributed across sensory qubits
        """
        qc = QuantumCircuit(self.num_qubits)

        # Initial Hadamard layer
        for q in range(self.num_qubits):
            qc.h(q)

        # Sensory qubits for input encoding (configured in __init__)
        valid_sensory = self.sensory_qubits

        # Data re-uploading: encode inputs before each reservoir layer
        for _layer in range(self.reservoir_depth):
            # Per-qubit asymmetric encoding on sensory qubits only
            for i, feature in enumerate(features):
                if i < len(valid_sensory):
                    qubit = valid_sensory[i]
                    angle = float(feature) * np.pi
                    if i % 2 == 0:
                        qc.ry(angle, qubit)  # Even features: RY (amplitude)
                    else:
                        qc.rz(angle, qubit)  # Odd features: RZ (phase)
                else:
                    qubit = valid_sensory[i % len(valid_sensory)]
                    angle = float(feature) * np.pi
                    if i % 2 == 0:
                        qc.ry(angle, qubit)
                    else:
                        qc.rz(angle, qubit)

            # Apply reservoir topology
            if self.use_random_topology:
                self._build_random_reservoir(qc)
            else:
                self._build_structured_reservoir(qc)

        # Statevector simulation (no measurement gates needed)
        sv = Statevector.from_instruction(qc)

        return np.asarray(sv.data, dtype=np.complex128)

    # =========================================================================
    # Feature Extraction
    # =========================================================================

    def _extract_features(self, statevector: np.ndarray) -> np.ndarray:
        """Extract X/Y/Z expectations and ZZ correlations from the statevector.

        Computes from the full complex statevector amplitudes:
        - <X_i> = sum_k 2*Re(psi_k* * psi_{k xor 2^i}) for each qubit i
        - <Y_i> = sum_k 2*Im(psi_{k xor 2^i}* * psi_k) for each qubit i
        - <Z_i> = sum_k (-1)^bit(k,i) |psi_k|^2 for each qubit i
        - <Z_i Z_j> = sum_k (-1)^(bit(k,i)+bit(k,j)) |psi_k|^2 for each pair (i,j)

        Uses pre-computed index arrays (_low_indices, _high_indices, _signs)
        from __init__ for vectorized numpy operations.

        Output dimension: 3N + N(N-1)/2 = 52 for 8 qubits.
        """
        n = self.num_qubits
        probabilities = np.abs(statevector) ** 2

        # X-expectations: <X_q> = 2 * sum_{k: bit(k,q)=0} Re(psi_k* * psi_{k|2^q})
        x_expectations = np.empty(n)
        for q in range(n):
            low = self._low_indices[q]
            high = self._high_indices[q]
            x_expectations[q] = 2.0 * np.sum(
                np.real(np.conj(statevector[low]) * statevector[high]),
            )

        # Y-expectations: <Y_q> = 2 * sum_{k: bit(k,q)=0} Im(psi_{k|2^q}* * psi_k)
        y_expectations = np.empty(n)
        for q in range(n):
            low = self._low_indices[q]
            high = self._high_indices[q]
            y_expectations[q] = 2.0 * np.sum(
                np.imag(np.conj(statevector[high]) * statevector[low]),
            )

        # Z-expectations: <Z_q> = sum_k (-1)^bit(k,q) |psi_k|^2
        z_expectations = self._signs @ probabilities

        # ZZ-correlations: <Z_i Z_j> for i < j
        zz_correlations = np.empty(n * (n - 1) // 2)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                zz_correlations[idx] = (self._signs[i] * self._signs[j]) @ probabilities
                idx += 1

        features = np.concatenate(
            [x_expectations, y_expectations, z_expectations, zz_correlations],
        )
        return features.astype(np.float32)
