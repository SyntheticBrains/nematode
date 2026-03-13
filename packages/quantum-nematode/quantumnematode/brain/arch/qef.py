"""
Quantum Entangled Features (QEF) Brain Architecture.

This architecture implements a parameterized quantum circuit with purposeful cross-modal
entanglement between sensory-modality qubit pairs and a PPO-trained classical actor-critic
readout. Unlike QRH's random reservoir, QEF uses configurable entanglement topologies
(modality-paired, ring, random) to encode cross-modal sensory interactions.

Key Features:
- **Uniform RY Encoding**: All qubits receive sensory input via RY(feature * pi) gates,
  maximizing information available for entanglement-driven correlations.
- **Configurable Entanglement Topology**: Three CZ-based topologies — modality_paired
  (default, maps food↔nociception and thermo↔mechano), ring, and seeded random.
- **CZ-Only Entanglement**: Intentionally uses CZ gates only (no CRY/CRZ controlled
  rotations as in QRH) to isolate entanglement topology structure as the variable
  under test.
- **Z + ZZ + cos/sin Feature Extraction**: Per-qubit Z expectations, pairwise ZZ
  correlations, and cos/sin nonlinear transforms of Z expectations. cos/sin replaces
  QRH's X/Y expectations because uniform RY encoding makes X/Y less informative.
- **Separable Ablation Mode**: `entanglement_enabled=False` skips all CZ gates for
  controlled A/B experiments isolating entanglement effects.
- **PPO Actor-Critic Readout**: Inherited from ReservoirHybridBase (identical to QRH).

Architecture:
    Sensory Input -> Uniform RY Encoding on ALL Qubits
    -> [RY + CZ Topology]^depth -> Statevector
    -> Z + ZZ + cos/sin Features -> PPO Readout

Feature dimension: 3N + N(N-1)/2 = 52 for 8 qubits
  - N Z expectations
  - N(N-1)/2 ZZ correlations
  - 2N cos/sin features (cos(Z_i), sin(Z_i))
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import Field, field_validator
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

DEFAULT_NUM_QUBITS = 8
DEFAULT_CIRCUIT_DEPTH = 2
DEFAULT_CIRCUIT_SEED = 42

# Validation constants
MIN_QUBITS = 2
MIN_CIRCUIT_DEPTH = 1

# =============================================================================
# Modality-Paired CZ Topology Constants
# =============================================================================
# 8-qubit mapping for cross-modal entanglement:
#   Qubits 0-1: food_chemotaxis (strength, angle)
#   Qubits 2-3: nociception (strength, angle)
#   Qubits 4-5: thermotaxis (strength, angle)
#   Qubits 6-7: mechanosensation (strength, angle)
#
# Cross-modal CZ pairs encode interacting sensory modalities:
#   (0, 2): food_chemotaxis ↔ nociception (approach vs avoidance)
#   (1, 3): food_chemotaxis ↔ nociception (angular correlation)
#   (4, 6): thermotaxis ↔ mechanosensation (environmental sensing)
#   (5, 7): thermotaxis ↔ mechanosensation (angular correlation)

MODALITY_PAIRED_CZ: list[tuple[int, int]] = [
    (0, 2),  # food_chemotaxis ↔ nociception (strength)
    (1, 3),  # food_chemotaxis ↔ nociception (angle)
    (4, 6),  # thermotaxis ↔ mechanosensation (strength)
    (5, 7),  # thermotaxis ↔ mechanosensation (angle)
]


def _compute_feature_dim(num_qubits: int) -> int:
    """Compute QEF feature dimension: 3N (Z + cos_z + sin_z) + N(N-1)/2 (ZZ correlations)."""
    return 3 * num_qubits + num_qubits * (num_qubits - 1) // 2


# =============================================================================
# Configuration
# =============================================================================


class QEFBrainConfig(ReservoirHybridBaseConfig):
    """Configuration for the QEFBrain architecture.

    Inherits PPO readout, LR scheduling, and sensory module config from
    ReservoirHybridBaseConfig. Adds quantum-specific fields for the
    entangled PQC feature extractor.
    """

    num_qubits: int = Field(
        default=DEFAULT_NUM_QUBITS,
        description="Number of qubits in the parameterized quantum circuit.",
    )
    circuit_depth: int = Field(
        default=DEFAULT_CIRCUIT_DEPTH,
        description="Number of encoding + entanglement layers (data re-uploading depth).",
    )
    circuit_seed: int = Field(
        default=DEFAULT_CIRCUIT_SEED,
        description="Seed for deterministic random topology construction.",
    )
    entanglement_topology: Literal["modality_paired", "ring", "random"] = Field(
        default="modality_paired",
        description=(
            "Entanglement topology for CZ gates: "
            "'modality_paired' (cross-modal pairs), 'ring' (nearest-neighbor), "
            "'random' (seeded random pairs)."
        ),
    )
    entanglement_enabled: bool = Field(
        default=True,
        description=(
            "Enable entanglement (CZ gates). When False, the circuit becomes a "
            "separable (product-state) PQC for ablation experiments."
        ),
    )

    encoding_mode: Literal["uniform", "sparse"] = Field(
        default="uniform",
        description=(
            "Input encoding strategy: 'uniform' encodes all qubits via RY(feature*pi), "
            "'sparse' encodes only the first input_dim qubits (like QRH) with "
            "asymmetric RY/RZ, leaving remaining qubits in H-superposition."
        ),
    )
    gate_mode: Literal["cz", "cry_crz"] = Field(
        default="cz",
        description=(
            "Entanglement gate type: 'cz' uses CZ-only phase gates, "
            "'cry_crz' uses parameterized controlled rotations (CRY/CRZ) "
            "with seeded random angles, matching QRH's gate richness."
        ),
    )
    feature_mode: Literal["z_cossin", "xyz"] = Field(
        default="z_cossin",
        description=(
            "Feature extraction channels: 'z_cossin' uses Z + ZZ + cos/sin(Z) "
            "(3N + N(N-1)/2 features), 'xyz' uses X + Y + Z + ZZ "
            "(3N + N(N-1)/2 features, matching QRH)."
        ),
    )
    trainable_entanglement: bool = Field(
        default=False,
        description="Reserved for future work. Raises NotImplementedError if True.",
    )

    @field_validator("num_qubits")
    @classmethod
    def validate_num_qubits(cls, v: int) -> int:
        """Validate num_qubits >= 2."""
        if v < MIN_QUBITS:
            msg = f"num_qubits must be >= {MIN_QUBITS}, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("circuit_depth")
    @classmethod
    def validate_circuit_depth(cls, v: int) -> int:
        """Validate circuit_depth >= 1."""
        if v < MIN_CIRCUIT_DEPTH:
            msg = f"circuit_depth must be >= {MIN_CIRCUIT_DEPTH}, got {v}"
            raise ValueError(msg)
        return v


# =============================================================================
# QEF Brain
# =============================================================================


class QEFBrain(ReservoirHybridBase):
    """Quantum Entangled Features brain architecture.

    Uses a parameterized quantum circuit with configurable cross-modal entanglement
    topology to generate Z + ZZ + cos/sin feature representations, with a PPO-trained
    classical actor-critic readout. The entanglement topology is the key variable
    under test — modality-paired encodes hypothesized sensory interactions.
    """

    _brain_name = "QEF"

    def __init__(
        self,
        config: QEFBrainConfig,
        num_actions: int = 4,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] | None = None,
    ) -> None:
        if config.trainable_entanglement:
            msg = (
                "Trainable entanglement is reserved for future work. "
                "Set trainable_entanglement=False."
            )
            raise NotImplementedError(msg)

        # Quantum parameters (must be set before super().__init__ computes feature_dim)
        self.num_qubits = config.num_qubits
        self.circuit_depth = config.circuit_depth
        self.circuit_seed = config.circuit_seed
        self.entanglement_topology = config.entanglement_topology
        self.entanglement_enabled = config.entanglement_enabled
        self.encoding_mode = config.encoding_mode
        self.gate_mode = config.gate_mode
        self.feature_mode = config.feature_mode

        # Compute feature dimension before calling base init
        feature_dim = self._compute_feature_dim()

        # Base class handles: seeding, sensory modules, input_dim, actor/critic,
        # LayerNorm, optimizer, LR scheduling, PPO params, buffer, state tracking
        super().__init__(config, feature_dim, num_actions, device, action_set)

        # For sparse encoding: compute sensory qubit indices (first input_dim qubits)
        if self.encoding_mode == "sparse":
            self._sensory_qubits = list(range(min(self.input_dim, self.num_qubits)))
        else:
            self._sensory_qubits = list(range(self.num_qubits))

        # Pre-compute topology pairs and gate angles
        self._cz_pairs: list[tuple[int, int]] = self._build_topology_pairs()

        # For cry_crz mode: pre-compute random rotation angles per CZ pair
        if self.gate_mode == "cry_crz":
            gate_rng = np.random.default_rng(self.circuit_seed)
            self._cry_angles = gate_rng.uniform(0.1, np.pi, size=len(self._cz_pairs))
            self._crz_angles = gate_rng.uniform(0.1, np.pi, size=len(self._cz_pairs))
        else:
            self._cry_angles = np.array([])
            self._crz_angles = np.array([])

        # Pre-compute sign array for vectorized Z and ZZ extraction.
        # _signs[q, k] = (-1)^bit(k, q) for computing <Z_q> = _signs[q] @ probs.
        num_states = 2**self.num_qubits
        bits = np.arange(num_states, dtype=np.int64)
        self._signs = np.array(
            [1.0 - 2.0 * ((bits >> q) & 1) for q in range(self.num_qubits)],
        )

        # For xyz feature mode: pre-compute low/high index arrays for X/Y expectations.
        # X/Y require off-diagonal statevector elements: pairs where bit q differs.
        if self.feature_mode == "xyz":
            self._low_indices: list[np.ndarray] = []
            self._high_indices: list[np.ndarray] = []
            for q in range(self.num_qubits):
                mask = 1 << q
                low = np.where((bits & mask) == 0)[0]
                self._low_indices.append(low)
                self._high_indices.append(low | mask)

        logger.info(
            f"QEFBrain initialized: {self.num_qubits} qubits, "
            f"topology={self.entanglement_topology}, "
            f"entanglement_enabled={self.entanglement_enabled}, "
            f"encoding={self.encoding_mode}, gate={self.gate_mode}, "
            f"features={self.feature_mode}, "
            f"circuit_depth={self.circuit_depth}, "
            f"feature_dim={self.feature_dim}, input_dim={self.input_dim}",
        )

    # =========================================================================
    # Abstract method implementations
    # =========================================================================

    def _compute_feature_dim(self) -> int:
        """Compute QEF feature dimension: 3N + N(N-1)/2."""
        return _compute_feature_dim(self.num_qubits)

    def _get_reservoir_features(self, sensory_features: np.ndarray) -> np.ndarray:
        """Run sensory features through the PQC and extract Z + ZZ + cos/sin features."""
        statevector = self._encode_and_run(sensory_features)
        return self._extract_features(statevector)

    def _create_copy_instance(
        self,
        config: ReservoirHybridBaseConfig,
    ) -> QEFBrain:
        """Construct a fresh QEFBrain for the copy() method."""
        return QEFBrain(
            config=config,  # type: ignore[arg-type]
            num_actions=self.num_actions,
            device=self._device_type,
            action_set=self._action_set,
        )

    # =========================================================================
    # Topology Construction
    # =========================================================================

    def _build_topology_pairs(self) -> list[tuple[int, int]]:
        """Build CZ pairs for the configured entanglement topology."""
        if self.entanglement_topology == "modality_paired":
            return self._build_modality_paired_topology()
        if self.entanglement_topology == "ring":
            return self._build_ring_topology()
        if self.entanglement_topology == "random":
            return self._build_random_topology()
        msg = f"Unknown entanglement topology: {self.entanglement_topology}"
        raise ValueError(msg)

    def _build_modality_paired_topology(self) -> list[tuple[int, int]]:
        """Build modality-paired CZ pairs, filtering to valid qubit indices."""
        return [
            (a, b) for a, b in MODALITY_PAIRED_CZ if a < self.num_qubits and b < self.num_qubits
        ]

    def _build_ring_topology(self) -> list[tuple[int, int]]:
        """Build ring CZ topology: (0,1), (1,2), ..., (N-2,N-1), (N-1,0)."""
        n = self.num_qubits
        return [(i, (i + 1) % n) for i in range(n)]

    def _build_random_topology(self) -> list[tuple[int, int]]:
        """Build seeded random CZ pairs with same count as modality-paired."""
        rng = np.random.default_rng(self.circuit_seed)
        all_pairs = list(itertools.combinations(range(self.num_qubits), 2))
        # Same number of CZ pairs as modality_paired (4 for 8 qubits)
        num_cz = len(MODALITY_PAIRED_CZ)
        max_pairs = len(all_pairs)
        num_cz = min(num_cz, max_pairs)
        chosen_indices = rng.choice(max_pairs, size=num_cz, replace=False)
        return [all_pairs[i] for i in chosen_indices]

    # =========================================================================
    # Circuit Construction
    # =========================================================================

    def _apply_entanglement(self, qc: QuantumCircuit) -> None:
        """Apply entanglement gates per configured topology and gate mode.

        Gate modes:
        - **cz**: CZ-only phase gates (original QEF).
        - **cry_crz**: Parameterized controlled rotations (CRY + CRZ) with
          seeded random angles, matching QRH's gate richness.

        Skipped entirely when entanglement_enabled is False (separable ablation).
        """
        if not self.entanglement_enabled:
            return
        if self.gate_mode == "cry_crz":
            for idx, (q_a, q_b) in enumerate(self._cz_pairs):
                qc.cry(float(self._cry_angles[idx]), q_a, q_b)
                qc.crz(float(self._crz_angles[idx]), q_a, q_b)
        else:
            for q_a, q_b in self._cz_pairs:
                qc.cz(q_a, q_b)

    def _encode_and_run(self, features: np.ndarray) -> np.ndarray:
        """Build and execute the PQC, returning the statevector.

        Constructs: H layer -> [encoding + CZ topology] x depth -> statevector.
        Uses Statevector simulation for exact feature computation.

        Encoding modes:
        - **uniform**: RY(feature*pi) on ALL qubits (feature index % num_qubits).
          Unencoded qubits remain in H-superposition.
        - **sparse**: RY/RZ on sensory qubits only (first input_dim qubits).
          Even features use RY, odd features use RZ (matching QRH pattern).
          Non-sensory qubits receive signal only through entanglement.
        """
        qc = QuantumCircuit(self.num_qubits)

        # Initial Hadamard layer
        for q in range(self.num_qubits):
            qc.h(q)

        # Data re-uploading: encode inputs before each entanglement layer
        for _layer in range(self.circuit_depth):
            if self.encoding_mode == "sparse":
                # Sparse: encode only on sensory qubits with asymmetric RY/RZ
                sensory = self._sensory_qubits
                for i, feature in enumerate(features):
                    qubit = sensory[i % len(sensory)]
                    angle = float(feature) * np.pi
                    if i % 2 == 0:
                        qc.ry(angle, qubit)
                    else:
                        qc.rz(angle, qubit)
            else:
                # Uniform: RY on all qubits
                for i, feature in enumerate(features):
                    qubit = i % self.num_qubits
                    angle = float(feature) * np.pi
                    qc.ry(angle, qubit)

            # Apply entanglement topology (skipped if entanglement_enabled=False)
            self._apply_entanglement(qc)

        # Statevector simulation (no measurement gates needed)
        sv = Statevector.from_instruction(qc)

        return np.asarray(sv.data, dtype=np.complex128)

    # =========================================================================
    # Feature Extraction
    # =========================================================================

    def _extract_features(self, statevector: np.ndarray) -> np.ndarray:
        """Extract features from statevector based on configured feature_mode.

        Feature modes:
        - **z_cossin** (default): Z + ZZ + cos/sin(Z) = 3N + N(N-1)/2 features.
          Output: [z_0..z_N-1, zz_01..zz_(N-1)N, cos_z_0..cos_z_N-1, sin_z_0..sin_z_N-1]
        - **xyz**: X + Y + Z + ZZ = 3N + N(N-1)/2 features (matching QRH).
          Output: [x_0..x_N-1, y_0..y_N-1, z_0..z_N-1, zz_01..zz_(N-1)N]

        Uses pre-computed index arrays from __init__ for vectorized operations.
        """
        n = self.num_qubits
        probabilities = np.abs(statevector) ** 2

        # Z-expectations: <Z_q> = sum_k (-1)^bit(k,q) |psi_k|^2
        z_expectations = self._signs @ probabilities

        # ZZ-correlations: <Z_i Z_j> for i < j
        zz_correlations = np.empty(n * (n - 1) // 2)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                zz_correlations[idx] = (self._signs[i] * self._signs[j]) @ probabilities
                idx += 1

        if self.feature_mode == "xyz":
            # X-expectations: <X_q> = 2 Re(sum_{k: bit(k,q)=0} psi_k* psi_{k|2^q})
            x_expectations = np.empty(n)
            for q in range(n):
                low = self._low_indices[q]
                high = self._high_indices[q]
                x_expectations[q] = 2.0 * np.sum(
                    np.real(np.conj(statevector[low]) * statevector[high]),
                )

            # Y-expectations: <Y_q> = 2 Im(sum_{k: bit(k,q)=0} psi_{k|2^q}* psi_k)
            y_expectations = np.empty(n)
            for q in range(n):
                low = self._low_indices[q]
                high = self._high_indices[q]
                y_expectations[q] = 2.0 * np.sum(
                    np.imag(np.conj(statevector[high]) * statevector[low]),
                )

            features = np.concatenate(
                [x_expectations, y_expectations, z_expectations, zz_correlations],
            )
        else:
            # cos/sin features from Z expectations
            cos_z = np.cos(z_expectations)
            sin_z = np.sin(z_expectations)

            features = np.concatenate(
                [z_expectations, zz_correlations, cos_z, sin_z],
            )
        return features.astype(np.float32)
