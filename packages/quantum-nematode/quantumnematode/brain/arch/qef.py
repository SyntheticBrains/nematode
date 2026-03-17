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
import torch
from pydantic import Field, field_validator, model_validator
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from torch import nn

from quantumnematode.brain.arch._quantum_reservoir import build_readout_network
from quantumnematode.brain.arch._reservoir_hybrid_base import (
    ReservoirHybridBase,
    ReservoirHybridBaseConfig,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.logging_config import logger

if TYPE_CHECKING:
    from quantumnematode.brain.actions import Action
    from quantumnematode.brain.arch import BrainParams

from quantumnematode.brain.actions import ActionData

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


def _get_cross_modal_pairs(num_qubits: int, input_dim: int) -> list[tuple[int, int]]:
    """Get cross-modal ZZ pairs based on qubit-to-modality mapping.

    Assumes uniform encoding cycles features across qubits. Modalities are
    determined by the sensory module structure (2 features per module for
    food/nociception, 3 for thermotaxis).
    """
    # Map each qubit to its modality index based on feature cycling
    # Default: food(0,1), noci(2,3), thermo(4,5,6), wrap(7->food again)
    module_sizes = [2, 2, 3] if input_dim >= 7 else [2, 2]  # noqa: PLR2004
    modality: list[int] = []
    mod_id = 0
    size_idx = 0
    remaining = module_sizes[0] if module_sizes else 1

    for _ in range(num_qubits):
        modality.append(mod_id)
        remaining -= 1
        if remaining == 0:
            mod_id += 1
            size_idx += 1
            remaining = module_sizes[size_idx] if size_idx < len(module_sizes) else 999

    return [
        (i, j)
        for i in range(num_qubits)
        for j in range(i + 1, num_qubits)
        if modality[i] != modality[j]
    ]


def _compute_feature_dim(
    num_qubits: int,
    *,
    include_zzz: bool = False,
    zz_mode: str = "all",
    include_cossin: bool = True,
    input_dim: int = 2,
) -> int:
    """Compute QEF feature dimension with configurable feature subsets."""
    # Z expectations always included
    dim = num_qubits

    # ZZ correlations
    if zz_mode == "cross_modal":
        dim += len(_get_cross_modal_pairs(num_qubits, input_dim))
    else:
        dim += num_qubits * (num_qubits - 1) // 2

    # cos/sin or X/Y features
    if include_cossin:
        dim += 2 * num_qubits

    # ZZZ three-body
    if include_zzz:
        dim += num_qubits * (num_qubits - 1) * (num_qubits - 2) // 6

    return dim


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
    hybrid_input: bool = Field(
        default=False,
        description=(
            "Enable hybrid input: concatenate raw sensory features with quantum "
            "features before the readout MLP. Gives the network direct access to "
            "actionable signals while still benefiting from quantum correlations."
        ),
    )
    hybrid_polynomial: bool = Field(
        default=False,
        description=(
            "Add classical pairwise polynomial features (x_i * x_j) to the hybrid "
            "input alongside raw + quantum features. Requires hybrid_input=True."
        ),
    )
    zz_mode: Literal["all", "cross_modal"] = Field(
        default="all",
        description=(
            "ZZ correlation mode: 'all' computes all N(N-1)/2 pairwise correlations, "
            "'cross_modal' computes only cross-modal pairs (food-noci, food-thermo, "
            "noci-thermo) to reduce noise from intra-modal correlations."
        ),
    )
    include_cossin: bool = Field(
        default=True,
        description="Include cos/sin(Z) features. Disable to reduce feature dimensionality.",
    )
    include_zzz: bool = Field(
        default=False,
        description=(
            "Include ZZZ three-body correlations in the quantum feature vector. "
            "Adds N(N-1)(N-2)/6 features (56 for 8 qubits). These capture "
            "genuinely quantum three-body entanglement effects."
        ),
    )
    feature_gating: Literal["none", "static", "context", "mixed"] = Field(
        default="none",
        description=(
            "Feature gating mode for quantum features: "
            "'none' = no gating, "
            "'static' = sigmoid(w) * quantum_features (learned per-dimension weights), "
            "'context' = sigmoid(MLP(raw_features)) * quantum_features "
            "(input-dependent gating based on current sensory state), "
            "'mixed' = average of static + context gates — combines robust static "
            "baseline with adaptive input-dependent modulation."
        ),
    )
    separate_critic: bool = Field(
        default=False,
        description=(
            "Give the critic raw sensory features directly instead of quantum "
            "features. Faster V(s) estimation since the critic doesn't need "
            "cross-modal quantum correlations."
        ),
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

    @model_validator(mode="after")
    def validate_hybrid_dependencies(self) -> QEFBrainConfig:
        """Validate that features requiring hybrid_input have it enabled."""
        if self.separate_critic and not self.hybrid_input:
            msg = "separate_critic requires hybrid_input=True"
            raise ValueError(msg)
        if self.hybrid_polynomial and not self.hybrid_input:
            msg = "hybrid_polynomial requires hybrid_input=True"
            raise ValueError(msg)
        return self


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
        self.hybrid_input = config.hybrid_input
        self.hybrid_polynomial = config.hybrid_polynomial
        self.zz_mode = config.zz_mode
        self.include_cossin = config.include_cossin
        self.include_zzz = config.include_zzz
        self.feature_gating = config.feature_gating  # "none", "static", or "context"
        self.separate_critic = config.separate_critic

        # Pre-compute input_dim for hybrid feature dim calculation
        # (super().__init__ also computes this, but we need it earlier)
        if config.sensory_modules is not None:
            from quantumnematode.brain.modules import get_classical_feature_dimension

            self._raw_input_dim = get_classical_feature_dimension(config.sensory_modules)
        else:
            self._raw_input_dim = 2

        # Compute feature dimension before calling base init
        feature_dim = self._compute_feature_dim()

        # Base class handles: seeding, sensory modules, input_dim, actor/critic,
        # LayerNorm, optimizer, LR scheduling, PPO params, buffer, state tracking
        super().__init__(config, feature_dim, num_actions, device, action_set)

        # Set up optional gating and separate critic components
        self._init_gating_and_critic(config)

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

        # Pre-compute hot-path caches and initialize state
        self._init_caches()

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
            f"feature_dim={self.feature_dim}, input_dim={self.input_dim}"
            f"{', hybrid_input=True' if self.hybrid_input else ''}"
            f"{f', gating={self.feature_gating}' if self.feature_gating != 'none' else ''}",
        )

    def _init_caches(self) -> None:
        """Pre-compute hot-path caches for feature extraction and gating."""
        self._cached_quantum_dim = self._quantum_feature_dim()
        if self.zz_mode == "cross_modal":
            self._cached_zz_pairs = _get_cross_modal_pairs(
                self.num_qubits,
                self._raw_input_dim,
            )
        else:
            self._cached_zz_pairs = [
                (i, j) for i in range(self.num_qubits) for j in range(i + 1, self.num_qubits)
            ]
        # Initialize state for separate_critic (avoids AttributeError on first call)
        self._last_raw_features: np.ndarray = np.zeros(
            self._raw_input_dim,
            dtype=np.float32,
        )

    def _init_gating_and_critic(self, config: QEFBrainConfig) -> None:
        """Initialize optional feature gating and separate critic components."""
        quantum_dim = self._quantum_feature_dim()

        if self.feature_gating in ("static", "mixed"):
            self.gate_weights = nn.Parameter(torch.zeros(quantum_dim, device=self.device))
            self.optimizer.add_param_group({"params": [self.gate_weights]})
        if self.feature_gating in ("context", "mixed"):
            # Context-dependent gating: raw_features → MLP → sigmoid → gate
            # Small MLP: raw_dim → 16 → quantum_dim
            self.gate_network = nn.Sequential(
                nn.Linear(self._raw_input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, quantum_dim),
            ).to(self.device)
            # Initialize final layer bias to 0 (sigmoid(0)=0.5, neutral gate)
            final_layer: nn.Linear = self.gate_network[-1]  # type: ignore[assignment]
            nn.init.zeros_(final_layer.bias)
            nn.init.xavier_normal_(final_layer.weight, gain=0.1)
            self.optimizer.add_param_group({"params": self.gate_network.parameters()})

        if self.separate_critic:
            self.critic = build_readout_network(
                input_dim=self._raw_input_dim,
                hidden_dim=config.readout_hidden_dim,
                output_dim=1,
                readout_type="mlp",
                num_layers=config.readout_num_layers,
            ).to(self.device)
            self.critic_norm = nn.LayerNorm(self._raw_input_dim).to(self.device)
            gating_params: list[torch.nn.Parameter] = []
            if self.feature_gating in ("static", "mixed"):
                gating_params.append(self.gate_weights)
            if self.feature_gating in ("context", "mixed"):
                gating_params.extend(self.gate_network.parameters())
            self.optimizer = torch.optim.Adam(
                list(self.actor.parameters())
                + list(self.critic.parameters())
                + list(self.feature_norm.parameters())
                + list(self.critic_norm.parameters())
                + gating_params,
                lr=config.actor_lr,
            )

    # =========================================================================
    # Abstract method implementations
    # =========================================================================

    def _quantum_feature_dim(self) -> int:
        """Compute quantum-only feature dimension."""
        return _compute_feature_dim(
            self.num_qubits,
            include_zzz=self.include_zzz,
            zz_mode=self.zz_mode,
            include_cossin=self.include_cossin,
            input_dim=self._raw_input_dim,
        )

    def _compute_feature_dim(self) -> int:
        """Compute QEF feature dimension, optionally including ZZZ and polynomial features."""
        quantum_dim = self._quantum_feature_dim()
        if self.hybrid_input:
            dim = self._raw_input_dim + quantum_dim
            if self.hybrid_polynomial:
                n = self._raw_input_dim
                dim += n * (n - 1) // 2  # pairwise products
            return dim
        return quantum_dim

    def _get_reservoir_features(self, sensory_features: np.ndarray) -> np.ndarray:
        """Run sensory features through the PQC and extract Z + ZZ + cos/sin features.

        If hybrid_input is enabled, concatenates raw sensory features with quantum
        features to give the readout MLP direct access to actionable signals.
        """
        statevector = self._encode_and_run(sensory_features)
        quantum_features = self._extract_features(statevector)

        # Store raw features for separate critic
        if self.separate_critic:
            self._last_raw_features = sensory_features

        if self.hybrid_input:
            parts = [sensory_features, quantum_features]
            if self.hybrid_polynomial:
                n = len(sensory_features)
                poly = np.array(
                    [
                        sensory_features[i] * sensory_features[j]
                        for i in range(n)
                        for j in range(i + 1, n)
                    ],
                    dtype=np.float32,
                )
                parts.append(poly)
            return np.concatenate(parts)
        return quantum_features

    def _apply_feature_gating(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learnable gating to quantum feature dimensions.

        Modes:
        - 'static': sigmoid(w) * quantum_features (per-dimension learned weights)
        - 'context': sigmoid(MLP(raw_features)) * quantum_features (input-dependent)

        For hybrid input, only gates the quantum portion. Raw features pass through.
        """
        if self.feature_gating == "none":
            return x

        if self.hybrid_input:
            raw = x[..., : self._raw_input_dim]
            quantum = x[..., self._raw_input_dim : self._raw_input_dim + self._cached_quantum_dim]
            gate = self._compute_gate(raw)
            parts = [raw, quantum * gate]
            # Polynomial features (if present) pass through ungated
            if self.hybrid_polynomial:
                poly = x[..., self._raw_input_dim + self._cached_quantum_dim :]
                parts.append(poly)
            return torch.cat(parts, dim=-1)

        # Non-hybrid: gate all features
        gate = self._compute_gate(x)
        return x * gate

    def _compute_gate(self, gate_input: torch.Tensor) -> torch.Tensor:
        """Compute gate values from static weights, context network, or both."""
        if self.feature_gating == "static":
            return torch.sigmoid(self.gate_weights)
        if self.feature_gating == "context":
            return torch.sigmoid(self.gate_network(gate_input))
        # mixed: average of static + context gates
        static_gate = torch.sigmoid(self.gate_weights)
        context_gate = torch.sigmoid(self.gate_network(gate_input))
        return (static_gate + context_gate) * 0.5

    def _get_critic_value(self, reservoir_features: torch.Tensor) -> torch.Tensor:
        """Compute critic value, using raw features if separate_critic is enabled."""
        if self.separate_critic:
            raw_x = torch.tensor(
                self._last_raw_features,
                dtype=torch.float32,
                device=self.device,
            )
            raw_x = self.critic_norm(raw_x)
            return self.critic(raw_x)
        return self.critic(reservoir_features)

    def _collect_trainable_params(self) -> list[torch.nn.Parameter]:
        """Collect all trainable parameters for gradient clipping in PPO update."""
        params = (
            list(self.actor.parameters())
            + list(self.critic.parameters())
            + list(self.feature_norm.parameters())
        )
        if self.feature_gating in ("static", "mixed"):
            params.append(self.gate_weights)
        if self.feature_gating in ("context", "mixed"):
            params.extend(self.gate_network.parameters())
        if self.separate_critic:
            params.extend(self.critic_norm.parameters())
        return params

    def _needs_custom_forward(self) -> bool:
        """Check if custom forward pass is needed for gating or separate critic."""
        return self.feature_gating != "none" or self.separate_critic

    def run_brain(
        self,
        params: BrainParams,
        reward: float | None = None,
        input_data: list[float] | None = None,
        *,
        top_only: bool,
        top_randomize: bool,
    ) -> list[ActionData]:
        """Override run_brain to support feature gating and separate critic."""
        if not self._needs_custom_forward():
            return super().run_brain(
                params,
                reward,
                input_data,
                top_only=top_only,
                top_randomize=top_randomize,
            )

        sensory_features = self.preprocess(params)
        reservoir_features = self._get_reservoir_features(sensory_features)

        x = torch.tensor(reservoir_features, dtype=torch.float32, device=self.device)
        x = self._apply_feature_gating(x)
        x_normed = self.feature_norm(x)
        logits = self.actor(x_normed)

        # Critic: use raw features if separate_critic, otherwise same as actor
        if self.separate_critic:
            raw_x = torch.tensor(
                self._last_raw_features,
                dtype=torch.float32,
                device=self.device,
            )
            value = self.critic(self.critic_norm(raw_x))
        else:
            value = self.critic(x_normed)

        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_idx = int(dist.sample().item())
        log_prob = dist.log_prob(torch.tensor(action_idx, device=self.device))

        action_name = self.action_set[action_idx]
        probs_np = probs.detach().cpu().numpy()
        self.current_probabilities = probs_np

        if self.buffer.position % 50 == 0:
            feat_min, feat_max = float(reservoir_features.min()), float(reservoir_features.max())
            logits_np = logits.detach().cpu().numpy()
            logger.debug(
                f"{self._brain_name} step {self.buffer.position}: "
                f"features=[{feat_min:.3f}, {feat_max:.3f}], "
                f"probs={probs_np}, logits=[{logits_np.min():.3f}, {logits_np.max():.3f}], "
                f"value={value.item():.4f}",
            )

        if self._deferred_ppo_update:
            logger.debug(
                f"{self._brain_name} executing deferred PPO update with correct "
                f"V(s_{{t+1}})={value.item():.4f}",
            )
            self._perform_ppo_update(bootstrap_value=value)
            self.buffer.reset()
            self._deferred_ppo_update = False

        self._pending_state = reservoir_features
        self._pending_action = action_idx
        self._pending_log_prob = log_prob
        self._pending_value = value
        self.last_value = value

        self.latest_data.action = ActionData(
            state=action_name,
            action=action_name,
            probability=probs_np[action_idx],
        )
        self.history_data.actions.append(self.latest_data.action)
        self.history_data.probabilities.append(float(probs_np[action_idx]))

        return [self.latest_data.action]

    def _perform_ppo_update(self, bootstrap_value: torch.Tensor | None = None) -> None:
        """Override PPO update to support feature gating and separate critic."""
        if not self._needs_custom_forward():
            super()._perform_ppo_update(bootstrap_value)
            return

        if len(self.buffer) == 0:
            return

        min_buffer_size = min(64, self.buffer.buffer_size // 2)
        if len(self.buffer) < min_buffer_size:
            logger.debug(
                f"{self._brain_name} skipping PPO update: "
                f"buffer={len(self.buffer)}/{self.buffer.buffer_size} < min={min_buffer_size}",
            )
            return

        last_value = (
            bootstrap_value
            if bootstrap_value is not None
            else (
                self.last_value
                if self.last_value is not None
                else torch.tensor([0.0], device=self.device)
            )
        )

        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value,
            self.gamma,
            self.gae_lambda,
        )
        # Note: advantage normalization happens inside get_minibatches (base class)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        num_updates = 0

        all_params = self._collect_trainable_params()

        for _ in range(self.ppo_epochs):
            for batch in self.buffer.get_minibatches(self.ppo_minibatches, returns, advantages):
                states = batch["states"]

                # Apply feature gating then normalize for actor
                gated_states = self._apply_feature_gating(states)
                normalized_states = self.feature_norm(gated_states)
                logits = self.actor(normalized_states)

                # Critic: raw features if separate, otherwise same as actor
                if self.separate_critic:
                    raw_states = states[:, : self._raw_input_dim]
                    critic_input = self.critic_norm(raw_states)
                    values = self.critic(critic_input).squeeze(-1)
                else:
                    values = self.critic(normalized_states).squeeze(-1)

                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch["actions"])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch["old_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch["advantages"]
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, batch["returns"])

                current_entropy_coeff = self._get_current_entropy_coeff()
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    - current_entropy_coeff * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy.item()
                num_updates += 1

        if num_updates > 0:
            avg_loss = total_policy_loss / num_updates
            avg_vloss = total_value_loss / num_updates
            avg_entropy = total_entropy_loss / num_updates
            self.latest_data.loss = avg_loss
            self.history_data.losses.append(avg_loss)
            logger.debug(
                f"{self._brain_name} PPO update: "
                f"policy_loss={avg_loss:.4f}, value_loss={avg_vloss:.4f}, "
                f"entropy={avg_entropy:.4f}, updates={num_updates}",
            )

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

    def _extract_features(self, statevector: np.ndarray) -> np.ndarray:  # noqa: C901, PLR0912
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

        # ZZ-correlations (using precomputed pairs from __init__)
        zz_pairs = self._cached_zz_pairs
        zz_correlations = np.empty(len(zz_pairs))
        for idx, (i, j) in enumerate(zz_pairs):
            zz_correlations[idx] = (self._signs[i] * self._signs[j]) @ probabilities

        # ZZZ three-body correlations: <Z_i Z_j Z_k> for i < j < k
        if self.include_zzz:
            num_zzz = n * (n - 1) * (n - 2) // 6
            zzz_correlations = np.empty(num_zzz)
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    sign_ij = self._signs[i] * self._signs[j]
                    for k in range(j + 1, n):
                        zzz_correlations[idx] = (sign_ij * self._signs[k]) @ probabilities
                        idx += 1
        else:
            zzz_correlations = np.empty(0)

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

            parts = [x_expectations, y_expectations, z_expectations, zz_correlations]
            if self.include_zzz:
                parts.append(zzz_correlations)
            features = np.concatenate(parts)
        else:
            parts = [z_expectations, zz_correlations]
            if self.include_cossin:
                parts.extend([np.cos(z_expectations), np.sin(z_expectations)])
            if self.include_zzz:
                parts.append(zzz_correlations)
            features = np.concatenate(parts)
        return features.astype(np.float32)
