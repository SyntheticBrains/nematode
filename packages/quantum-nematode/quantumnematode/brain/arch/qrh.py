"""
Quantum Reservoir Hybrid (QRH) Brain Architecture.

This architecture implements a structured quantum reservoir with C. elegans-inspired
topology and a PPO-trained classical actor-critic readout. It addresses the failure of
QRC's random reservoir (0% success) by using biologically-grounded topology, richer
Z/ZZ feature extraction, and PPO training.

Key Features:
- **Structured Quantum Reservoir**: C. elegans connectome topology (ASEL/ASER → AIY →
  AIA → AVA) mapped to 8 qubits with gap junction CZ gates and chemical synapse
  controlled rotations (CRY/CRZ)
- **Per-Qubit Input Encoding**: Only sensory qubits (ASEL/ASER) receive direct input;
  interneuron/command qubits receive signal exclusively through the topology, matching
  biological signal routing
- **X/Y/Z + ZZ Feature Extraction**: Per-qubit Pauli expectations ⟨X_i⟩, ⟨Y_i⟩, ⟨Z_i⟩
  + pairwise ZZ-correlations ⟨Z_i Z_j⟩ (52 features for 8 qubits) via exact statevector
  simulation. X/Y expectations capture phase information lost by Z-only measurement.
- **PPO Actor-Critic Readout**: Actor and critic MLPs trained with combined PPO loss
  (clipped surrogate + value loss + entropy bonus) via single optimizer
- **No Barren Plateaus**: Reservoir is fixed (no quantum parameters trained)

Architecture:
    Sensory Input → Per-Qubit Encoding → Structured Quantum Reservoir (FIXED)
    → X/Y/Z + ZZ Features → PPO Readout

The reservoir avoids barren plateaus entirely because no quantum parameters are
trained. Only the classical actor-critic readout (~5K params) is optimized.

References
----------
- Cook et al. (2019) "Whole-animal connectomes of both C. elegans sexes"
- Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

import numpy as np
import torch
from pydantic import Field, field_validator
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from torch import nn, optim

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, ClassicalBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch._quantum_reservoir import build_readout_network
from quantumnematode.brain.arch.dtypes import BrainConfig, DeviceType
from quantumnematode.brain.modules import (
    ModuleName,
    extract_classical_features,
    get_classical_feature_dimension,
)
from quantumnematode.env import Direction
from quantumnematode.logging_config import logger
from quantumnematode.utils.seeding import ensure_seed, get_rng, set_global_seed

# =============================================================================
# Default Hyperparameters
# =============================================================================

DEFAULT_NUM_RESERVOIR_QUBITS = 8
DEFAULT_RESERVOIR_DEPTH = 3
DEFAULT_RESERVOIR_SEED = 42
DEFAULT_SHOTS = 1024  # Retained for future QPU compatibility
DEFAULT_READOUT_HIDDEN_DIM = 64
DEFAULT_READOUT_NUM_LAYERS = 2
DEFAULT_ACTOR_LR = 0.0003
DEFAULT_CRITIC_LR = 0.0003
DEFAULT_GAMMA = 0.99
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_PPO_CLIP_EPSILON = 0.2
DEFAULT_PPO_EPOCHS = 4
DEFAULT_PPO_MINIBATCHES = 4
DEFAULT_PPO_BUFFER_SIZE = 512
DEFAULT_ENTROPY_COEFF = 0.01
DEFAULT_VALUE_LOSS_COEF = 0.5
DEFAULT_MAX_GRAD_NORM = 0.5

# Validation constants
MIN_RESERVOIR_QUBITS = 2
MIN_RESERVOIR_DEPTH = 1
MIN_READOUT_HIDDEN_DIM = 1
MIN_SHOTS = 100

# =============================================================================
# C. elegans Topology Constants
# =============================================================================
# 8-qubit mapping of the sensory-interneuron navigation subnetwork:
#   Qubits 0-1: ASEL/ASER (amphid sensory neurons — salt/food chemotaxis)
#   Qubits 2-3: AIYL/AIYR (first-layer interneurons — sensory integration)
#   Qubits 4-5: AIAL/AIAR (second-layer interneurons — navigation command)
#   Qubits 6-7: AVAL/AVAR (command interneurons — forward/reverse decision)
#
# Per-qubit input encoding:
#   Only sensory qubits (0-1) receive direct feature encoding, matching the
#   biological circuit where only ASE neurons sense the chemical gradient.
#   Interneuron and command qubits receive signal only through the topology.
#   ASEL (qubit 0) encodes gradient_strength via RY (on-response).
#   ASER (qubit 1) encodes relative_angle via RZ (off-response / phase).

# Sensory qubit indices — only these receive direct input encoding
SENSORY_QUBITS: list[int] = [0, 1]

# Gap junction CZ pairs (bidirectional electrical coupling):
# - Bilateral pairs: left-right connections within each neuron class
# - Feedforward: sensory → interneuron connections
GAP_JUNCTION_CZ_PAIRS: list[tuple[int, int]] = [
    # Bilateral pairs (symmetric left-right coupling)
    (0, 1),  # ASEL ↔ ASER
    (2, 3),  # AIYL ↔ AIYR
    (4, 5),  # AIAL ↔ AIAR
    (6, 7),  # AVAL ↔ AVAR
    # Feedforward excitation (sensory integration)
    (2, 4),  # AIYL → AIAL
    (3, 5),  # AIYR → AIAR
]

# Chemical synapse CRY/CRZ angles (directed, conditional signaling):
# Implemented as controlled rotations: the rotation on the target qubit is
# conditioned on the control qubit's state, modeling how postsynaptic response
# depends on presynaptic activity.
# Angles normalized from published synaptic weight ratios (Cook et al. 2019).
# Each tuple: (control_qubit, target_qubit, ry_angle, rz_angle)
CHEMICAL_SYNAPSE_ROTATIONS: list[tuple[int, int, float, float]] = [
    # ASE → AIY (sensory → first interneuron, strong excitatory)
    (0, 2, 0.8, 0.3),  # ASEL → AIYL
    (1, 3, 0.8, 0.3),  # ASER → AIYR
    # AIY → AIA (first → second interneuron, moderate excitatory)
    (2, 4, 0.5, 0.4),  # AIYL → AIAL
    (3, 5, 0.5, 0.4),  # AIYR → AIAR
    # AIA → AVA (second interneuron → command, strong excitatory)
    (4, 6, 0.7, 0.2),  # AIAL → AVAL
    (5, 7, 0.7, 0.2),  # AIAR → AVAR
    # Cross-lateral connections (weak inhibitory)
    (0, 3, 0.2, -0.3),  # ASEL → AIYR (cross-lateral)
    (1, 2, 0.2, -0.3),  # ASER → AIYL (cross-lateral)
]


def _compute_feature_dim(num_qubits: int) -> int:
    """Compute QRH feature dimension: 3N (X/Y/Z expectations) + N(N-1)/2 (ZZ correlations)."""
    return 3 * num_qubits + num_qubits * (num_qubits - 1) // 2


# =============================================================================
# Configuration
# =============================================================================


class QRHBrainConfig(BrainConfig):
    """Configuration for the QRHBrain architecture.

    Supports two modes for input feature extraction:

    1. **Legacy mode** (default): Uses 2 features (gradient_strength, relative_angle)
       - Set `sensory_modules=None` (default)

    2. **Unified sensory mode**: Uses modular feature extraction from brain/modules.py
       - Set `sensory_modules` to a list of ModuleName values
       - Uses extract_classical_features() which outputs semantic-preserving ranges
       - Each module contributes 2 features [strength, angle] in [0,1] and [-1,1]

    Attributes
    ----------
    num_reservoir_qubits : int
        Number of qubits in the quantum reservoir (default 8).
    reservoir_depth : int
        Number of entanglement layers in the reservoir circuit (default 3).
    reservoir_seed : int
        Seed for deterministic reservoir construction (default 42).
    shots : int
        Number of measurement shots, retained for future QPU compatibility (default 1024).
    readout_hidden_dim : int
        Hidden units per layer in actor/critic MLPs (default 64).
    readout_num_layers : int
        Number of hidden layers in actor/critic MLPs (default 2).
    actor_lr : float
        Learning rate for actor optimizer (default 3e-4).
    critic_lr : float
        Learning rate for critic optimizer (default 3e-4).
    gamma : float
        Discount factor for GAE computation (default 0.99).
    gae_lambda : float
        GAE lambda parameter (default 0.95).
    ppo_clip_epsilon : float
        PPO clipping parameter (default 0.2).
    ppo_epochs : int
        Number of PPO update epochs per buffer (default 4).
    ppo_minibatches : int
        Number of minibatches per PPO epoch (default 4).
    ppo_buffer_size : int
        Rollout buffer capacity in steps (default 512).
    entropy_coeff : float
        Entropy bonus coefficient for exploration (default 0.01).
    value_loss_coef : float
        Value loss coefficient (default 0.5).
    max_grad_norm : float
        Maximum gradient norm for clipping (default 0.5).
    use_random_topology : bool
        If True, use random reservoir topology for MI comparison (default False).
    sensory_modules : list[ModuleName] | None
        Sensory modules for unified feature extraction (default None = legacy mode).
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

    # Readout architecture
    readout_hidden_dim: int = Field(
        default=DEFAULT_READOUT_HIDDEN_DIM,
        description="Hidden units per layer in actor/critic MLPs.",
    )
    readout_num_layers: int = Field(
        default=DEFAULT_READOUT_NUM_LAYERS,
        description="Number of hidden layers in actor/critic MLPs.",
    )

    # Separate optimizer learning rates (see design Decision 5)
    actor_lr: float = Field(
        default=DEFAULT_ACTOR_LR,
        description="Learning rate for actor optimizer.",
    )
    critic_lr: float = Field(
        default=DEFAULT_CRITIC_LR,
        description="Learning rate for critic optimizer.",
    )

    # PPO training parameters
    gamma: float = Field(default=DEFAULT_GAMMA, description="Discount factor.")
    gae_lambda: float = Field(default=DEFAULT_GAE_LAMBDA, description="GAE lambda.")
    ppo_clip_epsilon: float = Field(
        default=DEFAULT_PPO_CLIP_EPSILON,
        description="PPO clipping parameter.",
    )
    ppo_epochs: int = Field(
        default=DEFAULT_PPO_EPOCHS,
        description="PPO update epochs per buffer.",
    )
    ppo_minibatches: int = Field(
        default=DEFAULT_PPO_MINIBATCHES,
        description="Minibatches per PPO epoch.",
    )
    ppo_buffer_size: int = Field(
        default=DEFAULT_PPO_BUFFER_SIZE,
        description="Rollout buffer capacity.",
    )
    entropy_coeff: float = Field(
        default=DEFAULT_ENTROPY_COEFF,
        description="Entropy bonus coefficient.",
    )
    value_loss_coef: float = Field(
        default=DEFAULT_VALUE_LOSS_COEF,
        description="Value loss coefficient.",
    )
    max_grad_norm: float = Field(
        default=DEFAULT_MAX_GRAD_NORM,
        description="Maximum gradient norm for clipping.",
    )

    # MI comparison toggle
    use_random_topology: bool = Field(
        default=False,
        description="Use random topology for MI comparison (default: structured).",
    )

    # Learning rate scheduling (optional)
    # Supports warmup followed by optional decay for more stable early learning.
    # - lr_warmup_episodes: Episodes to linearly increase LR from lr_warmup_start to actor_lr
    # - lr_warmup_start: Initial LR during warmup (default: 10% of actor_lr)
    # - lr_decay_episodes: Episodes after warmup to decay LR to lr_decay_end (None = no decay)
    # - lr_decay_end: Final LR after decay (default: 10% of actor_lr)
    lr_warmup_episodes: int = Field(
        default=0,
        description="Episodes to linearly increase LR (0 = no warmup).",
    )
    lr_warmup_start: float | None = Field(
        default=None,
        description="Initial LR during warmup (None = 10% of actor_lr).",
    )
    lr_decay_episodes: int | None = Field(
        default=None,
        description="Episodes after warmup to decay LR (None = no decay).",
    )
    lr_decay_end: float | None = Field(
        default=None,
        description="Final LR after decay (None = 10% of actor_lr).",
    )

    # Unified sensory feature extraction
    sensory_modules: list[ModuleName] | None = Field(
        default=None,
        description="Sensory modules for feature extraction (None = legacy mode).",
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
        if v < MIN_READOUT_HIDDEN_DIM:
            msg = f"readout_hidden_dim must be >= {MIN_READOUT_HIDDEN_DIM}, got {v}"
            raise ValueError(msg)
        return v


# =============================================================================
# Rollout Buffer (following mlpppo.py pattern)
# =============================================================================


class _RolloutBuffer:
    """Buffer for storing rollout experience for PPO updates."""

    def __init__(
        self,
        buffer_size: int,
        device: torch.device,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.buffer_size = buffer_size
        self.device = device
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self) -> None:
        """Clear all stored experience."""
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.position = 0

    def add(  # noqa: PLR0913
        self,
        state: np.ndarray,
        action: int,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,  # noqa: FBT001
    ) -> None:
        """Add a single experience to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.rewards.append(reward)
        self.dones.append(done)
        self.position += 1

    def is_full(self) -> bool:
        """Check if buffer has reached capacity."""
        return self.position >= self.buffer_size

    def __len__(self) -> int:
        return self.position

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        advantages = torch.zeros(len(self), device=self.device)
        last_gae = 0.0

        values = torch.stack(self.values).squeeze(-1)

        for t in reversed(range(len(self))):
            if t == len(self) - 1:
                next_value = last_value.item()
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = values[t + 1].item()
                next_non_terminal = 1.0 - float(self.dones[t])

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - values[t].item()
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        returns = advantages + values
        return returns, advantages

    def get_minibatches(
        self,
        num_minibatches: int,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Generate minibatches for training."""
        batch_size = len(self)
        minibatch_size = batch_size // num_minibatches

        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.stack(self.log_probs)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices_np = self.rng.permutation(batch_size)
        indices = torch.tensor(indices_np, device=self.device)

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_indices = indices[start:end]

            yield {
                "states": states[mb_indices],
                "actions": actions[mb_indices],
                "old_log_probs": old_log_probs[mb_indices],
                "returns": returns[mb_indices],
                "advantages": advantages[mb_indices],
            }


# =============================================================================
# QRH Brain
# =============================================================================


class QRHBrain(ClassicalBrain):
    """Quantum Reservoir Hybrid brain architecture.

    Uses a fixed structured quantum reservoir with C. elegans-inspired topology
    to generate Z/ZZ feature representations, with a PPO-trained classical
    actor-critic readout.
    """

    def __init__(  # noqa: PLR0915
        self,
        config: QRHBrainConfig,
        num_actions: int = 4,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] | None = None,
    ) -> None:
        """Initialize the QRHBrain.

        Parameters
        ----------
        config : QRHBrainConfig
            Configuration for the QRH brain architecture.
        num_actions : int
            Number of available actions (default 4).
        device : DeviceType
            Device for PyTorch operations (default CPU).
        action_set : list[Action] | None
            Custom action set (default is DEFAULT_ACTIONS).
        """
        super().__init__()

        self.config = config
        self.num_actions = num_actions
        self._device_type = device
        self.device = torch.device(device.value)
        self._action_set = action_set if action_set is not None else DEFAULT_ACTIONS

        # Initialize seeding
        self.seed = ensure_seed(config.seed)
        self.rng = get_rng(self.seed)
        set_global_seed(self.seed)
        logger.info(f"QRHBrain using seed: {self.seed}")

        # Store sensory modules for feature extraction
        self.sensory_modules = config.sensory_modules

        # Determine input dimension
        if config.sensory_modules is not None:
            self.input_dim = get_classical_feature_dimension(config.sensory_modules)
            logger.info(
                f"Using unified sensory modules: "
                f"{[m.value for m in config.sensory_modules]} "
                f"(input_dim={self.input_dim})",
            )
        else:
            self.input_dim = 2
            logger.info("Using legacy 2-feature preprocessing (gradient_strength, rel_angle)")

        # Initialize data tracking
        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()

        # Reservoir configuration
        self.num_qubits = config.num_reservoir_qubits
        self.reservoir_depth = config.reservoir_depth
        self.reservoir_seed = config.reservoir_seed
        self.use_random_topology = config.use_random_topology

        # Feature dimension: 3N + N(N-1)/2
        self.feature_dim = _compute_feature_dim(self.num_qubits)

        # Build actor and critic readout networks using shared utility
        self.actor = build_readout_network(
            input_dim=self.feature_dim,
            hidden_dim=config.readout_hidden_dim,
            output_dim=self.num_actions,
            readout_type="mlp",
            num_layers=config.readout_num_layers,
        ).to(self.device)

        self.critic = build_readout_network(
            input_dim=self.feature_dim,
            hidden_dim=config.readout_hidden_dim,
            output_dim=1,
            readout_type="mlp",
            num_layers=config.readout_num_layers,
        ).to(self.device)

        # Feature normalization (normalizes heterogeneous X/Y/Z/ZZ feature scales)
        self.feature_norm = nn.LayerNorm(self.feature_dim).to(self.device)

        # Single combined optimizer (matching MLPPPO pattern for stable training)
        self.optimizer = optim.Adam(
            list(self.actor.parameters())
            + list(self.critic.parameters())
            + list(self.feature_norm.parameters()),
            lr=config.actor_lr,
        )

        # Learning rate scheduling
        self.base_lr = config.actor_lr
        self.lr_warmup_episodes = config.lr_warmup_episodes
        self.lr_warmup_start = config.lr_warmup_start or (0.1 * config.actor_lr)
        self.lr_decay_episodes = config.lr_decay_episodes
        self.lr_decay_end = config.lr_decay_end or (0.1 * config.actor_lr)
        self.lr_scheduling_enabled = (
            self.lr_warmup_episodes > 0 or self.lr_decay_episodes is not None
        )
        if self.lr_scheduling_enabled:
            logger.info("QRH LR scheduling enabled:")
            if self.lr_warmup_episodes > 0:
                logger.info(
                    f"  warmup {self.lr_warmup_start:.6f} -> {self.base_lr:.6f} "
                    f"over {self.lr_warmup_episodes} episodes",
                )
            if self.lr_decay_episodes is not None:
                logger.info(
                    f"  decay {self.base_lr:.6f} -> {self.lr_decay_end:.6f} "
                    f"over {self.lr_decay_episodes} episodes",
                )

        # PPO parameters
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.clip_epsilon = config.ppo_clip_epsilon
        self.ppo_epochs = config.ppo_epochs
        self.ppo_minibatches = config.ppo_minibatches
        self.entropy_coeff = config.entropy_coeff
        self.value_loss_coef = config.value_loss_coef
        self.max_grad_norm = config.max_grad_norm

        # Rollout buffer
        self.buffer = _RolloutBuffer(config.ppo_buffer_size, self.device, rng=self.rng)

        # State tracking
        self.training = True
        self.current_probabilities: np.ndarray | None = None
        self.last_value: torch.Tensor | None = None
        self._pending_state: np.ndarray | None = None
        self._pending_action: int | None = None
        self._pending_log_prob: torch.Tensor | None = None
        self._pending_value: torch.Tensor | None = None

        # Episode tracking
        self._episode_count = 0

        # Pre-compute random topology if needed
        self._random_cz_pairs: list[tuple[int, int]] | None = None
        self._random_rotations: list[tuple[int, int, float, float]] | None = None
        if self.use_random_topology:
            self._random_cz_pairs, self._random_rotations = self._generate_random_topology()

        logger.info(
            f"QRHBrain initialized: {self.num_qubits} qubits, {self.reservoir_depth} layers, "
            f"{'random' if self.use_random_topology else 'structured'} topology, "
            f"feature_dim={self.feature_dim}, "
            f"actor/critic ({config.readout_hidden_dim}x{config.readout_num_layers})",
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
        # Gap junctions → CZ gates
        for q_a, q_b in GAP_JUNCTION_CZ_PAIRS:
            if q_a < self.num_qubits and q_b < self.num_qubits:
                qc.cz(q_a, q_b)

        # Chemical synapses → controlled RY/RZ rotations (CRY/CRZ)
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
        - ASEL gets feature 0 (gradient_strength) via RY — on-response
        - ASER gets feature 1 (relative_angle) via RZ — off-response / phase
        - Interneuron and command qubits receive signal only through topology
        - Additional features (if any) are distributed across sensory qubits

        Parameters
        ----------
        features : np.ndarray
            Input features to encode.

        Returns
        -------
        np.ndarray
            Statevector probabilities (|ψ_k|²), shape (2^num_qubits,).
        """
        qc = QuantumCircuit(self.num_qubits)

        # Initial Hadamard layer
        for q in range(self.num_qubits):
            qc.h(q)

        # Determine valid sensory qubits (those within qubit count)
        valid_sensory = [q for q in SENSORY_QUBITS if q < self.num_qubits]

        # Data re-uploading: encode inputs before each reservoir layer
        for _layer in range(self.reservoir_depth):
            # Per-qubit asymmetric encoding on sensory qubits only
            for i, feature in enumerate(features):
                if i < len(valid_sensory):
                    # Assign each feature to its corresponding sensory qubit
                    qubit = valid_sensory[i]
                    angle = float(feature) * np.pi
                    if i % 2 == 0:
                        qc.ry(angle, qubit)  # Even features: RY (amplitude)
                    else:
                        qc.rz(angle, qubit)  # Odd features: RZ (phase)
                else:
                    # Extra features wrap around sensory qubits
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

    def _extract_features(self, statevector: np.ndarray) -> np.ndarray:  # noqa: C901
        """Extract X/Y/Z expectations and ZZ correlations from the statevector.

        Computes from the full complex statevector amplitudes:
        - ⟨X_i⟩ = Σ_k 2·Re(ψ_k* · ψ_{k⊕2^i}) for each qubit i
        - ⟨Y_i⟩ = Σ_k 2·Im(ψ_{k⊕2^i}* · ψ_k) for each qubit i
        - ⟨Z_i⟩ = Σ_k (-1)^bit(k,i) |ψ_k|² for each qubit i
        - ⟨Z_i Z_j⟩ = Σ_k (-1)^(bit(k,i)+bit(k,j)) |ψ_k|² for each pair (i,j)

        Output dimension: 3N + N(N-1)/2 = 52 for 8 qubits.

        Parameters
        ----------
        statevector : np.ndarray
            Complex statevector amplitudes ψ_k, shape (2^num_qubits,).

        Returns
        -------
        np.ndarray
            Feature vector of shape (feature_dim,) with values in [-1, 1].
        """
        n = self.num_qubits
        num_states = len(statevector)
        probabilities = np.abs(statevector) ** 2

        # Pre-compute sign arrays for each qubit: sign[q][k] = (-1)^bit(k,q)
        signs = np.zeros((n, num_states))
        for q in range(n):
            for k in range(num_states):
                signs[q, k] = 1.0 - 2.0 * ((k >> q) & 1)

        # X-expectations: ⟨X_i⟩ = Σ_k 2·Re(ψ_k* · ψ_{k⊕2^i})
        # For each qubit i, X flips bit i: ⟨X_i⟩ = 2·Re(Σ_{k: bit_i=0} ψ_k* · ψ_{k+2^i})
        x_expectations = np.zeros(n)
        for q in range(n):
            mask = 1 << q
            exp_val = 0.0
            for k in range(num_states):
                if (k & mask) == 0:
                    exp_val += np.real(np.conj(statevector[k]) * statevector[k | mask])
            x_expectations[q] = 2.0 * exp_val

        # Y-expectations: ⟨Y_i⟩ = Σ_k 2·Im(ψ_{k⊕2^i}* · ψ_k) for bit_i=0
        # Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
        # ⟨Y_i⟩ = 2·Im(Σ_{k: bit_i=0} ψ_{k+2^i}* · ψ_k)
        y_expectations = np.zeros(n)
        for q in range(n):
            mask = 1 << q
            exp_val = 0.0
            for k in range(num_states):
                if (k & mask) == 0:
                    exp_val += np.imag(np.conj(statevector[k | mask]) * statevector[k])
            y_expectations[q] = 2.0 * exp_val

        # Z-expectations: ⟨Z_i⟩
        z_expectations = np.array([np.dot(signs[q], probabilities) for q in range(n)])

        # ZZ-correlations: ⟨Z_i Z_j⟩ for i < j
        zz_correlations = []
        for i in range(n):
            for j in range(i + 1, n):
                zz = np.dot(signs[i] * signs[j], probabilities)
                zz_correlations.append(zz)

        features = np.concatenate(
            [
                x_expectations,
                y_expectations,
                z_expectations,
                np.array(zz_correlations),
            ],
        )
        return features.astype(np.float32)

    # =========================================================================
    # Preprocessing
    # =========================================================================

    def preprocess(self, params: BrainParams) -> np.ndarray:
        """Preprocess BrainParams to extract features for the readout network.

        Two modes:
        1. **Unified sensory mode** (when sensory_modules is set):
           Uses extract_classical_features() which outputs semantic-preserving ranges.

        2. **Legacy mode** (default):
           Computes gradient strength [0, 1] and relative angle [-1, 1].

        Parameters
        ----------
        params : BrainParams
            Brain parameters containing sensory information.

        Returns
        -------
        np.ndarray
            Preprocessed features.
        """
        if self.sensory_modules is not None:
            return extract_classical_features(params, self.sensory_modules)

        # Legacy mode
        grad_strength = float(params.gradient_strength or 0.0)
        grad_direction = float(params.gradient_direction or 0.0)
        direction_map = {
            Direction.UP: np.pi / 2,
            Direction.DOWN: -np.pi / 2,
            Direction.LEFT: np.pi,
            Direction.RIGHT: 0.0,
        }
        agent_facing_angle = direction_map.get(params.agent_direction or Direction.UP, np.pi / 2)
        relative_angle = (grad_direction - agent_facing_angle + np.pi) % (2 * np.pi) - np.pi
        rel_angle_norm = relative_angle / np.pi

        return np.array([grad_strength, rel_angle_norm], dtype=np.float32)

    # =========================================================================
    # Action Selection
    # =========================================================================

    def _get_reservoir_features(self, sensory_features: np.ndarray) -> np.ndarray:
        """Run sensory features through reservoir and extract X/Y/Z + ZZ features."""
        statevector = self._encode_and_run(sensory_features)
        return self._extract_features(statevector)

    def run_brain(
        self,
        params: BrainParams,
        reward: float | None = None,  # noqa: ARG002
        input_data: list[float] | None = None,  # noqa: ARG002
        *,
        top_only: bool,  # noqa: ARG002
        top_randomize: bool,  # noqa: ARG002
    ) -> list[ActionData]:
        """Run the QRH brain and select an action.

        Pipeline: preprocess → reservoir → Z/ZZ features → actor → sample action.

        Parameters
        ----------
        params : BrainParams
            Brain parameters containing sensory information.
        reward : float | None
            Current reward (unused, kept for protocol compatibility).
        input_data : list[float] | None
            Optional input data (unused).
        top_only : bool
            Whether to return only top action (unused).
        top_randomize : bool
            Whether to randomize selection (unused).

        Returns
        -------
        list[ActionData]
            List containing the selected action.
        """
        # Preprocess sensory input
        sensory_features = self.preprocess(params)

        # Run through quantum reservoir and extract Z/ZZ features
        reservoir_features = self._get_reservoir_features(sensory_features)

        # Forward through actor network (with feature normalization)
        x = torch.tensor(reservoir_features, dtype=torch.float32, device=self.device)
        x = self.feature_norm(x)
        logits = self.actor(x)
        value = self.critic(x)

        # Compute action distribution
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_idx = int(dist.sample().item())
        log_prob = dist.log_prob(torch.tensor(action_idx, device=self.device))

        action_name = self.action_set[action_idx]
        probs_np = probs.detach().cpu().numpy()
        self.current_probabilities = probs_np

        # Diagnostic logging (sampled)
        if self.buffer.position % 50 == 0:
            feat_min, feat_max = float(reservoir_features.min()), float(reservoir_features.max())
            logits_np = logits.detach().cpu().numpy()
            logger.debug(
                f"QRH step {self.buffer.position}: "
                f"features=[{feat_min:.3f}, {feat_max:.3f}], "
                f"probs={probs_np}, logits=[{logits_np.min():.3f}, {logits_np.max():.3f}], "
                f"value={value.item():.4f}",
            )

        # Store for PPO buffer (will be committed when reward arrives in learn())
        self._pending_state = reservoir_features
        self._pending_action = action_idx
        self._pending_log_prob = log_prob
        self._pending_value = value
        self.last_value = value

        # Update tracking data
        self.latest_data.action = ActionData(
            state=action_name,
            action=action_name,
            probability=probs_np[action_idx],
        )
        self.history_data.actions.append(self.latest_data.action)
        self.history_data.probabilities.append(float(probs_np[action_idx]))

        return [self.latest_data.action]

    def learn(
        self,
        params: BrainParams,  # noqa: ARG002
        reward: float,
        *,
        episode_done: bool = False,
    ) -> None:
        """Add experience to buffer and perform PPO update when ready.

        Parameters
        ----------
        params : BrainParams
            Brain parameters (unused).
        reward : float
            Reward for the current step.
        episode_done : bool
            Whether the episode has ended.
        """
        # Add to buffer if we have pending state
        if (
            self._pending_state is not None
            and self._pending_action is not None
            and self._pending_log_prob is not None
            and self._pending_value is not None
        ):
            self.buffer.add(
                state=self._pending_state,
                action=self._pending_action,
                log_prob=self._pending_log_prob,
                value=self._pending_value,
                reward=reward,
                done=episode_done,
            )

        # Trigger PPO update when buffer is full or episode ends with enough data
        if self.buffer.is_full() or (episode_done and len(self.buffer) >= self.ppo_minibatches):
            logger.debug(
                f"QRH PPO update triggered: buffer={len(self.buffer)}/{self.buffer.buffer_size}, "
                f"episode_done={episode_done}",
            )
            self._perform_ppo_update()
            self.buffer.reset()

        # Store for history
        self.history_data.rewards.append(reward)

    def _get_current_lr(self) -> float:
        """Get the current learning rate based on episode count.

        Supports warmup followed by optional decay:
        - During warmup: linearly increases from lr_warmup_start to base_lr
        - After warmup (if decay enabled): linearly decreases from base_lr to lr_decay_end
        - Otherwise: returns base_lr
        """
        if not self.lr_scheduling_enabled:
            return self.base_lr

        episode = self._episode_count

        # Warmup phase
        if episode < self.lr_warmup_episodes:
            progress = episode / self.lr_warmup_episodes
            return self.lr_warmup_start + (self.base_lr - self.lr_warmup_start) * progress

        # Decay phase (if enabled)
        if self.lr_decay_episodes is not None:
            decay_start_episode = self.lr_warmup_episodes
            decay_episode = episode - decay_start_episode
            if decay_episode < self.lr_decay_episodes:
                progress = decay_episode / self.lr_decay_episodes
                return self.base_lr + (self.lr_decay_end - self.base_lr) * progress
            return self.lr_decay_end

        return self.base_lr

    def _update_learning_rate(self) -> None:
        """Update optimizer learning rate based on current schedule."""
        if not self.lr_scheduling_enabled:
            return

        new_lr = self._get_current_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        logger.debug(
            f"Episode {self._episode_count}: LR = {new_lr:.6f}",
        )

    def _perform_ppo_update(self) -> None:
        """Perform PPO update using collected experience."""
        if len(self.buffer) == 0:
            return

        # Get last value for GAE computation
        last_value = (
            self.last_value
            if self.last_value is not None
            else torch.tensor(
                [0.0],
                device=self.device,
            )
        )

        # Compute returns and advantages
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value,
            self.gamma,
            self.gae_lambda,
        )

        # Skip PPO update if buffer is too small — tiny batches produce unreliable
        # gradients with high value_loss variance, causing policy instability.
        # Use 64 as floor, but never more than half the configured buffer size
        # (so small-buffer configs in tests still work).
        min_buffer_size = min(64, self.buffer.buffer_size // 2)
        if len(self.buffer) < min_buffer_size:
            logger.debug(
                f"QRH skipping PPO update: buffer size {len(self.buffer)} < {min_buffer_size}",
            )
            self.buffer.reset()
            return

        # PPO update loop
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        num_updates = 0

        all_params = (
            list(self.actor.parameters())
            + list(self.critic.parameters())
            + list(self.feature_norm.parameters())
        )

        for _ in range(self.ppo_epochs):
            for batch in self.buffer.get_minibatches(self.ppo_minibatches, returns, advantages):
                # Normalize features (same transform as forward pass)
                normalized_states = self.feature_norm(batch["states"])

                # Get new action probabilities and values
                logits = self.actor(normalized_states)
                values = self.critic(normalized_states).squeeze(-1)

                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                new_log_probs = dist.log_prob(batch["actions"])
                entropy = dist.entropy().mean()

                # Compute ratio
                ratio = torch.exp(new_log_probs - batch["old_log_probs"])

                # Clipped surrogate objective
                surr1 = ratio * batch["advantages"]
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch["advantages"]
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, batch["returns"])

                # Combined loss (matching MLPPPO pattern)
                loss = (
                    policy_loss + self.value_loss_coef * value_loss - self.entropy_coeff * entropy
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
            self.latest_data.loss = avg_loss
            self.history_data.losses.append(avg_loss)

            logger.debug(
                f"QRH PPO update: policy_loss={total_policy_loss / num_updates:.4f}, "
                f"value_loss={total_value_loss / num_updates:.4f}, "
                f"entropy={total_entropy_loss / num_updates:.4f}",
            )

    # =========================================================================
    # Brain Protocol Methods
    # =========================================================================

    def update_memory(self, reward: float | None = None) -> None:
        """No-op for QRHBrain."""

    def prepare_episode(self) -> None:
        """Prepare for a new episode by clearing pending state."""
        self._pending_state = None
        self._pending_action = None
        self._pending_log_prob = None
        self._pending_value = None
        self.last_value = None

    def post_process_episode(
        self,
        *,
        episode_success: bool | None = None,  # noqa: ARG002
    ) -> None:
        """Post-process after each episode."""
        self._episode_count += 1

        # Update learning rate based on schedule (if enabled)
        self._update_learning_rate()

        # Clear pending state to prevent cross-episode contamination
        self._pending_state = None
        self._pending_action = None
        self._pending_log_prob = None
        self._pending_value = None

    def copy(self) -> QRHBrain:
        """Create an independent copy of the QRHBrain.

        The copy produces identical reservoir circuits (same seed, same topology)
        but has independent readout network weights.

        Returns
        -------
        QRHBrain
            Independent copy of this brain.
        """
        config_copy = QRHBrainConfig(
            **{**self.config.model_dump(), "seed": self.seed},
        )

        new_brain = QRHBrain(
            config=config_copy,
            num_actions=self.num_actions,
            device=self._device_type,
            action_set=self._action_set,
        )

        # Copy network weights (independent)
        new_brain.actor.load_state_dict(deepcopy(self.actor.state_dict()))
        new_brain.critic.load_state_dict(deepcopy(self.critic.state_dict()))
        new_brain.feature_norm.load_state_dict(deepcopy(self.feature_norm.state_dict()))

        # Copy optimizer state
        new_brain.optimizer.load_state_dict(deepcopy(self.optimizer.state_dict()))

        # Preserve episode counter
        new_brain._episode_count = self._episode_count

        return new_brain

    @property
    def action_set(self) -> list[Action]:
        """Get the list of available actions."""
        return self._action_set

    @action_set.setter
    def action_set(self, actions: list[Action]) -> None:
        """Set the list of available actions."""
        if len(actions) != self.num_actions:
            msg = (
                f"Cannot set action_set of length {len(actions)}: "
                f"readout network expects {self.num_actions} actions"
            )
            raise ValueError(msg)
        self._action_set = actions
