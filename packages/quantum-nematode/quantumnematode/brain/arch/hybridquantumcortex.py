"""
Hierarchical Hybrid Quantum Cortex Brain Architecture.

Combines a QSNN reflex layer (QLIF neurons, REINFORCE training) with a QSNN
cortex layer (grouped sensory QLIF neurons with shared hidden layer, REINFORCE
with critic-provided GAE advantages) and a classical critic for multi-objective
decision-making.

Architecture::

    Sensory Input (2-dim legacy)         Multi-sensory Input (8+ dim)
           |                                       |
           v                                       v
    QSNN Reflex (unchanged)              QSNN Cortex (NEW)
      S->H->M QLIF                         Grouped sensory QLIF
      ~212 quantum params                    -> Shared hidden QLIF
      Surrogate REINFORCE                    -> Output QLIF
      Output: 4 reflex logits               ~350-500 quantum params
           |                                 Surrogate REINFORCE + GAE
           |                                       |
           v                                       v
      Fusion: reflex_logits * trust + action_biases
           |
           v
      Action Selection (4 actions)

    Classical Critic: sensory_dim -> 64 -> 64 -> 1, Huber loss

Four-stage curriculum:
  1. QSNN reflex on foraging (REINFORCE)
  2. QSNN cortex + critic (reflex frozen, REINFORCE+GAE)
  3. Joint fine-tune (both QSNNs + critic)
  4. Multi-sensory scaling (same as 3, more modules)

References
----------
- Brand & Petruccione (2024). "A quantum leaky integrate-and-fire spiking neuron
  and network." npj Quantum Information, 10(1), 16.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from pydantic import Field, field_validator, model_validator

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, ClassicalBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch._hybrid_common import (
    _CortexRolloutBuffer,
    _ReinforceUpdateStats,
    adaptive_entropy_coef,
    exploration_schedule,
    fuse,
    init_critic_mlp,
    normalize_reward,
    preprocess_legacy,
    update_cortex_learning_rates,
)
from quantumnematode.brain.arch._qlif_layers import (
    LOGIT_SCALE,
    WEIGHT_INIT_SCALE,
    encode_sensory_spikes,
    execute_qlif_layer,
    execute_qlif_layer_differentiable,
    execute_qlif_layer_differentiable_cached,
    get_qiskit_backend,
)
from quantumnematode.brain.arch.dtypes import BrainConfig, DeviceType
from quantumnematode.brain.modules import (
    ModuleName,
    extract_classical_features,
    get_classical_feature_dimension,
)
from quantumnematode.logging_config import logger
from quantumnematode.utils.seeding import ensure_seed, get_rng, set_global_seed

# ──────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────

# QSNN reflex defaults (identical to HybridQuantumBrain)
DEFAULT_NUM_SENSORY_NEURONS = 8
DEFAULT_NUM_HIDDEN_NEURONS = 16
DEFAULT_NUM_MOTOR_NEURONS = 4
DEFAULT_MEMBRANE_TAU = 0.9
DEFAULT_THRESHOLD = 0.5
DEFAULT_REFRACTORY_PERIOD = 0
DEFAULT_SHOTS = 100
DEFAULT_NUM_QSNN_TIMESTEPS = 4
DEFAULT_SURROGATE_ALPHA = 1.0
DEFAULT_WEIGHT_CLIP = 3.0
DEFAULT_THETA_MOTOR_MAX_NORM = 2.0

# QSNN REINFORCE defaults
DEFAULT_QSNN_LR = 0.01
DEFAULT_QSNN_LR_DECAY_EPISODES = 400
DEFAULT_NUM_REINFORCE_EPOCHS = 2
DEFAULT_REINFORCE_WINDOW_SIZE = 20
DEFAULT_GAMMA = 0.99
DEFAULT_ADVANTAGE_CLIP = 2.0
DEFAULT_CLIP_EPSILON = 0.2
DEFAULT_EXPLORATION_EPSILON = 0.1
DEFAULT_EXPLORATION_DECAY_EPISODES = 80
DEFAULT_LR_MIN_FACTOR = 0.1

# Cortex QSNN defaults
DEFAULT_CORTEX_NEURONS_PER_GROUP = 4
DEFAULT_CORTEX_HIDDEN_NEURONS = 12
DEFAULT_CORTEX_OUTPUT_NEURONS = 8
DEFAULT_NUM_CORTEX_TIMESTEPS = 4
DEFAULT_CORTEX_SHOTS = 100
DEFAULT_NUM_MODES = 3
DEFAULT_CORTEX_LR = 0.01
DEFAULT_CRITIC_LR = 0.001
DEFAULT_NUM_CORTEX_REINFORCE_EPOCHS = 2
DEFAULT_PPO_BUFFER_SIZE = 512
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_ENTROPY_COEFF = 0.02
DEFAULT_MAX_GRAD_NORM = 0.5
DEFAULT_CORTEX_HIDDEN_DIM = 64
DEFAULT_CORTEX_NUM_LAYERS = 2

# Joint fine-tune
DEFAULT_JOINT_FINETUNE_LR_FACTOR = 0.1

# Entropy regulation
DEFAULT_ENTROPY_FLOOR = 0.5
DEFAULT_ENTROPY_BOOST_MAX = 20.0
DEFAULT_ENTROPY_CEILING_FRACTION = 0.95

# Reward normalization
DEFAULT_REWARD_NORM_ALPHA = 0.01

# Surrogate gradient clipping
SURROGATE_GRAD_CLIP = 1.0

# Validation
MIN_SENSORY_NEURONS = 1
MIN_HIDDEN_NEURONS = 1
MIN_MOTOR_NEURONS = 2
MIN_SHOTS = 100
MIN_REINFORCE_BATCH_SIZE = 2
MIN_CORTEX_NEURONS_PER_GROUP = 2
MIN_CORTEX_HIDDEN_NEURONS = 4
MIN_NUM_MODES = 2

# Normalization epsilon
NORM_EPS = 1e-8

# Training stages
STAGE_REFLEX_ONLY = 1
STAGE_CORTEX_ONLY = 2
STAGE_JOINT = 3
STAGE_MULTI_SENSORY = 4

# Mode logit scaling
MODE_LOGIT_SCALE = 2.0


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────


class HybridQuantumCortexBrainConfig(BrainConfig):
    """Configuration for the HybridQuantumCortexBrain architecture.

    Supports a QSNN reflex layer (legacy 2-feature) combined with a QSNN cortex
    layer (grouped sensory QLIF neurons) and a classical critic.
    """

    # QSNN reflex params
    num_sensory_neurons: int = Field(
        default=DEFAULT_NUM_SENSORY_NEURONS,
        description="Number of sensory layer neurons.",
    )
    num_hidden_neurons: int = Field(
        default=DEFAULT_NUM_HIDDEN_NEURONS,
        description="Number of hidden interneuron layer neurons.",
    )
    num_motor_neurons: int = Field(
        default=DEFAULT_NUM_MOTOR_NEURONS,
        description="Number of motor layer neurons matching action space.",
    )
    membrane_tau: float = Field(
        default=DEFAULT_MEMBRANE_TAU,
        description="Leak time constant for QLIF neurons in (0, 1].",
    )
    threshold: float = Field(
        default=DEFAULT_THRESHOLD,
        description="Firing threshold in (0, 1).",
    )
    refractory_period: int = Field(
        default=DEFAULT_REFRACTORY_PERIOD,
        description="Timesteps to suppress activity after firing.",
    )
    shots: int = Field(
        default=DEFAULT_SHOTS,
        description="Number of quantum measurement shots for reflex.",
    )
    num_qsnn_timesteps: int = Field(
        default=DEFAULT_NUM_QSNN_TIMESTEPS,
        description="Number of QLIF timesteps per reflex decision.",
    )
    surrogate_alpha: float = Field(
        default=DEFAULT_SURROGATE_ALPHA,
        description="Surrogate gradient sharpness parameter.",
    )
    logit_scale: float = Field(
        default=LOGIT_SCALE,
        description="Scaling factor for converting spike probs to action logits.",
    )
    weight_clip: float = Field(
        default=DEFAULT_WEIGHT_CLIP,
        description="Maximum absolute weight value for stability.",
    )
    theta_motor_max_norm: float = Field(
        default=DEFAULT_THETA_MOTOR_MAX_NORM,
        description="Max L2 norm for theta_motor vector.",
    )

    # Cortex QSNN params
    cortex_sensory_modules: list[ModuleName] | None = Field(
        default=None,
        description="List of sensory modules for cortex grouped QLIF processing.",
    )
    cortex_neurons_per_group: int = Field(
        default=DEFAULT_CORTEX_NEURONS_PER_GROUP,
        description="Number of QLIF neurons per sensory modality group.",
    )
    cortex_hidden_neurons: int = Field(
        default=DEFAULT_CORTEX_HIDDEN_NEURONS,
        description="Number of QLIF neurons in shared hidden layer.",
    )
    cortex_output_neurons: int = Field(
        default=DEFAULT_CORTEX_OUTPUT_NEURONS,
        description="Number of QLIF neurons in output layer.",
    )
    num_cortex_timesteps: int = Field(
        default=DEFAULT_NUM_CORTEX_TIMESTEPS,
        description="Number of QLIF timesteps per cortex decision.",
    )
    cortex_shots: int = Field(
        default=DEFAULT_CORTEX_SHOTS,
        description="Number of quantum measurement shots for cortex.",
    )
    num_modes: int = Field(
        default=DEFAULT_NUM_MODES,
        description="Number of modes for gating (forage, evade, explore).",
    )

    # Training stage
    training_stage: int = Field(
        default=1,
        description="Training stage: 1=reflex only, 2=cortex+critic, 3=joint, 4=multi-sensory.",
    )

    # QSNN REINFORCE params
    qsnn_lr: float = Field(
        default=DEFAULT_QSNN_LR,
        description="Learning rate for QSNN reflex REINFORCE.",
    )
    qsnn_lr_decay_episodes: int = Field(
        default=DEFAULT_QSNN_LR_DECAY_EPISODES,
        description="Episodes over which QSNN LR decays via cosine annealing.",
    )
    num_reinforce_epochs: int = Field(
        default=DEFAULT_NUM_REINFORCE_EPOCHS,
        description="Number of gradient passes per REINFORCE update window.",
    )
    reinforce_window_size: int = Field(
        default=DEFAULT_REINFORCE_WINDOW_SIZE,
        description="Steps per intra-episode REINFORCE update.",
    )
    gamma: float = Field(
        default=DEFAULT_GAMMA,
        description="Discount factor for returns.",
    )
    advantage_clip: float = Field(
        default=DEFAULT_ADVANTAGE_CLIP,
        description="Clamp normalized advantages to [-clip, +clip].",
    )
    clip_epsilon: float = Field(
        default=DEFAULT_CLIP_EPSILON,
        description="PPO-style clipping epsilon for REINFORCE policy ratio.",
    )
    exploration_epsilon: float = Field(
        default=DEFAULT_EXPLORATION_EPSILON,
        description="Initial exploration epsilon.",
    )
    exploration_decay_episodes: int = Field(
        default=DEFAULT_EXPLORATION_DECAY_EPISODES,
        description="Episodes over which exploration decays.",
    )
    lr_min_factor: float = Field(
        default=DEFAULT_LR_MIN_FACTOR,
        description="Minimum LR as fraction of initial LR.",
    )
    reward_norm_alpha: float = Field(
        default=DEFAULT_REWARD_NORM_ALPHA,
        description="EMA smoothing factor for running reward normalization.",
    )
    use_reward_normalization: bool = Field(
        default=True,
        description="Enable running reward normalization for QSNN REINFORCE.",
    )
    entropy_floor: float = Field(
        default=DEFAULT_ENTROPY_FLOOR,
        description="Entropy threshold below which adaptive entropy boost activates.",
    )
    entropy_boost_max: float = Field(
        default=DEFAULT_ENTROPY_BOOST_MAX,
        description="Maximum multiplier for entropy_coef when entropy is low.",
    )
    entropy_ceiling_fraction: float = Field(
        default=DEFAULT_ENTROPY_CEILING_FRACTION,
        description="Fraction of max entropy above which entropy bonus is suppressed.",
    )

    # Cortex training params
    cortex_lr: float = Field(
        default=DEFAULT_CORTEX_LR,
        description="Learning rate for cortex QSNN REINFORCE.",
    )
    critic_lr: float = Field(
        default=DEFAULT_CRITIC_LR,
        description="Learning rate for classical critic.",
    )
    num_cortex_reinforce_epochs: int = Field(
        default=DEFAULT_NUM_CORTEX_REINFORCE_EPOCHS,
        description="Number of REINFORCE epochs per cortex update.",
    )
    ppo_buffer_size: int = Field(
        default=DEFAULT_PPO_BUFFER_SIZE,
        description="Rollout buffer size for cortex REINFORCE+GAE.",
    )
    gae_lambda: float = Field(
        default=DEFAULT_GAE_LAMBDA,
        description="Lambda for GAE advantage estimation.",
    )
    entropy_coeff: float = Field(
        default=DEFAULT_ENTROPY_COEFF,
        description="Entropy regularization coefficient.",
    )
    max_grad_norm: float = Field(
        default=DEFAULT_MAX_GRAD_NORM,
        description="Maximum gradient norm for clipping.",
    )
    use_gae_advantages: bool = Field(
        default=True,
        description="Use critic-provided GAE advantages (True) or pure REINFORCE returns (False).",
    )

    # Joint fine-tune
    joint_finetune_lr_factor: float = Field(
        default=DEFAULT_JOINT_FINETUNE_LR_FACTOR,
        description="LR multiplier for QSNN reflex in stage 3/4.",
    )

    # Cortex LR scheduling
    cortex_lr_warmup_episodes: int = Field(
        default=0,
        description="Episodes to linearly increase cortex LR from warmup_start to base LR.",
    )
    cortex_lr_warmup_start: float | None = Field(
        default=None,
        description="Initial cortex LR during warmup. None = 0.1 * cortex_lr.",
    )
    cortex_lr_decay_episodes: int | None = Field(
        default=None,
        description="Episodes after warmup to decay cortex LR. None = no decay.",
    )
    cortex_lr_decay_end: float | None = Field(
        default=None,
        description="Final cortex LR after decay. None = 0.1 * cortex_lr.",
    )

    # Critic MLP params
    critic_hidden_dim: int = Field(
        default=DEFAULT_CORTEX_HIDDEN_DIM,
        description="Hidden layer dimension for critic MLP.",
    )
    critic_num_layers: int = Field(
        default=DEFAULT_CORTEX_NUM_LAYERS,
        description="Number of hidden layers in critic MLP.",
    )

    # Weight persistence
    reflex_weights_path: str | None = Field(
        default=None,
        description="Path to pre-trained reflex weights (.pt file) for stage 2/3/4.",
    )
    cortex_weights_path: str | None = Field(
        default=None,
        description="Path to pre-trained cortex weights (.pt file) for stage 3/4.",
    )
    critic_weights_path: str | None = Field(
        default=None,
        description="Path to pre-trained critic weights (.pt file) for stage 3/4.",
    )

    # Sensory modules for reflex (legacy mode uses None)
    sensory_modules: list[ModuleName] | None = Field(
        default=None,
        description="List of sensory modules for reflex feature extraction (None = legacy mode).",
    )

    # ── Validators ──

    @field_validator("num_sensory_neurons")
    @classmethod
    def validate_num_sensory_neurons(cls, v: int) -> int:
        """Validate num_sensory_neurons >= 1."""
        if v < MIN_SENSORY_NEURONS:
            msg = f"num_sensory_neurons must be >= {MIN_SENSORY_NEURONS}, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("num_hidden_neurons")
    @classmethod
    def validate_num_hidden_neurons(cls, v: int) -> int:
        """Validate num_hidden_neurons >= 1."""
        if v < MIN_HIDDEN_NEURONS:
            msg = f"num_hidden_neurons must be >= {MIN_HIDDEN_NEURONS}, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("num_motor_neurons")
    @classmethod
    def validate_num_motor_neurons(cls, v: int) -> int:
        """Validate num_motor_neurons >= 2."""
        if v < MIN_MOTOR_NEURONS:
            msg = f"num_motor_neurons must be >= {MIN_MOTOR_NEURONS}, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("membrane_tau")
    @classmethod
    def validate_membrane_tau(cls, v: float) -> float:
        """Validate membrane_tau in (0, 1]."""
        if not (0 < v <= 1):
            msg = f"membrane_tau must be in (0, 1], got {v}"
            raise ValueError(msg)
        return v

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate threshold in (0, 1)."""
        if not (0 < v < 1):
            msg = f"threshold must be in (0, 1), got {v}"
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

    @field_validator("training_stage")
    @classmethod
    def validate_training_stage(cls, v: int) -> int:
        """Validate training_stage is 1, 2, 3, or 4."""
        if v not in (STAGE_REFLEX_ONLY, STAGE_CORTEX_ONLY, STAGE_JOINT, STAGE_MULTI_SENSORY):
            msg = f"training_stage must be 1, 2, 3, or 4, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("num_reinforce_epochs")
    @classmethod
    def validate_num_reinforce_epochs(cls, v: int) -> int:
        """Validate num_reinforce_epochs >= 1."""
        if v < 1:
            msg = f"num_reinforce_epochs must be >= 1, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("cortex_neurons_per_group")
    @classmethod
    def validate_cortex_neurons_per_group(cls, v: int) -> int:
        """Validate cortex_neurons_per_group >= 2."""
        if v < MIN_CORTEX_NEURONS_PER_GROUP:
            msg = f"cortex_neurons_per_group must be >= {MIN_CORTEX_NEURONS_PER_GROUP}, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("cortex_hidden_neurons")
    @classmethod
    def validate_cortex_hidden_neurons(cls, v: int) -> int:
        """Validate cortex_hidden_neurons >= 4."""
        if v < MIN_CORTEX_HIDDEN_NEURONS:
            msg = f"cortex_hidden_neurons must be >= {MIN_CORTEX_HIDDEN_NEURONS}, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("num_modes")
    @classmethod
    def validate_num_modes(cls, v: int) -> int:
        """Validate num_modes >= 2."""
        if v < MIN_NUM_MODES:
            msg = f"num_modes must be >= {MIN_NUM_MODES}, got {v}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_cortex_modules_for_stage(self) -> HybridQuantumCortexBrainConfig:
        """Validate cortex_sensory_modules non-empty when training_stage >= 2."""
        if self.training_stage >= STAGE_CORTEX_ONLY and not self.cortex_sensory_modules:
            msg = (
                "cortex_sensory_modules must be non-empty when training_stage >= 2, "
                f"got {self.cortex_sensory_modules}"
            )
            raise ValueError(msg)
        return self


# ──────────────────────────────────────────────────────────────────────
# Brain Implementation
# ──────────────────────────────────────────────────────────────────────


class HybridQuantumCortexBrain(ClassicalBrain):
    """Hierarchical hybrid quantum brain with QSNN reflex + QSNN cortex + classical critic.

    The QSNN reflex layer provides proven reactive foraging behaviour via QLIF
    quantum circuits. The QSNN cortex layer uses grouped sensory QLIF neurons
    for multi-modal processing, trained via REINFORCE with critic-provided GAE
    advantages. Mode-gated fusion combines reflex and cortex outputs.
    """

    def __init__(  # noqa: PLR0915
        self,
        config: HybridQuantumCortexBrainConfig,
        num_actions: int = 4,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] | None = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.num_actions = num_actions
        self.device = torch.device(device.value)
        self._action_set = action_set if action_set is not None else DEFAULT_ACTIONS[:num_actions]

        if self.num_actions != len(self._action_set):
            msg = (
                f"num_actions ({self.num_actions}) does not match "
                f"action_set length ({len(self._action_set)})"
            )
            raise ValueError(msg)

        # Seeding
        self.seed = ensure_seed(config.seed)
        self.rng = get_rng(self.seed)
        set_global_seed(self.seed)
        logger.info(f"HybridQuantumCortexBrain using seed: {self.seed}")

        # Reflex sensory modules (always legacy 2-feature for reflex)
        self.sensory_modules = config.sensory_modules
        if config.sensory_modules is not None:
            self.reflex_input_dim = get_classical_feature_dimension(config.sensory_modules)
        else:
            self.reflex_input_dim = 2

        # Cortex sensory modules
        self.cortex_sensory_modules = config.cortex_sensory_modules
        if self.cortex_sensory_modules is not None:
            self.cortex_input_dim = get_classical_feature_dimension(self.cortex_sensory_modules)
            self.num_cortex_groups = len(self.cortex_sensory_modules)
        else:
            self.cortex_input_dim = 2
            self.num_cortex_groups = 1

        # Data tracking
        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()

        # Network configuration
        self.num_sensory = config.num_sensory_neurons
        self.num_hidden = config.num_hidden_neurons
        self.num_motor = config.num_motor_neurons
        self.membrane_tau = config.membrane_tau
        self.threshold = config.threshold
        self.refractory_period = config.refractory_period
        self.shots = config.shots
        self.num_qsnn_timesteps = config.num_qsnn_timesteps
        self.gamma = config.gamma
        self.weight_clip = config.weight_clip
        self.training_stage = config.training_stage
        self.num_modes = config.num_modes

        # Cortex QSNN configuration
        self.cortex_neurons_per_group = config.cortex_neurons_per_group
        self.cortex_hidden_neurons = config.cortex_hidden_neurons
        self.cortex_output_neurons = config.cortex_output_neurons
        self.num_cortex_timesteps = config.num_cortex_timesteps
        self.cortex_shots = config.cortex_shots

        # Leak angle
        self.leak_angle = (1 - self.membrane_tau) * np.pi

        # Initialize QSNN reflex layer
        self._init_reflex_weights()

        # Initialize QSNN cortex layer
        self._init_cortex_qsnn()

        # Initialize classical critic
        self._init_critic()

        # Initialize optimizers
        self._init_optimizers()

        # Load pre-trained weights
        if config.reflex_weights_path is not None:
            self._load_reflex_weights()
        elif self.training_stage >= STAGE_CORTEX_ONLY:
            logger.warning(
                "Stage >= 2 but no reflex_weights_path specified. "
                "Reflex will use random initialization.",
            )

        if config.cortex_weights_path is not None:
            self._load_cortex_weights()

        if config.critic_weights_path is not None:
            self._load_critic_weights()

        # Freeze reflex in stage 2
        if self.training_stage == STAGE_CORTEX_ONLY:
            self._freeze_reflex()

        # Rollout buffer for cortex REINFORCE+GAE
        self.rollout_buffer = _CortexRolloutBuffer(
            config.ppo_buffer_size,
            self.device,
            rng=self.rng,
        )

        # REINFORCE episode buffers (for reflex)
        self._init_episode_state()

        # Pending cortex data
        self._has_pending_cortex_data = False
        self._latest_returns: torch.Tensor | None = None

        # Counters
        self._step_count = 0
        self._episode_count = 0

        # Session ID for weight saving
        self._session_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        # Qiskit backends (lazy)
        self._reflex_backend = None
        self._cortex_backend = None

        # Running reward normalization (REINFORCE)
        self.reward_running_mean: float = 0.0
        self.reward_running_var: float = 1.0

        # Fusion tracking
        self._episode_qsnn_trusts: list[float] = []
        self._episode_mode_probs: list[list[float]] = []

        # Current action probabilities
        self.current_probabilities: np.ndarray | None = None
        self.training = True

        # Log parameter counts
        total_reflex_params = (
            self.num_sensory * self.num_hidden
            + self.num_hidden * self.num_motor
            + self.num_hidden
            + self.num_motor
        )
        total_cortex_params = self._count_cortex_params()
        critic_params = sum(p.numel() for p in self.critic.parameters())

        logger.info(
            f"HybridQuantumCortexBrain initialized: "
            f"reflex {self.num_sensory}->{self.num_hidden}->{self.num_motor}, "
            f"cortex groups={self.num_cortex_groups}x{self.cortex_neurons_per_group}, "
            f"hidden={self.cortex_hidden_neurons}, output={self.cortex_output_neurons}, "
            f"modes={self.num_modes}, stage={self.training_stage}, "
            f"params: reflex={total_reflex_params}, "
            f"cortex={total_cortex_params}, "
            f"critic={critic_params}",
        )

    # ──────────────────────────────────────────────────────────────────
    # QSNN reflex initialization (identical to HybridQuantumBrain)
    # ──────────────────────────────────────────────────────────────────

    def _init_reflex_weights(self) -> None:
        """Initialize QSNN reflex trainable weight matrices and neuron parameters."""
        self.W_sh = (
            torch.randn(self.num_sensory, self.num_hidden, device=self.device) * WEIGHT_INIT_SCALE
        )
        self.W_hm = (
            torch.randn(self.num_hidden, self.num_motor, device=self.device) * WEIGHT_INIT_SCALE
        )
        self.theta_hidden = torch.full(
            (self.num_hidden,),
            np.pi / 4,
            device=self.device,
        )
        self.theta_motor = torch.zeros(self.num_motor, device=self.device)

        # Enable gradients for surrogate gradient REINFORCE
        self.W_sh.requires_grad_(True)  # noqa: FBT003
        self.W_hm.requires_grad_(True)  # noqa: FBT003
        self.theta_hidden.requires_grad_(True)  # noqa: FBT003
        self.theta_motor.requires_grad_(True)  # noqa: FBT003

        # Refractory state
        self.refractory_hidden = np.zeros(self.num_hidden, dtype=np.int32)
        self.refractory_motor = np.zeros(self.num_motor, dtype=np.int32)

    # ──────────────────────────────────────────────────────────────────
    # QSNN cortex initialization (grouped sensory QLIF)
    # ──────────────────────────────────────────────────────────────────

    def _init_cortex_qsnn(self) -> None:
        """Initialize QSNN cortex with grouped sensory QLIF neurons."""
        # Weight matrix shape: (num_pre_spikes, num_post_neurons)
        # encode_sensory_spikes produces cortex_neurons_per_group spikes,
        # so pre-synaptic dim = cortex_neurons_per_group.
        self.cortex_group_weights: list[torch.Tensor] = []
        self.cortex_group_thetas: list[torch.Tensor] = []

        for _ in range(self.num_cortex_groups):
            w = (
                torch.randn(
                    self.cortex_neurons_per_group,
                    self.cortex_neurons_per_group,
                    device=self.device,
                )
                * WEIGHT_INIT_SCALE
            )
            w.requires_grad_(True)  # noqa: FBT003
            self.cortex_group_weights.append(w)

            theta = torch.full(
                (self.cortex_neurons_per_group,),
                np.pi / 4,
                device=self.device,
            )
            theta.requires_grad_(True)  # noqa: FBT003
            self.cortex_group_thetas.append(theta)

        # Shared hidden layer weights
        total_sensory_neurons = self.num_cortex_groups * self.cortex_neurons_per_group
        self.W_cortex_sh = (
            torch.randn(total_sensory_neurons, self.cortex_hidden_neurons, device=self.device)
            * WEIGHT_INIT_SCALE
        )
        self.W_cortex_sh.requires_grad_(True)  # noqa: FBT003
        self.theta_cortex_hidden = torch.full(
            (self.cortex_hidden_neurons,),
            np.pi / 4,
            device=self.device,
        )
        self.theta_cortex_hidden.requires_grad_(True)  # noqa: FBT003

        # Output layer weights
        self.W_cortex_ho = (
            torch.randn(self.cortex_hidden_neurons, self.cortex_output_neurons, device=self.device)
            * WEIGHT_INIT_SCALE
        )
        self.W_cortex_ho.requires_grad_(True)  # noqa: FBT003
        self.theta_cortex_output = torch.full(
            (self.cortex_output_neurons,),
            np.pi / 4,
            device=self.device,
        )
        self.theta_cortex_output.requires_grad_(True)  # noqa: FBT003

        # Cortex refractory states
        self.cortex_refractory_groups: list[np.ndarray] = [
            np.zeros(self.cortex_neurons_per_group, dtype=np.int32)
            for _ in range(self.num_cortex_groups)
        ]
        self.cortex_refractory_hidden = np.zeros(self.cortex_hidden_neurons, dtype=np.int32)
        self.cortex_refractory_output = np.zeros(self.cortex_output_neurons, dtype=np.int32)

    def _count_cortex_params(self) -> int:
        """Count total cortex QSNN parameters."""
        total = 0
        for w in self.cortex_group_weights:
            total += w.numel()
        for t in self.cortex_group_thetas:
            total += t.numel()
        total += self.W_cortex_sh.numel()
        total += self.theta_cortex_hidden.numel()
        total += self.W_cortex_ho.numel()
        total += self.theta_cortex_output.numel()
        return total

    def _get_cortex_params(self) -> list[torch.Tensor]:
        """Get all cortex QSNN parameters for optimizer."""
        params: list[torch.Tensor] = []
        params.extend(self.cortex_group_weights)
        params.extend(self.cortex_group_thetas)
        params.extend(
            [
                self.W_cortex_sh,
                self.theta_cortex_hidden,
                self.W_cortex_ho,
                self.theta_cortex_output,
            ],
        )
        return params

    # ──────────────────────────────────────────────────────────────────
    # Classical critic initialization
    # ──────────────────────────────────────────────────────────────────

    def _init_critic(self) -> None:
        """Initialize classical critic MLP."""
        self.critic = init_critic_mlp(
            input_dim=self.cortex_input_dim,
            hidden_dim=self.config.critic_hidden_dim,
            num_layers=self.config.critic_num_layers,
            device=self.device,
        )

    # ──────────────────────────────────────────────────────────────────
    # Optimizer initialization
    # ──────────────────────────────────────────────────────────────────

    def _init_optimizers(self) -> None:
        """Initialize separate optimizers for reflex, cortex, and critic."""
        qsnn_lr = self.config.qsnn_lr
        if self.training_stage in (STAGE_JOINT, STAGE_MULTI_SENSORY):
            qsnn_lr *= self.config.joint_finetune_lr_factor

        self.reflex_optimizer = torch.optim.Adam(
            [self.W_sh, self.W_hm, self.theta_hidden, self.theta_motor],
            lr=qsnn_lr,
        )
        self.reflex_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.reflex_optimizer,
            T_max=self.config.qsnn_lr_decay_episodes,
            eta_min=qsnn_lr * self.config.lr_min_factor,
        )

        # Cortex optimizer
        self.cortex_optimizer = torch.optim.Adam(
            self._get_cortex_params(),
            lr=self.config.cortex_lr,
        )

        # Cortex LR scheduling state
        self.cortex_base_lr = self.config.cortex_lr
        self.cortex_lr_warmup_episodes = self.config.cortex_lr_warmup_episodes
        self.cortex_lr_warmup_start = self.config.cortex_lr_warmup_start or (
            0.1 * self.config.cortex_lr
        )
        self.cortex_lr_decay_episodes = self.config.cortex_lr_decay_episodes
        self.cortex_lr_decay_end = self.config.cortex_lr_decay_end or (0.1 * self.config.cortex_lr)
        self.cortex_lr_scheduling_enabled = (
            self.cortex_lr_warmup_episodes > 0 or self.cortex_lr_decay_episodes is not None
        )

        # Critic optimizer
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config.critic_lr,
        )

    def _freeze_reflex(self) -> None:
        """Freeze reflex weights (stage 2)."""
        self.W_sh.requires_grad_(False)  # noqa: FBT003
        self.W_hm.requires_grad_(False)  # noqa: FBT003
        self.theta_hidden.requires_grad_(False)  # noqa: FBT003
        self.theta_motor.requires_grad_(False)  # noqa: FBT003
        logger.info("Reflex weights frozen for stage 2 training")

    def _init_episode_state(self) -> None:
        """Initialize REINFORCE episode tracking state."""
        self.episode_rewards: list[float] = []
        self.episode_actions: list[int] = []
        self.episode_features: list[np.ndarray] = []
        self.episode_old_log_probs: list[float] = []
        self._cached_spike_probs: list[list[dict[str, list[float]]]] = []
        self.baseline = 0.0
        self.baseline_alpha = 0.05

    # ──────────────────────────────────────────────────────────────────
    # Qiskit backends
    # ──────────────────────────────────────────────────────────────────

    def _get_reflex_backend(self):  # noqa: ANN202
        if self._reflex_backend is None:
            self._reflex_backend = get_qiskit_backend(
                DeviceType(self.device.type) if hasattr(self.device, "type") else DeviceType.CPU,
                seed=self.seed,
            )
        return self._reflex_backend

    def _get_cortex_backend(self):  # noqa: ANN202
        if self._cortex_backend is None:
            self._cortex_backend = get_qiskit_backend(
                DeviceType(self.device.type) if hasattr(self.device, "type") else DeviceType.CPU,
                seed=self.seed + 1 if self.seed is not None else None,
            )
        return self._cortex_backend

    # ──────────────────────────────────────────────────────────────────
    # QSNN reflex forward pass (identical to HybridQuantumBrain)
    # ──────────────────────────────────────────────────────────────────

    def _encode_sensory_spikes(self, features: np.ndarray) -> np.ndarray:
        return encode_sensory_spikes(features, self.num_sensory)

    def _reflex_timestep(self, features: np.ndarray) -> np.ndarray:
        """Execute one QSNN reflex timestep (non-differentiable)."""
        sensory_spikes = self._encode_sensory_spikes(features)

        hidden_spikes, self.refractory_hidden = execute_qlif_layer(
            sensory_spikes,
            self.W_sh,
            self.theta_hidden,
            self.refractory_hidden,
            backend=self._get_reflex_backend(),
            shots=self.shots,
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            leak_angle=self.leak_angle,
        )
        motor_spikes, self.refractory_motor = execute_qlif_layer(
            hidden_spikes,
            self.W_hm,
            self.theta_motor,
            self.refractory_motor,
            backend=self._get_reflex_backend(),
            shots=self.shots,
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            leak_angle=self.leak_angle,
        )
        return motor_spikes

    def _reflex_multi_timestep(self, features: np.ndarray) -> np.ndarray:
        """Execute multiple QSNN reflex timesteps and average motor spike probabilities."""
        motor_accumulator = np.zeros(self.num_motor)
        for _ in range(self.num_qsnn_timesteps):
            motor_spikes = self._reflex_timestep(features)
            motor_accumulator += motor_spikes
        return motor_accumulator / self.num_qsnn_timesteps

    def _reflex_timestep_differentiable(self, features: np.ndarray) -> torch.Tensor:
        """Execute one QSNN reflex timestep with gradient tracking."""
        sensory_spikes = self._encode_sensory_spikes(features)
        sensory_tensor = torch.tensor(sensory_spikes, dtype=torch.float32, device=self.device)

        hidden_spikes, self.refractory_hidden = execute_qlif_layer_differentiable(
            sensory_tensor,
            self.W_sh,
            self.theta_hidden,
            self.refractory_hidden,
            backend=self._get_reflex_backend(),
            shots=self.shots,
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            leak_angle=self.leak_angle,
            device=self.device,
        )
        motor_spikes, self.refractory_motor = execute_qlif_layer_differentiable(
            hidden_spikes,
            self.W_hm,
            self.theta_motor,
            self.refractory_motor,
            backend=self._get_reflex_backend(),
            shots=self.shots,
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            leak_angle=self.leak_angle,
            device=self.device,
        )
        return motor_spikes

    def _reflex_multi_timestep_differentiable(self, features: np.ndarray) -> torch.Tensor:
        """Execute multiple QSNN reflex timesteps with gradient tracking."""
        motor_accumulator = torch.zeros(self.num_motor, device=self.device)
        for _ in range(self.num_qsnn_timesteps):
            motor_spikes = self._reflex_timestep_differentiable(features)
            motor_accumulator = motor_accumulator + motor_spikes
        return motor_accumulator / self.num_qsnn_timesteps

    def _reflex_timestep_differentiable_caching(
        self,
        features: np.ndarray,
        cache_out: dict[str, list[float]],
    ) -> torch.Tensor:
        """Execute one QSNN reflex timestep with gradient tracking, caching spike probs."""
        sensory_spikes = self._encode_sensory_spikes(features)
        sensory_tensor = torch.tensor(sensory_spikes, dtype=torch.float32, device=self.device)

        hidden_spikes, self.refractory_hidden = execute_qlif_layer_differentiable(
            sensory_tensor,
            self.W_sh,
            self.theta_hidden,
            self.refractory_hidden,
            backend=self._get_reflex_backend(),
            shots=self.shots,
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            leak_angle=self.leak_angle,
            device=self.device,
        )
        cache_out["hidden"] = hidden_spikes.detach().cpu().tolist()

        motor_spikes, self.refractory_motor = execute_qlif_layer_differentiable(
            hidden_spikes,
            self.W_hm,
            self.theta_motor,
            self.refractory_motor,
            backend=self._get_reflex_backend(),
            shots=self.shots,
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            leak_angle=self.leak_angle,
            device=self.device,
        )
        cache_out["motor"] = motor_spikes.detach().cpu().tolist()

        return motor_spikes

    def _reflex_multi_timestep_differentiable_caching(
        self,
        features: np.ndarray,
        cache_out: list[dict[str, list[float]]],
    ) -> torch.Tensor:
        motor_accumulator = torch.zeros(self.num_motor, device=self.device)
        for _ in range(self.num_qsnn_timesteps):
            step_cache: dict[str, list[float]] = {}
            motor_spikes = self._reflex_timestep_differentiable_caching(features, step_cache)
            motor_accumulator = motor_accumulator + motor_spikes
            cache_out.append(step_cache)
        return motor_accumulator / self.num_qsnn_timesteps

    def _reflex_timestep_differentiable_cached(
        self,
        features: np.ndarray,
        cached_step: dict[str, list[float]],
    ) -> torch.Tensor:
        sensory_spikes = self._encode_sensory_spikes(features)
        sensory_tensor = torch.tensor(sensory_spikes, dtype=torch.float32, device=self.device)

        hidden_spikes, self.refractory_hidden = execute_qlif_layer_differentiable_cached(
            sensory_tensor,
            self.W_sh,
            self.theta_hidden,
            self.refractory_hidden,
            cached_spike_probs=cached_step["hidden"],
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            device=self.device,
        )
        motor_spikes, self.refractory_motor = execute_qlif_layer_differentiable_cached(
            hidden_spikes,
            self.W_hm,
            self.theta_motor,
            self.refractory_motor,
            cached_spike_probs=cached_step["motor"],
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            device=self.device,
        )
        return motor_spikes

    def _reflex_multi_timestep_differentiable_cached(
        self,
        features: np.ndarray,
        cached_timestep: list[dict[str, list[float]]],
    ) -> torch.Tensor:
        motor_accumulator = torch.zeros(self.num_motor, device=self.device)
        for step_idx in range(self.num_qsnn_timesteps):
            motor_spikes = self._reflex_timestep_differentiable_cached(
                features,
                cached_timestep[step_idx],
            )
            motor_accumulator = motor_accumulator + motor_spikes
        return motor_accumulator / self.num_qsnn_timesteps

    # ──────────────────────────────────────────────────────────────────
    # QSNN cortex forward pass (grouped sensory QLIF)
    # ──────────────────────────────────────────────────────────────────

    def _cortex_sensory_forward(
        self,
        cortex_features: np.ndarray,
    ) -> np.ndarray:
        """Execute per-group sensory QLIF layers (non-differentiable)."""
        features_per_module = (
            2 if self.cortex_sensory_modules is not None else self.cortex_input_dim
        )
        group_outputs: list[np.ndarray] = []

        for i in range(self.num_cortex_groups):
            group_feats = cortex_features[i * features_per_module : (i + 1) * features_per_module]
            sensory_spikes = encode_sensory_spikes(group_feats, self.cortex_neurons_per_group)

            group_spikes, self.cortex_refractory_groups[i] = execute_qlif_layer(
                sensory_spikes,
                self.cortex_group_weights[i],
                self.cortex_group_thetas[i],
                self.cortex_refractory_groups[i],
                backend=self._get_cortex_backend(),
                shots=self.cortex_shots,
                threshold=self.threshold,
                refractory_period=self.refractory_period,
                leak_angle=self.leak_angle,
            )
            group_outputs.append(group_spikes)

        return np.concatenate(group_outputs)

    def _cortex_sensory_forward_differentiable(
        self,
        cortex_features: np.ndarray,
    ) -> torch.Tensor:
        """Execute per-group sensory QLIF layers with gradient tracking."""
        features_per_module = (
            2 if self.cortex_sensory_modules is not None else self.cortex_input_dim
        )
        group_outputs: list[torch.Tensor] = []

        for i in range(self.num_cortex_groups):
            group_feats = cortex_features[i * features_per_module : (i + 1) * features_per_module]
            sensory_spikes = encode_sensory_spikes(group_feats, self.cortex_neurons_per_group)
            sensory_tensor = torch.tensor(
                sensory_spikes,
                dtype=torch.float32,
                device=self.device,
            )

            group_spikes, self.cortex_refractory_groups[i] = execute_qlif_layer_differentiable(
                sensory_tensor,
                self.cortex_group_weights[i],
                self.cortex_group_thetas[i],
                self.cortex_refractory_groups[i],
                backend=self._get_cortex_backend(),
                shots=self.cortex_shots,
                threshold=self.threshold,
                refractory_period=self.refractory_period,
                leak_angle=self.leak_angle,
                device=self.device,
            )
            group_outputs.append(group_spikes)

        return torch.cat(group_outputs)

    def _cortex_sensory_forward_differentiable_caching(
        self,
        cortex_features: np.ndarray,
        cache_out: dict[str, list[float]],
    ) -> torch.Tensor:
        """Execute per-group sensory QLIF layers with gradient tracking and caching."""
        features_per_module = (
            2 if self.cortex_sensory_modules is not None else self.cortex_input_dim
        )
        group_outputs: list[torch.Tensor] = []

        for i in range(self.num_cortex_groups):
            group_feats = cortex_features[i * features_per_module : (i + 1) * features_per_module]
            sensory_spikes = encode_sensory_spikes(group_feats, self.cortex_neurons_per_group)
            sensory_tensor = torch.tensor(
                sensory_spikes,
                dtype=torch.float32,
                device=self.device,
            )

            group_spikes, self.cortex_refractory_groups[i] = execute_qlif_layer_differentiable(
                sensory_tensor,
                self.cortex_group_weights[i],
                self.cortex_group_thetas[i],
                self.cortex_refractory_groups[i],
                backend=self._get_cortex_backend(),
                shots=self.cortex_shots,
                threshold=self.threshold,
                refractory_period=self.refractory_period,
                leak_angle=self.leak_angle,
                device=self.device,
            )
            cache_out[f"group_{i}"] = group_spikes.detach().cpu().tolist()
            group_outputs.append(group_spikes)

        return torch.cat(group_outputs)

    def _cortex_sensory_forward_differentiable_cached(
        self,
        cortex_features: np.ndarray,
        cached: dict[str, list[float]],
    ) -> torch.Tensor:
        """Execute per-group sensory QLIF layers reusing cached spike probs."""
        features_per_module = (
            2 if self.cortex_sensory_modules is not None else self.cortex_input_dim
        )
        group_outputs: list[torch.Tensor] = []

        for i in range(self.num_cortex_groups):
            group_feats = cortex_features[i * features_per_module : (i + 1) * features_per_module]
            sensory_spikes = encode_sensory_spikes(group_feats, self.cortex_neurons_per_group)
            sensory_tensor = torch.tensor(
                sensory_spikes,
                dtype=torch.float32,
                device=self.device,
            )

            group_spikes, self.cortex_refractory_groups[i] = (
                execute_qlif_layer_differentiable_cached(
                    sensory_tensor,
                    self.cortex_group_weights[i],
                    self.cortex_group_thetas[i],
                    self.cortex_refractory_groups[i],
                    cached_spike_probs=cached[f"group_{i}"],
                    threshold=self.threshold,
                    refractory_period=self.refractory_period,
                    device=self.device,
                )
            )
            group_outputs.append(group_spikes)

        return torch.cat(group_outputs)

    def _cortex_hidden_forward(
        self,
        sensory_concat: np.ndarray,
    ) -> np.ndarray:
        """Execute shared hidden QLIF layer (non-differentiable)."""
        hidden_spikes, self.cortex_refractory_hidden = execute_qlif_layer(
            sensory_concat,
            self.W_cortex_sh,
            self.theta_cortex_hidden,
            self.cortex_refractory_hidden,
            backend=self._get_cortex_backend(),
            shots=self.cortex_shots,
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            leak_angle=self.leak_angle,
        )
        return hidden_spikes

    def _cortex_hidden_forward_differentiable(
        self,
        sensory_concat: torch.Tensor,
    ) -> torch.Tensor:
        """Execute shared hidden QLIF layer with gradient tracking."""
        hidden_spikes, self.cortex_refractory_hidden = execute_qlif_layer_differentiable(
            sensory_concat,
            self.W_cortex_sh,
            self.theta_cortex_hidden,
            self.cortex_refractory_hidden,
            backend=self._get_cortex_backend(),
            shots=self.cortex_shots,
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            leak_angle=self.leak_angle,
            device=self.device,
        )
        return hidden_spikes

    def _cortex_output_forward(
        self,
        hidden_spikes: np.ndarray,
    ) -> np.ndarray:
        """Execute output QLIF layer (non-differentiable)."""
        output_spikes, self.cortex_refractory_output = execute_qlif_layer(
            hidden_spikes,
            self.W_cortex_ho,
            self.theta_cortex_output,
            self.cortex_refractory_output,
            backend=self._get_cortex_backend(),
            shots=self.cortex_shots,
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            leak_angle=self.leak_angle,
        )
        return output_spikes

    def _cortex_output_forward_differentiable(
        self,
        hidden_spikes: torch.Tensor,
    ) -> torch.Tensor:
        """Execute output QLIF layer with gradient tracking."""
        output_spikes, self.cortex_refractory_output = execute_qlif_layer_differentiable(
            hidden_spikes,
            self.W_cortex_ho,
            self.theta_cortex_output,
            self.cortex_refractory_output,
            backend=self._get_cortex_backend(),
            shots=self.cortex_shots,
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            leak_angle=self.leak_angle,
            device=self.device,
        )
        return output_spikes

    def _cortex_timestep(self, cortex_features: np.ndarray) -> np.ndarray:
        """Execute one cortex timestep (non-differentiable): sensory -> hidden -> output."""
        sensory_concat = self._cortex_sensory_forward(cortex_features)
        hidden_spikes = self._cortex_hidden_forward(sensory_concat)
        return self._cortex_output_forward(hidden_spikes)

    def _cortex_multi_timestep(self, cortex_features: np.ndarray) -> np.ndarray:
        """Execute multiple cortex timesteps and average output spike probabilities."""
        output_accumulator = np.zeros(self.cortex_output_neurons)
        for _ in range(self.num_cortex_timesteps):
            output_spikes = self._cortex_timestep(cortex_features)
            output_accumulator += output_spikes
        return output_accumulator / self.num_cortex_timesteps

    def _cortex_timestep_differentiable(
        self,
        cortex_features: np.ndarray,
    ) -> torch.Tensor:
        """Execute one cortex timestep with gradient tracking."""
        sensory_concat = self._cortex_sensory_forward_differentiable(cortex_features)
        hidden_spikes = self._cortex_hidden_forward_differentiable(sensory_concat)
        return self._cortex_output_forward_differentiable(hidden_spikes)

    def _cortex_multi_timestep_differentiable(
        self,
        cortex_features: np.ndarray,
    ) -> torch.Tensor:
        """Execute multiple cortex timesteps with gradient tracking."""
        output_accumulator = torch.zeros(self.cortex_output_neurons, device=self.device)
        for _ in range(self.num_cortex_timesteps):
            output_spikes = self._cortex_timestep_differentiable(cortex_features)
            output_accumulator = output_accumulator + output_spikes
        return output_accumulator / self.num_cortex_timesteps

    def _cortex_timestep_differentiable_caching(
        self,
        cortex_features: np.ndarray,
        cache_out: dict[str, list[float]],
    ) -> torch.Tensor:
        """Execute one cortex timestep with gradient tracking and caching."""
        sensory_concat = self._cortex_sensory_forward_differentiable_caching(
            cortex_features,
            cache_out,
        )

        hidden_spikes, self.cortex_refractory_hidden = execute_qlif_layer_differentiable(
            sensory_concat,
            self.W_cortex_sh,
            self.theta_cortex_hidden,
            self.cortex_refractory_hidden,
            backend=self._get_cortex_backend(),
            shots=self.cortex_shots,
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            leak_angle=self.leak_angle,
            device=self.device,
        )
        cache_out["hidden"] = hidden_spikes.detach().cpu().tolist()

        output_spikes, self.cortex_refractory_output = execute_qlif_layer_differentiable(
            hidden_spikes,
            self.W_cortex_ho,
            self.theta_cortex_output,
            self.cortex_refractory_output,
            backend=self._get_cortex_backend(),
            shots=self.cortex_shots,
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            leak_angle=self.leak_angle,
            device=self.device,
        )
        cache_out["output"] = output_spikes.detach().cpu().tolist()

        return output_spikes

    def _cortex_multi_timestep_differentiable_caching(
        self,
        cortex_features: np.ndarray,
        cache_out: list[dict[str, list[float]]],
    ) -> torch.Tensor:
        output_accumulator = torch.zeros(self.cortex_output_neurons, device=self.device)
        for _ in range(self.num_cortex_timesteps):
            step_cache: dict[str, list[float]] = {}
            output_spikes = self._cortex_timestep_differentiable_caching(
                cortex_features,
                step_cache,
            )
            output_accumulator = output_accumulator + output_spikes
            cache_out.append(step_cache)
        return output_accumulator / self.num_cortex_timesteps

    def _cortex_timestep_differentiable_cached(
        self,
        cortex_features: np.ndarray,
        cached_step: dict[str, list[float]],
    ) -> torch.Tensor:
        """Execute one cortex timestep reusing cached spike probs."""
        sensory_concat = self._cortex_sensory_forward_differentiable_cached(
            cortex_features,
            cached_step,
        )

        hidden_spikes, self.cortex_refractory_hidden = execute_qlif_layer_differentiable_cached(
            sensory_concat,
            self.W_cortex_sh,
            self.theta_cortex_hidden,
            self.cortex_refractory_hidden,
            cached_spike_probs=cached_step["hidden"],
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            device=self.device,
        )
        output_spikes, self.cortex_refractory_output = execute_qlif_layer_differentiable_cached(
            hidden_spikes,
            self.W_cortex_ho,
            self.theta_cortex_output,
            self.cortex_refractory_output,
            cached_spike_probs=cached_step["output"],
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            device=self.device,
        )
        return output_spikes

    def _cortex_multi_timestep_differentiable_cached(
        self,
        cortex_features: np.ndarray,
        cached_timesteps: list[dict[str, list[float]]],
    ) -> torch.Tensor:
        output_accumulator = torch.zeros(self.cortex_output_neurons, device=self.device)
        for step_idx in range(self.num_cortex_timesteps):
            output_spikes = self._cortex_timestep_differentiable_cached(
                cortex_features,
                cached_timesteps[step_idx],
            )
            output_accumulator = output_accumulator + output_spikes
        return output_accumulator / self.num_cortex_timesteps

    # ──────────────────────────────────────────────────────────────────
    # Cortex output mapping
    # ──────────────────────────────────────────────────────────────────

    def _map_cortex_output(
        self,
        output_probs: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
        """Map cortex output spike probs to action biases, mode logits, trust."""
        # Action biases: neurons 0..num_motor-1
        action_biases = (output_probs[: self.num_motor] - 0.5) * self.config.logit_scale

        # Mode logits: neurons num_motor..num_motor+num_modes-1
        mode_logits = (
            output_probs[self.num_motor : self.num_motor + self.num_modes] - 0.5
        ) * MODE_LOGIT_SCALE

        # Trust modulation: last neuron (raw spike probability)
        trust = output_probs[self.num_motor + self.num_modes]

        return action_biases, mode_logits, trust

    # ──────────────────────────────────────────────────────────────────
    # Fusion mechanism
    # ──────────────────────────────────────────────────────────────────

    def _fuse(
        self,
        reflex_logits: torch.Tensor,
        action_biases: torch.Tensor,
        mode_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, float, torch.Tensor]:
        """Apply mode-gated fusion (shared implementation)."""
        return fuse(reflex_logits, action_biases, mode_logits)

    # ──────────────────────────────────────────────────────────────────
    # Preprocessing
    # ──────────────────────────────────────────────────────────────────

    def _preprocess_legacy(self, params: BrainParams) -> np.ndarray:
        """Extract legacy 2-feature input (gradient_strength, relative_angle)."""
        return preprocess_legacy(params)

    def preprocess(self, params: BrainParams) -> np.ndarray:
        """Preprocess BrainParams to extract reflex features (always legacy 2-feature)."""
        return self._preprocess_legacy(params)

    def _preprocess_cortex(self, params: BrainParams) -> np.ndarray:
        """Preprocess BrainParams for cortex input via sensory modules."""
        if self.cortex_sensory_modules is not None:
            return extract_classical_features(params, self.cortex_sensory_modules)
        return self._preprocess_legacy(params)

    # ──────────────────────────────────────────────────────────────────
    # Exploration schedule
    # ──────────────────────────────────────────────────────────────────

    def _exploration_schedule(self) -> tuple[float, float]:
        return exploration_schedule(
            self._episode_count,
            self.config.exploration_epsilon,
            self.config.exploration_decay_episodes,
        )

    # ──────────────────────────────────────────────────────────────────
    # run_brain
    # ──────────────────────────────────────────────────────────────────

    def run_brain(
        self,
        params: BrainParams,
        reward: float | None = None,  # noqa: ARG002
        input_data: list[float] | None = None,  # noqa: ARG002
        *,
        top_only: bool,  # noqa: ARG002
        top_randomize: bool,  # noqa: ARG002
    ) -> list[ActionData]:
        """Run hybrid brain and select an action."""
        reflex_features = self.preprocess(params)

        # Reflex forward pass (always runs)
        motor_probs = self._reflex_multi_timestep(reflex_features)
        motor_probs = np.clip(motor_probs, 1e-8, 1.0 - 1e-8)
        reflex_logits_np = (motor_probs - 0.5) * self.config.logit_scale

        if self.training_stage == STAGE_REFLEX_ONLY:
            # Stage 1: reflex only, no cortex involvement
            logits = reflex_logits_np
            epsilon, temperature = self._exploration_schedule()
            scaled_logits = logits / temperature
            exp_probs = np.exp(scaled_logits - np.max(scaled_logits))
            softmax_probs = exp_probs / np.sum(exp_probs)
            uniform = np.ones(self.num_actions) / self.num_actions
            action_probs = (1 - epsilon) * softmax_probs + epsilon * uniform
        else:
            # Stage 2/3/4: cortex forward pass
            cortex_features = self._preprocess_cortex(params)
            cortex_output = self._cortex_multi_timestep(cortex_features)
            cortex_output = np.clip(cortex_output, 1e-8, 1.0 - 1e-8)

            action_biases_np, mode_logits_np, _trust_raw = self._map_cortex_output(cortex_output)

            reflex_logits_t = torch.tensor(
                reflex_logits_np,
                dtype=torch.float32,
                device=self.device,
            )
            action_biases_t = torch.tensor(
                action_biases_np,
                dtype=torch.float32,
                device=self.device,
            )
            mode_logits_t = torch.tensor(
                mode_logits_np,
                dtype=torch.float32,
                device=self.device,
            )

            final_logits, qsnn_trust, mode_probs = self._fuse(
                reflex_logits_t,
                action_biases_t,
                mode_logits_t,
            )

            # Track fusion diagnostics
            self._episode_qsnn_trusts.append(qsnn_trust)
            self._episode_mode_probs.append(mode_probs.detach().cpu().tolist())

            action_probs_t = torch.softmax(final_logits, dim=-1)
            action_probs = action_probs_t.detach().cpu().numpy()

            # Store rollout buffer data for cortex REINFORCE+GAE
            cortex_features_t = torch.tensor(
                cortex_features,
                dtype=torch.float32,
                device=self.device,
            )
            with torch.no_grad():
                value = self.critic(cortex_features_t).squeeze(-1)

            self._pending_cortex_state = cortex_features
            self._pending_cortex_value = value
            self._has_pending_cortex_data = True

        # Sample action
        action_probs = np.clip(action_probs, 1e-8, 1.0)
        action_probs = action_probs / action_probs.sum()
        action_idx = self.rng.choice(self.num_actions, p=action_probs)
        action_name = self.action_set[action_idx]

        # Store REINFORCE data (stage 1, 3, 4)
        if self.training_stage != STAGE_CORTEX_ONLY:
            self.episode_features.append(reflex_features)
            old_log_prob = float(np.log(action_probs[action_idx] + 1e-8))
            self.episode_old_log_probs.append(old_log_prob)

        # Store chosen action for rollout buffer
        if self.training_stage >= STAGE_CORTEX_ONLY:
            self._pending_cortex_action = action_idx
            # Log prob from fused distribution for REINFORCE
            self._pending_cortex_log_prob = torch.tensor(
                np.log(action_probs[action_idx] + NORM_EPS),
            )

        self.episode_actions.append(action_idx)
        self.current_probabilities = action_probs

        # Update tracking
        self.latest_data.action = ActionData(
            state=action_name,
            action=action_name,
            probability=action_probs[action_idx],
        )
        self.latest_data.probability = float(action_probs[action_idx])
        self.history_data.actions.append(self.latest_data.action)
        self.history_data.probabilities.append(float(action_probs[action_idx]))

        return [self.latest_data.action]

    # ──────────────────────────────────────────────────────────────────
    # learn
    # ──────────────────────────────────────────────────────────────────

    def learn(  # noqa: C901, PLR0912
        self,
        params: BrainParams,  # noqa: ARG002
        reward: float,
        *,
        episode_done: bool = False,
    ) -> None:
        """Accumulate rewards and trigger stage-dependent training."""
        # Reward normalization for REINFORCE (reflex stages)
        uses_reinforce = self.training_stage != STAGE_CORTEX_ONLY
        if uses_reinforce and self.config.use_reward_normalization:
            normalized_reward = self._normalize_reward(reward)
            self.episode_rewards.append(normalized_reward)
        else:
            self.episode_rewards.append(reward)

        self.history_data.rewards.append(reward)
        self._step_count += 1

        # Add to rollout buffer (stage 2, 3, 4)
        if self.training_stage >= STAGE_CORTEX_ONLY and self._has_pending_cortex_data:
            self.rollout_buffer.add(
                state=self._pending_cortex_state,
                action=self._pending_cortex_action,
                log_prob=self._pending_cortex_log_prob,
                value=self._pending_cortex_value,
                reward=reward,
                done=episode_done,
            )

        # Intra-episode reflex REINFORCE update (stage 1, 3, 4)
        window = self.config.reinforce_window_size
        if (
            self.training_stage != STAGE_CORTEX_ONLY
            and window > 0
            and self._step_count % window == 0
            and not episode_done
        ):
            self._reflex_reinforce_update()
            self.episode_rewards.clear()
            self.episode_actions.clear()
            self.episode_features.clear()
            self.episode_old_log_probs.clear()
            self._cached_spike_probs = []

        # Rollout buffer update (stage 2, 3, 4)
        if self.training_stage >= STAGE_CORTEX_ONLY:
            min_buffer_size = 2
            trigger_update = self.rollout_buffer.is_full() or (
                episode_done and len(self.rollout_buffer) >= min_buffer_size
            )
            if trigger_update:
                self._cortex_reinforce_update()
                self._critic_update()
                self.rollout_buffer.reset()

        if episode_done and len(self.episode_rewards) > 0:
            # Log episode summary
            total_reward = 0.0
            discount = 1.0
            for r in self.episode_rewards:
                total_reward += discount * r
                discount *= self.gamma

            logger.info(
                f"HybridQuantumCortex episode complete: episode={self._episode_count}, "
                f"stage={self.training_stage}, steps={self._step_count}, "
                f"total_reward={total_reward:.4f}, "
                f"W_sh_norm={torch.norm(self.W_sh).item():.4f}, "
                f"W_hm_norm={torch.norm(self.W_hm).item():.4f}",
            )

            # Final reflex REINFORCE update (stage 1, 3, 4)
            if self.training_stage != STAGE_CORTEX_ONLY:
                self._reflex_reinforce_update()

            # Log fusion diagnostics (stage 2, 3, 4)
            if self.training_stage >= STAGE_CORTEX_ONLY and self._episode_qsnn_trusts:
                mean_trust = np.mean(self._episode_qsnn_trusts)
                mode_means = np.mean(self._episode_mode_probs, axis=0).tolist()
                logger.info(
                    f"HybridQuantumCortex fusion: qsnn_trust_mean={mean_trust:.4f}, "
                    f"mode_dist={[f'{m:.3f}' for m in mode_means]}",
                )

            self._episode_count += 1

            # Step reflex LR scheduler (stage 1, 3, 4)
            if self.training_stage != STAGE_CORTEX_ONLY:
                self.reflex_scheduler.step()

            # Step cortex LR scheduler (stage 2, 3, 4)
            if self.training_stage >= STAGE_CORTEX_ONLY:
                self._update_cortex_learning_rate()

            # Auto-save weights
            if self.training_stage != STAGE_CORTEX_ONLY:
                self._save_reflex_weights(self._session_id)
            if self.training_stage >= STAGE_CORTEX_ONLY:
                self._save_cortex_weights(self._session_id)
                self._save_critic_weights(self._session_id)

            self._reset_episode()

    # ──────────────────────────────────────────────────────────────────
    # Reflex REINFORCE training
    # ──────────────────────────────────────────────────────────────────

    def _normalize_reward(self, reward: float) -> float:
        result, self.reward_running_mean, self.reward_running_var = normalize_reward(
            reward,
            self.reward_running_mean,
            self.reward_running_var,
            self.config.reward_norm_alpha,
        )
        return result

    def _adaptive_entropy_coef(self, entropy_val: float) -> float:
        return adaptive_entropy_coef(
            entropy_val,
            self.num_actions,
            entropy_coeff=self.config.entropy_coeff,
            entropy_floor=self.config.entropy_floor,
            entropy_boost_max=self.config.entropy_boost_max,
            entropy_ceiling_fraction=self.config.entropy_ceiling_fraction,
        )

    def _reflex_reinforce_update(self) -> None:
        """REINFORCE policy gradient update for QSNN reflex."""
        if len(self.episode_rewards) == 0:
            return
        num_steps = len(self.episode_rewards)
        if num_steps < MIN_REINFORCE_BATCH_SIZE:
            self.episode_rewards.clear()
            self.episode_features.clear()
            self.episode_actions.clear()
            self.episode_old_log_probs.clear()
            self._cached_spike_probs = []
            return

        # Compute discounted returns
        returns: list[float] = []
        discounted_return = 0.0
        for reward in reversed(self.episode_rewards):
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)

        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        raw_mean = returns_tensor.mean().item()

        self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * raw_mean

        if len(returns_tensor) > 1:
            advantages = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        else:
            advantages = returns_tensor - self.baseline

        advantages = torch.clamp(
            advantages,
            -self.config.advantage_clip,
            self.config.advantage_clip,
        )

        num_epochs = self.config.num_reinforce_epochs
        self._cached_spike_probs = []

        old_log_probs_t = torch.tensor(
            self.episode_old_log_probs[:num_steps],
            dtype=torch.float32,
            device=self.device,
        )

        for epoch in range(num_epochs):
            self._run_reflex_reinforce_epoch(
                num_steps,
                advantages,
                old_log_probs_t,
                _ReinforceUpdateStats(
                    raw_mean=raw_mean,
                    returns_tensor=returns_tensor,
                    epoch=epoch,
                    num_epochs=num_epochs,
                ),
            )

        self._clamp_reflex_weights()

    def _run_reflex_reinforce_epoch(
        self,
        num_steps: int,
        advantages: torch.Tensor,
        old_log_probs_t: torch.Tensor,
        stats: _ReinforceUpdateStats,
    ) -> None:
        """Run a single REINFORCE gradient epoch for reflex with PPO clipping."""
        self.refractory_hidden.fill(0)
        self.refractory_motor.fill(0)

        log_probs_list: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []

        for t in range(num_steps):
            features = self.episode_features[t]
            action_idx = self.episode_actions[t]

            if stats.num_epochs > 1 and stats.epoch == 0:
                timestep_cache: list[dict[str, list[float]]] = []
                motor_spikes = self._reflex_multi_timestep_differentiable_caching(
                    features,
                    timestep_cache,
                )
                self._cached_spike_probs.append(timestep_cache)
            elif stats.epoch > 0:
                motor_spikes = self._reflex_multi_timestep_differentiable_cached(
                    features,
                    self._cached_spike_probs[t],
                )
            else:
                motor_spikes = self._reflex_multi_timestep_differentiable(features)

            motor_clipped = torch.clamp(motor_spikes, 1e-8, 1.0 - 1e-8)
            logits = (motor_clipped - 0.5) * self.config.logit_scale

            epsilon, temperature = self._exploration_schedule()
            softmax_probs = torch.softmax(logits / temperature, dim=-1)
            uniform = torch.ones_like(softmax_probs) / self.num_actions
            action_probs = (1 - epsilon) * softmax_probs + epsilon * uniform

            log_prob = torch.log(action_probs[action_idx] + 1e-8)
            log_probs_list.append(log_prob)

            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10))
            entropies.append(entropy)

        log_probs = torch.stack(log_probs_list)
        mean_entropy = torch.stack(entropies).mean()
        effective_entropy_coef = self._adaptive_entropy_coef(mean_entropy.item())

        ratio = torch.exp(log_probs - old_log_probs_t)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon)
            * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean() - effective_entropy_coef * mean_entropy

        self.reflex_optimizer.zero_grad()
        policy_loss.backward()

        params_list = [self.W_sh, self.W_hm, self.theta_hidden, self.theta_motor]
        grad_norm = torch.nn.utils.clip_grad_norm_(params_list, max_norm=SURROGATE_GRAD_CLIP)
        self.reflex_optimizer.step()

        logger.debug(
            f"HybridQuantumCortex reflex REINFORCE: "
            f"loss={policy_loss.item():.4f}, "
            f"entropy={mean_entropy.item():.4f}, "
            f"grad_norm={grad_norm.item():.4f}, "
            f"W_sh_norm={torch.norm(self.W_sh).item():.4f}, "
            f"W_hm_norm={torch.norm(self.W_hm).item():.4f}, "
            f"theta_h_norm={torch.norm(self.theta_hidden).item():.4f}, "
            f"theta_m_norm={torch.norm(self.theta_motor).item():.4f}",
        )

    def _clamp_reflex_weights(self) -> None:
        """Clamp QSNN reflex weights after REINFORCE update."""
        with torch.no_grad():
            for p in [self.W_sh, self.W_hm, self.theta_hidden, self.theta_motor]:
                p.clamp_(-self.weight_clip, self.weight_clip)

            tm_norm = torch.norm(self.theta_motor)
            if tm_norm > self.config.theta_motor_max_norm:
                self.theta_motor.mul_(self.config.theta_motor_max_norm / tm_norm)

    # ──────────────────────────────────────────────────────────────────
    # Cortex REINFORCE+GAE training
    # ──────────────────────────────────────────────────────────────────

    def _compute_cortex_advantages(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns for cortex REINFORCE update."""
        buffer = self.rollout_buffer

        # Compute last value for GAE bootstrap
        last_state = buffer.states[-1]
        with torch.no_grad():
            last_state_t = torch.tensor(last_state, dtype=torch.float32, device=self.device)
            last_value = self.critic(last_state_t).squeeze(-1)

        # Compute returns and advantages via GAE
        returns, gae_advantages = buffer.compute_returns_and_advantages(
            last_value,
            self.gamma,
            self.config.gae_lambda,
        )

        if self.config.use_gae_advantages:
            advantages = gae_advantages
        else:
            # Fallback: use normalized discounted returns
            advantages = returns
            if advantages.std() > NORM_EPS:
                advantages = (advantages - advantages.mean()) / (advantages.std() + NORM_EPS)

        return returns, advantages

    def _cortex_reinforce_update(self) -> None:
        """REINFORCE update for cortex QSNN using GAE advantages from critic."""
        buffer = self.rollout_buffer
        if len(buffer) < MIN_REINFORCE_BATCH_SIZE:
            return

        returns, advantages = self._compute_cortex_advantages()
        self._latest_returns = returns

        states = buffer.states
        actions = buffer.actions

        num_epochs = self.config.num_cortex_reinforce_epochs
        cortex_cache: list[list[dict[str, list[float]]]] = []

        for epoch in range(num_epochs):
            self._reset_cortex_refractory()

            log_probs_list: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []

            for t in range(len(buffer)):
                cortex_features = states[t]
                action_idx = actions[t]

                if num_epochs > 1 and epoch == 0:
                    timestep_cache: list[dict[str, list[float]]] = []
                    output_spikes = self._cortex_multi_timestep_differentiable_caching(
                        cortex_features,
                        timestep_cache,
                    )
                    cortex_cache.append(timestep_cache)
                elif epoch > 0:
                    output_spikes = self._cortex_multi_timestep_differentiable_cached(
                        cortex_features,
                        cortex_cache[t],
                    )
                else:
                    output_spikes = self._cortex_multi_timestep_differentiable(cortex_features)

                output_clipped = torch.clamp(output_spikes, 1e-8, 1.0 - 1e-8)
                ab, _mode_logits, _trust = self._map_cortex_output(output_clipped)
                # output_clipped is a Tensor, so ab is guaranteed to be a Tensor
                action_biases = ab if isinstance(ab, torch.Tensor) else torch.tensor(ab)

                # Use action biases as policy logits for REINFORCE
                cortex_probs = torch.softmax(action_biases, dim=-1)
                log_prob = torch.log(cortex_probs[action_idx] + 1e-8)
                log_probs_list.append(log_prob)

                entropy = -torch.sum(cortex_probs * torch.log(cortex_probs + 1e-10))
                entropies.append(entropy)

            log_probs = torch.stack(log_probs_list)
            mean_entropy = torch.stack(entropies).mean()
            effective_entropy_coef = self._adaptive_entropy_coef(mean_entropy.item())

            # REINFORCE loss with detached advantages
            policy_loss = -(log_probs * advantages.detach()).mean()
            policy_loss = policy_loss - effective_entropy_coef * mean_entropy

            self.cortex_optimizer.zero_grad()
            policy_loss.backward()

            cortex_params = self._get_cortex_params()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                cortex_params,
                max_norm=self.config.max_grad_norm,
            )
            self.cortex_optimizer.step()

            logger.debug(
                f"HybridQuantumCortex cortex REINFORCE: "
                f"epoch={epoch}/{num_epochs}, "
                f"loss={policy_loss.item():.4f}, "
                f"entropy={mean_entropy.item():.4f}, "
                f"grad_norm={grad_norm.item():.4f}",
            )

        self._log_cortex_weight_norms()

    def _log_cortex_weight_norms(self) -> None:
        """Log cortex weight norms for diagnostics."""
        group_norms = [torch.norm(w).item() for w in self.cortex_group_weights]
        logger.debug(
            f"HybridQuantumCortex cortex weights: "
            f"group_norms={[f'{n:.4f}' for n in group_norms]}, "
            f"hidden_norm={torch.norm(self.W_cortex_sh).item():.4f}, "
            f"output_norm={torch.norm(self.W_cortex_ho).item():.4f}",
        )

    def _critic_update(self) -> None:
        """Train classical critic via Huber loss against target returns."""
        buffer = self.rollout_buffer
        if len(buffer) < MIN_REINFORCE_BATCH_SIZE:
            return

        states_t = torch.tensor(
            np.stack(buffer.states),
            dtype=torch.float32,
            device=self.device,
        )
        returns = self._latest_returns
        if returns is None:
            return

        # Forward pass through critic
        predicted = self.critic(states_t).squeeze(-1)

        # Huber loss
        loss = torch.nn.functional.huber_loss(predicted, returns)

        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            max_norm=self.config.max_grad_norm,
        )
        self.critic_optimizer.step()

        # Log explained variance
        target_var = returns.var()
        explained_var = (1.0 - (returns - predicted.detach()).var() / (target_var + 1e-8)).item()

        logger.debug(
            f"HybridQuantumCortex critic: "
            f"value_loss={loss.item():.4f}, "
            f"explained_variance={explained_var:.4f}",
        )

    # ──────────────────────────────────────────────────────────────────
    # Cortex LR scheduling
    # ──────────────────────────────────────────────────────────────────

    def _update_cortex_learning_rate(self) -> None:
        """Update cortex optimizer learning rates based on current schedule."""
        update_cortex_learning_rates(
            scheduling_enabled=self.cortex_lr_scheduling_enabled,
            episode_count=self._episode_count,
            base_lr=self.cortex_base_lr,
            warmup_episodes=self.cortex_lr_warmup_episodes,
            warmup_start=self.cortex_lr_warmup_start,
            decay_episodes=self.cortex_lr_decay_episodes,
            decay_end=self.cortex_lr_decay_end,
            cortex_actor_optimizer=self.cortex_optimizer,
            cortex_critic_optimizer=self.critic_optimizer,
        )

    # ──────────────────────────────────────────────────────────────────
    # Weight persistence
    # ──────────────────────────────────────────────────────────────────

    def _save_reflex_weights(self, session_id: str) -> None:
        """Save QSNN reflex weights to disk."""
        export_dir = Path("exports") / session_id
        export_dir.mkdir(parents=True, exist_ok=True)
        save_path = export_dir / "reflex_weights.pt"

        weights_dict = {
            "W_sh": self.W_sh.detach().cpu(),
            "W_hm": self.W_hm.detach().cpu(),
            "theta_hidden": self.theta_hidden.detach().cpu(),
            "theta_motor": self.theta_motor.detach().cpu(),
        }
        torch.save(weights_dict, save_path)
        logger.info(f"Reflex weights saved to {save_path}")

    def _load_reflex_weights(self) -> None:
        """Load pre-trained QSNN reflex weights from disk."""
        weights_path = self.config.reflex_weights_path
        if weights_path is None:
            return

        path = Path(weights_path)
        if not path.exists():
            msg = f"Reflex weights file not found: {weights_path}"
            raise FileNotFoundError(msg)

        weights_dict = torch.load(path, weights_only=True)

        expected_shapes = {
            "W_sh": (self.num_sensory, self.num_hidden),
            "W_hm": (self.num_hidden, self.num_motor),
            "theta_hidden": (self.num_hidden,),
            "theta_motor": (self.num_motor,),
        }
        for key, expected_shape in expected_shapes.items():
            if key not in weights_dict:
                msg = f"Missing key '{key}' in reflex weights file"
                raise ValueError(msg)
            actual_shape = tuple(weights_dict[key].shape)
            if actual_shape != expected_shape:
                msg = f"Shape mismatch for '{key}': expected {expected_shape}, got {actual_shape}"
                raise ValueError(msg)

        with torch.no_grad():
            self.W_sh.copy_(weights_dict["W_sh"].to(self.device))
            self.W_hm.copy_(weights_dict["W_hm"].to(self.device))
            self.theta_hidden.copy_(weights_dict["theta_hidden"].to(self.device))
            self.theta_motor.copy_(weights_dict["theta_motor"].to(self.device))

        logger.info(f"Reflex weights loaded from {weights_path}")

    def _save_cortex_weights(self, session_id: str) -> None:
        """Save QSNN cortex weights to disk."""
        export_dir = Path("exports") / session_id
        export_dir.mkdir(parents=True, exist_ok=True)
        save_path = export_dir / "cortex_weights.pt"

        weights_dict: dict[str, torch.Tensor] = {}
        for i, w in enumerate(self.cortex_group_weights):
            weights_dict[f"W_group_{i}"] = w.detach().cpu()
        for i, t in enumerate(self.cortex_group_thetas):
            weights_dict[f"theta_group_{i}"] = t.detach().cpu()
        weights_dict["W_cortex_sh"] = self.W_cortex_sh.detach().cpu()
        weights_dict["theta_cortex_hidden"] = self.theta_cortex_hidden.detach().cpu()
        weights_dict["W_cortex_ho"] = self.W_cortex_ho.detach().cpu()
        weights_dict["theta_cortex_output"] = self.theta_cortex_output.detach().cpu()

        torch.save(weights_dict, save_path)
        logger.info(f"Cortex weights saved to {save_path}")

    def _load_cortex_weights(self) -> None:
        """Load pre-trained QSNN cortex weights from disk."""
        weights_path = self.config.cortex_weights_path
        if weights_path is None:
            return

        path = Path(weights_path)
        if not path.exists():
            msg = f"Cortex weights file not found: {weights_path}"
            raise FileNotFoundError(msg)

        weights_dict = torch.load(path, weights_only=True)

        with torch.no_grad():
            for i in range(self.num_cortex_groups):
                key_w = f"W_group_{i}"
                key_t = f"theta_group_{i}"
                if key_w not in weights_dict:
                    msg = f"Missing key '{key_w}' in cortex weights file"
                    raise ValueError(msg)
                if key_t not in weights_dict:
                    msg = f"Missing key '{key_t}' in cortex weights file"
                    raise ValueError(msg)

                expected_w = tuple(self.cortex_group_weights[i].shape)
                actual_w = tuple(weights_dict[key_w].shape)
                if actual_w != expected_w:
                    msg = f"Shape mismatch for '{key_w}': expected {expected_w}, got {actual_w}"
                    raise ValueError(msg)

                self.cortex_group_weights[i].copy_(weights_dict[key_w].to(self.device))
                self.cortex_group_thetas[i].copy_(weights_dict[key_t].to(self.device))

            for key, tensor in [
                ("W_cortex_sh", self.W_cortex_sh),
                ("theta_cortex_hidden", self.theta_cortex_hidden),
                ("W_cortex_ho", self.W_cortex_ho),
                ("theta_cortex_output", self.theta_cortex_output),
            ]:
                if key not in weights_dict:
                    msg = f"Missing key '{key}' in cortex weights file"
                    raise ValueError(msg)
                expected_shape = tuple(tensor.shape)
                actual_shape = tuple(weights_dict[key].shape)
                if actual_shape != expected_shape:
                    msg = (
                        f"Shape mismatch for '{key}': expected {expected_shape}, got {actual_shape}"
                    )
                    raise ValueError(msg)
                tensor.copy_(weights_dict[key].to(self.device))

        logger.info(f"Cortex weights loaded from {weights_path}")

    def _save_critic_weights(self, session_id: str) -> None:
        """Save critic weights to disk."""
        export_dir = Path("exports") / session_id
        export_dir.mkdir(parents=True, exist_ok=True)
        save_path = export_dir / "critic_weights.pt"
        torch.save(self.critic.state_dict(), save_path)
        logger.info(f"Critic weights saved to {save_path}")

    def _load_critic_weights(self) -> None:
        """Load pre-trained critic weights from disk."""
        weights_path = self.config.critic_weights_path
        if weights_path is None:
            return

        path = Path(weights_path)
        if not path.exists():
            msg = f"Critic weights file not found: {weights_path}"
            raise FileNotFoundError(msg)

        state_dict = torch.load(path, weights_only=True)
        self.critic.load_state_dict(state_dict)
        logger.info(f"Critic weights loaded from {weights_path}")

    # ──────────────────────────────────────────────────────────────────
    # Session management
    # ──────────────────────────────────────────────────────────────────

    def set_session_id(self, session_id: str) -> None:
        """Set the session ID for weight saving."""
        self._session_id = session_id

    # ──────────────────────────────────────────────────────────────────
    # Episode management
    # ──────────────────────────────────────────────────────────────────

    def _reset_episode(self) -> None:
        """Reset episode state."""
        self.episode_rewards.clear()
        self.episode_actions.clear()
        self.episode_features.clear()
        self.episode_old_log_probs.clear()
        self._cached_spike_probs = []
        self._episode_qsnn_trusts.clear()
        self._episode_mode_probs.clear()

        self._has_pending_cortex_data = False

        # Reset both QSNN refractory states
        self.refractory_hidden.fill(0)
        self.refractory_motor.fill(0)
        self._reset_cortex_refractory()

        self._step_count = 0

    def _reset_cortex_refractory(self) -> None:
        """Reset cortex QSNN refractory states."""
        for ref in self.cortex_refractory_groups:
            ref.fill(0)
        self.cortex_refractory_hidden.fill(0)
        self.cortex_refractory_output.fill(0)

    # ──────────────────────────────────────────────────────────────────
    # Protocol methods
    # ──────────────────────────────────────────────────────────────────

    def update_memory(self, reward: float | None = None) -> None:
        """No-op for HybridQuantumCortexBrain."""

    def prepare_episode(self) -> None:
        """Prepare for a new episode."""
        self._reset_episode()

    def post_process_episode(self, *, episode_success: bool | None = None) -> None:
        """Post-process the episode (no-op, learning happens in learn())."""

    def copy(self) -> HybridQuantumCortexBrain:
        """Create a copy of this brain."""
        config_copy = HybridQuantumCortexBrainConfig(
            **{**self.config.model_dump(), "seed": self.seed},
        )
        new_brain = HybridQuantumCortexBrain(
            config=config_copy,
            num_actions=self.num_actions,
            device=DeviceType(self.device.type),
            action_set=self._action_set,
        )

        # Copy reflex weights
        with torch.no_grad():
            new_brain.W_sh.copy_(self.W_sh)
            new_brain.W_hm.copy_(self.W_hm)
            new_brain.theta_hidden.copy_(self.theta_hidden)
            new_brain.theta_motor.copy_(self.theta_motor)

        # Copy cortex weights
        with torch.no_grad():
            for i in range(self.num_cortex_groups):
                new_brain.cortex_group_weights[i].copy_(self.cortex_group_weights[i])
                new_brain.cortex_group_thetas[i].copy_(self.cortex_group_thetas[i])
            new_brain.W_cortex_sh.copy_(self.W_cortex_sh)
            new_brain.theta_cortex_hidden.copy_(self.theta_cortex_hidden)
            new_brain.W_cortex_ho.copy_(self.W_cortex_ho)
            new_brain.theta_cortex_output.copy_(self.theta_cortex_output)

        # Copy critic state dict
        new_brain.critic.load_state_dict(self.critic.state_dict())

        # Copy learning state
        new_brain.baseline = self.baseline
        new_brain._episode_count = self._episode_count
        new_brain.reward_running_mean = self.reward_running_mean
        new_brain.reward_running_var = self.reward_running_var

        return new_brain

    @property
    def action_set(self) -> list[Action]:
        """Return the set of available actions."""
        return self._action_set

    @action_set.setter
    def action_set(self, actions: list[Action]) -> None:
        if len(actions) != self.num_actions:
            msg = (
                f"Cannot set action_set of length {len(actions)}: "
                f"network expects {self.num_actions} actions"
            )
            raise ValueError(msg)
        self._action_set = actions
