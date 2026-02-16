"""
Hierarchical Hybrid Quantum Brain Architecture.

Combines a QSNN reflex layer (QLIF neurons, REINFORCE training) with a classical
cortex MLP (PPO training) for multi-objective decision-making. The QSNN provides
proven reactive foraging behaviour (73.9% success), while the cortex learns
strategic mode gating (forage/evade/explore) to modulate trust in the reflex.

Architecture::

    Sensory Input (8-dim)
           |
           v
    QSNN Reflex Layer (QLIFNetwork from _qlif_layers.py)
      sensory -> hidden -> motor QLIF neurons
      ~212 quantum params, surrogate gradient REINFORCE
      Output: 4-dim reflex logits
           |
           v                    Sensory Input (8-dim)
           |                          |
           v                          v
      +---------+            Classical Cortex (MLP)
      |  Fusion |<--------   sensory -> hidden -> 7
      +---------+            (4 action biases + 3 mode logits)
           |                  PPO training, ~5K actor params
           v
      final_logits = reflex_logits * qsnn_trust + cortex_biases
           |
           v
      Action Selection (4 actions)

Three-stage curriculum:
  1. QSNN reflex on foraging (REINFORCE)
  2. Freeze QSNN, train cortex (PPO)
  3. Optional joint fine-tune

References
----------
- Brand & Petruccione (2024). "A quantum leaky integrate-and-fire spiking neuron
  and network." npj Quantum Information, 10(1), 16.
- Schulman et al. (2017). "Proximal Policy Optimization Algorithms."
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from pydantic import Field, field_validator
from torch import nn

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, ClassicalBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
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
from quantumnematode.env import Direction
from quantumnematode.logging_config import logger
from quantumnematode.utils.seeding import ensure_seed, get_rng, set_global_seed

if TYPE_CHECKING:
    from collections.abc import Iterator

# ──────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────

# QSNN reflex defaults
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

# Cortex PPO defaults
DEFAULT_CORTEX_HIDDEN_DIM = 64
DEFAULT_CORTEX_NUM_LAYERS = 2
DEFAULT_NUM_MODES = 3
DEFAULT_CORTEX_ACTOR_LR = 0.001
DEFAULT_CORTEX_CRITIC_LR = 0.001
DEFAULT_PPO_CLIP_EPSILON = 0.2
DEFAULT_PPO_EPOCHS = 4
DEFAULT_PPO_MINIBATCHES = 4
DEFAULT_PPO_BUFFER_SIZE = 512
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_ENTROPY_COEFF = 0.01
DEFAULT_MAX_GRAD_NORM = 0.5

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

# Training stages
STAGE_QSNN_ONLY = 1
STAGE_CORTEX_ONLY = 2
STAGE_JOINT = 3


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────


class HybridQuantumBrainConfig(BrainConfig):
    """Configuration for the HybridQuantumBrain architecture.

    Supports two modes for input feature extraction:

    1. **Legacy mode** (default): Uses 2 features (gradient_strength, relative_angle)
       - Set `sensory_modules=None` (default)

    2. **Unified sensory mode**: Uses modular feature extraction from brain/modules.py
       - Set `sensory_modules` to a list of ModuleName values
       - Each module contributes 2 features [strength, angle]
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
        description="Number of quantum measurement shots.",
    )
    num_qsnn_timesteps: int = Field(
        default=DEFAULT_NUM_QSNN_TIMESTEPS,
        description="Number of QLIF timesteps per decision.",
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

    # Cortex params
    cortex_hidden_dim: int = Field(
        default=DEFAULT_CORTEX_HIDDEN_DIM,
        description="Hidden layer dimension for cortex MLP.",
    )
    cortex_num_layers: int = Field(
        default=DEFAULT_CORTEX_NUM_LAYERS,
        description="Number of hidden layers in cortex MLPs.",
    )
    num_modes: int = Field(
        default=DEFAULT_NUM_MODES,
        description="Number of modes for gating (forage, evade, explore).",
    )

    # Training stage
    training_stage: int = Field(
        default=1,
        description="Training stage: 1=QSNN only, 2=cortex only (QSNN frozen), 3=joint.",
    )

    # QSNN REINFORCE params
    qsnn_lr: float = Field(
        default=DEFAULT_QSNN_LR,
        description="Learning rate for QSNN REINFORCE.",
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

    # Cortex PPO params
    cortex_actor_lr: float = Field(
        default=DEFAULT_CORTEX_ACTOR_LR,
        description="Learning rate for cortex actor.",
    )
    cortex_critic_lr: float = Field(
        default=DEFAULT_CORTEX_CRITIC_LR,
        description="Learning rate for cortex critic.",
    )
    ppo_clip_epsilon: float = Field(
        default=DEFAULT_PPO_CLIP_EPSILON,
        description="PPO clipping epsilon.",
    )
    ppo_epochs: int = Field(
        default=DEFAULT_PPO_EPOCHS,
        description="Number of PPO gradient epochs per buffer update.",
    )
    ppo_minibatches: int = Field(
        default=DEFAULT_PPO_MINIBATCHES,
        description="Number of minibatches per PPO epoch.",
    )
    ppo_buffer_size: int = Field(
        default=DEFAULT_PPO_BUFFER_SIZE,
        description="Rollout buffer size for cortex PPO.",
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

    # Joint fine-tune
    joint_finetune_lr_factor: float = Field(
        default=DEFAULT_JOINT_FINETUNE_LR_FACTOR,
        description="LR multiplier for QSNN in stage 3 (10x lower).",
    )

    # Weight persistence
    qsnn_weights_path: str | None = Field(
        default=None,
        description="Path to pre-trained QSNN weights (.pt file) for stage 2/3.",
    )

    # Sensory modules
    sensory_modules: list[ModuleName] | None = Field(
        default=None,
        description="List of sensory modules for feature extraction (None = legacy mode).",
    )

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
        """Validate training_stage is 1, 2, or 3."""
        if v not in (STAGE_QSNN_ONLY, STAGE_CORTEX_ONLY, STAGE_JOINT):
            msg = f"training_stage must be 1, 2, or 3, got {v}"
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


# ──────────────────────────────────────────────────────────────────────
# Internal data structures
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _ReinforceUpdateStats:
    """Bundled statistics for REINFORCE optimizer step logging."""

    raw_mean: float
    returns_tensor: torch.Tensor
    epoch: int
    num_epochs: int


class _CortexRolloutBuffer:
    """Rollout buffer for cortex PPO training."""

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
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.rewards.append(reward)
        self.dones.append(done)
        self.position += 1

    def is_full(self) -> bool:
        return self.position >= self.buffer_size

    def __len__(self) -> int:
        return self.position

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros(len(self), device=self.device)
        last_gae = 0.0
        values = torch.stack(self.values).reshape(-1)

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
        batch_size = len(self)
        minibatch_size = batch_size // num_minibatches

        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.stack(self.log_probs)

        if len(advantages) > 1:
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


# ──────────────────────────────────────────────────────────────────────
# Brain Implementation
# ──────────────────────────────────────────────────────────────────────


class HybridQuantumBrain(ClassicalBrain):
    """Hierarchical hybrid quantum brain combining QSNN reflex with classical cortex.

    The QSNN reflex layer provides proven reactive foraging behaviour via QLIF
    quantum circuits. The classical cortex MLP learns strategic mode gating
    (forage/evade/explore) to modulate trust in the reflex, plus additive action
    biases for direct direction preference.
    """

    def __init__(  # noqa: PLR0915
        self,
        config: HybridQuantumBrainConfig,
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
        logger.info(f"HybridQuantumBrain using seed: {self.seed}")

        # Sensory modules
        self.sensory_modules = config.sensory_modules
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

        # Leak angle
        self.leak_angle = (1 - self.membrane_tau) * np.pi

        # Initialize QSNN reflex layer
        self._init_qsnn_weights()

        # Initialize cortex actor + critic
        self._init_cortex()

        # Initialize optimizers
        self._init_optimizers()

        # Load pre-trained QSNN weights if specified
        if config.qsnn_weights_path is not None:
            self._load_qsnn_weights()
        elif self.training_stage >= STAGE_CORTEX_ONLY:
            logger.warning(
                "Stage >= 2 but no qsnn_weights_path specified. "
                "QSNN will use random initialization.",
            )

        # Freeze QSNN in stage 2
        if self.training_stage == STAGE_CORTEX_ONLY:
            self._freeze_qsnn()

        # PPO rollout buffer for cortex
        self.ppo_buffer = _CortexRolloutBuffer(
            config.ppo_buffer_size,
            self.device,
            rng=self.rng,
        )

        # REINFORCE episode buffers
        self._init_episode_state()

        # Pending cortex data (set per-step in run_brain, consumed in learn)
        self._has_pending_cortex_data = False

        # Counters
        self._step_count = 0
        self._episode_count = 0

        # Session ID for weight saving (set via set_session_id, fallback to timestamp)
        self._session_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        # Qiskit backend (lazy)
        self._backend = None

        # Running reward normalization (REINFORCE)
        self.reward_running_mean: float = 0.0
        self.reward_running_var: float = 1.0

        # Fusion tracking
        self._episode_qsnn_trusts: list[float] = []
        self._episode_mode_probs: list[list[float]] = []

        # Current action probabilities
        self.current_probabilities: np.ndarray | None = None
        self.training = True

        total_qsnn_params = (
            self.num_sensory * self.num_hidden
            + self.num_hidden * self.num_motor
            + self.num_hidden
            + self.num_motor
        )
        cortex_actor_params = sum(p.numel() for p in self.cortex_actor.parameters())
        cortex_critic_params = sum(p.numel() for p in self.cortex_critic.parameters())

        logger.info(
            f"HybridQuantumBrain initialized: "
            f"QSNN {self.num_sensory}->{self.num_hidden}->{self.num_motor}, "
            f"cortex hidden={config.cortex_hidden_dim}x{config.cortex_num_layers}, "
            f"modes={self.num_modes}, stage={self.training_stage}, "
            f"params: QSNN={total_qsnn_params}, "
            f"cortex_actor={cortex_actor_params}, "
            f"cortex_critic={cortex_critic_params}",
        )

    def _init_qsnn_weights(self) -> None:
        """Initialize QSNN trainable weight matrices and neuron parameters."""
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

    def _init_cortex(self) -> None:
        """Initialize cortex actor and critic MLPs."""
        cortex_output_dim = self.num_motor + self.num_modes  # action biases + mode logits

        # Actor: sensory -> hidden -> (action_biases + mode_logits)
        actor_layers: list[nn.Module] = [
            nn.Linear(self.input_dim, self.config.cortex_hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(self.config.cortex_num_layers - 1):
            actor_layers += [
                nn.Linear(self.config.cortex_hidden_dim, self.config.cortex_hidden_dim),
                nn.ReLU(),
            ]
        actor_layers.append(nn.Linear(self.config.cortex_hidden_dim, cortex_output_dim))
        self.cortex_actor = nn.Sequential(*actor_layers).to(self.device)

        # Critic: sensory -> hidden -> V(s)
        critic_layers: list[nn.Module] = [
            nn.Linear(self.input_dim, self.config.cortex_hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(self.config.cortex_num_layers - 1):
            critic_layers += [
                nn.Linear(self.config.cortex_hidden_dim, self.config.cortex_hidden_dim),
                nn.ReLU(),
            ]
        critic_layers.append(nn.Linear(self.config.cortex_hidden_dim, 1))
        self.cortex_critic = nn.Sequential(*critic_layers).to(self.device)

        # Orthogonal initialization
        def init_weights(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        self.cortex_actor.apply(init_weights)
        self.cortex_critic.apply(init_weights)

    def _init_optimizers(self) -> None:
        """Initialize separate optimizers for QSNN and cortex."""
        qsnn_lr = self.config.qsnn_lr
        if self.training_stage == STAGE_JOINT:
            qsnn_lr *= self.config.joint_finetune_lr_factor

        self.qsnn_optimizer = torch.optim.Adam(
            [self.W_sh, self.W_hm, self.theta_hidden, self.theta_motor],
            lr=qsnn_lr,
        )
        self.qsnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.qsnn_optimizer,
            T_max=self.config.qsnn_lr_decay_episodes,
            eta_min=qsnn_lr * self.config.lr_min_factor,
        )

        self.cortex_actor_optimizer = torch.optim.Adam(
            self.cortex_actor.parameters(),
            lr=self.config.cortex_actor_lr,
        )
        self.cortex_critic_optimizer = torch.optim.Adam(
            self.cortex_critic.parameters(),
            lr=self.config.cortex_critic_lr,
        )

    def _freeze_qsnn(self) -> None:
        """Freeze QSNN weights (stage 2)."""
        self.W_sh.requires_grad_(False)  # noqa: FBT003
        self.W_hm.requires_grad_(False)  # noqa: FBT003
        self.theta_hidden.requires_grad_(False)  # noqa: FBT003
        self.theta_motor.requires_grad_(False)  # noqa: FBT003
        logger.info("QSNN weights frozen for stage 2 training")

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
    # Qiskit backend
    # ──────────────────────────────────────────────────────────────────

    def _get_backend(self):  # noqa: ANN202
        if self._backend is None:
            self._backend = get_qiskit_backend(
                DeviceType(self.device.type) if hasattr(self.device, "type") else DeviceType.CPU,
                seed=self.seed,
            )
        return self._backend

    # ──────────────────────────────────────────────────────────────────
    # QSNN forward pass
    # ──────────────────────────────────────────────────────────────────

    def _encode_sensory_spikes(self, features: np.ndarray) -> np.ndarray:
        return encode_sensory_spikes(features, self.num_sensory)

    def _timestep(self, features: np.ndarray) -> np.ndarray:
        """Execute one QSNN timestep (non-differentiable)."""
        sensory_spikes = self._encode_sensory_spikes(features)

        hidden_spikes, self.refractory_hidden = execute_qlif_layer(
            sensory_spikes,
            self.W_sh,
            self.theta_hidden,
            self.refractory_hidden,
            backend=self._get_backend(),
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
            backend=self._get_backend(),
            shots=self.shots,
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            leak_angle=self.leak_angle,
        )
        return motor_spikes

    def _multi_timestep(self, features: np.ndarray) -> np.ndarray:
        """Execute multiple QSNN timesteps and average motor spike probabilities."""
        motor_accumulator = np.zeros(self.num_motor)
        for _ in range(self.num_qsnn_timesteps):
            motor_spikes = self._timestep(features)
            motor_accumulator += motor_spikes
        return motor_accumulator / self.num_qsnn_timesteps

    def _timestep_differentiable(self, features: np.ndarray) -> torch.Tensor:
        """Execute one QSNN timestep with gradient tracking."""
        sensory_spikes = self._encode_sensory_spikes(features)
        sensory_tensor = torch.tensor(sensory_spikes, dtype=torch.float32, device=self.device)

        hidden_spikes, self.refractory_hidden = execute_qlif_layer_differentiable(
            sensory_tensor,
            self.W_sh,
            self.theta_hidden,
            self.refractory_hidden,
            backend=self._get_backend(),
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
            backend=self._get_backend(),
            shots=self.shots,
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            leak_angle=self.leak_angle,
            device=self.device,
        )
        return motor_spikes

    def _multi_timestep_differentiable(self, features: np.ndarray) -> torch.Tensor:
        """Execute multiple QSNN timesteps with gradient tracking and average."""
        motor_accumulator = torch.zeros(self.num_motor, device=self.device)
        for _ in range(self.num_qsnn_timesteps):
            motor_spikes = self._timestep_differentiable(features)
            motor_accumulator = motor_accumulator + motor_spikes
        return motor_accumulator / self.num_qsnn_timesteps

    def _timestep_differentiable_caching(
        self,
        features: np.ndarray,
        cache_out: dict[str, list[float]],
    ) -> torch.Tensor:
        """Execute one QSNN timestep with gradient tracking, caching spike probs."""
        sensory_spikes = self._encode_sensory_spikes(features)
        sensory_tensor = torch.tensor(sensory_spikes, dtype=torch.float32, device=self.device)

        hidden_spikes, self.refractory_hidden = execute_qlif_layer_differentiable(
            sensory_tensor,
            self.W_sh,
            self.theta_hidden,
            self.refractory_hidden,
            backend=self._get_backend(),
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
            backend=self._get_backend(),
            shots=self.shots,
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            leak_angle=self.leak_angle,
            device=self.device,
        )
        cache_out["motor"] = motor_spikes.detach().cpu().tolist()

        return motor_spikes

    def _multi_timestep_differentiable_caching(
        self,
        features: np.ndarray,
        cache_out: list[dict[str, list[float]]],
    ) -> torch.Tensor:
        motor_accumulator = torch.zeros(self.num_motor, device=self.device)
        for _ in range(self.num_qsnn_timesteps):
            step_cache: dict[str, list[float]] = {}
            motor_spikes = self._timestep_differentiable_caching(features, step_cache)
            motor_accumulator = motor_accumulator + motor_spikes
            cache_out.append(step_cache)
        return motor_accumulator / self.num_qsnn_timesteps

    def _timestep_differentiable_cached(
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

    def _multi_timestep_differentiable_cached(
        self,
        features: np.ndarray,
        cached_timestep: list[dict[str, list[float]]],
    ) -> torch.Tensor:
        motor_accumulator = torch.zeros(self.num_motor, device=self.device)
        for step_idx in range(self.num_qsnn_timesteps):
            motor_spikes = self._timestep_differentiable_cached(
                features,
                cached_timestep[step_idx],
            )
            motor_accumulator = motor_accumulator + motor_spikes
        return motor_accumulator / self.num_qsnn_timesteps

    # ──────────────────────────────────────────────────────────────────
    # Cortex forward pass
    # ──────────────────────────────────────────────────────────────────

    def _cortex_forward(self, sensory_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run cortex actor forward pass, returning action biases and mode logits."""
        cortex_out = self.cortex_actor(sensory_input)
        action_biases = cortex_out[: self.num_motor]
        mode_logits = cortex_out[self.num_motor :]
        return action_biases, mode_logits

    def _cortex_value(self, sensory_input: torch.Tensor) -> torch.Tensor:
        """Get critic value estimate from sensory input."""
        return self.cortex_critic(sensory_input).squeeze(-1)

    # ──────────────────────────────────────────────────────────────────
    # Fusion mechanism
    # ──────────────────────────────────────────────────────────────────

    def _fuse(
        self,
        reflex_logits: torch.Tensor,
        action_biases: torch.Tensor,
        mode_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, float, torch.Tensor]:
        """Apply mode-gated fusion.

        Returns (final_logits, qsnn_trust, mode_probs).
        """
        mode_probs = torch.softmax(mode_logits, dim=-1)
        qsnn_trust = mode_probs[0]  # forage mode trusts QSNN
        final_logits = reflex_logits * qsnn_trust + action_biases
        return final_logits, float(qsnn_trust.item()), mode_probs

    # ──────────────────────────────────────────────────────────────────
    # Preprocessing
    # ──────────────────────────────────────────────────────────────────

    def preprocess(self, params: BrainParams) -> np.ndarray:
        """Preprocess BrainParams to extract features."""
        if self.sensory_modules is not None:
            return extract_classical_features(params, self.sensory_modules)

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

    # ──────────────────────────────────────────────────────────────────
    # Exploration schedule
    # ──────────────────────────────────────────────────────────────────

    def _exploration_schedule(self) -> tuple[float, float]:
        progress = min(1.0, self._episode_count / max(self.config.exploration_decay_episodes, 1))
        current_epsilon = self.config.exploration_epsilon * (1.0 - progress * 0.7)
        current_temperature = 1.5 - 0.5 * progress
        return current_epsilon, current_temperature

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
        features = self.preprocess(params)

        # QSNN forward pass (always runs regardless of stage)
        motor_probs = self._multi_timestep(features)
        motor_probs = np.clip(motor_probs, 1e-8, 1.0 - 1e-8)
        reflex_logits_np = (motor_probs - 0.5) * self.config.logit_scale

        if self.training_stage == STAGE_QSNN_ONLY:
            # Stage 1: QSNN only, no cortex involvement
            logits = reflex_logits_np
            epsilon, temperature = self._exploration_schedule()
            scaled_logits = logits / temperature
            exp_probs = np.exp(scaled_logits - np.max(scaled_logits))
            softmax_probs = exp_probs / np.sum(exp_probs)
            uniform = np.ones(self.num_actions) / self.num_actions
            action_probs = (1 - epsilon) * softmax_probs + epsilon * uniform
        else:
            # Stage 2/3: cortex fusion
            sensory_t = torch.tensor(features, dtype=torch.float32, device=self.device)
            reflex_logits_t = torch.tensor(
                reflex_logits_np,
                dtype=torch.float32,
                device=self.device,
            )

            action_biases, mode_logits = self._cortex_forward(sensory_t)
            final_logits, qsnn_trust, mode_probs = self._fuse(
                reflex_logits_t,
                action_biases,
                mode_logits,
            )

            # Track fusion diagnostics
            self._episode_qsnn_trusts.append(qsnn_trust)
            self._episode_mode_probs.append(mode_probs.detach().cpu().tolist())

            action_probs_t = torch.softmax(final_logits, dim=-1)
            action_probs = action_probs_t.detach().cpu().numpy()

            # Store PPO data for cortex training.
            # Use cortex-only log_probs (from action_biases) so they match
            # what _perform_ppo_update recomputes. The fused distribution
            # drives action selection, but PPO trains the cortex policy.
            cortex_logits = action_biases.detach()
            cortex_probs = torch.softmax(cortex_logits, dim=-1)
            cortex_log_probs = torch.log(cortex_probs + 1e-8)
            with torch.no_grad():
                value = self._cortex_value(sensory_t)
            self._pending_cortex_state = features
            self._pending_cortex_log_prob_dist = cortex_log_probs
            self._pending_cortex_value = value

        # Sample action
        action_probs = np.clip(action_probs, 1e-8, 1.0)
        action_probs = action_probs / action_probs.sum()
        action_idx = self.rng.choice(self.num_actions, p=action_probs)
        action_name = self.action_set[action_idx]

        # Store REINFORCE data (stage 1 and 3)
        if self.training_stage in (STAGE_QSNN_ONLY, STAGE_JOINT):
            self.episode_features.append(features)
            old_log_prob = float(np.log(action_probs[action_idx] + 1e-8))
            self.episode_old_log_probs.append(old_log_prob)

        # Store cortex PPO chosen-action log_prob for the PPO buffer
        if self.training_stage >= STAGE_CORTEX_ONLY:
            self._pending_cortex_action = action_idx
            self._pending_cortex_chosen_log_prob = self._pending_cortex_log_prob_dist[action_idx]
            self._has_pending_cortex_data = True

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

    def learn(  # noqa: C901
        self,
        params: BrainParams,  # noqa: ARG002
        reward: float,
        *,
        episode_done: bool = False,
    ) -> None:
        """Accumulate rewards and trigger stage-dependent training."""
        # Reward normalization for REINFORCE (stage 1 and 3)
        uses_reinforce = self.training_stage in (STAGE_QSNN_ONLY, STAGE_JOINT)
        if uses_reinforce and self.config.use_reward_normalization:
            normalized_reward = self._normalize_reward(reward)
            self.episode_rewards.append(normalized_reward)
        else:
            self.episode_rewards.append(reward)

        self.history_data.rewards.append(reward)
        self._step_count += 1

        # Add to PPO buffer (stage 2 and 3)
        if self.training_stage >= STAGE_CORTEX_ONLY and self._has_pending_cortex_data:
            self.ppo_buffer.add(
                state=self._pending_cortex_state,
                action=self._pending_cortex_action,
                log_prob=self._pending_cortex_chosen_log_prob,
                value=self._pending_cortex_value,
                reward=reward,
                done=episode_done,
            )

        # Intra-episode REINFORCE update (stage 1 and 3)
        window = self.config.reinforce_window_size
        if (
            self.training_stage in (STAGE_QSNN_ONLY, STAGE_JOINT)
            and window > 0
            and self._step_count % window == 0
            and not episode_done
        ):
            self._reinforce_update()
            self.episode_rewards.clear()
            self.episode_actions.clear()
            self.episode_features.clear()
            self.episode_old_log_probs.clear()
            self._cached_spike_probs = []

        # PPO buffer update (stage 2 and 3)
        if self.training_stage >= STAGE_CORTEX_ONLY:
            trigger_ppo = self.ppo_buffer.is_full() or (
                episode_done and len(self.ppo_buffer) >= self.config.ppo_minibatches
            )
            if trigger_ppo:
                self._perform_ppo_update()
                self.ppo_buffer.reset()

        if episode_done and len(self.episode_rewards) > 0:
            # Log episode summary
            total_reward = 0.0
            discount = 1.0
            for r in self.episode_rewards:
                total_reward += discount * r
                discount *= self.gamma

            logger.info(
                f"HybridQuantum episode complete: episode={self._episode_count}, "
                f"stage={self.training_stage}, steps={self._step_count}, "
                f"total_reward={total_reward:.4f}, "
                f"W_sh_norm={torch.norm(self.W_sh).item():.4f}, "
                f"W_hm_norm={torch.norm(self.W_hm).item():.4f}",
            )

            # Final REINFORCE update (stage 1 and 3)
            if self.training_stage in (STAGE_QSNN_ONLY, STAGE_JOINT):
                self._reinforce_update()

            # Log fusion diagnostics (stage 2/3)
            if self.training_stage >= STAGE_CORTEX_ONLY and self._episode_qsnn_trusts:
                mean_trust = np.mean(self._episode_qsnn_trusts)
                mode_means = np.mean(self._episode_mode_probs, axis=0).tolist()
                logger.info(
                    f"HybridQuantum fusion: qsnn_trust_mean={mean_trust:.4f}, "
                    f"mode_dist={[f'{m:.3f}' for m in mode_means]}",
                )

            self._episode_count += 1

            # Step QSNN LR scheduler (stage 1 and 3)
            if self.training_stage in (STAGE_QSNN_ONLY, STAGE_JOINT):
                self.qsnn_scheduler.step()

            # Auto-save QSNN weights when QSNN is being trained
            if self.training_stage in (STAGE_QSNN_ONLY, STAGE_JOINT):
                self._save_qsnn_weights(self._session_id)

            self._reset_episode()

    # ──────────────────────────────────────────────────────────────────
    # QSNN REINFORCE training
    # ──────────────────────────────────────────────────────────────────

    def _normalize_reward(self, reward: float) -> float:
        alpha = self.config.reward_norm_alpha
        self.reward_running_mean = (1 - alpha) * self.reward_running_mean + alpha * reward
        diff = reward - self.reward_running_mean
        self.reward_running_var = (1 - alpha) * self.reward_running_var + alpha * diff * diff
        running_std = np.sqrt(self.reward_running_var)
        return (reward - self.reward_running_mean) / (running_std + 1e-8)

    def _adaptive_entropy_coef(self, entropy_val: float) -> float:
        max_entropy = np.log(self.num_actions)
        entropy_ceiling = self.config.entropy_ceiling_fraction * max_entropy
        if entropy_val < self.config.entropy_floor:
            ratio = 1.0 - entropy_val / self.config.entropy_floor
            entropy_scale = 1.0 + ratio * (self.config.entropy_boost_max - 1.0)
        elif entropy_val > entropy_ceiling:
            ratio = (entropy_val - entropy_ceiling) / (max_entropy - entropy_ceiling)
            entropy_scale = max(0.0, 1.0 - ratio)
        else:
            entropy_scale = 1.0
        return self.config.entropy_coeff * entropy_scale

    def _reinforce_update(self) -> None:
        """REINFORCE policy gradient update for QSNN."""
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
            self._run_reinforce_epoch(
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

        self._clamp_weights()

    def _run_reinforce_epoch(
        self,
        num_steps: int,
        advantages: torch.Tensor,
        old_log_probs_t: torch.Tensor,
        stats: _ReinforceUpdateStats,
    ) -> None:
        """Run a single REINFORCE gradient epoch with PPO clipping."""
        self.refractory_hidden.fill(0)
        self.refractory_motor.fill(0)

        log_probs_list: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []

        for t in range(num_steps):
            features = self.episode_features[t]
            action_idx = self.episode_actions[t]

            if stats.num_epochs > 1 and stats.epoch == 0:
                timestep_cache: list[dict[str, list[float]]] = []
                motor_spikes = self._multi_timestep_differentiable_caching(
                    features,
                    timestep_cache,
                )
                self._cached_spike_probs.append(timestep_cache)
            elif stats.epoch > 0:
                motor_spikes = self._multi_timestep_differentiable_cached(
                    features,
                    self._cached_spike_probs[t],
                )
            else:
                motor_spikes = self._multi_timestep_differentiable(features)

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

        self.qsnn_optimizer.zero_grad()
        policy_loss.backward()

        params_list = [self.W_sh, self.W_hm, self.theta_hidden, self.theta_motor]
        grad_norm = torch.nn.utils.clip_grad_norm_(params_list, max_norm=SURROGATE_GRAD_CLIP)
        self.qsnn_optimizer.step()

        epoch_str = f"epoch={stats.epoch}/{stats.num_epochs}, " if stats.num_epochs > 1 else ""
        logger.debug(
            f"HybridQuantum QSNN REINFORCE: {epoch_str}"
            f"loss={policy_loss.item():.4f}, "
            f"entropy={mean_entropy.item():.4f}, "
            f"grad_norm={grad_norm.item():.4f}, "
            f"W_sh_norm={torch.norm(self.W_sh).item():.4f}, "
            f"W_hm_norm={torch.norm(self.W_hm).item():.4f}, "
            f"theta_h_norm={torch.norm(self.theta_hidden).item():.4f}, "
            f"theta_m_norm={torch.norm(self.theta_motor).item():.4f}",
        )

    def _clamp_weights(self) -> None:
        """Clamp QSNN weights after REINFORCE update."""
        with torch.no_grad():
            for p in [self.W_sh, self.W_hm, self.theta_hidden, self.theta_motor]:
                p.clamp_(-self.weight_clip, self.weight_clip)

            tm_norm = torch.norm(self.theta_motor)
            if tm_norm > self.config.theta_motor_max_norm:
                self.theta_motor.mul_(self.config.theta_motor_max_norm / tm_norm)

    # ──────────────────────────────────────────────────────────────────
    # Cortex PPO training
    # ──────────────────────────────────────────────────────────────────

    def _perform_ppo_update(self) -> None:
        """Perform PPO update using collected cortex experience."""
        if len(self.ppo_buffer) == 0:
            return

        # Get last value for GAE
        if self._has_pending_cortex_data:
            with torch.no_grad():
                sensory_t = torch.tensor(
                    self._pending_cortex_state,
                    dtype=torch.float32,
                    device=self.device,
                )
                last_value = self._cortex_value(sensory_t)
        else:
            last_value = torch.tensor(0.0, device=self.device)

        returns, advantages = self.ppo_buffer.compute_returns_and_advantages(
            last_value,
            self.gamma,
            self.config.gae_lambda,
        )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        num_updates = 0

        for _ in range(self.config.ppo_epochs):
            for batch in self.ppo_buffer.get_minibatches(
                self.config.ppo_minibatches,
                returns,
                advantages,
            ):
                # Cortex actor forward pass — use action_biases as logits
                # since reflex_logits are fixed (QSNN frozen in stage 2,
                # or detached). The cortex learns to produce action
                # probabilities that complement the (frozen) reflex.
                cortex_out = self.cortex_actor(batch["states"])
                logits = cortex_out[:, : self.num_motor]
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                new_log_probs = dist.log_prob(batch["actions"])
                entropy = dist.entropy().mean()

                # Values
                values = self.cortex_critic(batch["states"]).squeeze(-1)

                # PPO clipped surrogate
                ratio = torch.exp(new_log_probs - batch["old_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = (
                    torch.clamp(
                        ratio,
                        1 - self.config.ppo_clip_epsilon,
                        1 + self.config.ppo_clip_epsilon,
                    )
                    * batch["advantages"]
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (Huber for robustness)
                value_loss = nn.functional.smooth_l1_loss(values, batch["returns"])

                # Combined loss
                loss = policy_loss + 0.5 * value_loss - self.config.entropy_coeff * entropy

                # Optimize actor
                self.cortex_actor_optimizer.zero_grad()
                self.cortex_critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.cortex_actor.parameters(),
                    self.config.max_grad_norm,
                )
                nn.utils.clip_grad_norm_(
                    self.cortex_critic.parameters(),
                    self.config.max_grad_norm,
                )
                self.cortex_actor_optimizer.step()
                self.cortex_critic_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy.item()
                num_updates += 1

        if num_updates > 0:
            # Compute explained variance
            with torch.no_grad():
                all_states = torch.tensor(
                    np.array(self.ppo_buffer.states),
                    dtype=torch.float32,
                    device=self.device,
                )
                predicted = self.cortex_critic(all_states).squeeze(-1)
                target_var = returns.var()
                explained_var = (1.0 - (returns - predicted).var() / (target_var + 1e-8)).item()

            approx_kl = (
                0.5
                * (
                    torch.stack(self.ppo_buffer.log_probs).mean() - total_entropy_loss / num_updates
                ).item()
            )

            logger.info(
                f"HybridQuantum cortex PPO: "
                f"policy_loss={total_policy_loss / num_updates:.4f}, "
                f"value_loss={total_value_loss / num_updates:.4f}, "
                f"entropy={total_entropy_loss / num_updates:.4f}, "
                f"explained_var={explained_var:.4f}, "
                f"approx_kl={approx_kl:.4f}",
            )

    # ──────────────────────────────────────────────────────────────────
    # Session management
    # ──────────────────────────────────────────────────────────────────

    def set_session_id(self, session_id: str) -> None:
        """Set the session ID for weight saving.

        Called by the simulation runner to align weight export paths
        with the simulation's export directory.
        """
        self._session_id = session_id

    # ──────────────────────────────────────────────────────────────────
    # Weight persistence
    # ──────────────────────────────────────────────────────────────────

    def _save_qsnn_weights(self, session_id: str) -> None:
        """Save QSNN weights to disk."""
        export_dir = Path("exports") / session_id
        export_dir.mkdir(parents=True, exist_ok=True)
        save_path = export_dir / "qsnn_weights.pt"

        weights_dict = {
            "W_sh": self.W_sh.detach().cpu(),
            "W_hm": self.W_hm.detach().cpu(),
            "theta_hidden": self.theta_hidden.detach().cpu(),
            "theta_motor": self.theta_motor.detach().cpu(),
        }
        torch.save(weights_dict, save_path)
        logger.info(f"QSNN weights saved to {save_path}")

    def _load_qsnn_weights(self) -> None:
        """Load pre-trained QSNN weights from disk."""
        weights_path = self.config.qsnn_weights_path
        if weights_path is None:
            return

        path = Path(weights_path)
        if not path.exists():
            msg = f"QSNN weights file not found: {weights_path}"
            raise FileNotFoundError(msg)

        weights_dict = torch.load(path, weights_only=True)

        # Validate shapes
        expected_shapes = {
            "W_sh": (self.num_sensory, self.num_hidden),
            "W_hm": (self.num_hidden, self.num_motor),
            "theta_hidden": (self.num_hidden,),
            "theta_motor": (self.num_motor,),
        }
        for key, expected_shape in expected_shapes.items():
            if key not in weights_dict:
                msg = f"Missing key '{key}' in QSNN weights file"
                raise ValueError(msg)
            actual_shape = tuple(weights_dict[key].shape)
            if actual_shape != expected_shape:
                msg = f"Shape mismatch for '{key}': expected {expected_shape}, got {actual_shape}"
                raise ValueError(msg)

        # Assign weights
        with torch.no_grad():
            self.W_sh.copy_(weights_dict["W_sh"].to(self.device))
            self.W_hm.copy_(weights_dict["W_hm"].to(self.device))
            self.theta_hidden.copy_(weights_dict["theta_hidden"].to(self.device))
            self.theta_motor.copy_(weights_dict["theta_motor"].to(self.device))

        logger.info(f"QSNN weights loaded from {weights_path}")

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

        # Clear pending cortex data flag to prevent stale state leaking
        self._has_pending_cortex_data = False

        # Reset QSNN refractory states
        self.refractory_hidden.fill(0)
        self.refractory_motor.fill(0)

        self._step_count = 0

    # ──────────────────────────────────────────────────────────────────
    # Protocol methods
    # ──────────────────────────────────────────────────────────────────

    def update_memory(self, reward: float | None = None) -> None:
        """No-op for HybridQuantumBrain."""

    def prepare_episode(self) -> None:
        """Prepare for a new episode."""
        self._reset_episode()

    def post_process_episode(self, *, episode_success: bool | None = None) -> None:
        """Post-process the episode (no-op, learning happens in learn())."""

    def copy(self) -> HybridQuantumBrain:
        """Create a copy of this brain."""
        config_copy = HybridQuantumBrainConfig(
            **{**self.config.model_dump(), "seed": self.seed},
        )
        new_brain = HybridQuantumBrain(
            config=config_copy,
            num_actions=self.num_actions,
            device=DeviceType(self.device.type),
            action_set=self._action_set,
        )

        # Copy QSNN weights
        with torch.no_grad():
            new_brain.W_sh.copy_(self.W_sh)
            new_brain.W_hm.copy_(self.W_hm)
            new_brain.theta_hidden.copy_(self.theta_hidden)
            new_brain.theta_motor.copy_(self.theta_motor)

        # Copy cortex state dicts
        new_brain.cortex_actor.load_state_dict(self.cortex_actor.state_dict())
        new_brain.cortex_critic.load_state_dict(self.cortex_critic.state_dict())

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
