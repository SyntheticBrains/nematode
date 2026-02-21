"""
Hierarchical Hybrid Classical Brain Architecture (QSNN Ablation).

Classical ablation control for the HybridQuantum brain. Replaces the QSNN
reflex layer (QLIF quantum circuits) with a small classical MLP reflex of
comparable parameter count, keeping everything else identical: same cortex MLP,
same mode-gated fusion, same 3-stage curriculum, same REINFORCE + PPO training.

Architecture::

    Sensory Input (2-dim legacy)
           |
           v
    Classical Reflex MLP (nn.Sequential)
      2 -> 16 -> 4 (ReLU hidden)
      ~116 classical params, REINFORCE training
      Output: 4-dim reflex logits
           |
           v                    Sensory Input (multi-objective)
           |                          |
           v                          v
      +---------+            Classical Cortex (MLP)
      |  Fusion |<--------   sensory -> hidden -> 7
      +---------+            (4 action biases + 3 mode logits)
           |                  PPO training, ~5K actor params
           v
      final_logits = reflex_logits * reflex_trust + cortex_biases
           |
           v
      Action Selection (4 actions)

Three-stage curriculum (identical to HybridQuantum):
  1. Reflex MLP on foraging (REINFORCE)
  2. Freeze reflex, train cortex (PPO)
  3. Optional joint fine-tune

This brain exists solely to answer: "Is the QSNN quantum component necessary,
or does the hybrid architecture + curriculum explain the performance gains?"
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from pydantic import Field, field_validator
from torch import nn

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, ClassicalBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch._hybrid_common import (
    _CortexRolloutBuffer,
    _ReinforceUpdateStats,
    adaptive_entropy_coef,
    cortex_forward,
    cortex_value,
    exploration_schedule,
    fuse,
    get_cortex_lr,
    init_cortex_mlps,
    load_cortex_weights,
    normalize_reward,
    perform_ppo_update,
    preprocess_legacy,
    save_cortex_weights,
    update_cortex_learning_rates,
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

# Reflex MLP defaults
DEFAULT_REFLEX_HIDDEN_DIM = 16
DEFAULT_REFLEX_INPUT_DIM = 2  # legacy 2-feature mode
DEFAULT_NUM_MOTOR = 4
DEFAULT_LOGIT_SCALE = 5.0

# Reflex REINFORCE defaults
DEFAULT_REFLEX_LR = 0.01
DEFAULT_REFLEX_LR_DECAY_EPISODES = 400
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

# Reflex gradient clipping
REFLEX_GRAD_CLIP = 1.0

# Validation
MIN_REINFORCE_BATCH_SIZE = 2

# Training stages
STAGE_REFLEX_ONLY = 1
STAGE_CORTEX_ONLY = 2
STAGE_JOINT = 3


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────


class HybridClassicalBrainConfig(BrainConfig):
    """Configuration for the HybridClassicalBrain architecture.

    Classical ablation of HybridQuantum: replaces QSNN reflex with a small
    classical MLP reflex. Supports the same 3-stage curriculum and
    mode-gated fusion.

    Supports two modes for cortex input feature extraction:

    1. **Legacy mode** (default): Uses 2 features (gradient_strength, relative_angle)
       - Set ``sensory_modules=None`` (default)

    2. **Unified sensory mode**: Uses modular feature extraction from brain/modules.py
       - Set ``sensory_modules`` to a list of ModuleName values
       - Each module contributes 2 features [strength, angle]
    """

    # Reflex MLP params
    reflex_hidden_dim: int = Field(
        default=DEFAULT_REFLEX_HIDDEN_DIM,
        description="Hidden layer dimension for reflex MLP.",
    )
    logit_scale: float = Field(
        default=DEFAULT_LOGIT_SCALE,
        description="Scaling factor for converting reflex output to action logits.",
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
        description="Training stage: 1=reflex only, 2=cortex only (reflex frozen), 3=joint.",
    )

    # Reflex REINFORCE params
    reflex_lr: float = Field(
        default=DEFAULT_REFLEX_LR,
        description="Learning rate for reflex REINFORCE.",
    )
    reflex_lr_decay_episodes: int = Field(
        default=DEFAULT_REFLEX_LR_DECAY_EPISODES,
        description="Episodes over which reflex LR decays via cosine annealing.",
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
        description="Enable running reward normalization for reflex REINFORCE.",
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
        description="LR multiplier for reflex in stage 3 (10x lower).",
    )

    # Cortex LR scheduling (optional warmup + decay)
    cortex_lr_warmup_episodes: int = Field(
        default=0,
        description="Episodes to linearly increase cortex LR from warmup_start to base LR. "
        "0 = no warmup.",
    )
    cortex_lr_warmup_start: float | None = Field(
        default=None,
        description="Initial cortex LR during warmup. None = 0.1 * cortex_actor_lr.",
    )
    cortex_lr_decay_episodes: int | None = Field(
        default=None,
        description="Episodes after warmup to decay cortex LR to decay_end. None = no decay.",
    )
    cortex_lr_decay_end: float | None = Field(
        default=None,
        description="Final cortex LR after decay. None = 0.1 * cortex_actor_lr.",
    )

    # Weight persistence
    reflex_weights_path: str | None = Field(
        default=None,
        description="Path to pre-trained reflex weights (.pt file) for stage 2/3.",
    )
    cortex_weights_path: str | None = Field(
        default=None,
        description="Path to pre-trained cortex weights (.pt file) for stage 3.",
    )

    # Sensory modules
    sensory_modules: list[ModuleName] | None = Field(
        default=None,
        description="List of sensory modules for feature extraction (None = legacy mode).",
    )

    @field_validator("training_stage")
    @classmethod
    def validate_training_stage(cls, v: int) -> int:
        """Validate training_stage is 1, 2, or 3."""
        if v not in (STAGE_REFLEX_ONLY, STAGE_CORTEX_ONLY, STAGE_JOINT):
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

    @field_validator("reflex_hidden_dim")
    @classmethod
    def validate_reflex_hidden_dim(cls, v: int) -> int:
        """Validate reflex_hidden_dim >= 1."""
        if v < 1:
            msg = f"reflex_hidden_dim must be >= 1, got {v}"
            raise ValueError(msg)
        return v


# ──────────────────────────────────────────────────────────────────────
# Brain Implementation
# ──────────────────────────────────────────────────────────────────────


class HybridClassicalBrain(ClassicalBrain):
    """Hierarchical hybrid classical brain — QSNN ablation control.

    Replaces the QSNN reflex layer from HybridQuantumBrain with a small
    classical MLP reflex of comparable parameter count (~116 vs 92 params).
    Everything else is identical: cortex MLP, mode-gated fusion, 3-stage
    curriculum, REINFORCE + PPO training.

    Purpose: isolate whether the QSNN quantum component or the hybrid
    architecture/curriculum is responsible for the observed performance gains.
    """

    def __init__(  # noqa: PLR0915
        self,
        config: HybridClassicalBrainConfig,
        num_actions: int = 4,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] | None = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.num_actions = num_actions
        self.num_motor = DEFAULT_NUM_MOTOR
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
        logger.info(f"HybridClassicalBrain using seed: {self.seed}")

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
        self.gamma = config.gamma
        self.training_stage = config.training_stage
        self.num_modes = config.num_modes

        # Initialize classical reflex MLP
        self._init_reflex()

        # Initialize cortex actor + critic
        self._init_cortex()

        # Initialize optimizers
        self._init_optimizers()

        # Load pre-trained reflex weights if specified
        if config.reflex_weights_path is not None:
            self._load_reflex_weights()
        elif self.training_stage >= STAGE_CORTEX_ONLY:
            logger.warning(
                "Stage >= 2 but no reflex_weights_path specified. "
                "Reflex will use random initialization.",
            )

        # Load pre-trained cortex weights if specified (stage 3)
        if config.cortex_weights_path is not None:
            self._load_cortex_weights()

        # Freeze reflex in stage 2
        if self.training_stage == STAGE_CORTEX_ONLY:
            self._freeze_reflex()

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

        # Running reward normalization (REINFORCE)
        self.reward_running_mean: float = 0.0
        self.reward_running_var: float = 1.0

        # Fusion tracking
        self._episode_reflex_trusts: list[float] = []
        self._episode_mode_probs: list[list[float]] = []

        # Current action probabilities
        self.current_probabilities: np.ndarray | None = None
        self.training = True

        reflex_params = sum(p.numel() for p in self.reflex_mlp.parameters())
        cortex_actor_params = sum(p.numel() for p in self.cortex_actor.parameters())
        cortex_critic_params = sum(p.numel() for p in self.cortex_critic.parameters())

        logger.info(
            f"HybridClassicalBrain initialized: "
            f"reflex hidden={config.reflex_hidden_dim}, "
            f"cortex hidden={config.cortex_hidden_dim}x{config.cortex_num_layers}, "
            f"modes={self.num_modes}, stage={self.training_stage}, "
            f"params: reflex={reflex_params}, "
            f"cortex_actor={cortex_actor_params}, "
            f"cortex_critic={cortex_critic_params}",
        )

    def _init_reflex(self) -> None:
        """Initialize the classical reflex MLP.

        Architecture: Linear(2 -> hidden) -> ReLU -> Linear(hidden -> 4)
        Matches QSNN's legacy 2-feature input and 4 motor outputs.
        """
        self.reflex_mlp = nn.Sequential(
            nn.Linear(DEFAULT_REFLEX_INPUT_DIM, self.config.reflex_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.reflex_hidden_dim, self.num_motor),
        ).to(self.device)

        # Orthogonal initialization (same as cortex)
        def init_weights(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        self.reflex_mlp.apply(init_weights)

    def _init_cortex(self) -> None:
        """Initialize cortex actor and critic MLPs."""
        self.cortex_actor, self.cortex_critic = init_cortex_mlps(
            input_dim=self.input_dim,
            cortex_hidden_dim=self.config.cortex_hidden_dim,
            cortex_num_layers=self.config.cortex_num_layers,
            num_motor=self.num_motor,
            num_modes=self.num_modes,
            device=self.device,
        )

    def _init_optimizers(self) -> None:
        """Initialize separate optimizers for reflex and cortex."""
        reflex_lr = self.config.reflex_lr
        if self.training_stage == STAGE_JOINT:
            reflex_lr *= self.config.joint_finetune_lr_factor

        self.reflex_optimizer = torch.optim.Adam(
            self.reflex_mlp.parameters(),
            lr=reflex_lr,
        )
        self.reflex_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.reflex_optimizer,
            T_max=self.config.reflex_lr_decay_episodes,
            eta_min=reflex_lr * self.config.lr_min_factor,
        )

        # Cortex LR scheduling state
        self.cortex_base_lr = self.config.cortex_actor_lr
        self.cortex_lr_warmup_episodes = self.config.cortex_lr_warmup_episodes
        self.cortex_lr_warmup_start = self.config.cortex_lr_warmup_start or (
            0.1 * self.config.cortex_actor_lr
        )
        self.cortex_lr_decay_episodes = self.config.cortex_lr_decay_episodes
        self.cortex_lr_decay_end = self.config.cortex_lr_decay_end or (
            0.1 * self.config.cortex_actor_lr
        )
        self.cortex_lr_scheduling_enabled = (
            self.cortex_lr_warmup_episodes > 0 or self.cortex_lr_decay_episodes is not None
        )

        # Start at warmup LR if warmup is enabled
        initial_actor_lr = (
            self.cortex_lr_warmup_start
            if self.cortex_lr_warmup_episodes > 0
            else self.config.cortex_actor_lr
        )
        initial_critic_lr = (
            self.cortex_lr_warmup_start
            if self.cortex_lr_warmup_episodes > 0
            else self.config.cortex_critic_lr
        )

        self.cortex_actor_optimizer = torch.optim.Adam(
            self.cortex_actor.parameters(),
            lr=initial_actor_lr,
        )
        self.cortex_critic_optimizer = torch.optim.Adam(
            self.cortex_critic.parameters(),
            lr=initial_critic_lr,
        )

        if self.cortex_lr_scheduling_enabled:
            schedule_desc = []
            if self.cortex_lr_warmup_episodes > 0:
                schedule_desc.append(
                    f"warmup {self.cortex_lr_warmup_start:.6f} -> {self.cortex_base_lr:.6f} "
                    f"over {self.cortex_lr_warmup_episodes} episodes",
                )
            if self.cortex_lr_decay_episodes is not None:
                schedule_desc.append(
                    f"decay {self.cortex_base_lr:.6f} -> {self.cortex_lr_decay_end:.6f} "
                    f"over {self.cortex_lr_decay_episodes} episodes",
                )
            logger.info(f"Cortex LR scheduling enabled: {', then '.join(schedule_desc)}")

    def _freeze_reflex(self) -> None:
        """Freeze reflex MLP weights (stage 2)."""
        for param in self.reflex_mlp.parameters():
            param.requires_grad_(False)  # noqa: FBT003
        logger.info("Reflex MLP weights frozen for stage 2 training")

    def _get_cortex_lr(self) -> float:
        """Get the current cortex learning rate based on episode count."""
        return get_cortex_lr(
            scheduling_enabled=self.cortex_lr_scheduling_enabled,
            episode_count=self._episode_count,
            base_lr=self.cortex_base_lr,
            warmup_episodes=self.cortex_lr_warmup_episodes,
            warmup_start=self.cortex_lr_warmup_start,
            decay_episodes=self.cortex_lr_decay_episodes,
            decay_end=self.cortex_lr_decay_end,
        )

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
            cortex_actor_optimizer=self.cortex_actor_optimizer,
            cortex_critic_optimizer=self.cortex_critic_optimizer,
        )

    def _init_episode_state(self) -> None:
        """Initialize REINFORCE episode tracking state."""
        self.episode_rewards: list[float] = []
        self.episode_actions: list[int] = []
        self.episode_features: list[np.ndarray] = []
        self.episode_old_log_probs: list[float] = []
        self.baseline = 0.0
        self.baseline_alpha = 0.05

    # ──────────────────────────────────────────────────────────────────
    # Reflex forward pass
    # ──────────────────────────────────────────────────────────────────

    def _reflex_forward(self, features: np.ndarray) -> np.ndarray:
        """Run the classical reflex MLP and return motor logits (non-differentiable).

        Applies sigmoid to map raw MLP output to [0, 1] range (matching QSNN's
        spike probability output), then converts to logits via the same
        ``(prob - 0.5) * logit_scale`` formula used by HybridQuantum.
        """
        with torch.no_grad():
            features_t = torch.tensor(features, dtype=torch.float32, device=self.device)
            raw_output = self.reflex_mlp(features_t)
            motor_probs = torch.sigmoid(raw_output)
        return motor_probs.cpu().numpy()

    def _reflex_forward_differentiable(self, features: np.ndarray) -> torch.Tensor:
        """Run the classical reflex MLP with gradient tracking.

        Returns motor probabilities (sigmoid-mapped) as a differentiable tensor.
        """
        features_t = torch.tensor(features, dtype=torch.float32, device=self.device)
        raw_output = self.reflex_mlp(features_t)
        return torch.sigmoid(raw_output)

    # ──────────────────────────────────────────────────────────────────
    # Cortex forward pass
    # ──────────────────────────────────────────────────────────────────

    def _cortex_forward(self, sensory_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run cortex actor forward pass, returning action biases and mode logits."""
        return cortex_forward(self.cortex_actor, sensory_input, self.num_motor)

    def _cortex_value(self, sensory_input: torch.Tensor) -> torch.Tensor:
        """Get critic value estimate from sensory input."""
        return cortex_value(self.cortex_critic, sensory_input)

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

        Returns (final_logits, reflex_trust, mode_probs).
        """
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
        """Preprocess BrainParams for cortex input.

        Uses unified sensory modules when configured (multi-objective),
        otherwise falls back to the same legacy 2-feature input as the reflex.
        """
        if self.sensory_modules is not None:
            return extract_classical_features(params, self.sensory_modules)
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

        # Reflex forward pass (always runs regardless of stage)
        motor_probs = self._reflex_forward(reflex_features)
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
            # Stage 2/3: cortex gets its own features (may include predator info)
            cortex_features = self._preprocess_cortex(params)
            sensory_t = torch.tensor(cortex_features, dtype=torch.float32, device=self.device)
            reflex_logits_t = torch.tensor(
                reflex_logits_np,
                dtype=torch.float32,
                device=self.device,
            )

            action_biases, mode_logits = self._cortex_forward(sensory_t)
            final_logits, reflex_trust, mode_probs = self._fuse(
                reflex_logits_t,
                action_biases,
                mode_logits,
            )

            # Track fusion diagnostics
            self._episode_reflex_trusts.append(reflex_trust)
            self._episode_mode_probs.append(mode_probs.detach().cpu().tolist())

            action_probs_t = torch.softmax(final_logits, dim=-1)
            action_probs = action_probs_t.detach().cpu().numpy()

            # Store PPO data for cortex training.
            cortex_logits = action_biases.detach()
            cortex_probs = torch.softmax(cortex_logits, dim=-1)
            cortex_log_probs = torch.log(cortex_probs + 1e-8)
            with torch.no_grad():
                value = self._cortex_value(sensory_t)
            self._pending_cortex_state = cortex_features
            self._pending_cortex_log_prob_dist = cortex_log_probs
            self._pending_cortex_value = value

        # Sample action
        action_probs = np.clip(action_probs, 1e-8, 1.0)
        action_probs = action_probs / action_probs.sum()
        action_idx = self.rng.choice(self.num_actions, p=action_probs)
        action_name = self.action_set[action_idx]

        # Store REINFORCE data (stage 1 and 3)
        if self.training_stage in (STAGE_REFLEX_ONLY, STAGE_JOINT):
            self.episode_features.append(reflex_features)
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

    def learn(  # noqa: C901, PLR0912
        self,
        params: BrainParams,  # noqa: ARG002
        reward: float,
        *,
        episode_done: bool = False,
    ) -> None:
        """Accumulate rewards and trigger stage-dependent training."""
        # Reward normalization for REINFORCE (stage 1 and 3)
        uses_reinforce = self.training_stage in (STAGE_REFLEX_ONLY, STAGE_JOINT)
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
            self.training_stage in (STAGE_REFLEX_ONLY, STAGE_JOINT)
            and window > 0
            and self._step_count % window == 0
            and not episode_done
        ):
            self._reinforce_update()
            self.episode_rewards.clear()
            self.episode_actions.clear()
            self.episode_features.clear()
            self.episode_old_log_probs.clear()

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
                f"HybridClassical episode complete: episode={self._episode_count}, "
                f"stage={self.training_stage}, steps={self._step_count}, "
                f"total_reward={total_reward:.4f}",
            )

            # Final REINFORCE update (stage 1 and 3)
            if self.training_stage in (STAGE_REFLEX_ONLY, STAGE_JOINT):
                self._reinforce_update()

            # Log fusion diagnostics (stage 2/3)
            if self.training_stage >= STAGE_CORTEX_ONLY and self._episode_reflex_trusts:
                mean_trust = np.mean(self._episode_reflex_trusts)
                mode_means = np.mean(self._episode_mode_probs, axis=0).tolist()
                logger.info(
                    f"HybridClassical fusion: reflex_trust_mean={mean_trust:.4f}, "
                    f"mode_dist={[f'{m:.3f}' for m in mode_means]}",
                )

            self._episode_count += 1

            # Step reflex LR scheduler (stage 1 and 3)
            if self.training_stage in (STAGE_REFLEX_ONLY, STAGE_JOINT):
                self.reflex_scheduler.step()

            # Step cortex LR scheduler (stage 2 and 3)
            if self.training_stage >= STAGE_CORTEX_ONLY:
                self._update_cortex_learning_rate()

            # Auto-save reflex weights when reflex is being trained
            if self.training_stage in (STAGE_REFLEX_ONLY, STAGE_JOINT):
                self._save_reflex_weights(self._session_id)

            # Auto-save cortex weights when cortex is being trained
            if self.training_stage >= STAGE_CORTEX_ONLY:
                self._save_cortex_weights(self._session_id)

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

    def _reinforce_update(self) -> None:
        """REINFORCE policy gradient update for reflex MLP."""
        if len(self.episode_rewards) == 0:
            return
        num_steps = len(self.episode_rewards)
        if num_steps < MIN_REINFORCE_BATCH_SIZE:
            self.episode_rewards.clear()
            self.episode_features.clear()
            self.episode_actions.clear()
            self.episode_old_log_probs.clear()
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

    def _run_reinforce_epoch(
        self,
        num_steps: int,
        advantages: torch.Tensor,
        old_log_probs_t: torch.Tensor,
        stats: _ReinforceUpdateStats,
    ) -> None:
        """Run a single REINFORCE gradient epoch with PPO clipping."""
        log_probs_list: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []

        for t in range(num_steps):
            features = self.episode_features[t]
            action_idx = self.episode_actions[t]

            # Classical reflex forward pass with gradients
            motor_probs = self._reflex_forward_differentiable(features)
            motor_clipped = torch.clamp(motor_probs, 1e-8, 1.0 - 1e-8)
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

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.reflex_mlp.parameters(),
            max_norm=REFLEX_GRAD_CLIP,
        )
        self.reflex_optimizer.step()

        epoch_str = f"epoch={stats.epoch}/{stats.num_epochs}, " if stats.num_epochs > 1 else ""
        logger.debug(
            f"HybridClassical reflex REINFORCE: {epoch_str}"
            f"loss={policy_loss.item():.4f}, "
            f"entropy={mean_entropy.item():.4f}, "
            f"grad_norm={grad_norm.item():.4f}",
        )

    # ──────────────────────────────────────────────────────────────────
    # Cortex PPO training
    # ──────────────────────────────────────────────────────────────────

    def _perform_ppo_update(self) -> None:
        """Perform PPO update using collected cortex experience."""
        perform_ppo_update(
            ppo_buffer=self.ppo_buffer,
            cortex_actor=self.cortex_actor,
            cortex_critic=self.cortex_critic,
            cortex_actor_optimizer=self.cortex_actor_optimizer,
            cortex_critic_optimizer=self.cortex_critic_optimizer,
            num_motor=self.num_motor,
            gamma=self.gamma,
            gae_lambda=self.config.gae_lambda,
            ppo_epochs=self.config.ppo_epochs,
            ppo_minibatches=self.config.ppo_minibatches,
            ppo_clip_epsilon=self.config.ppo_clip_epsilon,
            entropy_coeff=self.config.entropy_coeff,
            max_grad_norm=self.config.max_grad_norm,
            device=self.device,
            has_pending_cortex_data=self._has_pending_cortex_data,
            pending_cortex_state=getattr(self, "_pending_cortex_state", None),
            brain_name="HybridClassical",
        )

    # ──────────────────────────────────────────────────────────────────
    # Session management
    # ──────────────────────────────────────────────────────────────────

    def set_session_id(self, session_id: str) -> None:
        """Set the session ID for weight saving."""
        self._session_id = session_id

    # ──────────────────────────────────────────────────────────────────
    # Weight persistence
    # ──────────────────────────────────────────────────────────────────

    def _save_reflex_weights(self, session_id: str) -> None:
        """Save reflex MLP weights to disk."""
        export_dir = Path("exports") / session_id
        export_dir.mkdir(parents=True, exist_ok=True)
        save_path = export_dir / "reflex_weights.pt"
        torch.save(self.reflex_mlp.state_dict(), save_path)
        logger.info(f"Reflex weights saved to {save_path}")

    def _load_reflex_weights(self) -> None:
        """Load pre-trained reflex MLP weights from disk."""
        weights_path = self.config.reflex_weights_path
        if weights_path is None:
            return

        path = Path(weights_path)
        if not path.exists():
            msg = f"Reflex weights file not found: {weights_path}"
            raise FileNotFoundError(msg)

        state_dict = torch.load(path, weights_only=True)
        self.reflex_mlp.load_state_dict(state_dict)
        logger.info(f"Reflex weights loaded from {weights_path}")

    def _save_cortex_weights(self, session_id: str) -> None:
        """Save cortex actor and critic weights to disk."""
        save_cortex_weights(
            self.cortex_actor,
            self.cortex_critic,
            session_id,
            brain_name="HybridClassical",
        )

    def _load_cortex_weights(self) -> None:
        """Load pre-trained cortex weights from disk."""
        weights_path = self.config.cortex_weights_path
        if weights_path is None:
            return
        load_cortex_weights(
            self.cortex_actor,
            self.cortex_critic,
            weights_path,
            brain_name="HybridClassical",
        )

    # ──────────────────────────────────────────────────────────────────
    # Episode management
    # ──────────────────────────────────────────────────────────────────

    def _reset_episode(self) -> None:
        """Reset episode state."""
        self.episode_rewards.clear()
        self.episode_actions.clear()
        self.episode_features.clear()
        self.episode_old_log_probs.clear()
        self._episode_reflex_trusts.clear()
        self._episode_mode_probs.clear()

        # Clear pending cortex data flag to prevent stale state leaking
        self._has_pending_cortex_data = False

        self._step_count = 0

    # ──────────────────────────────────────────────────────────────────
    # Protocol methods
    # ──────────────────────────────────────────────────────────────────

    def update_memory(self, reward: float | None = None) -> None:
        """No-op for HybridClassicalBrain."""

    def prepare_episode(self) -> None:
        """Prepare for a new episode."""
        self._reset_episode()

    def post_process_episode(self, *, episode_success: bool | None = None) -> None:
        """Post-process the episode (no-op, learning happens in learn())."""

    def copy(self) -> HybridClassicalBrain:
        """Create a copy of this brain."""
        config_copy = HybridClassicalBrainConfig(
            **{**self.config.model_dump(), "seed": self.seed},
        )
        new_brain = HybridClassicalBrain(
            config=config_copy,
            num_actions=self.num_actions,
            device=DeviceType(self.device.type),
            action_set=self._action_set,
        )

        # Copy reflex weights
        new_brain.reflex_mlp.load_state_dict(self.reflex_mlp.state_dict())

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
