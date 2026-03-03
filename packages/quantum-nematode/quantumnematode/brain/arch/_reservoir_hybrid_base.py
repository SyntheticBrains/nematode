"""
Reservoir Hybrid Base — shared PPO infrastructure for reservoir-based brains.

Provides ReservoirHybridBase, an abstract base class for brain architectures that
combine a fixed (non-trainable) reservoir with a PPO-trained classical actor-critic
readout. Subclasses implement reservoir-specific logic via two abstract methods:

- ``_get_reservoir_features(sensory_features) -> np.ndarray``
- ``_compute_feature_dim() -> int``

Concrete subclasses:
- QRHBrain: Quantum reservoir (Qiskit statevector, C. elegans topology)
- CRHBrain: Classical reservoir (Echo State Network)

All PPO training, action selection, rollout buffering, LR scheduling, episode
tracking, and brain protocol methods live here.
"""

from __future__ import annotations

import abc
from copy import deepcopy
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from collections.abc import Iterator

import numpy as np
import torch
from pydantic import Field
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
# Default Hyperparameters (shared across all reservoir hybrid brains)
# =============================================================================

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
MIN_READOUT_HIDDEN_DIM = 1


# =============================================================================
# Configuration
# =============================================================================


class ReservoirHybridBaseConfig(BrainConfig):
    """Shared configuration for reservoir hybrid brains (QRH, CRH).

    Contains all PPO readout, LR scheduling, and sensory module config fields.
    Subclass configs add reservoir-specific fields.
    """

    # Readout architecture
    readout_hidden_dim: int = Field(
        default=DEFAULT_READOUT_HIDDEN_DIM,
        description="Hidden units per layer in actor/critic MLPs.",
    )
    readout_num_layers: int = Field(
        default=DEFAULT_READOUT_NUM_LAYERS,
        description="Number of hidden layers in actor/critic MLPs.",
    )

    # Optimizer learning rates
    actor_lr: float = Field(
        default=DEFAULT_ACTOR_LR,
        description="Learning rate for optimizer.",
    )
    critic_lr: float = Field(
        default=DEFAULT_CRITIC_LR,
        description="Learning rate for critic (unused with combined optimizer, "
        "kept for config compat).",
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

    # Learning rate scheduling (optional)
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

    # Entropy coefficient decay (optional)
    entropy_coeff_end: float | None = Field(
        default=None,
        description="Final entropy coefficient after decay (None = no decay).",
    )
    entropy_decay_episodes: int | None = Field(
        default=None,
        description="Episodes over which entropy_coeff linearly decays to entropy_coeff_end.",
    )

    # Unified sensory feature extraction
    sensory_modules: list[ModuleName] | None = Field(
        default=None,
        description="Sensory modules for feature extraction (None = legacy mode).",
    )


# =============================================================================
# Rollout Buffer
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
# ReservoirHybridBase
# =============================================================================


class ReservoirHybridBase(ClassicalBrain):
    """Abstract base class for reservoir hybrid brain architectures.

    Subclasses must:
    - Set ``_brain_name`` class attribute (e.g., ``"QRH"``, ``"CRH"``)
    - Implement ``_get_reservoir_features(sensory_features) -> np.ndarray``
    - Implement ``_compute_feature_dim() -> int``
    - Implement ``_create_copy_instance(config) -> Self``
    """

    _brain_name: str = "ReservoirHybrid"  # Override in subclass

    @abc.abstractmethod
    def _get_reservoir_features(self, sensory_features: np.ndarray) -> np.ndarray:
        """Transform sensory features through the reservoir.

        Parameters
        ----------
        sensory_features : np.ndarray
            Preprocessed sensory input.

        Returns
        -------
        np.ndarray
            Reservoir feature vector for the readout network.
        """

    @abc.abstractmethod
    def _compute_feature_dim(self) -> int:
        """Return the reservoir feature vector dimension."""

    @abc.abstractmethod
    def _create_copy_instance(
        self,
        config: ReservoirHybridBaseConfig,
    ) -> Self:
        """Construct a fresh instance for the copy() method.

        The base class ``copy()`` calls this to build a new instance, then
        deep-copies the readout weights into it.
        """

    def __init__(  # noqa: PLR0915
        self,
        config: ReservoirHybridBaseConfig,
        feature_dim: int,
        num_actions: int = 4,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] | None = None,
    ) -> None:
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
        logger.info(f"{self._brain_name}Brain using seed: {self.seed}")

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

        # Store feature dimension (computed by subclass before calling super().__init__)
        self.feature_dim = feature_dim

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

        # Feature normalization
        self.feature_norm = nn.LayerNorm(self.feature_dim).to(self.device)

        # Single combined optimizer
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
            logger.info(f"{self._brain_name} LR scheduling enabled:")
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
        self.entropy_coeff_end = config.entropy_coeff_end
        self.entropy_decay_episodes = config.entropy_decay_episodes
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
        self._deferred_ppo_update: bool = False

        # Episode tracking
        self._episode_count = 0

    # =========================================================================
    # Preprocessing
    # =========================================================================

    def preprocess(self, params: BrainParams) -> np.ndarray:
        """Preprocess BrainParams to extract features for the reservoir.

        Two modes:
        1. **Unified sensory mode** (when sensory_modules is set):
           Uses extract_classical_features() which outputs semantic-preserving ranges.
        2. **Legacy mode** (default):
           Computes gradient strength [0, 1] and relative angle [-1, 1].
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

    def run_brain(
        self,
        params: BrainParams,
        reward: float | None = None,  # noqa: ARG002
        input_data: list[float] | None = None,  # noqa: ARG002
        *,
        top_only: bool,  # noqa: ARG002
        top_randomize: bool,  # noqa: ARG002
    ) -> list[ActionData]:
        """Run the brain and select an action.

        Pipeline: preprocess -> reservoir -> features -> actor -> sample action.
        """
        # Preprocess sensory input
        sensory_features = self.preprocess(params)

        # Run through reservoir and extract features
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
                f"{self._brain_name} step {self.buffer.position}: "
                f"features=[{feat_min:.3f}, {feat_max:.3f}], "
                f"probs={probs_np}, logits=[{logits_np.min():.3f}, {logits_np.max():.3f}], "
                f"value={value.item():.4f}",
            )

        # If a mid-episode buffer flush was deferred, perform the PPO update now that
        # V(s_{t+1}) is available from this forward pass, then clear the buffer.
        if self._deferred_ppo_update:
            logger.debug(
                f"{self._brain_name} executing deferred PPO update with correct "
                f"V(s_{{t+1}})={value.item():.4f}",
            )
            self._perform_ppo_update(bootstrap_value=value)
            self.buffer.reset()
            self._deferred_ppo_update = False

        # Store for PPO buffer (committed when reward arrives in learn())
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

    # =========================================================================
    # Learning
    # =========================================================================

    def learn(
        self,
        params: BrainParams,  # noqa: ARG002
        reward: float,
        *,
        episode_done: bool = False,
    ) -> None:
        """Add experience to buffer and perform PPO update when ready."""
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

        # Trigger PPO update when buffer is full or episode ends with enough data.
        # Mid-episode buffer flushes (is_full and not episode_done) are deferred to the
        # next run_brain() call so that V(s_{t+1}) — the correct GAE bootstrap value —
        # is available from the subsequent forward pass rather than using V(s_t).
        if episode_done and len(self.buffer) >= self.ppo_minibatches:
            logger.debug(
                f"{self._brain_name} PPO update triggered (episode done): "
                f"buffer={len(self.buffer)}/{self.buffer.buffer_size}",
            )
            self._perform_ppo_update()
            self.buffer.reset()
        elif self.buffer.is_full():
            logger.debug(
                f"{self._brain_name} buffer full mid-episode — deferring PPO update "
                f"until next run_brain() to obtain correct V(s_{{t+1}}) bootstrap",
            )
            self._deferred_ppo_update = True

        # Store for history
        self.history_data.rewards.append(reward)

    def _get_current_lr(self) -> float:
        """Get the current learning rate based on episode count."""
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

    def _get_current_entropy_coeff(self) -> float:
        """Get current entropy coefficient with optional linear decay schedule."""
        if self.entropy_decay_episodes is None or self.entropy_coeff_end is None:
            return self.entropy_coeff
        if self._episode_count >= self.entropy_decay_episodes:
            return self.entropy_coeff_end
        progress = self._episode_count / self.entropy_decay_episodes
        return self.entropy_coeff + progress * (self.entropy_coeff_end - self.entropy_coeff)

    def _perform_ppo_update(
        self,
        bootstrap_value: torch.Tensor | None = None,
    ) -> None:
        """Perform PPO update using collected experience.

        Parameters
        ----------
        bootstrap_value : torch.Tensor | None
            V(s_{t+1}) to use for GAE bootstrapping at non-terminal buffer boundaries.
            When ``None``, falls back to ``self.last_value`` (used for terminal flushes
            where the bootstrap is zeroed out by ``next_non_terminal`` anyway).
        """
        if len(self.buffer) == 0:
            return

        # Get last value for GAE computation.
        # ``bootstrap_value`` is V(s_{t+1}), provided by the deferred-update path in
        # run_brain(). For terminal episode flushes the done flag zeros out the bootstrap,
        # so self.last_value (= V(s_t)) is equally correct there.
        last_value = (
            bootstrap_value
            if bootstrap_value is not None
            else (
                self.last_value
                if self.last_value is not None
                else torch.tensor([0.0], device=self.device)
            )
        )

        # Compute returns and advantages
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value,
            self.gamma,
            self.gae_lambda,
        )

        # Skip PPO update if buffer is too small
        min_buffer_size = min(64, self.buffer.buffer_size // 2)
        if len(self.buffer) < min_buffer_size:
            logger.debug(
                f"{self._brain_name} skipping PPO update: "
                f"buffer size {len(self.buffer)} < {min_buffer_size}",
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

                # Combined loss
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
            self.latest_data.loss = avg_loss
            self.history_data.losses.append(avg_loss)

            logger.debug(
                f"{self._brain_name} PPO update: "
                f"policy_loss={total_policy_loss / num_updates:.4f}, "
                f"value_loss={total_value_loss / num_updates:.4f}, "
                f"entropy={total_entropy_loss / num_updates:.4f}, "
                f"entropy_coeff={self._get_current_entropy_coeff():.5f}",
            )

    # =========================================================================
    # Brain Protocol Methods
    # =========================================================================

    def update_memory(self, reward: float | None = None) -> None:
        """No-op for reservoir hybrid brains."""

    def prepare_episode(self) -> None:
        """Prepare for a new episode by clearing pending state."""
        self._pending_state = None
        self._pending_action = None
        self._pending_log_prob = None
        self._pending_value = None
        self.last_value = None
        self._deferred_ppo_update = False

    def post_process_episode(
        self,
        *,
        episode_success: bool | None = None,  # noqa: ARG002
    ) -> None:
        """Post-process after each episode."""
        self._episode_count += 1
        self._update_learning_rate()

        # Clear pending state to prevent cross-episode contamination
        self._pending_state = None
        self._pending_action = None
        self._pending_log_prob = None
        self._pending_value = None

    def copy(self) -> Self:
        """Create an independent copy of this brain.

        Uses construct-then-copy-weights pattern:
        1. Serialize config via model_dump()
        2. Call subclass _create_copy_instance() to build fresh instance
        3. Deep-copy actor/critic/feature_norm/optimizer state dicts
        4. Copy _episode_count
        """
        config_copy = type(self.config)(
            **{**self.config.model_dump(), "seed": self.seed},
        )

        new_brain = self._create_copy_instance(config_copy)

        # Copy network weights (independent)
        new_brain.actor.load_state_dict(deepcopy(self.actor.state_dict()))
        new_brain.critic.load_state_dict(deepcopy(self.critic.state_dict()))
        new_brain.feature_norm.load_state_dict(deepcopy(self.feature_norm.state_dict()))

        # Copy optimizer state
        new_brain.optimizer.load_state_dict(deepcopy(self.optimizer.state_dict()))

        # Preserve episode counter
        new_brain._episode_count = self._episode_count  # noqa: SLF001

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
