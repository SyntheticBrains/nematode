"""
Proximal Policy Optimization (PPO) Brain Architecture.

This architecture implements PPO, a state-of-the-art policy gradient algorithm that uses
clipped surrogate objectives for stable learning.

Key Features:
- **Actor-Critic**: Separate networks for policy (actor) and value estimation (critic)
- **Clipped Surrogate Objective**: Prevents large policy updates for stable training
- **Generalized Advantage Estimation (GAE)**: Reduces variance in advantage computation
- **Rollout Buffer**: Collects experience for batch updates
- **Multiple Epochs**: Performs multiple gradient steps per rollout for sample efficiency

Architecture:
- Input: State features from configurable sensory modules
- Actor: MLP producing action probabilities via softmax
- Critic: MLP producing value estimate (single scalar)
- Output: Action selection from categorical distribution

The MLP PPO brain learns by:
1. Collecting rollout_buffer_size steps of experience
2. Computing GAE advantages using value estimates
3. Performing num_epochs of minibatch updates
4. Using clipped surrogate objective to constrain policy changes
5. Jointly optimizing policy loss, value loss, and entropy bonus

References
----------
- Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import nn, optim

if TYPE_CHECKING:
    from quantumnematode.brain.weights import WeightComponent
    from quantumnematode.initializers._initializer import ParameterInitializer

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, ClassicalBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch._ppo_buffer import RolloutBuffer
from quantumnematode.brain.arch.dtypes import BrainConfig, DeviceType
from quantumnematode.brain.modules import (
    ModuleName,
    extract_classical_features,
    get_classical_feature_dimension,
)
from quantumnematode.logging_config import logger
from quantumnematode.utils.seeding import ensure_seed, get_rng, set_global_seed

# Default hyperparameters
DEFAULT_ACTOR_HIDDEN_DIM = 64
DEFAULT_CRITIC_HIDDEN_DIM = 64
DEFAULT_NUM_HIDDEN_LAYERS = 2
DEFAULT_LEARNING_RATE = 0.0003
DEFAULT_GAMMA = 0.99
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_CLIP_EPSILON = 0.2
DEFAULT_VALUE_LOSS_COEF = 0.5
DEFAULT_ENTROPY_COEF = 0.01
DEFAULT_NUM_EPOCHS = 4
DEFAULT_NUM_MINIBATCHES = 4
DEFAULT_ROLLOUT_BUFFER_SIZE = 2048
DEFAULT_MAX_GRAD_NORM = 0.5

EPISODE_LOG_INTERVAL = 25


class MLPPPOBrainConfig(BrainConfig):
    """Configuration for the MLPPPOBrain architecture.

    Uses modular feature extraction via sensory_modules (required).

    Each module contributes 2 features [strength, angle] in [0,1] and [-1,1].
    ``input_dim`` is auto-computed as ``len(sensory_modules) * 2``.

    Example config:
        >>> config = MLPPPOBrainConfig(
        ...     sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        ... )
        >>> # input_dim will be 4 (2 modules * 2 features each)
    """

    # Network architecture
    actor_hidden_dim: int = DEFAULT_ACTOR_HIDDEN_DIM
    critic_hidden_dim: int = DEFAULT_CRITIC_HIDDEN_DIM
    num_hidden_layers: int = DEFAULT_NUM_HIDDEN_LAYERS

    # Learning parameters
    learning_rate: float = DEFAULT_LEARNING_RATE
    gamma: float = DEFAULT_GAMMA
    gae_lambda: float = DEFAULT_GAE_LAMBDA

    # PPO-specific parameters
    clip_epsilon: float = DEFAULT_CLIP_EPSILON
    value_loss_coef: float = DEFAULT_VALUE_LOSS_COEF
    entropy_coef: float = DEFAULT_ENTROPY_COEF
    num_epochs: int = DEFAULT_NUM_EPOCHS
    num_minibatches: int = DEFAULT_NUM_MINIBATCHES
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM

    # Rollout buffer
    rollout_buffer_size: int = DEFAULT_ROLLOUT_BUFFER_SIZE

    # Sensory feature extraction (required)
    sensory_modules: list[ModuleName]

    # Learning rate scheduling (optional)
    # Supports warmup followed by optional decay for more stable early learning.
    # - lr_warmup_episodes: Episodes to linearly increase LR from lr_warmup_start to learning_rate
    # - lr_warmup_start: Initial LR during warmup (default: 10% of learning_rate)
    # - lr_decay_episodes: Episodes after warmup to decay LR to lr_decay_end (None = no decay)
    # - lr_decay_end: Final LR after decay (default: 10% of learning_rate)
    lr_warmup_episodes: int = 0  # 0 = no warmup
    lr_warmup_start: float | None = None  # None = 0.1 * learning_rate
    lr_decay_episodes: int | None = None  # None = no decay after warmup
    lr_decay_end: float | None = None  # None = 0.1 * learning_rate

    # Feature expansion for ablation experiments
    # - "none": raw sensory features only (default)
    # - "polynomial": raw + all pairwise products (x_i * x_j for i < j)
    # - "polynomial3": raw + pairwise + triple products (degree-3 polynomial)
    # - "random_projection": raw + fixed random projection (7 raw + 52 projected = 59 total)
    feature_expansion: Literal["none", "polynomial", "polynomial3", "random_projection"] = "none"
    feature_expansion_dim: int = 52  # number of projected features to ADD (total = raw + this)
    feature_expansion_seed: int = 42  # seed for reproducible random projection
    feature_gating: bool = False  # learnable sigmoid gate on expanded features


class MLPPPOBrain(ClassicalBrain):
    """
    Proximal Policy Optimization (PPO) brain architecture.

    Uses actor-critic networks with clipped surrogate objective for stable learning.
    This is a SOTA classical baseline for comparison with quantum approaches.
    """

    def __init__(  # noqa: PLR0913, PLR0915
        self,
        config: MLPPPOBrainConfig,
        input_dim: int | None = None,
        num_actions: int = 4,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] = DEFAULT_ACTIONS,
        *,
        parameter_initializer: ParameterInitializer | None = None,
    ) -> None:
        super().__init__()

        # Initialize seeding for reproducibility
        self.seed = ensure_seed(config.seed)
        self.rng = get_rng(self.seed)
        set_global_seed(self.seed)  # Set global numpy/torch seeds
        logger.info(f"MLPPPOBrain using seed: {self.seed}")

        # Store sensory modules for feature extraction
        self.sensory_modules = config.sensory_modules

        # Determine input dimension from sensory modules
        computed_dim = get_classical_feature_dimension(config.sensory_modules)
        if input_dim is not None and input_dim != computed_dim:
            logger.warning(
                f"input_dim={input_dim} overridden by sensory_modules (computed: {computed_dim})",
            )
        self.input_dim = computed_dim
        logger.info(
            f"Using classical feature extraction with modules: "
            f"{[m.value for m in config.sensory_modules]} "
            f"(input_dim={self.input_dim}, features=[strength, angle] per module)",
        )

        # Feature expansion for ablation experiments
        self._raw_input_dim = self.input_dim
        self.feature_expansion = config.feature_expansion
        self._feature_gating = config.feature_gating
        self._init_feature_expansion(config)

        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()
        self.num_actions = num_actions
        self.device = torch.device(device.to_torch_device_str())
        self._action_set = action_set

        # Store config
        self.config = config
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.clip_epsilon = config.clip_epsilon
        self.value_loss_coef = config.value_loss_coef
        self.entropy_coef = config.entropy_coef
        self.num_epochs = config.num_epochs
        self.num_minibatches = config.num_minibatches
        self.max_grad_norm = config.max_grad_norm

        # Store LR scheduling config
        self._init_lr_schedule(config)

        # Build networks
        self.actor = self._build_network(
            config.actor_hidden_dim,
            config.num_hidden_layers,
            output_dim=num_actions,
            input_dim_override=self.input_dim,
        ).to(self.device)

        self.critic = self._build_network(
            config.critic_hidden_dim,
            config.num_hidden_layers,
            output_dim=1,
            input_dim_override=self.input_dim,
        ).to(self.device)

        # Initialize parameters
        self._initialize_parameters(parameter_initializer)

        # Feature gating: learnable sigmoid gate on expanded features
        if self._feature_gating:
            if config.feature_expansion == "none":
                msg = "feature_gating requires feature_expansion != 'none' (no features to gate)"
                raise ValueError(msg)
            expansion_dim = self.input_dim - self._raw_input_dim
            self.gate_weights = nn.Parameter(
                torch.zeros(expansion_dim, device=self.device),
            )
            logger.info(f"Feature gating enabled on {expansion_dim} expanded features")

        # Single optimizer for all trainable components
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        if self._feature_gating:
            params.append(self.gate_weights)
        self.optimizer = optim.Adam(params, lr=config.learning_rate)

        # Rollout buffer (pass RNG for reproducible minibatch shuffling)
        self.buffer = RolloutBuffer(config.rollout_buffer_size, self.device, rng=self.rng)

        # State tracking
        self.training = True
        self.current_probabilities = None
        self.last_value: torch.Tensor | None = None
        self.pending_reward: float | None = None

        # Episode tracking
        self._episode_count = 0
        self._current_episode_rewards: list[float] = []

    def _init_feature_expansion(self, config: MLPPPOBrainConfig) -> None:
        """Initialize feature expansion for ablation experiments."""
        if config.feature_expansion == "polynomial":
            n = self._raw_input_dim
            poly_dim = n * (n - 1) // 2
            self.input_dim = n + poly_dim
            logger.info(
                f"Polynomial feature expansion: {n} raw + {poly_dim} pairwise = "
                f"{self.input_dim} total features",
            )
        elif config.feature_expansion == "polynomial3":
            n = self._raw_input_dim
            pair_dim = n * (n - 1) // 2
            triple_dim = n * (n - 1) * (n - 2) // 6
            self.input_dim = n + pair_dim + triple_dim
            logger.info(
                f"Degree-3 polynomial expansion: {n} raw + {pair_dim} pairwise + "
                f"{triple_dim} triple = {self.input_dim} total features",
            )
        elif config.feature_expansion == "random_projection":
            rng = np.random.default_rng(config.feature_expansion_seed)
            self._projection_matrix = rng.standard_normal(
                (self._raw_input_dim, config.feature_expansion_dim),
            ).astype(np.float32) / np.sqrt(self._raw_input_dim)
            self.input_dim = self._raw_input_dim + config.feature_expansion_dim
            logger.info(
                f"Random projection expansion: {self._raw_input_dim} raw + "
                f"{config.feature_expansion_dim} projected = {self.input_dim} total features",
            )

    def _init_lr_schedule(self, config: MLPPPOBrainConfig) -> None:
        """Initialize learning rate scheduling parameters."""
        self.base_lr = config.learning_rate
        self.lr_warmup_episodes = config.lr_warmup_episodes
        self.lr_warmup_start = config.lr_warmup_start or (0.1 * config.learning_rate)
        self.lr_decay_episodes = config.lr_decay_episodes
        self.lr_decay_end = config.lr_decay_end or (0.1 * config.learning_rate)
        self.lr_scheduling_enabled = (
            self.lr_warmup_episodes > 0 or self.lr_decay_episodes is not None
        )

        if self.lr_scheduling_enabled:
            schedule_desc = []
            if self.lr_warmup_episodes > 0:
                schedule_desc.append(
                    f"warmup {self.lr_warmup_start:.6f} -> {self.base_lr:.6f} "
                    f"over {self.lr_warmup_episodes} episodes",
                )
            if self.lr_decay_episodes is not None:
                schedule_desc.append(
                    f"decay {self.base_lr:.6f} -> {self.lr_decay_end:.6f} "
                    f"over {self.lr_decay_episodes} episodes",
                )
            logger.info(f"LR scheduling enabled: {', then '.join(schedule_desc)}")

    def _build_network(
        self,
        hidden_dim: int,
        num_hidden_layers: int,
        output_dim: int,
        input_dim_override: int | None = None,
    ) -> nn.Sequential:
        """Build an MLP network."""
        in_dim = input_dim_override if input_dim_override is not None else self.input_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)

    def _initialize_parameters(self, parameter_initializer: ParameterInitializer | None) -> None:
        """Initialize network parameters with orthogonal initialization."""
        param_count = 0

        def init_weights(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        self.actor.apply(init_weights)
        self.critic.apply(init_weights)

        # Count parameters
        for param in self.actor.parameters():
            param_count += param.numel()
        for param in self.critic.parameters():
            param_count += param.numel()

        logger.info(f"MLPPPOBrain initialized with {param_count:,} total parameters")
        if parameter_initializer is not None:
            logger.info("Custom parameter initializer provided but using orthogonal init")

    def preprocess(self, params: BrainParams) -> np.ndarray:
        """Preprocess BrainParams to extract features for the neural network.

        Uses extract_classical_features() which outputs semantic-preserving ranges:
        - strength: [0, 1] where 0 = no signal
        - angle: [-1, 1] where 0 = aligned with agent heading
        """
        raw_features = extract_classical_features(params, self.sensory_modules)
        return self._apply_feature_expansion(raw_features)

    def _apply_feature_expansion(self, raw_features: np.ndarray) -> np.ndarray:
        """Apply feature expansion for ablation experiments."""
        if self.feature_expansion == "polynomial":
            n = len(raw_features)
            pairs = [raw_features[i] * raw_features[j] for i in range(n) for j in range(i + 1, n)]
            expanded = np.concatenate([raw_features, np.array(pairs, dtype=np.float32)])
        elif self.feature_expansion == "polynomial3":
            n = len(raw_features)
            pairs = [raw_features[i] * raw_features[j] for i in range(n) for j in range(i + 1, n)]
            triples = [
                raw_features[i] * raw_features[j] * raw_features[k]
                for i in range(n)
                for j in range(i + 1, n)
                for k in range(j + 1, n)
            ]
            expanded = np.concatenate(
                [
                    raw_features,
                    np.array(pairs, dtype=np.float32),
                    np.array(triples, dtype=np.float32),
                ],
            )
        elif self.feature_expansion == "random_projection":
            projected = raw_features @ self._projection_matrix
            expanded = np.concatenate([raw_features, projected])
        else:
            return raw_features

        return expanded

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

    def _apply_torch_gating(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learnable sigmoid gating to expanded feature dimensions."""
        if not self._feature_gating:
            return x
        gate = torch.sigmoid(self.gate_weights)
        raw = x[..., : self._raw_input_dim]
        expanded = x[..., self._raw_input_dim :]
        return torch.cat([raw, expanded * gate], dim=-1)

    def forward_actor(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through actor network."""
        return self.actor(self._apply_torch_gating(x))

    def forward_critic(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through critic network."""
        return self.critic(self._apply_torch_gating(x))

    def get_action_and_value(
        self,
        state: np.ndarray,
        action: int | None = None,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value for a state.

        Args:
            state: Preprocessed state features
            action: If provided, compute log_prob for this action

        Returns
        -------
            Tuple of (action, log_prob, entropy, value)
        """
        x = torch.tensor(state, dtype=torch.float32, device=self.device)

        features = self._apply_torch_gating(x)

        # Get action logits and value
        logits = self.actor(features)
        value = self.critic(features)

        # Compute action distribution
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if action is None:
            action = int(dist.sample().item())

        log_prob = dist.log_prob(torch.tensor(action, device=self.device))
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def run_brain(
        self,
        params: BrainParams,
        reward: float | None = None,
        input_data: list[float] | None = None,  # noqa: ARG002
        *,
        top_only: bool,  # noqa: ARG002
        top_randomize: bool,  # noqa: ARG002
    ) -> list[ActionData]:
        """Run the actor network and select an action."""
        # Store pending reward from previous step
        if reward is not None:
            self.pending_reward = reward

        x = self.preprocess(params)

        # Get action and value
        action_idx, log_prob, _entropy, value = self.get_action_and_value(x)
        action_name = self.action_set[action_idx]

        # Store value for next step's buffer update
        self.last_value = value

        # Get probabilities for tracking
        features = self._apply_torch_gating(
            torch.tensor(x, dtype=torch.float32, device=self.device),
        )
        logits = self.actor(features)
        probs = torch.softmax(logits, dim=-1)
        probs_np = probs.detach().cpu().numpy()
        self.current_probabilities = probs_np

        # Store current step info for buffer (will be added when reward arrives)
        self._pending_state = x
        self._pending_action = action_idx
        self._pending_log_prob = log_prob
        self._pending_value = value

        # Update latest data
        self.latest_data.action = ActionData(
            state=action_name,
            action=action_name,
            probability=probs_np[action_idx],
        )

        # Update history
        self.history_data.actions.append(self.latest_data.action)
        self.history_data.probabilities.append(float(probs_np[action_idx]))

        return [ActionData(state=action_name, action=action_name, probability=probs_np[action_idx])]

    def learn(
        self,
        params: BrainParams,  # noqa: ARG002
        reward: float,
        *,
        episode_done: bool = False,
    ) -> None:
        """Add experience to buffer and update if buffer is full."""
        self._current_episode_rewards.append(reward)

        # Add to buffer if we have pending state
        if hasattr(self, "_pending_state"):
            self.buffer.add(
                state=self._pending_state,
                action=self._pending_action,
                log_prob=self._pending_log_prob,
                value=self._pending_value,
                reward=reward,
                done=episode_done,
            )

        # Check if buffer is full or episode ended with enough data
        if self.buffer.is_full() or (episode_done and len(self.buffer) >= self.num_minibatches):
            self._perform_ppo_update()
            self.buffer.reset()

        # Store for history
        self.history_data.rewards.append(reward)

    def _perform_ppo_update(self) -> None:
        """Perform PPO update using collected experience."""
        if len(self.buffer) == 0:
            return

        # Get last value for GAE computation
        if self.last_value is not None:
            last_value = self.last_value
        else:
            last_value = torch.tensor([0.0], device=self.device)

        # Compute returns and advantages
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value,
            self.gamma,
            self.gae_lambda,
        )

        self._perform_standard_ppo_update(returns, advantages)

    def _perform_standard_ppo_update(
        self,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> None:
        """Perform PPO update with shuffled minibatches."""
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        num_updates = 0

        for _ in range(self.num_epochs):
            for batch in self.buffer.get_minibatches(self.num_minibatches, returns, advantages):
                gated_states = self._apply_torch_gating(batch["states"])
                logits = self.actor(gated_states)
                values = self.critic(gated_states).squeeze(-1)

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
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                all_params = list(self.actor.parameters()) + list(self.critic.parameters())
                if self._feature_gating:
                    all_params.append(self.gate_weights)
                nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy.item()
                num_updates += 1

        if num_updates > 0:
            self.latest_data.loss = total_policy_loss / num_updates
            logger.debug(
                f"PPO update: policy_loss={total_policy_loss / num_updates:.4f}, "
                f"value_loss={total_value_loss / num_updates:.4f}, "
                f"entropy={total_entropy_loss / num_updates:.4f}",
            )

    def update_memory(self, reward: float | None = None) -> None:
        """No-op for MLPPPOBrain."""

    def prepare_episode(self) -> None:
        """Prepare for a new episode."""

    def post_process_episode(
        self,
        *,
        episode_success: bool | None = None,  # noqa: ARG002
    ) -> None:
        """Post-process after each episode."""
        self._episode_count += 1

        # Update learning rate based on schedule (if enabled)
        self._update_learning_rate()

        # Reset tracking
        self._current_episode_rewards.clear()

    # ------------------------------------------------------------------
    # Weight persistence (WeightPersistence protocol)
    # ------------------------------------------------------------------

    def get_weight_components(
        self,
        *,
        components: set[str] | None = None,
    ) -> dict[str, WeightComponent]:
        """Return weight components for persistence.

        Components
        ----------
        ``"policy"``
            Actor network state_dict.
        ``"value"``
            Critic network state_dict.
        ``"optimizer"``
            Joint optimizer state_dict.
        ``"training_state"``
            Episode count and other training metadata.
        """
        from quantumnematode.brain.weights import WeightComponent

        all_components: dict[str, WeightComponent] = {
            "policy": WeightComponent(
                name="policy",
                state=self.actor.state_dict(),
            ),
            "value": WeightComponent(
                name="value",
                state=self.critic.state_dict(),
            ),
            "optimizer": WeightComponent(
                name="optimizer",
                state=self.optimizer.state_dict(),
            ),
            "training_state": WeightComponent(
                name="training_state",
                state={"episode_count": self._episode_count},
            ),
        }

        if self._feature_gating:
            all_components["gate_weights"] = WeightComponent(
                name="gate_weights",
                state={"gate_weights": self.gate_weights.data.clone()},
            )

        if components is None:
            return all_components

        unknown = components - set(all_components)
        if unknown:
            msg = f"Unknown weight components: {unknown}. Valid components: {set(all_components)}"
            raise ValueError(msg)
        return {k: v for k, v in all_components.items() if k in components}

    def load_weight_components(
        self,
        components: dict[str, WeightComponent],
    ) -> None:
        """Load weight components into this brain.

        Network state is loaded before optimizer state.  The PPO rollout
        buffer is reset to discard stale experience.
        """
        # Load networks first (catches shape mismatches before optimizer)
        if "policy" in components:
            self.actor.load_state_dict(components["policy"].state)
        if "value" in components:
            self.critic.load_state_dict(components["value"].state)

        # Gate weights
        if "gate_weights" in components and self._feature_gating:
            self.gate_weights.data.copy_(components["gate_weights"].state["gate_weights"])

        # Optimizer state only after networks succeed
        if "optimizer" in components:
            self.optimizer.load_state_dict(components["optimizer"].state)

        # Training state
        if "training_state" in components:
            ts = components["training_state"].state
            if "episode_count" in ts:
                self._episode_count = int(ts["episode_count"])
                self._update_learning_rate()

        # Reset buffer to prevent stale experience
        self.buffer.reset()

        logger.info(
            "MLPPPOBrain weights loaded (components: %s, episode_count=%d)",
            list(components.keys()),
            self._episode_count,
        )

    def copy(self) -> MLPPPOBrain:
        """MLPPPOBrain does not support copying."""
        error_msg = "MLPPPOBrain does not support copying. Use deepcopy if needed."
        raise NotImplementedError(error_msg)

    @property
    def action_set(self) -> list[Action]:
        """Get the list of actions."""
        return self._action_set

    @action_set.setter
    def action_set(self, actions: list[Action]) -> None:
        self._action_set = actions

    def build_brain(self) -> None:
        """Not applicable to MLPPPOBrain."""
        error_msg = "MLPPPOBrain does not have a quantum circuit."
        raise NotImplementedError(error_msg)

    def update_parameters(
        self,
        gradients: list[float],
        reward: float | None = None,
        learning_rate: float = 0.01,
        **kwargs: object,
    ) -> None:
        """Not used - PPO uses its own optimizer."""
