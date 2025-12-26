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
- Input: 2D state features (gradient strength, relative direction to goal)
- Actor: MLP producing action probabilities via softmax
- Critic: MLP producing value estimate (single scalar)
- Output: Action selection from categorical distribution

The PPO brain learns by:
1. Collecting rollout_buffer_size steps of experience
2. Computing GAE advantages using value estimates
3. Performing num_epochs of minibatch updates
4. Using clipped surrogate objective to constrain policy changes
5. Jointly optimizing policy loss, value loss, and entropy bonus

References
----------
- Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
"""

from collections.abc import Iterator

import numpy as np
import torch
from torch import nn, optim

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, ClassicalBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch.dtypes import BrainConfig, DeviceType
from quantumnematode.env import Direction
from quantumnematode.initializers._initializer import ParameterInitializer
from quantumnematode.logging_config import logger
from quantumnematode.monitoring.overfitting_detector import create_overfitting_detector_for_brain

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


class PPOBrainConfig(BrainConfig):
    """Configuration for the PPOBrain architecture."""

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


class RolloutBuffer:
    """Buffer for storing rollout experience for PPO updates."""

    def __init__(self, buffer_size: int, device: torch.device) -> None:
        self.buffer_size = buffer_size
        self.device = device
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
        """Return the current buffer size."""
        return self.position

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns.

        Args:
            last_value: Value estimate for the state after the last step
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns
        -------
            Tuple of (returns, advantages) tensors
        """
        advantages = torch.zeros(len(self), device=self.device)
        last_gae = 0.0

        # Convert values to tensor for easier indexing
        values = torch.stack(self.values).squeeze()

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
        """
        Generate minibatches for training.

        Yields
        ------
            Dictionary with states, actions, old_log_probs, returns, advantages
        """
        batch_size = len(self)
        minibatch_size = batch_size // num_minibatches

        # Convert to tensors
        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.stack(self.log_probs)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Generate random indices
        indices = torch.randperm(batch_size, device=self.device)

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


class PPOBrain(ClassicalBrain):
    """
    Proximal Policy Optimization (PPO) brain architecture.

    Uses actor-critic networks with clipped surrogate objective for stable learning.
    This is a SOTA classical baseline for comparison with quantum approaches.
    """

    def __init__(  # noqa: PLR0913
        self,
        config: PPOBrainConfig,
        input_dim: int,
        num_actions: int,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] = DEFAULT_ACTIONS,
        *,
        parameter_initializer: ParameterInitializer | None = None,
    ) -> None:
        super().__init__()

        logger.info(f"Initializing PPOBrain with config: {config}")

        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.device = torch.device(device.value)
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

        # Build networks
        self.actor = self._build_network(
            config.actor_hidden_dim,
            config.num_hidden_layers,
            output_dim=num_actions,
        ).to(self.device)

        self.critic = self._build_network(
            config.critic_hidden_dim,
            config.num_hidden_layers,
            output_dim=1,
        ).to(self.device)

        # Initialize parameters
        self._initialize_parameters(parameter_initializer)

        # Single optimizer for both networks
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config.learning_rate,
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(config.rollout_buffer_size, self.device)

        # State tracking
        self.training = True
        self.current_probabilities = None
        self.last_value: torch.Tensor | None = None
        self.pending_reward: float | None = None

        # Overfitting detection
        self.overfitting_detector = create_overfitting_detector_for_brain("ppo")
        self.overfit_detector_episode_count = 0
        self.overfit_detector_current_episode_actions: list[str] = []
        self.overfit_detector_current_episode_positions: list[tuple[int, int]] = []
        self.overfit_detector_current_episode_rewards: list[float] = []

    def _build_network(
        self,
        hidden_dim: int,
        num_hidden_layers: int,
        output_dim: int,
    ) -> nn.Sequential:
        """Build an MLP network."""
        layers: list[nn.Module] = [nn.Linear(self.input_dim, hidden_dim), nn.ReLU()]
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

        logger.info(f"PPOBrain initialized with {param_count:,} total parameters")
        if parameter_initializer is not None:
            logger.info("Custom parameter initializer provided but using orthogonal init")

    def preprocess(self, params: BrainParams) -> np.ndarray:
        """
        Preprocess BrainParams to extract features for the neural network.

        Matches MLPBrain preprocessing exactly:
        - Gradient strength (float, [0, 1])
        - Normalized relative angle to goal ([-1, 1])
        """
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

        features = [grad_strength, rel_angle_norm]
        return np.array(features, dtype=np.float32)

    def forward_actor(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through actor network."""
        return self.actor(x)

    def forward_critic(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through critic network."""
        return self.critic(x)

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

        # Get action logits and value
        logits = self.forward_actor(x)
        value = self.forward_critic(x)

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
        logits = self.forward_actor(torch.tensor(x, dtype=torch.float32, device=self.device))
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

        # Track for overfitting detection
        self._track_episode_metrics(params, probs_np, action_name)

        # Update history
        self.history_data.actions.append(self.latest_data.action)
        self.history_data.probabilities.append(float(probs_np[action_idx]))

        return [ActionData(state=action_name, action=action_name, probability=probs_np[action_idx])]

    def _track_episode_metrics(
        self,
        params: BrainParams,
        probs_np: np.ndarray,
        action_name: str,
    ) -> None:
        """Track metrics for overfitting detection."""
        self.overfit_detector_current_episode_actions.append(action_name)
        if params.agent_position is not None:
            pos = (int(params.agent_position[0]), int(params.agent_position[1]))
            self.overfit_detector_current_episode_positions.append(pos)
        self.overfitting_detector.update_learning_metrics(loss=None, policy_probs=probs_np)

    def learn(
        self,
        params: BrainParams,  # noqa: ARG002
        reward: float,
        *,
        episode_done: bool = False,
    ) -> None:
        """Add experience to buffer and update if buffer is full."""
        self.overfit_detector_current_episode_rewards.append(reward)

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

        # PPO update loop
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        num_updates = 0

        for _ in range(self.num_epochs):
            for batch in self.buffer.get_minibatches(self.num_minibatches, returns, advantages):
                # Get new action probabilities and values
                logits = self.actor(batch["states"])
                values = self.critic(batch["states"]).squeeze()

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
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy.item()
                num_updates += 1

        # Store average loss for tracking
        if num_updates > 0:
            avg_loss = total_policy_loss / num_updates
            self.latest_data.loss = avg_loss

            # Update overfitting detector
            dummy_probs = np.zeros(self.num_actions)
            self.overfitting_detector.update_learning_metrics(avg_loss, dummy_probs)

            logger.debug(
                f"PPO update: policy_loss={total_policy_loss / num_updates:.4f}, "
                f"value_loss={total_value_loss / num_updates:.4f}, "
                f"entropy={total_entropy_loss / num_updates:.4f}",
            )

    def update_memory(self, reward: float | None = None) -> None:
        """No-op for PPOBrain."""

    def prepare_episode(self) -> None:
        """Prepare for a new episode."""

    def post_process_episode(
        self,
        *,
        episode_success: bool | None = None,  # noqa: ARG002
    ) -> None:
        """Post-process after each episode."""
        self._complete_episode_tracking()

    def _complete_episode_tracking(self) -> None:
        """Complete episode tracking for overfitting detection."""
        total_steps = len(self.overfit_detector_current_episode_rewards)
        total_reward = sum(self.overfit_detector_current_episode_rewards)

        self.overfitting_detector.update_performance_metrics(total_steps, total_reward)

        if (
            self.overfit_detector_current_episode_actions
            and self.overfit_detector_current_episode_positions
        ):
            start_pos = (
                self.overfit_detector_current_episode_positions[0]
                if self.overfit_detector_current_episode_positions
                else (0, 0)
            )
            self.overfitting_detector.update_behavioral_metrics(
                self.overfit_detector_current_episode_actions.copy(),
                self.overfit_detector_current_episode_positions.copy(),
                start_pos,
            )

        self.overfit_detector_episode_count += 1

        if self.overfit_detector_episode_count % EPISODE_LOG_INTERVAL == 0:
            self.overfitting_detector.log_overfitting_analysis()

        # Reset tracking
        self.overfit_detector_current_episode_actions.clear()
        self.overfit_detector_current_episode_positions.clear()
        self.overfit_detector_current_episode_rewards.clear()

    def copy(self) -> "PPOBrain":
        """PPOBrain does not support copying."""
        error_msg = "PPOBrain does not support copying. Use deepcopy if needed."
        raise NotImplementedError(error_msg)

    @property
    def action_set(self) -> list[Action]:
        """Get the list of actions."""
        return self._action_set

    @action_set.setter
    def action_set(self, actions: list[Action]) -> None:
        self._action_set = actions

    def build_brain(self) -> None:
        """Not applicable to PPOBrain."""
        error_msg = "PPOBrain does not have a quantum circuit."
        raise NotImplementedError(error_msg)

    def update_parameters(
        self,
        gradients: list[float],
        reward: float | None = None,
        learning_rate: float = 0.01,
        **kwargs: object,
    ) -> None:
        """Not used - PPO uses its own optimizer."""
