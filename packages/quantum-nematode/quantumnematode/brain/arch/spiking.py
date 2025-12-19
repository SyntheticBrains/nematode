"""
Spiking Neural Network (SNN) Brain Architecture with Surrogate Gradient Descent.

This architecture implements a biologically plausible spiking neural network using
Leaky Integrate-and-Fire (LIF) neurons with surrogate gradient descent for learning.
The approach combines biological realism with effective gradient-based optimization.

Key Features
------------
- **Temporal Dynamics**: LIF neurons with membrane potential integration
- **Surrogate Gradients**: Differentiable spike approximation enables backpropagation
- **Policy Gradient Learning**: REINFORCE algorithm with baseline subtraction
- **Relative Angle Encoding**: Proper directional features for navigation
- **Dense Learning Signals**: Every timestep contributes to gradient updates

Architecture
------------
- Input: 2 features (gradient strength, relative angle to goal)
- Hidden: Multiple LIF layers with recurrent membrane dynamics
- Output: 4 action neurons (forward, left, right, stay)

The SNN brain learns by:
1. Encoding state features as constant input currents
2. Simulating LIF neural dynamics for a fixed number of timesteps
3. Accumulating spikes across time to compute action probabilities
4. Updating network parameters using policy gradients (REINFORCE)
5. Maintaining baseline for variance reduction

This approach provides biological plausibility while enabling effective learning
through standard reinforcement learning algorithms.

References
----------
- Neftci et al. (2019). "Surrogate Gradient Learning in Spiking Neural Networks"
- Williams (1992). "Simple Statistical Gradient-Following Algorithms for
  Connectionist Reinforcement Learning" (REINFORCE)
- SpikingJelly: https://github.com/fangwei123456/spikingjelly
"""

import numpy as np
import torch

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, ClassicalBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch._spiking_layers import SpikingPolicyNetwork
from quantumnematode.brain.arch.dtypes import BrainConfig, DeviceType
from quantumnematode.env import Direction
from quantumnematode.initializers._initializer import ParameterInitializer
from quantumnematode.logging_config import logger
from quantumnematode.monitoring.overfitting_detector import create_overfitting_detector_for_brain

# Default configuration parameters
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_NUM_TIMESTEPS = 50
DEFAULT_NUM_HIDDEN_LAYERS = 2
DEFAULT_TAU_M = 20.0
DEFAULT_V_THRESHOLD = 1.0
DEFAULT_V_RESET = 0.0
DEFAULT_V_REST = 0.0
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_LR_DECAY_RATE = 0.0
DEFAULT_GAMMA = 0.99
DEFAULT_BASELINE_ALPHA = 0.05
DEFAULT_ENTROPY_BETA = 0.01
DEFAULT_ENTROPY_BETA_FINAL = 0.01
DEFAULT_ENTROPY_DECAY_EPISODES = 0
DEFAULT_SURROGATE_ALPHA = 10.0
DEFAULT_WEIGHT_INIT = "kaiming"  # kaiming, xavier, or default


class SpikingBrainConfig(BrainConfig):
    """Configuration for the SpikingBrain architecture."""

    # Network topology
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    num_hidden_layers: int = DEFAULT_NUM_HIDDEN_LAYERS
    num_timesteps: int = DEFAULT_NUM_TIMESTEPS

    # LIF neuron parameters
    tau_m: float = DEFAULT_TAU_M
    v_threshold: float = DEFAULT_V_THRESHOLD
    v_reset: float = DEFAULT_V_RESET
    v_rest: float = DEFAULT_V_REST

    # Learning parameters
    learning_rate: float = DEFAULT_LEARNING_RATE
    lr_decay_rate: float = DEFAULT_LR_DECAY_RATE
    gamma: float = DEFAULT_GAMMA
    baseline_alpha: float = DEFAULT_BASELINE_ALPHA
    entropy_beta: float = DEFAULT_ENTROPY_BETA
    entropy_beta_final: float = DEFAULT_ENTROPY_BETA_FINAL
    entropy_decay_episodes: int = DEFAULT_ENTROPY_DECAY_EPISODES

    # Surrogate gradient parameters
    surrogate_alpha: float = DEFAULT_SURROGATE_ALPHA

    # Weight initialization method
    weight_init: str = DEFAULT_WEIGHT_INIT


class SpikingBrain(ClassicalBrain):
    """
    Spiking neural network brain with surrogate gradient descent.

    Implements biologically plausible LIF neuron dynamics with gradient-based
    learning through surrogate gradient approximation. Uses REINFORCE policy
    gradient algorithm for reinforcement learning tasks.

    Parameters
    ----------
    config : SpikingBrainConfig
        Configuration for network architecture and learning
    input_dim : int
        Dimension of input features (typically 2: gradient strength, relative angle)
    num_actions : int
        Number of possible actions (typically 4: forward, left, right, stay)
    device : DeviceType
        Computing device (CPU or GPU)
    action_set : list[Action]
        Available actions for the agent
    parameter_initializer : ParameterInitializer | None
        Optional custom parameter initialization (not used with PyTorch auto-init)

    Attributes
    ----------
    policy : SpikingPolicyNetwork
        Spiking neural network for computing action probabilities
    optimizer : torch.optim.Adam
        Adam optimizer for gradient descent
    episode_states : list
        State observations collected during current episode
    episode_actions : list
        Actions taken during current episode
    episode_action_probs : list
        Detached action probabilities for entropy calculation
    episode_rewards : list
        Rewards received during current episode
    baseline : float
        Running average of returns for variance reduction
    """

    def __init__(  # noqa: PLR0913
        self,
        config: SpikingBrainConfig,
        input_dim: int,
        num_actions: int,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] = DEFAULT_ACTIONS,
        *,
        parameter_initializer: ParameterInitializer | None = None,
    ) -> None:
        super().__init__()

        logger.info(f"Initializing SpikingBrain with surrogate gradients: {config}")

        self.config = config
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.device = torch.device(device.value)
        self._action_set = action_set

        # Initialize data structures
        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()

        # Create spiking policy network
        self.policy = SpikingPolicyNetwork(
            input_dim=input_dim,
            hidden_dim=config.hidden_size,
            output_dim=num_actions,
            num_timesteps=config.num_timesteps,
            num_hidden_layers=config.num_hidden_layers,
            tau_m=config.tau_m,
            v_threshold=config.v_threshold,
            v_reset=config.v_reset,
            v_rest=config.v_rest,
            surrogate_alpha=config.surrogate_alpha,
        ).to(self.device)

        # Apply weight initialization
        self._initialize_weights(config.weight_init)

        # Optimizer (Adam for stable convergence)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate,
        )

        # Episode buffers for policy gradient learning
        # Note: We store states and actions, then recompute log_probs during learn()
        # to avoid memory leaks from storing computation graphs
        self.episode_states: list[np.ndarray] = []
        self.episode_actions: list[int] = []
        self.episode_action_probs: list[torch.Tensor] = []  # Detached, for entropy only
        self.episode_rewards: list[float] = []

        # Baseline for variance reduction (running average of returns)
        self.baseline = 0.0

        # Episode counter for decay schedules
        self.episode_count = 0
        self.initial_learning_rate = config.learning_rate
        self.initial_entropy_beta = config.entropy_beta

        # Overfitting detection
        self.overfitting_detector = create_overfitting_detector_for_brain("spiking")
        self.overfit_detector_episode_count = 0
        self.overfit_detector_current_episode_actions: list[Action] = []
        self.overfit_detector_current_episode_positions: list[tuple[float, float]] = []
        self.overfit_detector_current_episode_rewards: list[float] = []

        # Log parameter count
        total_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        logger.info(f"SpikingBrain initialized with {total_params:,} trainable parameters")

        if parameter_initializer is not None:
            logger.info(
                "Custom parameter initializer provided but using PyTorch default initialization",
            )

    @property
    def action_set(self) -> list[Action]:
        """Return the set of available actions."""
        return self._action_set

    def preprocess(self, params: BrainParams) -> np.ndarray:
        """
        Preprocess brain parameters into state vector.

        Computes relative angle between agent orientation and goal direction,
        matching the preprocessing used by MLPBrain for fair comparison.

        Parameters
        ----------
        params : BrainParams
            Brain parameters containing environmental state

        Returns
        -------
        np.ndarray
            Preprocessed state vector [gradient_strength, relative_angle_normalized]
            where relative_angle_normalized is in [-1, 1]
        """
        # Feature 1: Gradient strength [0, 1]
        grad_strength = float(params.gradient_strength or 0.0)
        grad_strength = max(0.0, min(1.0, grad_strength))

        # Feature 2: Relative angle to goal [-1, 1]
        if params.gradient_direction is not None and params.agent_direction is not None:
            # Map agent direction to angle (radians)
            direction_map = {
                Direction.UP: 0.5 * np.pi,
                Direction.RIGHT: 0.0,
                Direction.DOWN: 1.5 * np.pi,
                Direction.LEFT: np.pi,
            }
            agent_facing_angle = direction_map[params.agent_direction]

            # Compute relative angle: goal direction - agent facing direction
            # Normalize to [-π, π]
            relative_angle = (params.gradient_direction - agent_facing_angle + np.pi) % (
                2 * np.pi
            ) - np.pi

            # Normalize to [-1, 1] for network input
            rel_angle_normalized = relative_angle / np.pi
        else:
            rel_angle_normalized = 0.0

        return np.array([grad_strength, rel_angle_normalized], dtype=np.float32)

    def run_brain(
        self,
        params: BrainParams,
        reward: float | None = None,  # noqa: ARG002
        input_data: list[float] | None = None,  # noqa: ARG002
        *,
        top_only: bool,  # noqa: ARG002
        top_randomize: bool,  # noqa: ARG002
    ) -> list[ActionData]:
        """
        Run the spiking neural network and select an action.

        Forward pass through the spiking network, simulating LIF dynamics for
        num_timesteps to accumulate spikes and compute action probabilities.

        Parameters
        ----------
        params : BrainParams
            Brain parameters containing environmental state
        reward : float | None
            Optional reward signal (not used in forward pass)
        input_data : list[float] | None
            Optional input data (not used)
        top_only : bool
            Whether to return only top action (not used, always samples)
        top_randomize : bool
            Whether to randomize top actions (not used)

        Returns
        -------
        list[ActionData]
            List containing single selected action with probability
        """
        # Preprocess state
        state = self.preprocess(params)

        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Forward pass through spiking network
        with torch.no_grad() if not self.policy.training else torch.enable_grad():
            action_logits = self.policy(state_tensor)
            action_probs = torch.softmax(action_logits, dim=-1)

            # Monitor spike rates for debugging (only occasionally to avoid spam)
            # 10% sampling rate for diagnostics
            spike_rate_sample_prob = 0.10
            if (
                hasattr(self.policy, "last_spike_rates")
                and torch.rand(1).item() < spike_rate_sample_prob
            ):
                spike_rates = self.policy.last_spike_rates
                logger.debug(
                    f"Spike rates - min: {spike_rates.min():.3f}, "
                    f"max: {spike_rates.max():.3f}, mean: {spike_rates.mean():.3f}, "
                    f"std: {spike_rates.std():.3f}",
                )

        # Sample action from categorical distribution
        action_dist = torch.distributions.Categorical(action_probs)
        action_idx = action_dist.sample()

        # Store for policy gradient learning
        # Store states and actions only - we'll recompute log_probs during learn()
        # This prevents memory leak from computation graph retention
        self.episode_states.append(state)
        action_idx_int = int(action_idx.item())
        self.episode_actions.append(action_idx_int)
        # Store detached action probs for entropy calculation (doesn't need gradients)
        self.episode_action_probs.append(action_probs.squeeze(0).detach())

        # Get selected action and probability
        selected_action = self.action_set[action_idx_int]
        probability = action_probs[0, action_idx_int].item()

        # Log action probabilities to diagnose policy collapse
        probs_list = action_probs.squeeze(0).tolist()
        action_entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum().item()
        logger.debug(
            f"Action probs: {[f'{p:.3f}' for p in probs_list]}, "
            f"Selected: {selected_action}, Entropy: {action_entropy:.4f}",
        )

        # Store data for tracking
        state_str = f"grad_str:{state[0]:.3f},rel_angle:{state[1]:.3f}"
        action_data = ActionData(
            state=state_str,
            action=selected_action,
            probability=probability,
        )
        self.latest_data.action = action_data
        self.latest_data.probability = probability

        # Update overfitting detector
        if params.agent_position is not None:
            self.overfit_detector_current_episode_positions.append(params.agent_position)
        self.overfit_detector_current_episode_actions.append(selected_action)

        logger.debug(
            f"SpikingBrain selected action {selected_action} with probability {probability:.3f}",
        )

        return [action_data]

    def learn(  # noqa: PLR0915
        self,
        params: BrainParams,  # noqa: ARG002
        reward: float,
        *,
        episode_done: bool,
    ) -> None:
        """
        Update network parameters using policy gradient (REINFORCE).

        Computes discounted returns, normalizes for variance reduction, and
        updates policy network using policy gradient with baseline subtraction.

        Parameters
        ----------
        params : BrainParams
            Brain parameters (not used in learning)
        reward : float
            Reward signal for current timestep
        episode_done : bool
            Whether the episode is complete
        """
        self.episode_rewards.append(reward)
        self.overfit_detector_current_episode_rewards.append(reward)

        if episode_done and self.episode_rewards:
            logger.info(
                f"Episode complete - performing policy gradient update with "
                f"{len(self.episode_rewards)} timesteps",
            )
            # Compute discounted returns backward through episode
            returns: list[float] = []
            g_value = 0.0
            for r in reversed(self.episode_rewards):
                g_value = r + self.config.gamma * g_value
                returns.insert(0, g_value)

            # Convert to tensor
            returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)

            # Normalize returns for variance reduction
            if len(returns) > 1:
                returns_mean = returns_tensor.mean()
                returns_std = returns_tensor.std()
                returns_tensor = (returns_tensor - returns_mean) / (returns_std + 1e-8)
            else:
                returns_mean = returns_tensor.item()
                returns_std = 0.0

            # Update baseline (running average of mean episode return)
            self.baseline = (
                self.config.baseline_alpha * returns_mean
                + (1 - self.config.baseline_alpha) * self.baseline
            )

            # Compute advantages
            # After normalization, returns already have mean≈0, so we use them directly
            # as advantages The baseline is tracked separately for monitoring but not
            # subtracted from normalized returns
            advantages = returns_tensor

            # Recompute log_probs with fresh forward passes to enable gradient flow
            # This is more memory efficient than storing computation graphs
            log_probs_list = []
            for state, action_idx in zip(self.episode_states, self.episode_actions, strict=False):
                state_tensor = torch.tensor(
                    state,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                action_logits = self.policy(state_tensor)
                action_probs = torch.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                log_prob = action_dist.log_prob(torch.tensor(action_idx, device=self.device))
                log_probs_list.append(log_prob)

            # Compute policy loss: -Σ log_prob(a_t) * advantage_t
            log_probs = torch.stack(log_probs_list)
            policy_loss = -(log_probs * advantages).sum()

            # Apply entropy decay schedule
            current_entropy_beta = self._get_current_entropy_beta()

            # Add entropy regularization for exploration
            entropy = torch.tensor(0.0, device=self.device)
            if current_entropy_beta > 0:
                # Entropy: -Σ p(a) * log(p(a)) for each timestep, averaged
                action_probs = torch.stack(self.episode_action_probs)
                entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()

                # Log entropy for diagnostics
                logger.debug(
                    f"Episode entropy: {entropy.item():.4f}, "
                    f"entropy_beta: {current_entropy_beta:.4f}, "
                    f"entropy contribution: {current_entropy_beta * entropy.item():.4f}",
                )

                # Subtract entropy to maximize it (entropy bonus)
                policy_loss = policy_loss - current_entropy_beta * entropy

            # Backpropagation
            self.optimizer.zero_grad()
            policy_loss.backward()

            # Clip individual gradient values first to prevent explosion
            # This prevents inf gradients from surrogate gradient backward pass
            for param in self.policy.parameters():
                if param.grad is not None:
                    param.grad.clamp_(-1.0, 1.0)

            # Then clip gradient norm for overall stability
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)

            # Log gradient norm for diagnostics
            logger.debug(f"Gradient norm: {grad_norm.item():.6f}")

            # Update parameters
            self.optimizer.step()

            # Apply learning rate decay schedule
            self._apply_lr_decay()

            # Track learning data for export
            self.latest_data.loss = policy_loss.item()
            self.latest_data.learning_rate = self.optimizer.param_groups[0]["lr"]

            # Append to history for CSV export
            if self.latest_data.loss is not None:
                self.history_data.losses.append(self.latest_data.loss)
            if self.latest_data.learning_rate is not None:
                self.history_data.learning_rates.append(self.latest_data.learning_rate)

            # Log learning statistics
            episode_return = sum(self.episode_rewards)
            logger.debug(
                f"Policy gradient update: loss={policy_loss.item():.4f}, "
                f"episode_return={episode_return:.2f}, baseline={self.baseline:.2f}, "
                f"entropy={entropy.item():.4f}, returns_std={returns_std:.4f}",
            )

            # Increment episode counter
            self.episode_count += 1

            # Clear episode buffers
            self.episode_states.clear()
            self.episode_actions.clear()
            self.episode_action_probs.clear()
            self.episode_rewards.clear()

    def update_memory(self, reward: float | None) -> None:
        """
        Update memory with reward information.

        Parameters
        ----------
        reward : float | None
            Reward signal to store
        """
        if reward is not None:
            self.latest_data.reward = reward
            self.history_data.rewards.append(reward)

    def prepare_episode(self) -> None:
        """Prepare for a new episode (no-op for SpikingBrain)."""

    def post_process_episode(self, *, episode_success: bool | None = None) -> None:  # noqa: ARG002
        """
        Perform post-episode processing and cleanup.

        Parameters
        ----------
        episode_success : bool | None
            Whether the episode was successful (not used)
        """
        # Update overfitting detector
        if (
            self.overfit_detector_current_episode_actions
            and self.overfit_detector_current_episode_positions
            and self.overfit_detector_current_episode_rewards
        ):
            try:
                total_reward = sum(self.overfit_detector_current_episode_rewards)
                num_steps = len(self.overfit_detector_current_episode_actions)

                # Convert actions to strings
                action_sequence = [
                    action.value for action in self.overfit_detector_current_episode_actions
                ]

                # Update performance metrics
                if hasattr(self.overfitting_detector, "update_performance_metrics"):
                    self.overfitting_detector.update_performance_metrics(num_steps, total_reward)

                # Update behavioral metrics if we have position data
                if (
                    hasattr(self.overfitting_detector, "update_behavioral_metrics")
                    and self.overfit_detector_current_episode_positions
                ):
                    start_pos = self.overfit_detector_current_episode_positions[0]
                    # Convert float positions to int for behavioral metrics
                    int_positions = [
                        (int(x), int(y)) for x, y in self.overfit_detector_current_episode_positions
                    ]
                    int_start_pos = (int(start_pos[0]), int(start_pos[1]))
                    self.overfitting_detector.update_behavioral_metrics(
                        action_sequence,
                        int_positions,
                        int_start_pos,
                    )

            except (AttributeError, ValueError, TypeError) as e:
                logger.warning(f"Overfitting detector update failed: {e}")

        # Clear episode tracking
        self.overfit_detector_current_episode_actions.clear()
        self.overfit_detector_current_episode_positions.clear()
        self.overfit_detector_current_episode_rewards.clear()
        self.overfit_detector_episode_count += 1

        logger.debug("Episode post-processing complete")

    def _get_current_entropy_beta(self) -> float:
        """
        Get current entropy beta value based on decay schedule.

        Returns
        -------
        float
            Current entropy regularization coefficient
        """
        if self.config.entropy_decay_episodes <= 0:
            # No decay, return initial value
            return self.initial_entropy_beta

        # Linear decay from initial to final over specified episodes
        progress = min(self.episode_count / self.config.entropy_decay_episodes, 1.0)
        return (
            self.initial_entropy_beta
            - (self.initial_entropy_beta - self.config.entropy_beta_final) * progress
        )

    def _apply_lr_decay(self) -> None:
        """Apply exponential learning rate decay if configured."""
        if self.config.lr_decay_rate > 0:
            # Exponential decay: lr_t = lr_0 * (1 - decay_rate)^t
            decay_factor = (1.0 - self.config.lr_decay_rate) ** self.episode_count
            new_lr = self.initial_learning_rate * decay_factor

            # Update optimizer learning rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr

    def _initialize_weights(self, method: str) -> None:
        """
        Initialize network weights using specified method.

        Parameters
        ----------
        method : str
            Initialization method: "kaiming", "xavier", or "default"
            - kaiming: Variance-preserving for ReLU-like activations (recommended for spiking)
            - xavier: Variance-preserving for tanh/sigmoid activations
            - default: PyTorch default (uniform based on fan-in)
        """
        if method == "default":
            # PyTorch already initialized, nothing to do
            logger.info("Using PyTorch default weight initialization")
            return

        logger.info(f"Initializing weights with {method} method")

        # Find output layer (last linear layer)
        linear_layers = [
            module for module in self.policy.modules() if isinstance(module, torch.nn.Linear)
        ]
        output_layer = linear_layers[-1] if linear_layers else None

        for _name, module in self.policy.named_modules():
            if isinstance(module, torch.nn.Linear):
                if method == "kaiming":
                    # Kaiming/He initialization for ReLU-like activations
                    # Good for spiking neurons which have piecewise activation
                    torch.nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

                    # Scale down output layer to prevent extreme logits
                    # Output layer needs smaller weights for balanced softmax probabilities
                    if module is output_layer:
                        with torch.no_grad():
                            module.weight.mul_(0.01)  # Scale down by 100x
                        logger.info(
                            "Scaled down output layer weights by 0.01 for balanced probabilities",
                        )

                elif method == "xavier":
                    # Xavier/Glorot initialization for tanh/sigmoid
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

                    # Scale down output layer for balanced probabilities
                    if module is output_layer:
                        with torch.no_grad():
                            module.weight.mul_(0.1)  # Scale down by 10x
                        logger.info(
                            "Scaled down output layer weights by 0.1 for balanced probabilities",
                        )

                else:
                    logger.warning(f"Unknown initialization method: {method}, using default")

    def copy(self) -> "SpikingBrain":
        """
        Create a copy of the brain.

        Returns
        -------
        SpikingBrain
            New SpikingBrain instance with copied network parameters
        """
        new_brain = SpikingBrain(
            config=self.config,
            input_dim=self.input_dim,
            num_actions=self.num_actions,
            device=DeviceType(self.device.type),
            action_set=self._action_set.copy(),
        )

        # Copy network parameters
        new_brain.policy.load_state_dict(self.policy.state_dict())

        # Copy optimizer state
        new_brain.optimizer.load_state_dict(self.optimizer.state_dict())

        # Copy baseline and episode counter for decay schedules
        new_brain.baseline = self.baseline
        new_brain.episode_count = self.episode_count

        return new_brain
