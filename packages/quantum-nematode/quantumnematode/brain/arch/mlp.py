"""
Policy Gradient Multi-Layer Perceptron (MLP) Brain Architecture.

This architecture implements a REINFORCE-style policy gradient approach using a classical
multi-layer perceptron to learn optimal action policies for navigation tasks.

Key Features:
- **Policy Gradients**: Directly optimizes action selection policy using REINFORCE algorithm
- **Baseline Subtraction**: Uses running average baseline for variance reduction
- **Entropy Regularization**: Encourages exploration through entropy bonus
- **Experience Buffering**: Collects episode trajectories for batch updates
- **Adaptive Learning**: Learning rate scheduling and gradient clipping for stability

Architecture:
- Input: 2D state features (gradient strength, relative direction to goal)
- Hidden: Configurable MLP layers with ReLU activation
- Output: Action probabilities via softmax (forward, left, right, stay)

The MLP brain learns by:
1. Collecting complete episode trajectories of state-action-reward sequences
2. Computing discounted returns for each time step
3. Updating policy to increase probability of good actions (high return)
4. Using baseline subtraction and entropy regularization for stability
5. Performing batch updates for more stable learning

This approach learns policies directly but can be less sample-efficient than value-based
methods like Q-learning, especially for environments with sparse or delayed rewards.
"""

import numpy as np
import torch  # pyright: ignore[reportMissingImports]
from torch import nn, optim  # pyright: ignore[reportMissingImports]

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, ClassicalBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch.dtypes import BrainConfig, DeviceType
from quantumnematode.env import Direction
from quantumnematode.initializers._initializer import ParameterInitializer
from quantumnematode.logging_config import logger
from quantumnematode.monitoring.overfitting_detector import create_overfitting_detector_for_brain

DEFAULT_HIDDEN_DIM = 64
DEFAULT_NUM_HIDDEN_LAYERS = 2
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_LR_SCHEDULER_STEP_SIZE = 100
DEFAULT_LR_SCHEDULER_GAMMA = 0.9
DEFAULT_ENTROPY_BETA = 0.01
DEFAULT_BASELINE = 0.0
DEFAULT_BASELINE_ALPHA = 0.05
DEFAULT_GAMMA = 0.99

# Episode logging interval constant
# TODO: Make this configurable
EPISODE_LOG_INTERVAL = 25


class MLPBrainConfig(BrainConfig):
    """Configuration for the MLPBrain architecture."""

    baseline: float = DEFAULT_BASELINE
    baseline_alpha: float = DEFAULT_BASELINE_ALPHA
    entropy_beta: float = DEFAULT_ENTROPY_BETA
    gamma: float = DEFAULT_GAMMA
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    learning_rate: float = DEFAULT_LEARNING_RATE
    lr_scheduler_step_size: int = DEFAULT_LR_SCHEDULER_STEP_SIZE
    lr_scheduler_gamma: float = DEFAULT_LR_SCHEDULER_GAMMA
    num_hidden_layers: int = DEFAULT_NUM_HIDDEN_LAYERS


class MLPBrain(ClassicalBrain):
    """
    Classical multi-layer perceptron (MLP) brain architecture.

    Uses a simple MLP policy network with optional GPU acceleration.
    """

    def __init__(  # noqa: PLR0913
        self,
        config: MLPBrainConfig,
        input_dim: int,
        num_actions: int,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] = DEFAULT_ACTIONS,
        *,
        lr_scheduler: bool | None = None,
        parameter_initializer: ParameterInitializer | None = None,
    ) -> None:
        super().__init__()

        logger.info(
            f"Using configuration: {config}",
        )

        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.device = torch.device(device.value)
        self.entropy_beta = config.entropy_beta
        self.policy = self._build_network(config.hidden_dim, config.num_hidden_layers).to(
            self.device,
        )

        # Initialize parameters with custom initializer or log default initialization
        self._initialize_parameters(parameter_initializer)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        if lr_scheduler is None:
            lr_scheduler = True
        self.lr_scheduler = (
            optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.lr_scheduler_step_size,
                gamma=config.lr_scheduler_gamma,
            )
            if lr_scheduler
            else None
        )
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

        self.current_probabilities = None

        self.training = True
        self._action_set = action_set

        # Baseline for variance reduction in policy gradient
        self.baseline = config.baseline
        self.baseline_alpha = config.baseline_alpha  # Smoothing factor for running average

        # Discount factor for future rewards
        self.gamma = config.gamma

        # Episode buffer for batch learning
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []

        # Learning frequency control
        self.steps_since_update = 0
        self.update_frequency = 5  # Update every N steps for stability

        # Overfitting detection
        self.overfitting_detector = create_overfitting_detector_for_brain("mlp")
        self.overfit_detector_episode_count = 0
        self.overfit_detector_current_episode_actions = []
        self.overfit_detector_current_episode_positions = []
        self.overfit_detector_current_episode_rewards = []

    def _build_network(self, hidden_dim: int, num_hidden_layers: int) -> nn.Sequential:
        layers = [nn.Linear(self.input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, self.num_actions))
        return nn.Sequential(*layers)

    def _initialize_parameters(self, parameter_initializer: ParameterInitializer | None) -> None:
        """
        Initialize network parameters and log the initialization details.

        Args:
            parameter_initializer: Optional parameter initializer to use.
                                 If None, uses PyTorch's default initialization.
        """
        param_count = 0
        param_details = []

        with torch.no_grad():
            for name, param in self.policy.named_parameters():
                param_count += param.numel()

                if parameter_initializer is not None:
                    # NOTE: This is a placeholder for future custom initialization logic
                    logger.info(
                        f"Custom parameter initialization not fully implemented for MLP {name}",
                    )
                    # Keep PyTorch default for now, but log that custom initializer was provided
                    param_details.append(
                        f"  {name}: shape {list(param.shape)}, "
                        f"mean={param.mean().item():.6f}, std={param.std().item():.6f} "
                        f"(custom initializer provided but using PyTorch default)",
                    )
                else:
                    # Log PyTorch's default initialization
                    param_details.append(
                        f"  {name}: shape {list(param.shape)}, "
                        f"mean={param.mean().item():.6f}, std={param.std().item():.6f} "
                        f"(PyTorch default)",
                    )

        logger.info("MLPBrain parameter initialization complete:")
        logger.info(f"  Total parameters: {param_count:,}")
        logger.info("  Parameter details:")
        for detail in param_details:
            logger.info(detail)

    def preprocess(self, params: BrainParams) -> np.ndarray:
        """
        Preprocess BrainParams to extract features for the neural network.

        Convert BrainParams to a flat numpy array for the NN input, using relative angle to goal.

        Parameters
        ----------
            params: BrainParams containing agent state, gradient strength, and direction.

        Returns
        -------
            np.ndarray: Preprocessed features as a numpy array.
        -----------
        Features are:
            - Gradient strength (float, [0, 1])
            - Normalised relative angle to goal ([-1, 1])
        """
        # Use gradient_strength as-is (assumed [0, 1])
        grad_strength = float(params.gradient_strength or 0.0)

        # Compute relative angle to goal ([-pi, pi])
        grad_direction = float(params.gradient_direction or 0.0)
        direction_map = {
            Direction.UP: np.pi / 2,
            Direction.DOWN: -np.pi / 2,
            Direction.LEFT: np.pi,
            Direction.RIGHT: 0.0,
        }
        agent_facing_angle = direction_map.get(params.agent_direction or Direction.UP, np.pi / 2)
        relative_angle = (grad_direction - agent_facing_angle + np.pi) % (2 * np.pi) - np.pi
        # Normalise relative angle to [-1, 1]
        rel_angle_norm = relative_angle / np.pi

        features = [grad_strength, rel_angle_norm]
        return np.array(features, dtype=np.float32)

    def forward(self, x: np.ndarray) -> torch.Tensor:
        """
        Forward pass through the policy network.

        Args:
            x: Input features as a numpy array.

        Returns
        -------
            logits: Output logits from the policy network.
        """
        x_tensor = torch.from_numpy(x).float().to(self.device)
        return self.policy(x_tensor)

    def build_brain(self):  # noqa: ANN201
        """
        Build the brain architecture.

        This method is not applicable to MLPBrain as it does not have a quantum circuit.
        """
        error_msg = (
            "MLPBrain does not have a quantum circuit. "
            "This method is not applicable to classical architectures."
        )
        raise NotImplementedError(error_msg)

    def run_brain(
        self,
        params: BrainParams,
        reward: float | None = None,  # noqa: ARG002
        input_data: list[float] | None = None,  # noqa: ARG002
        *,
        top_only: bool,  # noqa: ARG002
        top_randomize: bool,  # noqa: ARG002
    ) -> list[ActionData]:
        """Run the policy network and select an action."""
        x = self.preprocess(params)
        logits = self.forward(x)

        # Add exploration noise to logits for better exploration
        if self.training:
            noise_std = 0.1  # Small amount of noise
            noise = torch.normal(0, noise_std, size=logits.shape).to(self.device)
            logits = logits + noise

        probs = torch.softmax(logits, dim=-1)
        probs_np = probs.detach().cpu().numpy()

        # Use temperature sampling for better exploration
        temperature = 1.2 if self.training else 1.0
        probs_temp = torch.softmax(logits / temperature, dim=-1)
        probs_temp_np = probs_temp.detach().cpu().numpy()

        rng = np.random.default_rng()
        action_idx = rng.choice(self.num_actions, p=probs_temp_np)
        action_name = self.action_set[action_idx]

        # Store log probability for learning (using original probs, not temperature)
        log_prob = torch.log(probs[action_idx] + 1e-8)

        self.latest_data.action = ActionData(
            state=action_name,
            action=action_name,
            probability=probs_np[action_idx],
        )

        # Store for episode-based learning
        self.episode_states.append(x)
        self.episode_actions.append(action_idx)
        self.episode_log_probs.append(log_prob)

        # Track for overfitting detection
        self._track_episode_metrics(params, probs_np, action_name)

        self.current_probabilities = probs_np
        first_prob = float(probs_np[action_idx])
        self.latest_data.probability = first_prob
        self.history_data.actions.append(self.latest_data.action)
        self.history_data.probabilities.append(first_prob)

        actions = {name: int(i == action_idx) for i, name in enumerate(self.action_set)}
        return self._get_most_probable_action(actions)

    def _track_episode_metrics(
        self,
        params: BrainParams,
        probs_np: np.ndarray,
        action_name: str,
    ) -> None:
        """Track metrics for the current episode."""
        self.overfit_detector_current_episode_actions.append(action_name)
        if params.agent_position is not None:
            self.overfit_detector_current_episode_positions.append(params.agent_position)

        # Track policy outputs for consistency analysis
        self.overfitting_detector.update_learning_metrics(loss=None, policy_probs=probs_np)

    def _get_most_probable_action(
        self,
        counts: dict,
    ) -> list[ActionData]:
        """Return the most probable action (or sampled action)."""
        # Counts is a one-hot dict from run_brain
        action_name = max(counts.items(), key=lambda x: x[1])[0]
        idx = self.action_set.index(action_name)
        prob = self.current_probabilities[idx] if self.current_probabilities is not None else 1.0
        return [ActionData(state=action_name, action=action_name, probability=prob)]

    def update_parameters(
        self,
        gradients: list[float],
        reward: float | None = None,
        learning_rate: float = 0.01,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """
        Update the parameters of the policy network.

        This method is not used in MLPBrain as it uses PyTorch autograd.
        """

    def compute_discounted_return(self, rewards: list[float], gamma: float = 0.99) -> float:
        """
        Compute the discounted return for a list of rewards.

        Args:
            rewards: List of rewards from the current episode (most recent first).
            gamma: Discount factor.

        Returns
        -------
            Discounted return (float).
        """
        g = 0.0
        for r in reversed(rewards):
            g = r + gamma * g
        return g

    def learn(
        self,
        params: BrainParams,  # noqa: ARG002
        reward: float,
        *,
        episode_done: bool = False,
    ) -> None:
        """Perform policy gradient learning with episode buffering."""
        # Store the reward for this step
        self.episode_rewards.append(reward)
        self.overfit_detector_current_episode_rewards.append(reward)
        self.steps_since_update += 1

        # Check if episode is done (goal reached or max steps)
        episode_done = episode_done or len(self.episode_rewards) >= self.update_frequency

        if episode_done and len(self.episode_rewards) > 0:
            self._perform_policy_update()
            self._reset_episode_buffer()

        # Store for history tracking
        if hasattr(self.latest_data, "loss") and self.latest_data.loss is not None:
            self.history_data.losses.append(self.latest_data.loss)
        self.history_data.rewards.append(reward)

    def _perform_policy_update(self) -> None:
        """Perform the actual policy gradient update using collected episode data."""
        if len(self.episode_states) == 0:
            return

        # Ensure all lists have the same length
        min_length = min(
            len(self.episode_states),
            len(self.episode_actions),
            len(self.episode_rewards),
            len(self.episode_log_probs),
        )

        if min_length == 0:
            return

        # Truncate all lists to the same length
        self.episode_states = self.episode_states[:min_length]
        self.episode_actions = self.episode_actions[:min_length]
        self.episode_rewards = self.episode_rewards[:min_length]
        self.episode_log_probs = self.episode_log_probs[:min_length]

        # Compute discounted returns
        returns = []
        discounted_return = 0
        for reward in reversed(self.episode_rewards):
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)

        # Convert to tensor and normalize
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Update baseline (moving average of returns)
        mean_return = returns.mean().item()
        self.baseline = (
            1 - self.baseline_alpha
        ) * self.baseline + self.baseline_alpha * mean_return

        # Compute advantages
        advantages = returns - self.baseline

        # Compute policy loss
        policy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        entropy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for t in range(len(self.episode_log_probs)):
            # Policy gradient loss
            policy_loss = policy_loss - self.episode_log_probs[t] * advantages[t]

            # Entropy regularization (recompute for current state)
            x = self.episode_states[t]
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            entropy_loss = entropy_loss - self.entropy_beta * entropy

        # Total loss
        total_loss = (policy_loss + entropy_loss) / len(self.episode_log_probs)

        # Ensure we have a tensor and clip loss to prevent extreme updates
        if not isinstance(total_loss, torch.Tensor):
            total_loss = torch.tensor(total_loss, device=self.device, requires_grad=True)
        total_loss = torch.clamp(total_loss, -10.0, 10.0)

        # Backprop and update
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)

        self.optimizer.step()

        if self.lr_scheduler:
            self.lr_scheduler.step()

        # Store loss for tracking
        self.latest_data.loss = total_loss.item()

        # Update overfitting detector with loss information
        dummy_probs = np.zeros(self.num_actions)  # Placeholder since we track probs elsewhere
        # TODO: This may not be working properly, loss reported looks incorrect later in sessions
        self.overfitting_detector.update_learning_metrics(total_loss.item(), dummy_probs)

    def _reset_episode_buffer(self) -> None:
        """Reset the episode buffer for the next episode."""
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_log_probs.clear()
        self.steps_since_update = 0

    def update_memory(
        self,
        reward: float | None = None,  # noqa: ARG002
    ) -> None:
        """No-op for MLPBrain."""
        return

    def post_process_episode(self) -> None:
        """Post-process the brain's state after each episode."""
        # Update overfitting detector with episode data
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

        # Log overfitting analysis every EPISODE_LOG_INTERVAL episodes
        if self.overfit_detector_episode_count % EPISODE_LOG_INTERVAL == 0:
            self.overfitting_detector.log_overfitting_analysis()

        # Reset overfitting tracking for new episode
        self.overfit_detector_current_episode_actions.clear()
        self.overfit_detector_current_episode_positions.clear()
        self.overfit_detector_current_episode_rewards.clear()

    def copy(self) -> "MLPBrain":
        """
        Create a copy of the MLPBrain instance.

        MLPBrain does not support copying as it is a simple neural network.
        Use deepcopy if needed.
        """
        error_msg = "MLPBrain does not support copying. Use deepcopy if needed."
        raise NotImplementedError(error_msg)

    @property
    def action_set(self) -> list[Action]:
        """Get the list of actions."""
        return self._action_set

    @action_set.setter
    def action_set(self, actions: list[Action]) -> None:
        self._action_set = actions
