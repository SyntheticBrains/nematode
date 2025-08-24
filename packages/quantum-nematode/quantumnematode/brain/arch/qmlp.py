"""
Q-Learning Multi-Layer Perceptron (QMLP) Brain Architecture.

This architecture implements a Deep Q-Network (DQN) approach using a classical
multi-layer perceptron to learn optimal action-value functions for navigation tasks.

Key Features:
- **Deep Q-Learning**: Uses neural networks to approximate Q-values for state-action pairs
- **Target Networks**: Maintains separate target network for stable Q-learning updates
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation with decaying epsilon
- **Experience Replay**: Uses experience buffer for stable batch learning
- **Gradient Clipping**: Prevents training instability through bounded gradient updates

Architecture:
- Input: 2D state features (gradient strength, relative direction to goal)
- Hidden: Configurable MLP layers with ReLU activation
- Output: Q-values for each possible action (forward, left, right, stay)

The QMLP brain learns by:
1. Observing current state and selecting action via epsilon-greedy policy
2. Storing experience (state, action, reward, next_state) in replay buffer
3. Sampling random batch from buffer for learning
4. Computing Q-learning targets using target network
5. Updating main network to minimize TD error
6. Periodically copying weights to target network for stability

This approach provides more stable and sample-efficient learning compared
to policy gradient methods for discrete action spaces like grid navigation.
"""

import random
from collections import deque

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


class QMLPBrainConfig(BrainConfig):
    """Configuration for the Q-learning MLP Brain architecture."""

    hidden_dim: int = 64
    learning_rate: float = 0.001
    epsilon: float = 0.1  # Exploration rate
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    gamma: float = 0.95  # Discount factor
    target_update_freq: int = 100  # How often to update target network
    num_hidden_layers: int = 2
    buffer_size: int = 10000  # Experience replay buffer size
    batch_size: int = 32  # Batch size for training


class QMLPBrain(ClassicalBrain):
    """
    Q-learning based MLP brain architecture.

    Uses epsilon-greedy exploration and experience replay for more stable learning.
    """

    def __init__(  # noqa: PLR0913
        self,
        config: QMLPBrainConfig,
        input_dim: int,
        num_actions: int,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] = DEFAULT_ACTIONS,
        parameter_initializer: ParameterInitializer | None = None,
    ) -> None:
        super().__init__()

        logger.info(f"Using Q-MLP configuration: {config}")

        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.device = torch.device(device.value)

        # Q-networks
        self.q_network = self._build_network(config.hidden_dim, config.num_hidden_layers).to(
            self.device,
        )
        self.target_q_network = self._build_network(config.hidden_dim, config.num_hidden_layers).to(
            self.device,
        )

        # Initialize parameters with custom initializer or log default initialization
        self._initialize_parameters(parameter_initializer)

        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.MSELoss()

        # Q-learning parameters
        self.epsilon = config.epsilon
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_min = config.epsilon_min
        self.gamma = config.gamma
        self.target_update_freq = config.target_update_freq
        self.update_count = 0

        # Experience replay buffer
        self.buffer_size = config.buffer_size
        self.batch_size = config.batch_size
        self.experience_buffer = deque(maxlen=self.buffer_size)

        self.satiety = 1.0
        self.training = True
        self._action_set = action_set

        # Store last state-action for learning
        self.last_state = None
        self.last_action = None
        self.last_q_values = None

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
            for name, param in self.q_network.named_parameters():
                param_count += param.numel()

                if parameter_initializer is not None:
                    # NOTE: This is a placeholder for future custom initialization logic
                    logger.info(
                        f"Custom parameter initialization not fully implemented for QMLP {name}",
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

        logger.info("QMLPBrain parameter initialization complete:")
        logger.info(f"  Total parameters: {param_count:,}")
        logger.info("  Parameter details:")
        for detail in param_details:
            logger.info(detail)

    def preprocess(self, params: BrainParams) -> np.ndarray:
        """Preprocess brain parameters into feature vector."""
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

    def forward(self, x: np.ndarray) -> torch.Tensor:
        """Forward pass through the Q-network."""
        x_tensor = torch.from_numpy(x).float().to(self.device)
        return self.q_network(x_tensor)

    def run_brain(
        self,
        params: BrainParams,
        reward: float | None = None,  # noqa: ARG002
        input_data: list[float] | None = None,  # noqa: ARG002
        *,
        top_only: bool,  # noqa: ARG002
        top_randomize: bool,  # noqa: ARG002
    ) -> list[ActionData]:
        """Run epsilon-greedy action selection."""
        state = self.preprocess(params)
        q_values = self.forward(state)

        # Epsilon-greedy action selection
        rng = np.random.default_rng()
        if self.training and rng.random() < self.epsilon:
            action_idx = rng.integers(0, self.num_actions)
        else:
            action_idx = torch.argmax(q_values).item()

        action_name = self.action_set[int(action_idx)]

        # Store for learning
        self.last_state = state
        self.last_action = action_idx
        self.last_q_values = q_values.detach()

        prob = 1.0 - self.epsilon + self.epsilon / self.num_actions if self.training else 1.0

        self.latest_data.action = ActionData(
            state=action_name,
            action=action_name,
            probability=prob,
        )

        self.latest_data.probability = prob
        self.history_data.actions.append(self.latest_data.action)
        self.history_data.probabilities.append(prob)

        actions = {name: int(i == action_idx) for i, name in enumerate(self.action_set)}
        return self._get_most_probable_action(actions)

    def _get_most_probable_action(self, counts: dict) -> list[ActionData]:
        """Return the selected action."""
        action_name = max(counts.items(), key=lambda x: x[1])[0]
        return [
            ActionData(
                state=action_name,
                action=action_name,
                probability=self.latest_data.probability or 1.0,
            ),
        ]

    def learn(
        self,
        params: BrainParams,
        reward: float,
    ) -> None:
        """Q-learning update with experience replay."""
        if self.last_state is None or self.last_action is None:
            return  # No previous state to learn from

        # Store experience in buffer
        current_state = self.preprocess(params)
        is_terminal = reward >= 1.0

        experience = (
            self.last_state.copy(),  # state
            self.last_action,  # action
            reward,  # reward
            current_state.copy(),  # next_state
            is_terminal,  # done
        )
        self.experience_buffer.append(experience)

        # Only train if we have enough experiences
        if len(self.experience_buffer) < self.batch_size:
            return

        # Sample batch from experience buffer
        batch = random.sample(self.experience_buffer, self.batch_size)
        states = torch.tensor([exp[0] for exp in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([exp[1] for exp in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([exp[2] for exp in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(
            [exp[3] for exp in batch],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor([exp[4] for exp in batch], dtype=torch.bool, device=self.device)

        # Current Q-values for selected actions
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_q_network(next_states).max(1)[0]
            # Set next Q-value to 0 for terminal states
            next_q_values[dones] = 0.0
            target_q_values = rewards + (self.gamma * next_q_values)

        # Compute loss
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)

        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Store history
        self.latest_data.loss = loss.item()
        self.history_data.rewards.append(reward)
        if self.latest_data.loss is not None:
            self.history_data.losses.append(self.latest_data.loss)

    def update_parameters(
        self,
        gradients: list[float],
        reward: float | None = None,
        learning_rate: float = 0.01,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Not used in Q-learning MLP."""

    def update_memory(
        self,
        reward: float | None = None,  # noqa: ARG002
    ) -> None:
        """No-op for Q-MLP."""
        return

    def post_process_episode(self) -> None:
        """Post-process the brain's state after each episode."""
        # Not implemented
        return

    def build_brain(self):  # noqa: ANN201
        """Not applicable to Q-MLP brain."""
        error_msg = "Q-MLPBrain does not have a quantum circuit."
        raise NotImplementedError(error_msg)

    def copy(self) -> "QMLPBrain":
        """Not implemented."""
        error_msg = "Q-MLPBrain does not support copying."
        raise NotImplementedError(error_msg)

    @property
    def action_set(self) -> list[Action]:
        """Get the list of actions."""
        return self._action_set

    @action_set.setter
    def action_set(self, actions: list[Action]) -> None:
        self._action_set = actions
