"""
Q-Learning Multi-Layer Perceptron (QMLP) Brain Architecture.

This architecture implements a Deep Q-Network (DQN) approach using a classical
multi-layer perceptron to learn optimal action-value functions for navigation tasks.

Key Features:
- **Deep Q-Learning**: Uses neural networks to approximate Q-values for state-action pairs
- **Target Networks**: Maintains separate target network for stable Q-learning updates
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation with decaying epsilon
- **Experience-Based Learning**: Updates Q-values based on observed state transitions and rewards
- **Gradient Clipping**: Prevents training instability through bounded gradient updates

Architecture:
- Input: 2D state features (gradient strength, relative direction to goal)
- Hidden: Configurable MLP layers with ReLU activation
- Output: Q-values for each possible action (forward, left, right, stay)

The QMLP brain learns by:
1. Observing current state and selecting action via epsilon-greedy policy
2. Experiencing reward and next state from environment
3. Computing Q-learning target: reward + gamma * max(Q(next_state))
4. Updating main network to minimize TD error
5. Periodically copying weights to target network for stability

This approach typically provides more stable and sample-efficient learning compared
to policy gradient methods for discrete action spaces like grid navigation.
"""

import numpy as np
import torch  # pyright: ignore[reportMissingImports]
from torch import nn, optim  # pyright: ignore[reportMissingImports]

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, ClassicalBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch.dtypes import BrainConfig, DeviceType
from quantumnematode.env import Direction
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


class QMLPBrain(ClassicalBrain):
    """
    Q-learning based MLP brain architecture.

    Uses epsilon-greedy exploration and experience replay for more stable learning.
    """

    def __init__(
        self,
        config: QMLPBrainConfig,
        input_dim: int,
        num_actions: int,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] = DEFAULT_ACTIONS,
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
        """Q-learning update."""
        if self.last_state is None or self.last_action is None:
            return  # No previous state to learn from

        # Current state Q-values (for next Q-value)
        current_state = self.preprocess(params)
        with torch.no_grad():
            current_q_values = self.forward(current_state)
            next_q_value = torch.max(current_q_values).item()

        # Q-learning target
        target_q_value = reward + self.gamma * next_q_value

        # Recompute Q-values for the previous state to get gradients
        last_state_tensor = torch.from_numpy(self.last_state).float().to(self.device)
        previous_q_values = self.q_network(last_state_tensor)
        current_q_value = previous_q_values[int(self.last_action)]

        # Compute loss
        target_tensor = torch.tensor(target_q_value, device=self.device, dtype=torch.float32)
        loss = self.loss_fn(current_q_value, target_tensor)

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
