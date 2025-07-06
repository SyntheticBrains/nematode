"""
Classical Multi-Layer Perceptron (MLP) Brain Architecture.

This architecture uses a simple MLP policy network to process the agent's state and
select actions based on learned policies.
It supports GPU acceleration via PyTorch and includes features for training,
action selection, and reward-based learning.
"""

import numpy as np
import torch  # pyright: ignore[reportMissingImports]
from torch import nn, optim  # pyright: ignore[reportMissingImports]

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, ClassicalBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.env import Direction
from quantumnematode.logging_config import logger


class MLPBrain(ClassicalBrain):
    """
    Classical multi-layer perceptron (MLP) brain architecture.

    Uses a simple MLP policy network with optional GPU acceleration.
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        num_actions: int,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        device: str = "cpu",
        learning_rate: float = 0.01,
        action_set: list[Action] = DEFAULT_ACTIONS,
        lr_scheduler: bool | None = None,
        entropy_beta: float = 0.01,
    ) -> None:
        super().__init__()

        logger.info(
            "Initializing MLPBrain with input_dim=%d, num_actions=%d, hidden_dim=%d, "
            "num_hidden_layers=%d, device=%s, learning_rate=%.4f, entropy_beta=%.4f",
            input_dim,
            num_actions,
            hidden_dim,
            num_hidden_layers,
            device,
            learning_rate,
            entropy_beta,
        )

        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.device = torch.device(device)
        self.entropy_beta = entropy_beta
        self.policy = self._build_network(hidden_dim, num_hidden_layers).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        if lr_scheduler is None:
            lr_scheduler = False
        self.lr_scheduler = (
            optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
            if lr_scheduler
            else None
        )
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.satiety = 1.0

        self.current_probabilities = None

        self.training = True
        self._action_set = action_set

        # Baseline for variance reduction in policy gradient
        self.baseline = 0.0
        self.baseline_alpha = 0.05  # Smoothing factor for running average

    def _build_network(self, hidden_dim: int, num_hidden_layers: int) -> nn.Sequential:
        layers = [nn.Linear(self.input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, self.num_actions))
        return nn.Sequential(*layers)

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
        probs = torch.softmax(logits, dim=-1)
        probs_np = probs.detach().cpu().numpy()
        rng = np.random.default_rng()
        action_idx = rng.choice(self.num_actions, p=probs_np)
        action_name = self.action_set[action_idx]

        self.latest_data.action = ActionData(
            state=action_name,
            action=action_name,
            probability=probs_np[action_idx],
        )

        self.current_probabilities = probs_np
        first_prob = float(probs_np[action_idx])
        self.latest_data.probability = first_prob
        self.history_data.actions.append(self.latest_data.action)
        self.history_data.probabilities.append(first_prob)

        actions = {name: int(i == action_idx) for i, name in enumerate(self.action_set)}
        return self._get_most_probable_action(actions)

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
        params: BrainParams,
        action_idx: int,
        reward: float,
        episode_rewards: list[float] | None = None,
        gamma: float = 0.99,
    ) -> None:
        """Perform a policy gradient update step with baseline and discounted return."""
        x = self.preprocess(params)
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        log_prob = torch.log(probs[action_idx] + 1e-8)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))

        # Use discounted return if provided, else use immediate reward
        if episode_rewards is None and self.history_data.rewards is not None:
            episode_rewards = self.history_data.rewards
        if episode_rewards is not None:
            g = self.compute_discounted_return(episode_rewards, gamma=gamma)
        else:
            g = reward

        # Subtract baseline from return
        advantage = g - self.baseline
        loss = -log_prob * advantage - self.entropy_beta * entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        self.latest_data.loss = loss.item()

        self.history_data.rewards.append(reward)
        self.history_data.losses.append(self.latest_data.loss)

        # Update baseline (running average)
        self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * g

    def update_memory(
        self,
        reward: float | None = None,  # noqa: ARG002
    ) -> None:
        """No-op for MLPBrain."""
        return

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
