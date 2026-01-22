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

from typing import Literal

import numpy as np
import torch

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, ClassicalBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch._spiking_layers import OutputMode, SpikingPolicyNetwork
from quantumnematode.brain.arch.dtypes import BrainConfig, DeviceType
from quantumnematode.env import Direction
from quantumnematode.logging_config import logger
from quantumnematode.utils.seeding import ensure_seed, get_rng, set_global_seed

# Default configuration parameters
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_NUM_TIMESTEPS = 100
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
DEFAULT_SURROGATE_ALPHA = 1.0
# Weight initialization method options
WeightInitMethod = Literal["orthogonal", "kaiming", "xavier", "default", "orthogonal_kaiming_input"]
DEFAULT_WEIGHT_INIT: WeightInitMethod = "orthogonal"

DEFAULT_BATCH_SIZE = 1  # Number of episodes to accumulate before gradient update

# Adaptive exploration parameters - decay from exploration to exploitation
DEFAULT_EXPLORATION_LOGIT_CLAMP = 2.0  # Initial logit clamp (tighter = more exploration)
DEFAULT_EXPLORATION_LOGIT_CLAMP_FINAL = 5.0  # Final logit clamp (looser = more exploitation)
DEFAULT_EXPLORATION_NOISE_STD = 0.3  # Initial noise standard deviation
DEFAULT_EXPLORATION_NOISE_STD_FINAL = 0.05  # Final noise (small but non-zero)
DEFAULT_EXPLORATION_TEMPERATURE = 1.5  # Initial temperature
DEFAULT_EXPLORATION_TEMPERATURE_FINAL = 1.0  # Final temperature
DEFAULT_EXPLORATION_DECAY_EPISODES = 30  # Episodes over which to decay exploration


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
    weight_init: WeightInitMethod = DEFAULT_WEIGHT_INIT

    # Batch learning (accumulate multiple episodes before gradient update)
    batch_size: int = DEFAULT_BATCH_SIZE

    # Warmup episodes - skip gradient updates for first N episodes
    # Allows random policy to explore before learning corrupts it
    warmup_episodes: int = 0

    # Adaptive exploration - decay from high exploration to exploitation
    exploration_logit_clamp: float = DEFAULT_EXPLORATION_LOGIT_CLAMP
    exploration_logit_clamp_final: float = DEFAULT_EXPLORATION_LOGIT_CLAMP_FINAL
    exploration_noise_std: float = DEFAULT_EXPLORATION_NOISE_STD
    exploration_noise_std_final: float = DEFAULT_EXPLORATION_NOISE_STD_FINAL
    exploration_temperature: float = DEFAULT_EXPLORATION_TEMPERATURE
    exploration_temperature_final: float = DEFAULT_EXPLORATION_TEMPERATURE_FINAL
    exploration_decay_episodes: int = DEFAULT_EXPLORATION_DECAY_EPISODES

    # Output mode for temporal spike aggregation
    # - "accumulator": Sum spikes over all timesteps (default, original behavior)
    # - "final": Use only final timestep's spike pattern (less smoothing)
    # - "membrane": Use final membrane potentials (continuous, most variation)
    output_mode: OutputMode = "accumulator"

    # Temporal modulation - prevents LIF neurons from reaching steady state
    # by varying input current over timesteps with sinusoidal pattern
    temporal_modulation: bool = False
    modulation_amplitude: float = 0.3  # Fraction of input to modulate
    modulation_period: int = 20  # Period in timesteps (~5 cycles over 100 timesteps)

    # Population coding - encode inputs using Gaussian tuning curves
    # Creates distinct activation patterns for different input values
    population_coding: bool = False
    neurons_per_feature: int = 8  # Number of encoding neurons per input feature
    population_sigma: float = 0.25  # Width of Gaussian tuning curves

    # Advantage clipping - prevents catastrophic gradient updates from large negative returns
    # Clips normalized advantages to [-clip, +clip] range
    advantage_clip: float = 0.0  # 0 = disabled, recommended: 2.0-3.0

    # Action probability floor - prevents entropy collapse by ensuring minimum probability
    # for all actions. Clamps probabilities to [min_action_prob, 1-min_action_prob*(n-1)]
    min_action_prob: float = 0.0  # 0 = disabled, recommended: 0.01-0.05

    # Return clipping - clips total episode return to [-clip, +clip] range
    # This limits the magnitude of gradient updates from extreme episodes
    return_clip: float = 0.0  # 0 = disabled, recommended: 50.0

    # Intra-episode update frequency - perform gradient updates every N steps
    # instead of only at episode end. This provides denser learning signals.
    # 0 = disabled (update only at episode end), recommended: 5 (like MLP brain)
    update_frequency: int = 0

    # Separated gradients - use separate food and predator gradient inputs
    # instead of the combined gradient. Enables the network to learn distinct
    # responses to appetitive (food) vs aversive (predator) signals.
    # When enabled, input_dim becomes 4: [food_strength, food_angle, pred_strength, pred_angle]
    use_separated_gradients: bool = False


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

    def __init__(
        self,
        config: SpikingBrainConfig,
        input_dim: int,
        num_actions: int,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] = DEFAULT_ACTIONS,
    ) -> None:
        super().__init__()

        # Initialize seeding for reproducibility
        self.seed = ensure_seed(config.seed)
        self.rng = get_rng(self.seed)
        set_global_seed(self.seed)  # Set global numpy/torch seeds
        logger.info(f"SpikingBrain using seed: {self.seed}")

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
            output_mode=config.output_mode,
            temporal_modulation=config.temporal_modulation,
            modulation_amplitude=config.modulation_amplitude,
            modulation_period=config.modulation_period,
            population_coding=config.population_coding,
            neurons_per_feature=config.neurons_per_feature,
            population_sigma=config.population_sigma,
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

        # Batch learning buffers - accumulate multiple episodes before gradient update
        # Each element is a complete episode's data
        self.batch_episodes_states: list[list[np.ndarray]] = []
        self.batch_episodes_actions: list[list[int]] = []
        self.batch_episodes_action_probs: list[list[torch.Tensor]] = []
        self.batch_episodes_rewards: list[list[float]] = []
        self.batch_episode_count = 0  # Track episodes in current batch
        self.total_episodes_seen = 0  # Track total episodes for warmup

        # Baseline for variance reduction (running average of returns)
        self.baseline = 0.0

        # Intra-episode update tracking (for update_frequency feature)
        self.steps_since_update = 0

        # Episode counter for decay schedules
        self.episode_count = 0
        self.initial_learning_rate = config.learning_rate
        self.initial_entropy_beta = config.entropy_beta

        # Log parameter count
        total_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        logger.info(f"SpikingBrain initialized with {total_params:,} trainable parameters")

    @property
    def action_set(self) -> list[Action]:
        """Return the set of available actions."""
        return self._action_set

    def preprocess(self, params: BrainParams) -> np.ndarray:
        """
        Preprocess brain parameters into state vector.

        Computes relative angle between agent orientation and goal direction,
        matching the preprocessing used by MLPBrain for fair comparison.

        When use_separated_gradients is enabled, returns 4 features:
        [food_strength, food_rel_angle, predator_strength, predator_rel_angle]

        Parameters
        ----------
        params : BrainParams
            Brain parameters containing environmental state

        Returns
        -------
        np.ndarray
            Preprocessed state vector. Either:
            - 2 features: [gradient_strength, relative_angle_normalized] (combined)
            - 4 features: [food_strength, food_angle, pred_strength, pred_angle] (separated)
            All angles normalized to [-1, 1]
        """
        # Map agent direction to angle (radians)
        direction_map = {
            Direction.UP: 0.5 * np.pi,
            Direction.RIGHT: 0.0,
            Direction.DOWN: 1.5 * np.pi,
            Direction.LEFT: np.pi,
        }
        agent_facing_angle = (
            direction_map[params.agent_direction] if params.agent_direction is not None else 0.0
        )

        def compute_relative_angle(direction: float | None) -> float:
            """Compute relative angle normalized to [-1, 1]."""
            if direction is None:
                return 0.0
            # Compute relative angle: goal direction - agent facing direction
            # Normalize to [-π, π]
            relative_angle = (direction - agent_facing_angle + np.pi) % (2 * np.pi) - np.pi
            # Normalize to [-1, 1] for network input
            return relative_angle / np.pi

        # Use separated gradients if configured and available
        if self.config.use_separated_gradients:
            # Food gradient features
            food_strength = float(params.food_gradient_strength or 0.0)
            food_strength = max(0.0, min(1.0, food_strength))
            food_rel_angle = compute_relative_angle(params.food_gradient_direction)

            # Predator gradient features
            pred_strength = float(params.predator_gradient_strength or 0.0)
            pred_strength = max(0.0, min(1.0, pred_strength))
            pred_rel_angle = compute_relative_angle(params.predator_gradient_direction)

            return np.array(
                [food_strength, food_rel_angle, pred_strength, pred_rel_angle],
                dtype=np.float32,
            )

        # Default: use combined gradient (2 features)
        grad_strength = float(params.gradient_strength or 0.0)
        grad_strength = max(0.0, min(1.0, grad_strength))
        rel_angle_normalized = compute_relative_angle(params.gradient_direction)

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
        with torch.no_grad():
            action_logits = self.policy(state_tensor)

            # Adaptive exploration: decay from high exploration to exploitation over episodes
            # This allows early random exploration while enabling confident exploitation later
            if self.config.exploration_decay_episodes > 0:
                decay_progress = min(
                    1.0,
                    self.total_episodes_seen / self.config.exploration_decay_episodes,
                )
            else:
                decay_progress = 1.0  # No decay, use final values

            # Interpolate exploration parameters
            current_clamp = self.config.exploration_logit_clamp + decay_progress * (
                self.config.exploration_logit_clamp_final - self.config.exploration_logit_clamp
            )
            current_noise_std = self.config.exploration_noise_std + decay_progress * (
                self.config.exploration_noise_std_final - self.config.exploration_noise_std
            )
            current_temperature = self.config.exploration_temperature + decay_progress * (
                self.config.exploration_temperature_final - self.config.exploration_temperature
            )

            # 1. Clamp logits - tight early (force exploration), loose later (allow confidence)
            action_logits = torch.clamp(
                action_logits,
                min=-current_clamp,
                max=current_clamp,
            )

            # 2. Add noise during training - high early, low later
            if self.policy.training and current_noise_std > 0:
                noise = torch.normal(
                    0,
                    current_noise_std,
                    size=action_logits.shape,
                    device=self.device,
                )
                action_logits = action_logits + noise

            # 3. Temperature sampling - high early (random), low later (greedy)
            temperature = current_temperature if self.policy.training else 1.0
            action_probs = torch.softmax(action_logits / temperature, dim=-1)

            # 4. Action probability floor - prevent entropy collapse
            if self.config.min_action_prob > 0:
                num_actions = action_probs.shape[-1]
                min_prob = self.config.min_action_prob
                max_prob = 1.0 - min_prob * (num_actions - 1)
                action_probs = torch.clamp(action_probs, min=min_prob, max=max_prob)
                # Renormalize to ensure sum = 1
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

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

        logger.debug(
            f"SpikingBrain selected action {selected_action} with probability {probability:.3f}",
        )

        return [action_data]

    def learn(
        self,
        params: BrainParams,  # noqa: ARG002
        reward: float,
        *,
        episode_done: bool,
    ) -> None:
        """
        Update network parameters using policy gradient (REINFORCE) with batch learning.

        When batch_size > 1, accumulates experiences across multiple episodes before
        performing a gradient update. This prevents single bad episodes from corrupting
        the policy and allows mixing of successful and failed experiences.

        When update_frequency > 0, performs intra-episode updates every N steps,
        providing denser learning signals similar to MLP brain.

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
        self.steps_since_update += 1

        # Intra-episode updates: perform gradient update every N steps (like MLP brain)
        if (
            self.config.update_frequency > 0
            and self.steps_since_update >= self.config.update_frequency
            and len(self.episode_states) >= self.config.update_frequency
        ):
            self._perform_intra_episode_update()
            self.steps_since_update = 0

        if episode_done and self.episode_rewards:
            self.total_episodes_seen += 1

            # During warmup, just clear buffers without learning
            if self.total_episodes_seen <= self.config.warmup_episodes:
                logger.debug(
                    f"Warmup period - skipping learning "
                    f"({self.total_episodes_seen}/{self.config.warmup_episodes})",
                )
                self.episode_states.clear()
                self.episode_actions.clear()
                self.episode_action_probs.clear()
                self.episode_rewards.clear()
                return

            # Store completed episode in batch buffers
            self.batch_episodes_states.append(list(self.episode_states))
            self.batch_episodes_actions.append(list(self.episode_actions))
            self.batch_episodes_action_probs.append(list(self.episode_action_probs))
            self.batch_episodes_rewards.append(list(self.episode_rewards))
            self.batch_episode_count += 1

            # Clear current episode buffers
            self.episode_states.clear()
            self.episode_actions.clear()
            self.episode_action_probs.clear()
            self.episode_rewards.clear()

            logger.debug(
                "Episode complete - added to batch "
                f"({self.batch_episode_count}/{self.config.batch_size})",
            )

            # Only perform gradient update when batch is full
            if self.batch_episode_count >= self.config.batch_size:
                self._perform_batch_gradient_update()

    def _perform_intra_episode_update(self) -> None:
        """
        Perform gradient update using recent steps within the current episode.

        This provides denser learning signals by updating every N steps instead of
        waiting until episode end. Similar to MLP brain's update_frequency=5 approach.
        Uses the last update_frequency steps of data for the update.
        """
        n_steps = self.config.update_frequency

        # Get the last n_steps of data
        recent_states = self.episode_states[-n_steps:]
        recent_actions = self.episode_actions[-n_steps:]
        recent_action_probs = self.episode_action_probs[-n_steps:]
        recent_rewards = self.episode_rewards[-n_steps:]

        if len(recent_states) < n_steps:
            return  # Not enough data yet

        # Compute discounted returns for recent steps
        returns: list[float] = []
        g_value = 0.0
        for r in reversed(recent_rewards):
            g_value = r + self.config.gamma * g_value
            returns.insert(0, g_value)

        # Apply return clipping
        if self.config.return_clip > 0:
            clip_val = self.config.return_clip
            returns = [max(-clip_val, min(clip_val, r)) for r in returns]

        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize returns for variance reduction
        if len(returns) > 1:
            returns_mean = returns_tensor.mean()
            returns_std = returns_tensor.std()
            returns_tensor = (returns_tensor - returns_mean) / (returns_std + 1e-8)

            # Update baseline
            self.baseline = (
                self.config.baseline_alpha * returns_mean.item()
                + (1 - self.config.baseline_alpha) * self.baseline
            )

        advantages = returns_tensor

        # Apply advantage clipping
        if self.config.advantage_clip > 0:
            advantages = torch.clamp(
                advantages,
                -self.config.advantage_clip,
                self.config.advantage_clip,
            )

        # Recompute log_probs with fresh forward passes
        log_probs_list = []
        for state, action_idx in zip(recent_states, recent_actions, strict=False):
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            action_logits = self.policy(state_tensor)
            action_probs = torch.softmax(action_logits, dim=-1)
            action_probs = self._apply_probability_floor(action_probs)

            action_dist = torch.distributions.Categorical(action_probs)
            log_prob = action_dist.log_prob(torch.tensor(action_idx, device=self.device))
            log_probs_list.append(log_prob)

        # Compute policy loss
        log_probs = torch.stack(log_probs_list)
        policy_loss = -(log_probs * advantages).mean()

        # Add entropy regularization
        current_entropy_beta = self._get_current_entropy_beta()
        entropy = torch.tensor(0.0, device=self.device)
        if current_entropy_beta > 0 and recent_action_probs:
            action_probs_stacked = torch.stack(recent_action_probs)
            entropy = (
                -(action_probs_stacked * torch.log(action_probs_stacked + 1e-8)).sum(dim=-1).mean()
            )
            policy_loss = policy_loss - current_entropy_beta * entropy

        # Clip loss
        policy_loss = torch.clamp(policy_loss, -10.0, 10.0)

        # Backpropagation
        self.optimizer.zero_grad()
        policy_loss.backward()

        # Clip gradients
        for param in self.policy.parameters():
            if param.grad is not None:
                param.grad.clamp_(-1.0, 1.0)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)

        # Update parameters
        self.optimizer.step()

        logger.debug(
            f"Intra-episode update: loss={policy_loss.item():.4f}, "
            f"steps={n_steps}, entropy={entropy.item():.4f}",
        )

    def _perform_batch_gradient_update(self) -> None:  # noqa: C901, PLR0915
        """
        Perform gradient update using accumulated batch of episodes.

        Computes gradients across all episodes in the batch, averaging the loss
        to prevent any single episode from dominating the update.
        """
        total_timesteps = sum(len(ep) for ep in self.batch_episodes_rewards)
        logger.info(
            f"Batch complete - performing policy gradient update with "
            f"{self.batch_episode_count} episodes, {total_timesteps} total timesteps",
        )

        # Compute returns and advantages for each episode
        all_returns: list[torch.Tensor] = []
        all_states: list[np.ndarray] = []
        all_actions: list[int] = []
        all_action_probs: list[torch.Tensor] = []

        for ep_idx in range(self.batch_episode_count):
            ep_rewards = self.batch_episodes_rewards[ep_idx]
            ep_states = self.batch_episodes_states[ep_idx]
            ep_actions = self.batch_episodes_actions[ep_idx]
            ep_action_probs = self.batch_episodes_action_probs[ep_idx]

            # Compute discounted returns backward through episode
            returns: list[float] = []
            g_value = 0.0
            for r in reversed(ep_rewards):
                g_value = r + self.config.gamma * g_value
                returns.insert(0, g_value)

            # Apply return clipping to limit extreme episode returns
            if self.config.return_clip > 0:
                clip_val = self.config.return_clip
                returns = [max(-clip_val, min(clip_val, r)) for r in returns]

            returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)

            # Normalize returns within episode for variance reduction
            if len(returns) > 1:
                returns_mean = returns_tensor.mean()
                returns_std = returns_tensor.std()
                returns_tensor = (returns_tensor - returns_mean) / (returns_std + 1e-8)

                # Update baseline (running average of mean episode return)
                self.baseline = (
                    self.config.baseline_alpha * returns_mean.item()
                    + (1 - self.config.baseline_alpha) * self.baseline
                )

            all_returns.append(returns_tensor)
            all_states.extend(ep_states)
            all_actions.extend(ep_actions)
            all_action_probs.extend(ep_action_probs)

        # Concatenate all returns (advantages)
        advantages = torch.cat(all_returns)

        # Apply advantage clipping to prevent catastrophic gradient updates
        if self.config.advantage_clip > 0:
            advantages = torch.clamp(
                advantages,
                -self.config.advantage_clip,
                self.config.advantage_clip,
            )

        # Recompute log_probs with fresh forward passes to enable gradient flow
        log_probs_list = []
        for state, action_idx in zip(all_states, all_actions, strict=False):
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            action_logits = self.policy(state_tensor)
            action_probs = torch.softmax(action_logits, dim=-1)
            action_probs = self._apply_probability_floor(action_probs)

            action_dist = torch.distributions.Categorical(action_probs)
            log_prob = action_dist.log_prob(torch.tensor(action_idx, device=self.device))
            log_probs_list.append(log_prob)

        # Compute policy loss: -Σ log_prob(a_t) * advantage_t
        # Average over batch to normalize gradient magnitude
        log_probs = torch.stack(log_probs_list)
        policy_loss = -(
            log_probs * advantages
        ).mean()  # mean instead of sum for batch normalization

        # Apply entropy decay schedule
        current_entropy_beta = self._get_current_entropy_beta()

        # Add entropy regularization for exploration
        entropy = torch.tensor(0.0, device=self.device)
        if current_entropy_beta > 0 and all_action_probs:
            # Entropy: -Σ p(a) * log(p(a)) for each timestep, averaged
            action_probs_stacked = torch.stack(all_action_probs)
            entropy = (
                -(action_probs_stacked * torch.log(action_probs_stacked + 1e-8)).sum(dim=-1).mean()
            )

            # Log entropy for diagnostics
            logger.debug(
                f"Batch entropy: {entropy.item():.4f}, "
                f"entropy_beta: {current_entropy_beta:.4f}, "
                f"entropy contribution: {current_entropy_beta * entropy.item():.4f}",
            )

            # Subtract entropy to maximize it (entropy bonus)
            policy_loss = policy_loss - current_entropy_beta * entropy

        # Clip loss to prevent extreme updates (matching MLP brain)
        policy_loss = torch.clamp(policy_loss, -10.0, 10.0)

        # Backpropagation
        self.optimizer.zero_grad()
        policy_loss.backward()

        # Clip individual gradient values first to prevent explosion
        for param in self.policy.parameters():
            if param.grad is not None:
                param.grad.clamp_(-1.0, 1.0)

        # Then clip gradient norm for overall stability (0.5 matches MLP brain)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)

        # Log gradient norm for diagnostics
        logger.debug(f"Gradient norm: {grad_norm.item():.6f}")

        # Update parameters
        self.optimizer.step()

        # Apply learning rate decay schedule (once per batch, not per episode)
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
        total_return = sum(sum(ep) for ep in self.batch_episodes_rewards)
        avg_return = total_return / self.batch_episode_count
        logger.debug(
            f"Batch gradient update: loss={policy_loss.item():.4f}, "
            f"avg_episode_return={avg_return:.2f}, baseline={self.baseline:.2f}, "
            f"entropy={entropy.item():.4f}, episodes={self.batch_episode_count}",
        )

        # Increment episode counter by batch size (for decay schedules)
        self.episode_count += self.batch_episode_count

        # Clear batch buffers
        self.batch_episodes_states.clear()
        self.batch_episodes_actions.clear()
        self.batch_episodes_action_probs.clear()
        self.batch_episodes_rewards.clear()
        self.batch_episode_count = 0

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

    def _apply_probability_floor(self, action_probs: torch.Tensor) -> torch.Tensor:
        """
        Apply probability floor to prevent entropy collapse.

        Clamps action probabilities to ensure no action has probability below
        min_action_prob, then renormalizes to sum to 1.

        Parameters
        ----------
        action_probs : torch.Tensor
            Raw action probabilities from softmax

        Returns
        -------
        torch.Tensor
            Clamped and renormalized action probabilities
        """
        if self.config.min_action_prob <= 0:
            return action_probs

        num_actions = action_probs.shape[-1]
        min_prob = self.config.min_action_prob
        max_prob = 1.0 - min_prob * (num_actions - 1)
        action_probs = torch.clamp(action_probs, min=min_prob, max=max_prob)
        return action_probs / action_probs.sum(dim=-1, keepdim=True)

    def _initialize_weights(self, method: WeightInitMethod) -> None:  # noqa: C901
        """
        Initialize network weights using specified method.

        Parameters
        ----------
        method : WeightInitMethod
            Initialization method:
            - "orthogonal": Preserves gradient norms through layers (best for deep SNNs)
            - "kaiming": Variance-preserving for ReLU-like activations
            - "xavier": Variance-preserving for tanh/sigmoid activations
            - "default": PyTorch default (uniform based on fan-in)
            - "orthogonal_kaiming_input": Kaiming for input layer (good for population
              coded inputs), orthogonal for hidden layers
        """
        if method == "default":
            # PyTorch already initialized, nothing to do
            logger.info("Using PyTorch default weight initialization")
            return

        logger.info(f"Initializing weights with {method} method")

        # Special case: hybrid initialization for population coding
        if method == "orthogonal_kaiming_input":
            self._initialize_hybrid_weights()
            return

        for _name, module in self.policy.named_modules():
            if isinstance(module, torch.nn.Linear):
                if method == "orthogonal":
                    # Orthogonal initialization - preserves gradient norms through layers
                    # Excellent for deep networks and recurrent-like architectures (SNNs)
                    # Gain of 1.0 preserves input variance
                    torch.nn.init.orthogonal_(module.weight, gain=1.0)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

                elif method == "kaiming":
                    # Kaiming/He initialization for ReLU-like activations
                    # Good for spiking neurons which have piecewise activation
                    torch.nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

                elif method == "xavier":
                    # Xavier/Glorot initialization for tanh/sigmoid
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

        # Scale down output layer weights for kaiming init
        # This produces reasonable initial logit magnitudes from spike accumulators
        if method == "kaiming" and hasattr(self.policy, "output_layer"):
            self.policy.output_layer.weight.data *= 0.01
            logger.info("Scaled down output layer weights by 0.01 for balanced probabilities")

    def _initialize_hybrid_weights(self) -> None:
        """
        Hybrid initialization: Kaiming for input layer, orthogonal for hidden layers.

        This is optimized for population-coded inputs:
        - Input layer uses Kaiming initialization which is well-suited for the
          Gaussian-shaped population-coded features (similar to ReLU activations)
        - Hidden layers use orthogonal initialization which preserves gradient norms
          through the LIF spiking dynamics
        - Output layer uses orthogonal with small scale for balanced initial probabilities
        """
        logger.info("Using hybrid initialization: Kaiming for input, orthogonal for hidden")

        # Input layer: Kaiming initialization (good for population-coded Gaussian features)
        if hasattr(self.policy, "input_layer"):
            torch.nn.init.kaiming_uniform_(
                self.policy.input_layer.weight,
                nonlinearity="relu",
            )
            if self.policy.input_layer.bias is not None:
                torch.nn.init.zeros_(self.policy.input_layer.bias)
            logger.info("Input layer: Kaiming initialization")

        # Hidden layers: Orthogonal initialization (preserves gradient norms in LIF dynamics)
        if hasattr(self.policy, "hidden_layers"):
            for i, layer in enumerate(self.policy.hidden_layers):
                # LIFLayer contains a Linear layer internally
                if hasattr(layer, "fc") and isinstance(layer.fc, torch.nn.Linear):
                    torch.nn.init.orthogonal_(layer.fc.weight, gain=1.0)
                    if layer.fc.bias is not None:
                        torch.nn.init.zeros_(layer.fc.bias)
                    logger.info(f"Hidden layer {i}: Orthogonal initialization")

        # Output layer: Orthogonal with smaller gain for balanced initial action probabilities
        if hasattr(self.policy, "output_layer"):
            torch.nn.init.orthogonal_(self.policy.output_layer.weight, gain=0.5)
            if self.policy.output_layer.bias is not None:
                torch.nn.init.zeros_(self.policy.output_layer.bias)
            logger.info("Output layer: Orthogonal initialization (gain=0.5)")

    def copy(self) -> "SpikingBrain":
        """
        Create a copy of the brain.

        Returns
        -------
        SpikingBrain
            New SpikingBrain instance with copied network parameters
        """
        # Create a config copy with the resolved seed to ensure reproducibility
        config_with_seed = SpikingBrainConfig(
            **{**self.config.model_dump(), "seed": self.seed},
        )
        new_brain = SpikingBrain(
            config=config_with_seed,
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
