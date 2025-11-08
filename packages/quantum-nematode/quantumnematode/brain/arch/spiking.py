"""
Spiking Neural Network (SNN) Brain Architecture.

This architecture implements a biologically plausible spiking neural network using
Leaky Integrate-and-Fire (LIF) neurons with Spike-Timing Dependent Plasticity (STDP)
learning for navigation tasks.

Key Features:
- **Temporal Dynamics**: Information encoded in spike timing and patterns
- **LIF Neurons**: Biologically realistic neuron model with membrane potential dynamics
- **STDP Learning**: Reward-modulated spike-timing dependent plasticity
- **Rate Coding**: Input encoding via Poisson spike trains
- **Sparse Computation**: Event-driven neural dynamics

Architecture:
- Input: 2 neurons encoding state features (gradient strength, relative direction)
- Hidden: Configurable fully-connected layer with LIF neurons
- Output: 4 neurons for action selection (forward, left, right, stay)

The SNN brain learns by:
1. Encoding continuous state values as Poisson spike trains
2. Simulating LIF neural dynamics for a fixed time window
3. Recording spike times and computing action probabilities from output rates
4. Updating synaptic weights using reward-modulated STDP
5. Maintaining homeostatic weight normalization for stability

This approach provides biological realism while maintaining compatibility with the
existing reinforcement learning framework.
"""

import numpy as np
import torch  # pyright: ignore[reportMissingImports]

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, ClassicalBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch.dtypes import BrainConfig, DeviceType
from quantumnematode.initializers._initializer import ParameterInitializer
from quantumnematode.logging_config import logger
from quantumnematode.monitoring.overfitting_detector import create_overfitting_detector_for_brain

# Default configuration parameters
DEFAULT_HIDDEN_SIZE = 32
DEFAULT_SIMULATION_DURATION = 100.0  # ms
DEFAULT_TIME_STEP = 1.0  # ms
DEFAULT_TAU_M = 20.0  # membrane time constant (ms)
DEFAULT_V_THRESHOLD = 1.0  # spike threshold
DEFAULT_V_RESET = 0.0  # reset potential
DEFAULT_V_REST = 0.0  # resting potential
DEFAULT_MAX_RATE = 100.0  # max input spike rate (Hz)
DEFAULT_MIN_RATE = 0.0  # min input spike rate (Hz)
DEFAULT_TAU_PLUS = 20.0  # STDP time constant (ms)
DEFAULT_TAU_MINUS = 20.0  # STDP time constant (ms)
DEFAULT_A_PLUS = 0.01  # STDP strength
DEFAULT_A_MINUS = 0.01  # STDP strength
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_REWARD_SCALING = 1.0
DEFAULT_WEIGHT_MEAN = 0.1
DEFAULT_WEIGHT_STD = 0.05
DEFAULT_WEIGHT_CLIP = 1.0


class SpikingBrainConfig(BrainConfig):
    """Configuration for the SpikingBrain architecture."""

    # Network topology
    hidden_size: int = DEFAULT_HIDDEN_SIZE

    # Simulation parameters
    simulation_duration: float = DEFAULT_SIMULATION_DURATION
    time_step: float = DEFAULT_TIME_STEP

    # LIF neuron parameters
    tau_m: float = DEFAULT_TAU_M
    v_threshold: float = DEFAULT_V_THRESHOLD
    v_reset: float = DEFAULT_V_RESET
    v_rest: float = DEFAULT_V_REST

    # Encoding parameters
    max_rate: float = DEFAULT_MAX_RATE
    min_rate: float = DEFAULT_MIN_RATE

    # STDP parameters
    tau_plus: float = DEFAULT_TAU_PLUS
    tau_minus: float = DEFAULT_TAU_MINUS
    a_plus: float = DEFAULT_A_PLUS
    a_minus: float = DEFAULT_A_MINUS
    learning_rate: float = DEFAULT_LEARNING_RATE
    reward_scaling: float = DEFAULT_REWARD_SCALING

    # Weight initialization
    weight_mean: float = DEFAULT_WEIGHT_MEAN
    weight_std: float = DEFAULT_WEIGHT_STD
    weight_clip: float = DEFAULT_WEIGHT_CLIP


class LIFNeuron:
    """Leaky Integrate-and-Fire neuron model."""

    def __init__(
        self,
        tau_m: float = DEFAULT_TAU_M,
        v_threshold: float = DEFAULT_V_THRESHOLD,
        v_reset: float = DEFAULT_V_RESET,
        v_rest: float = DEFAULT_V_REST,
    ) -> None:
        self.tau_m = tau_m
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.v_membrane = v_rest
        self.last_spike_time = -np.inf

    def step(self, input_current: float, dt: float) -> bool:
        """
        Update neuron state and check for spike.

        Args:
            input_current: Input current for this time step
            dt: Time step duration (ms)

        Returns
        -------
            True if neuron spikes, False otherwise
        """
        # Update membrane potential using Euler integration
        dv_dt = (self.v_rest - self.v_membrane + input_current) / self.tau_m
        self.v_membrane += dv_dt * dt

        # Check for spike
        if self.v_membrane >= self.v_threshold:
            self.v_membrane = self.v_reset
            return True
        return False

    def reset(self) -> None:
        """Reset neuron to resting state."""
        self.v_membrane = self.v_rest
        self.last_spike_time = -np.inf


class STDPRule:
    """Spike-Timing Dependent Plasticity learning rule."""

    def __init__(
        self,
        tau_plus: float = DEFAULT_TAU_PLUS,
        tau_minus: float = DEFAULT_TAU_MINUS,
        a_plus: float = DEFAULT_A_PLUS,
        a_minus: float = DEFAULT_A_MINUS,
    ) -> None:
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus

    def compute_weight_change(self, delta_t: float, reward_signal: float) -> float:
        """
        Compute weight change based on spike timing difference.

        Args:
            delta_t: Time difference between pre and post spikes (post - pre)
            reward_signal: Reward modulation signal

        Returns
        -------
            Weight change amount
        """
        if delta_t > 0:
            # Post after pre -> potentiation
            dw = self.a_plus * np.exp(-delta_t / self.tau_plus)
        else:
            # Pre after post -> depression
            dw = -self.a_minus * np.exp(delta_t / self.tau_minus)

        # Modulate by reward signal
        return dw * reward_signal


class SpikingBrain(ClassicalBrain):
    """
    Spiking neural network brain architecture using LIF neurons and STDP learning.

    Implements biologically plausible neural dynamics with temporal spike patterns
    for decision-making in navigation tasks.
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

        logger.info(f"Initializing SpikingBrain with config: {config}")

        self.config = config
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.device = torch.device(device.value)
        self._action_set = action_set

        # Initialize data structures
        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()

        # Network dimensions
        self.hidden_size = config.hidden_size
        self.total_neurons = input_dim + config.hidden_size + num_actions

        # Create neuron populations
        self.input_neurons = [
            LIFNeuron(config.tau_m, config.v_threshold, config.v_reset, config.v_rest)
            for _ in range(input_dim)
        ]
        self.hidden_neurons = [
            LIFNeuron(config.tau_m, config.v_threshold, config.v_reset, config.v_rest)
            for _ in range(config.hidden_size)
        ]
        self.output_neurons = [
            LIFNeuron(config.tau_m, config.v_threshold, config.v_reset, config.v_rest)
            for _ in range(num_actions)
        ]

        # Initialize synaptic weights
        self._initialize_weights(parameter_initializer)

        # STDP learning rule
        self.stdp_rule = STDPRule(config.tau_plus, config.tau_minus, config.a_plus, config.a_minus)

        # Spike history tracking
        self.spike_times = {
            "input": [[] for _ in range(input_dim)],
            "hidden": [[] for _ in range(config.hidden_size)],
            "output": [[] for _ in range(num_actions)],
        }

        # Episode tracking
        self.episode_spike_patterns = []
        self.episode_rewards = []

        # Overfitting detection
        self.overfitting_detector = create_overfitting_detector_for_brain("spiking")
        self.overfit_detector_episode_count = 0
        self.overfit_detector_current_episode_actions = []
        self.overfit_detector_current_episode_positions = []
        self.overfit_detector_current_episode_rewards = []

        logger.info(f"SpikingBrain initialized with {self.total_neurons} total neurons")

    def _initialize_weights(self, parameter_initializer: ParameterInitializer | None) -> None:
        """Initialize synaptic weight matrices."""
        # Input to hidden weights
        self.weights_input_hidden = torch.normal(
            self.config.weight_mean,
            self.config.weight_std,
            (self.input_dim, self.config.hidden_size),
            device=self.device,
        )

        # Hidden to output weights
        self.weights_hidden_output = torch.normal(
            self.config.weight_mean,
            self.config.weight_std,
            (self.config.hidden_size, self.num_actions),
            device=self.device,
        )

        # Optional recurrent connections within hidden layer
        self.weights_hidden_hidden = torch.normal(
            self.config.weight_mean * 0.5,  # Smaller recurrent weights
            self.config.weight_std * 0.5,
            (self.config.hidden_size, self.config.hidden_size),
            device=self.device,
        )

        # Clip weights to prevent instability
        self._clip_weights()

        total_params = (
            self.weights_input_hidden.numel()
            + self.weights_hidden_output.numel()
            + self.weights_hidden_hidden.numel()
        )

        logger.info(f"Initialized {total_params:,} synaptic weights")
        if parameter_initializer is not None:
            logger.info(
                "Custom parameter initializer provided but using standard normal initialization",
            )

    def _clip_weights(self) -> None:
        """Clip weights to maintain stability."""
        with torch.no_grad():
            self.weights_input_hidden.clamp_(-self.config.weight_clip, self.config.weight_clip)
            self.weights_hidden_output.clamp_(-self.config.weight_clip, self.config.weight_clip)
            self.weights_hidden_hidden.clamp_(-self.config.weight_clip, self.config.weight_clip)

    def _encode_state_to_spikes(self, state: list[float]) -> list[list[float]]:
        """
        Encode continuous state values to Poisson spike trains.

        Args:
            state: List of continuous state values

        Returns
        -------
            List of spike times for each input neuron
        """
        spike_trains = []
        dt = self.config.time_step
        duration = self.config.simulation_duration
        num_steps = int(duration / dt)
        rng = np.random.default_rng()

        for value in state:
            # Convert continuous value to firing rate
            rate = self.config.min_rate + (self.config.max_rate - self.config.min_rate) * max(
                0,
                min(1, value),
            )

            # Generate Poisson spike train
            spike_times = []
            for step in range(num_steps):
                prob = rate * dt / 1000.0  # Convert to probability per time step
                if rng.random() < prob:
                    spike_times.append(step * dt)

            spike_trains.append(spike_times)

        return spike_trains

    def _simulate_network(self, input_spike_trains: list[list[float]]) -> tuple[list[int], dict]:
        """
        Simulate the spiking neural network for one decision period.

        Args:
            input_spike_trains: Spike times for each input neuron

        Returns
        -------
            Tuple of (output spike counts, detailed spike history)
        """
        dt = self.config.time_step
        duration = self.config.simulation_duration
        num_steps = int(duration / dt)

        # Reset all neurons
        for neuron in self.input_neurons + self.hidden_neurons + self.output_neurons:
            neuron.reset()

        # Track spikes
        input_spikes = [[] for _ in range(self.input_dim)]
        hidden_spikes = [[] for _ in range(self.config.hidden_size)]
        output_spikes = [[] for _ in range(self.num_actions)]

        # Simulate time steps
        for step in range(num_steps):
            current_time = step * dt

            # Process input spikes
            input_currents_hidden = torch.zeros(self.config.hidden_size, device=self.device)
            for i, spike_times in enumerate(input_spike_trains):
                if any(abs(t - current_time) < dt / 2 for t in spike_times):
                    input_spikes[i].append(current_time)
                    # Add current to connected hidden neurons
                    input_currents_hidden += self.weights_input_hidden[i, :]

            # Update hidden neurons
            hidden_currents_output = torch.zeros(self.num_actions, device=self.device)
            hidden_currents_recurrent = torch.zeros(self.config.hidden_size, device=self.device)

            for i, neuron in enumerate(self.hidden_neurons):
                total_current = input_currents_hidden[i] + hidden_currents_recurrent[i]
                if neuron.step(total_current.item(), dt):
                    hidden_spikes[i].append(current_time)
                    # Add current to output neurons
                    hidden_currents_output += self.weights_hidden_output[i, :]
                    # Add recurrent current to other hidden neurons
                    hidden_currents_recurrent += self.weights_hidden_hidden[i, :]

            # Update output neurons
            for i, neuron in enumerate(self.output_neurons):
                if neuron.step(hidden_currents_output[i].item(), dt):
                    output_spikes[i].append(current_time)

        # Count output spikes
        output_spike_counts = [len(spikes) for spikes in output_spikes]

        # Store spike history
        spike_history = {
            "input": input_spikes,
            "hidden": hidden_spikes,
            "output": output_spikes,
            "duration": duration,
        }

        return output_spike_counts, spike_history

    def _decode_action_probabilities(self, spike_counts: list[int]) -> np.ndarray:
        """
        Convert output spike counts to action probabilities.

        Args:
            spike_counts: Number of spikes from each output neuron

        Returns
        -------
            Normalized action probabilities
        """
        # Convert counts to rates (spikes per second)
        rates = np.array(spike_counts) / (self.config.simulation_duration / 1000.0)

        # Add small epsilon to prevent zero probabilities
        rates = rates + 1e-8

        # Apply softmax to get probabilities
        exp_rates = np.exp(rates - np.max(rates))
        return exp_rates / np.sum(exp_rates)

    @property
    def action_set(self) -> list[Action]:
        """Return the set of available actions."""
        return self._action_set

    def preprocess(self, params: BrainParams) -> list[float]:
        """
        Preprocess brain parameters into state vector for encoding.

        Args:
            params: Brain parameters containing environmental state

        Returns
        -------
            Preprocessed state vector
        """
        state = []

        # Add gradient strength (normalized to [0, 1])
        if params.gradient_strength is not None:
            state.append(max(0, min(1, params.gradient_strength)))
        else:
            state.append(0.0)

        # Add gradient direction (normalized to [0, 1])
        if params.gradient_direction is not None:
            # Normalize angle to [0, 1]
            normalized_direction = (params.gradient_direction % (2 * np.pi)) / (2 * np.pi)
            state.append(normalized_direction)
        else:
            state.append(0.0)

        return state

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

        Args:
            params: Brain parameters containing environmental state
            reward: Optional reward signal (not used in forward pass)
            input_data: Optional input data (not used)
            top_only: Whether to return only top action (not used)
            top_randomize: Whether to randomize top actions (not used)

        Returns
        -------
            List of selected actions with probabilities
        """
        # Preprocess state
        state = self.preprocess(params)

        # Encode state to spike trains
        input_spike_trains = self._encode_state_to_spikes(state)

        # Simulate network
        output_spike_counts, spike_history = self._simulate_network(input_spike_trains)

        # Decode action probabilities
        probabilities = self._decode_action_probabilities(output_spike_counts)

        # Select action based on probabilities
        rng = np.random.default_rng()
        action_idx = rng.choice(len(probabilities), p=probabilities)
        selected_action = self.action_set[action_idx]
        selected_probability = probabilities[action_idx]

        # Store data for learning
        state_str = f"grad_str:{state[0]:.3f},grad_dir:{state[1]:.3f}"
        self.latest_data.action = ActionData(
            state=state_str,
            action=selected_action,
            probability=selected_probability,
        )
        self.latest_data.probability = selected_probability

        # Store spike history for STDP learning
        self.episode_spike_patterns.append(spike_history)

        # Update overfitting detector
        if params.agent_position is not None:
            self.overfit_detector_current_episode_positions.append(params.agent_position)
        self.overfit_detector_current_episode_actions.append(selected_action)

        logger.debug(
            f"SpikingBrain selected action {selected_action} "
            f"with probability {selected_probability:.3f}",
        )
        logger.debug(f"Output spike counts: {output_spike_counts}")

        return [
            ActionData(state=state_str, action=selected_action, probability=selected_probability),
        ]

    def learn(
        self,
        params: BrainParams,  # noqa: ARG002
        reward: float,
        *,
        episode_done: bool,
    ) -> None:
        """
        Update synaptic weights using reward-modulated STDP.

        Args:
            params: Brain parameters
            reward: Reward signal for learning
            episode_done: Whether the episode is complete
        """
        self.episode_rewards.append(reward)
        self.overfit_detector_current_episode_rewards.append(reward)

        if episode_done:
            self._update_weights_stdp()
            self._clip_weights()

            # Clear episode data
            self.episode_spike_patterns.clear()
            self.episode_rewards.clear()

    def _update_weights_stdp(self) -> None:  # noqa: C901, PLR0912
        """Update synaptic weights using STDP rule with reward modulation."""
        if not self.episode_spike_patterns or not self.episode_rewards:
            return

        # Use average reward as modulation signal
        avg_reward = float(np.mean(self.episode_rewards)) * self.config.reward_scaling

        with torch.no_grad():
            total_weight_change = 0.0

            # Update input-to-hidden weights
            for pattern in self.episode_spike_patterns:
                for i, input_spikes in enumerate(pattern["input"]):
                    for j, hidden_spikes in enumerate(pattern["hidden"]):
                        # Compute weight changes for all spike pairs
                        weight_change = 0.0
                        for t_pre in input_spikes:
                            for t_post in hidden_spikes:
                                delta_t = t_post - t_pre
                                if (
                                    abs(delta_t)
                                    < max(self.config.tau_plus, self.config.tau_minus) * 3
                                ):
                                    weight_change += self.stdp_rule.compute_weight_change(
                                        delta_t,
                                        avg_reward,
                                    )

                        self.weights_input_hidden[i, j] += self.config.learning_rate * weight_change
                        total_weight_change += abs(weight_change)

            # Update hidden-to-output weights
            for pattern in self.episode_spike_patterns:
                for i, hidden_spikes in enumerate(pattern["hidden"]):
                    for j, output_spikes in enumerate(pattern["output"]):
                        weight_change = 0.0
                        for t_pre in hidden_spikes:
                            for t_post in output_spikes:
                                delta_t = t_post - t_pre
                                if (
                                    abs(delta_t)
                                    < max(self.config.tau_plus, self.config.tau_minus) * 3
                                ):
                                    weight_change += self.stdp_rule.compute_weight_change(
                                        delta_t,
                                        avg_reward,
                                    )

                        self.weights_hidden_output[i, j] += (
                            self.config.learning_rate * weight_change
                        )
                        total_weight_change += abs(weight_change)

        logger.debug(
            f"STDP weight update: total change = {total_weight_change:.6f}, "
            f"avg_reward = {avg_reward:.3f}",
        )

    def update_memory(self, reward: float | None) -> None:
        """
        Update memory with reward information.

        Args:
            reward: Reward signal to store
        """
        if reward is not None:
            self.latest_data.reward = reward
            self.history_data.rewards.append(reward)

    def post_process_episode(self) -> None:
        """Perform post-episode processing and cleanup."""
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
                    self.overfitting_detector.update_behavioral_metrics(
                        action_sequence,
                        self.overfit_detector_current_episode_positions,
                        start_pos,
                    )

            except (AttributeError, ValueError, TypeError) as e:
                logger.warning(f"Overfitting detector update failed: {e}")

        # Clear episode tracking
        self.overfit_detector_current_episode_actions.clear()
        self.overfit_detector_current_episode_positions.clear()
        self.overfit_detector_current_episode_rewards.clear()
        self.overfit_detector_episode_count += 1

        logger.debug("Episode post-processing complete")

    def copy(self) -> "SpikingBrain":
        """
        Create a copy of the brain.

        Returns
        -------
            New SpikingBrain instance with copied weights
        """
        new_brain = SpikingBrain(
            config=self.config,
            input_dim=self.input_dim,
            num_actions=self.num_actions,
            device=DeviceType(self.device.type),
            action_set=self._action_set.copy(),
        )

        # Copy weights
        new_brain.weights_input_hidden = self.weights_input_hidden.clone()
        new_brain.weights_hidden_output = self.weights_hidden_output.clone()
        new_brain.weights_hidden_hidden = self.weights_hidden_hidden.clone()

        return new_brain
