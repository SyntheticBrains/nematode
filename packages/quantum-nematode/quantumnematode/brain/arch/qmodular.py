"""
Quantum Modular Q-Learning Brain (QModularBrain).

This hybrid quantum-classical implementation combines:
- Quantum circuit feature extraction from the modular brain
- Q-learning with experience replay and target networks
- Progressive quantum-classical action selection blending
- Confidence-based Q-guidance with numerical stability optimizations
- Enhanced reward shaping with proximity-based bonuses
- Fixed quantum parameter initialization for stable learning

Key advantages:
- Hybrid architecture leverages both quantum measurement diversity and classical
Q-learning stability
- Progressive Q-guidance blends quantum and classical decisions based on experience and confidence
- Experience replay smooths quantum measurement noise while preserving quantum advantages
- Target networks with gradient clipping provide stable optimization targets
- Enhanced reward shaping reduces overshooting and improves precision
- Numerical stability optimizations (safe softmax, gradient clipping) ensure robust training
- Single fixed quantum initialization allows Q-network to learn stable feature associations
- Adaptive learning with multiple passes for faster convergence when struggling
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from torch import nn, optim

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch.dtypes import BrainConfig, DeviceType
from quantumnematode.brain.modules import (
    DEFAULT_MODULES,
    SENSORY_MODULES,
    ModuleName,
    count_total_qubits,
)
from quantumnematode.logging_config import logger
from quantumnematode.optimizers.learning_rate import DynamicLearningRate
from quantumnematode.utils.seeding import ensure_seed, get_rng

# Defaults for Q-learning specific parameters
DEFAULT_BUFFER_SIZE = 1800
DEFAULT_BATCH_SIZE = 32
DEFAULT_TARGET_UPDATE_FREQUENCY = 25
DEFAULT_EPSILON_DECAY = 0.996
DEFAULT_MIN_EPSILON = 0.01
DEFAULT_NUM_LAYERS = 2
DEFAULT_SEED = None

# Defaults for reward shaping and thresholds
DEFAULT_NEGATIVE_REWARD_THRESHOLD = -0.01
DEFAULT_MIN_STATE_FEATURES_FOR_PROXIMITY = 6
DEFAULT_HIGH_GRADIENT_THRESHOLD = 0.8
DEFAULT_MEDIUM_GRADIENT_THRESHOLD = 0.5


if TYPE_CHECKING:
    from quantumnematode.initializers import (
        ManualParameterInitializer,
        RandomPiUniformInitializer,
        RandomSmallUniformInitializer,
        ZeroInitializer,
    )


class QModularBrainConfig(BrainConfig):
    """Configuration for the QModularBrain architecture."""

    modules: dict[ModuleName, list[int]] = (
        DEFAULT_MODULES  # Mapping of module names to qubit indices
    )
    num_layers: int = DEFAULT_NUM_LAYERS  # Number of layers in the quantum circuit
    seed: int | None = DEFAULT_SEED  # Random seed for reproducibility

    # Q-learning specific parameters
    buffer_size: int = DEFAULT_BUFFER_SIZE  # Experience replay buffer size
    batch_size: int = DEFAULT_BATCH_SIZE  # Batch size for Q-learning updates
    target_update_freq: int = DEFAULT_TARGET_UPDATE_FREQUENCY  # Target network update frequency
    epsilon_decay: float = DEFAULT_EPSILON_DECAY  # Epsilon decay rate for exploration
    min_epsilon: float = DEFAULT_MIN_EPSILON  # Minimum epsilon value

    # Reward shaping and threshold parameters
    negative_reward_threshold: float = (
        DEFAULT_NEGATIVE_REWARD_THRESHOLD  # Threshold for negative reward classification
    )
    # Minimum state features needed for proximity calculation
    min_state_features_for_proximity: int = DEFAULT_MIN_STATE_FEATURES_FOR_PROXIMITY
    high_gradient_threshold: float = (
        DEFAULT_HIGH_GRADIENT_THRESHOLD  # Gradient strength threshold for high proximity bonus
    )
    medium_gradient_threshold: float = (
        DEFAULT_MEDIUM_GRADIENT_THRESHOLD  # Gradient strength threshold for medium proximity bonus
    )


class QModularBrain:
    """
    Quantum Modular Brain using Q-Learning with Experience Replay.

    Combines quantum circuit feature extraction with stable Q-learning dynamics.
    """

    def __init__(  # noqa: PLR0913
        self,
        config: QModularBrainConfig,
        shots: int = 1024,
        device: DeviceType = DeviceType.CPU,
        learning_rate: DynamicLearningRate | None = None,
        parameter_initializer: ZeroInitializer  # noqa: ARG002
        | RandomPiUniformInitializer
        | RandomSmallUniformInitializer
        | ManualParameterInitializer
        | None = None,
        action_set: list[Action] = DEFAULT_ACTIONS,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        target_update_freq: int = DEFAULT_TARGET_UPDATE_FREQUENCY,
    ) -> None:
        """Initialize the QModularBrain."""
        # TODO: Add tracking metrics
        # TODO: Implement Qiskit runtime and Qiskit Functions
        # TODO: Use or remove LR
        # TODO: Use or remove parameter initializer
        self.config = config
        self.shots = shots
        self.device = device
        self.action_set = action_set
        self.num_qubits = count_total_qubits(config.modules)
        self.modules = config.modules or deepcopy(DEFAULT_MODULES)
        self.num_layers = config.num_layers

        # Q-learning parameters
        self.buffer_size = buffer_size if buffer_size != DEFAULT_BUFFER_SIZE else config.buffer_size
        self.batch_size = batch_size if batch_size != DEFAULT_BATCH_SIZE else config.batch_size
        self.target_update_freq = (
            target_update_freq
            if target_update_freq != DEFAULT_TARGET_UPDATE_FREQUENCY
            else config.target_update_freq
        )
        self.epsilon = 1.0
        self.epsilon_decay = config.epsilon_decay
        self.min_epsilon = config.min_epsilon

        # Reward shaping and threshold parameters from config
        self.negative_reward_threshold = config.negative_reward_threshold
        self.min_state_features_for_proximity = config.min_state_features_for_proximity
        self.high_gradient_threshold = config.high_gradient_threshold
        self.medium_gradient_threshold = config.medium_gradient_threshold

        # Experience replay buffer
        self.experience_buffer = deque(maxlen=buffer_size)
        self.update_count = 0
        self._step_count = 0

        # Initialize Brain protocol required attributes
        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()

        # Learning rate
        self.learning_rate = learning_rate or DynamicLearningRate()

        # Quantum circuit components
        self.parameters = {}
        self.parameter_values = {}
        self._circuit_cache = None
        self._backend = None

        # Initialize seeding for reproducibility
        self.seed = ensure_seed(config.seed)
        self.rng = get_rng(self.seed)
        self.quantum_seed = self.seed  # Preserve for quantum parameter initialization
        logger.info(f"QModularBrain using seed: {self.seed}")

        # Initialize quantum parameters and Q-networks
        self._initialize_quantum_parameters()
        self._initialize_q_networks()

        # Training state
        self.episode_count = 0
        self.step_count = 0

        logger.info(
            f"QModularBrain initialized with {self.num_qubits} qubits, "
            f"{len(self.action_set)} actions",
        )

    def _initialize_quantum_parameters(self) -> None:
        """Initialize quantum circuit parameters for stable Q-learning."""
        self._initialize_fixed_parameters(self.quantum_seed)

        # Create parameter structure
        for layer in range(self.num_layers):
            self.parameters[f"rx_{layer + 1}"] = [
                Parameter(f"θ_rx{layer + 1}_{i}") for i in range(self.num_qubits)
            ]
            self.parameters[f"ry_{layer + 1}"] = [
                Parameter(f"θ_ry{layer + 1}_{i}") for i in range(self.num_qubits)
            ]
            self.parameters[f"rz_{layer + 1}"] = [
                Parameter(f"θ_rz{layer + 1}_{i}") for i in range(self.num_qubits)
            ]

    def _initialize_fixed_parameters(self, seed: int | None) -> None:
        """Initialize quantum parameters with an optional seed."""
        # Recreate generator with an optional seed for consistency
        rng = np.random.Generator(np.random.PCG64(seed))
        self.parameter_values = {}

        for layer in range(self.num_layers):
            for axis in ["rx", "ry", "rz"]:
                param_names = [f"θ_{axis}{layer + 1}_{i}" for i in range(self.num_qubits)]
                for param_name in param_names:
                    # Use initialization range that works well for quantum circuits
                    self.parameter_values[param_name] = rng.uniform(-0.1, 0.1)

    def _initialize_q_networks(self) -> None:
        """Initialize Q-networks for value function approximation."""
        # Simplified input: focus on classical features for more stable learning
        # Input size: quantum measurement outcomes (2^num_qubits) + environment features
        quantum_feature_size = 2**self.num_qubits
        env_feature_size = 4  # position, chemotaxis features
        input_size = quantum_feature_size + env_feature_size

        # Optimized network architecture for better learning
        # TODO: Make configurable
        hidden_size = 64  # Increased capacity for better function approximation
        output_size = len(self.action_set)

        # Main Q-network with improved architecture
        self.q_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),  # Additional layer for better representation
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
        )

        # Target Q-network (for stable learning)
        self.target_q_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
        )

        # Copy main network to target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        # Optimizer with optimized learning rate, momentum, and stability
        # TODO: Make configurable
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=0.002,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Learning rate scheduler for better convergence
        # TODO: Make configurable
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=500,
            gamma=0.9,
        )

    def build_quantum_circuit(
        self,
        input_params: dict[ModuleName, dict[str, float]] | None = None,
    ) -> QuantumCircuit:
        """Build the quantum circuit for feature extraction."""
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)

        # Initial superposition
        for q in range(self.num_qubits):
            qc.h(q)

        # Parameterized layers
        for layer in range(self.num_layers):
            # Feature encoding
            for module, qubit_indices in self.modules.items():
                features = input_params.get(module, {}) if input_params else {}
                for _, q in enumerate(qubit_indices):
                    if q < self.num_qubits:  # Safety check
                        rx = features.get("rx", 0.0)
                        ry = features.get("ry", 0.0)
                        rz = features.get("rz", 0.0)

                        # Apply parameterized gates
                        qc.rx(rx + self.parameters[f"rx_{layer + 1}"][q], q)
                        qc.ry(ry + self.parameters[f"ry_{layer + 1}"][q], q)
                        qc.rz(rz + self.parameters[f"rz_{layer + 1}"][q], q)

            # Entanglement
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    qc.cz(i, j)

        # Measurement
        qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc

    def get_backend(self) -> AerSimulator:
        """Get or create the quantum backend."""
        if self._backend is None:
            self._backend = AerSimulator(seed_simulator=self.seed)
        return self._backend

    def extract_quantum_features(self, brain_params: BrainParams) -> np.ndarray:
        """Extract quantum features from the current state."""
        # Build quantum circuit with current parameters
        input_params = self._prepare_input_params(brain_params)
        qc = self.build_quantum_circuit(input_params)

        # Create parameter mapping for binding
        param_dict = {}
        param_list = list(qc.parameters)

        # Map parameters to values
        param_idx = 0
        for layer in range(self.num_layers):
            for axis in ["rx", "ry", "rz"]:
                for i in range(self.num_qubits):
                    param_name = f"θ_{axis}{layer + 1}_{i}"
                    if param_name in self.parameter_values and param_idx < len(param_list):
                        param_dict[param_list[param_idx]] = self.parameter_values[param_name]
                        param_idx += 1

        # Bind parameters using assign_parameters
        bound_circuit = qc.assign_parameters(param_dict)

        # Execute circuit
        backend = self.get_backend()
        transpiled_qc = transpile(bound_circuit, backend, seed_transpiler=self.seed)
        job = backend.run(transpiled_qc, shots=self.shots)
        result = job.result()
        counts = result.get_counts(0)

        # Convert counts to probability distribution
        total_shots = sum(counts.values())
        quantum_features = np.zeros(2**self.num_qubits)

        for bitstring, count in counts.items():
            index = int(bitstring, 2)
            quantum_features[index] = count / total_shots

        return quantum_features

    def _prepare_input_params(
        self,
        brain_params: BrainParams,
    ) -> dict[ModuleName, dict[str, float]]:
        """Prepare input parameters for quantum circuit."""
        # Extract relevant features from brain_params
        input_params: dict[ModuleName, dict[str, float]] = {}

        for module in self.modules:
            input_params[module] = SENSORY_MODULES[module].to_quantum_dict(brain_params)

        return input_params

    def get_state_features(self, brain_params: BrainParams) -> np.ndarray:
        """Get combined quantum and classical features for Q-network input."""
        # Extract quantum features
        quantum_features = self.extract_quantum_features(brain_params)

        # Extract classical environment features
        env_features = np.zeros(4)

        # Use gradient information (which represents direction to food)
        grad_strength = brain_params.gradient_strength or 0.0
        grad_direction = brain_params.gradient_direction or 0.0

        # Position features
        # TODO: Get grid size from agent's environment, currently set to 10 here
        grid_size = 10.0
        if brain_params.agent_position:
            env_features[0] = brain_params.agent_position[0] / grid_size
            env_features[1] = brain_params.agent_position[1] / grid_size

        # Gradient features (direction and strength to food)
        env_features[2] = grad_strength
        env_features[3] = grad_direction / (2 * np.pi)  # Normalize angle

        # Combine features
        return np.concatenate([quantum_features, env_features])

    def decide_action(self, brain_params: BrainParams) -> Action:  # noqa: PLR0915
        """Decide action using quantum circuit measurements with Q-learning guidance."""
        # Run quantum circuit to get measurement outcomes
        input_params = self._prepare_input_params(brain_params)
        qc = self.build_quantum_circuit(input_params)

        # Create parameter mapping for binding
        param_dict = {}
        param_list = list(qc.parameters)

        # Map parameters to values
        param_idx = 0
        for layer in range(self.num_layers):
            for axis in ["rx", "ry", "rz"]:
                for i in range(self.num_qubits):
                    param_name = f"θ_{axis}{layer + 1}_{i}"
                    if param_name in self.parameter_values and param_idx < len(param_list):
                        param_dict[param_list[param_idx]] = self.parameter_values[param_name]
                        param_idx += 1

        # Bind parameters and run circuit
        bound_circuit = qc.assign_parameters(param_dict)
        backend = self.get_backend()
        transpiled_qc = transpile(bound_circuit, backend, seed_transpiler=self.seed)
        job = backend.run(transpiled_qc, shots=self.shots)
        result = job.result()
        counts = result.get_counts(0)

        # Map quantum measurement outcomes to actions (like modular brain)
        num_states = 2**self.num_qubits
        binary_states = [f"{{:0{self.num_qubits}b}}".format(i) for i in range(num_states)]
        action_map = {
            state: self.action_set[i % len(self.action_set)]
            for i, state in enumerate(binary_states)
        }

        # Filter valid counts
        valid_counts = {k: v for k, v in counts.items() if k in action_map}
        if not valid_counts:
            # Fallback to random action if no valid measurements
            return self.action_set[self.rng.integers(0, len(self.action_set))]

        # Enhanced action selection with precision improvements
        if len(self.experience_buffer) >= self.batch_size and self.rng.random() > self.epsilon:
            # Get Q-values to guide action selection
            state_features = self.get_state_features(brain_params)
            state_tensor = torch.FloatTensor(state_features).unsqueeze(0)

            with torch.no_grad():
                q_values = self.q_network(state_tensor).squeeze()
                q_values_np = q_values.numpy()

            # Modify quantum probabilities based on Q-values with precision improvements
            total_counts = sum(valid_counts.values())
            quantum_probs = {k: v / total_counts for k, v in valid_counts.items()}

            # Calculate Q-value confidence for precision control
            q_max = np.max(q_values_np)
            q_min = np.min(q_values_np)
            q_confidence = (q_max - q_min) if q_max != q_min else 0.0

            # Progressive Q-guidance that increases with both experience and confidence
            experience_factor = len(self.experience_buffer) / self.buffer_size
            confidence_factor = min(1.0, q_confidence / 3.0)
            q_weight = min(
                0.7,
                0.1 + 0.3 * experience_factor + 0.3 * confidence_factor,
            )

            # Enhanced blending for precision with numerical stability
            blended_probs = {}

            # Safe softmax calculation with numerical stability
            q_values_shifted = q_values_np - np.max(
                q_values_np,
            )  # Subtract max for numerical stability
            q_softmax = np.exp(q_values_shifted) / np.sum(np.exp(q_values_shifted))

            for state, quantum_prob in quantum_probs.items():
                action = action_map[state]
                action_idx = self.action_set.index(action)

                # Blend with enhanced precision
                blended_prob = (1 - q_weight) * quantum_prob + q_weight * q_softmax[action_idx]
                blended_probs[state] = blended_prob

            # Renormalize
            total_prob = sum(blended_probs.values())
            if total_prob > 0:
                blended_probs = {k: v / total_prob for k, v in blended_probs.items()}

                # Sample from blended probabilities with numerical stability
                states = list(blended_probs.keys())
                probs = list(blended_probs.values())

                # Ensure probabilities sum to exactly 1.0
                probs = np.array(probs)
                probs = probs / probs.sum()

                chosen_state = self.rng.choice(states, p=probs)
                action = action_map[chosen_state]
            else:
                # Fallback to pure quantum selection
                states = list(quantum_probs.keys())
                probs = list(quantum_probs.values())

                # Ensure probabilities sum to exactly 1.0
                probs = np.array(probs)
                probs = probs / probs.sum()

                chosen_state = self.rng.choice(states, p=probs)
                action = action_map[chosen_state]
        else:
            # Pure quantum selection (exploration phase)
            total_counts = sum(valid_counts.values())
            quantum_probs = {k: v / total_counts for k, v in valid_counts.items()}

            states = list(quantum_probs.keys())
            probs = list(quantum_probs.values())

            # Ensure probabilities sum to exactly 1.0
            probs = np.array(probs)
            probs = probs / probs.sum()

            chosen_state = self.rng.choice(states, p=probs)
            action = action_map[chosen_state]

        self.step_count += 1
        return action

    def store_experience(
        self,
        state: np.ndarray,
        action: Action,
        reward: float,
        next_state: np.ndarray,
        *,
        done: bool,
    ) -> None:
        """Store experience in replay buffer with enhanced reward shaping to reduce overshooting."""
        action_idx = self.action_set.index(action)

        # Enhanced reward shaping for precision and consistency
        shaped_reward = reward

        if reward > 0:
            # Strong amplification for success (goal reached)
            shaped_reward = reward * 10.0  # Increased further for clearer signal
        elif reward < self.negative_reward_threshold:
            # Gentler negative rewards to encourage exploration but discourage bad paths
            shaped_reward = max(reward * 0.1, -0.05)
        else:
            # Small positive reward for neutral steps to encourage progress
            shaped_reward = 0.005

        # Add proximity-based reward shaping to reduce overshooting
        # Extract position information from state features if available
        if len(state) >= self.min_state_features_for_proximity:  # quantum features + 4 env features
            gradient_strength = state[-2]

            # Bonus for being close to goal with strong gradient
            if gradient_strength > self.high_gradient_threshold:  # Very close to goal
                proximity_bonus = 0.02
                shaped_reward += proximity_bonus
            elif gradient_strength > self.medium_gradient_threshold:  # Moderately close
                proximity_bonus = 0.01
                shaped_reward += proximity_bonus

        experience = (state, action_idx, shaped_reward, next_state, done)
        self.experience_buffer.append(experience)

    def learn_from_experience(self) -> None:
        """Learn from stored experiences using Q-learning."""
        if len(self.experience_buffer) < self.batch_size:
            return

        # Sample random batch
        batch_indices = self.rng.choice(
            len(self.experience_buffer),
            size=self.batch_size,
            replace=False,
        )
        batch = [self.experience_buffer[i] for i in batch_indices]
        states, actions, rewards, next_states, dones = zip(*batch, strict=False)

        # Convert to tensors (convert to numpy arrays first for performance)
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(np.array(actions))
        rewards_tensor = torch.FloatTensor(np.array(rewards))
        next_states_tensor = torch.FloatTensor(np.array(next_states))
        dones_tensor = torch.BoolTensor(np.array(dones))

        # Current Q-values
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1))

        # Next Q-values from target network with improved discount factor
        with torch.no_grad():
            next_q_values = self.target_q_network(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + (
                0.95 * next_q_values * ~dones_tensor
            )  # Lower gamma for faster learning

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize with gradient clipping for stability
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            max_norm=1.0,
        )  # Gradient clipping
        self.optimizer.step()
        self.scheduler.step()  # Update learning rate

        self.update_count += 1

        # Update target network
        if self.update_count % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            logger.debug(f"Target network updated at step {self.update_count}")

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def end_episode(self, final_reward: float) -> None:
        """Handle end of episode."""
        self.episode_count += 1

        # Log episode completion with final reward
        if self.episode_count % 10 == 0:
            recent_rewards = [exp[2] for exp in list(self.experience_buffer)[-50:]]
            if recent_rewards:
                avg_reward = sum(recent_rewards) / len(recent_rewards)
                logger.info(
                    f"Episode {self.episode_count}, final reward: {final_reward:.3f}, "
                    f"avg recent reward: {avg_reward:.3f}, "
                    f"buffer size: {len(self.experience_buffer)}",
                )
            else:
                logger.info(
                    f"Episode {self.episode_count}, final reward: {final_reward:.3f}, "
                    f"buffer size: {len(self.experience_buffer)}",
                )

    def run_brain(
        self,
        params: BrainParams,
        reward: float | None = None,
        input_data: list[float] | None = None,  # noqa: ARG002
        *,
        top_only: bool = True,
        top_randomize: bool = True,  # noqa: ARG002
    ) -> list[ActionData]:
        """
        Run the quantum Q-learning brain for the given parameters.

        Args:
            params: BrainParams for the agent/environment state.
            reward: Optional reward for Q-learning updates.
            input_data: Optional input data (unused).
            top_only: If True, return only the most probable action.
            top_randomize: If True, randomly select among top actions.

        Returns
        -------
            list[ActionData]: List of ActionData with action and probability.
        """
        # Increment step counter
        self._step_count += 1

        # Extract quantum features for the current state
        state_features = self.get_state_features(params)

        # Decide on action using Q-learning
        action = self.decide_action(params)

        # Store experience for learning if we have a previous state-action pair
        if hasattr(self, "_last_state") and hasattr(self, "_last_action") and reward is not None:
            next_state_features = state_features
            done = reward >= 1.0  # Assume episode done when goal reached
            self.store_experience(
                self._last_state,
                self._last_action,
                reward,
                next_state_features,
                done=done,
            )

            # Optimized learning frequency for faster adaptation
            if len(self.experience_buffer) >= self.batch_size:
                self.learn_from_experience()

                # More aggressive learning when performance is poor
                if (
                    reward < self.negative_reward_threshold
                    and len(self.experience_buffer) >= self.batch_size
                ):
                    self.learn_from_experience()  # Double learning when struggling

                # Additional learning every few steps for faster adaptation
                if self._step_count % 5 == 0:
                    self.learn_from_experience()

        # Store current state and action for next iteration
        self._last_state = state_features
        self._last_action = action

        # Handle episode completion
        if reward is not None and reward >= 1.0:
            self.end_episode(reward)
            # Clear stored state/action since episode is done
            if hasattr(self, "_last_state"):
                delattr(self, "_last_state")
            if hasattr(self, "_last_action"):
                delattr(self, "_last_action")

        # Convert action to ActionData format
        # Create probability distribution for the action
        # For Q-learning, we can use a confidence-based probability
        state_str = f"step_{getattr(self, '_step_count', 0)}"  # Simple state representation

        if top_only:
            # Return single action with high probability
            action_data = ActionData(state=state_str, action=action, probability=0.9)
            return [action_data]
        # Return all possible actions with computed probabilities
        action_probs = []
        for possible_action in [Action.LEFT, Action.RIGHT, Action.FORWARD, Action.STAY]:
            # Higher probability for chosen action
            prob = 0.7 if possible_action == action else 0.15
            action_probs.append(
                ActionData(state=state_str, action=possible_action, probability=prob),
            )
        return action_probs

    def update_memory(self, reward: float | None) -> None:
        """
        Update internal memory based on reward.

        Args:
            reward: Reward signal for updating memory.
        """
        # Reserved for future brain-internal memory mechanisms

    def prepare_episode(self) -> None:
        """Prepare for a new episode (no-op for QModularBrain)."""

    def post_process_episode(self, *, episode_success: bool | None = None) -> None:  # noqa: ARG002
        """Post-process the brain's state after each episode."""
        # Not implemented
        return

    def inspect_circuit(self) -> QuantumCircuit:
        """Inspect the quantum circuit structure."""
        # Build a circuit with default input params for inspection
        default_params = {module: {"rx": 0.0, "ry": 0.0, "rz": 0.0} for module in self.modules}
        return self.build_quantum_circuit(default_params)

    def copy(self) -> QModularBrain:
        """Create a copy of the brain."""
        # TODO: Copy entire state
        # Create a config copy with the resolved seed to ensure reproducibility
        config_with_seed = QModularBrainConfig(
            **{**self.config.model_dump(), "seed": self.seed},
        )
        new_brain = QModularBrain(
            config=config_with_seed,
            shots=self.shots,
            device=self.device,
            action_set=self.action_set,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            target_update_freq=self.target_update_freq,
        )
        new_brain.parameter_values = deepcopy(self.parameter_values)
        return new_brain
