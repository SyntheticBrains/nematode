"""Dynamic Quantum Brain Architecture."""

import math
from copy import deepcopy

import numpy as np  # pyright: ignore[reportMissingImports]
from qiskit import QuantumCircuit, transpile  # pyright: ignore[reportMissingImports]
from qiskit.circuit import Parameter  # pyright: ignore[reportMissingImports]
from qiskit_aer import AerSimulator  # pyright: ignore[reportMissingImports]

from quantumnematode.brain._brain import Brain, BrainParams
from quantumnematode.initializers import (
    RandomPiUniformInitializer,
    RandomSmallUniformInitializer,
    ZeroInitializer,
)
from quantumnematode.logging_config import logger
from quantumnematode.models import ActionData
from quantumnematode.optimizer.gradient_methods import GradientCalculationMethod, compute_gradients
from quantumnematode.optimizer.learning_rate import (
    AdamLearningRate,
    DynamicLearningRate,
    PerformanceBasedLearningRate,
)

EXPLORATION_MIN = 0.6  # Minimum exploration factor
EXPLORATION_MAX = 1.0  # Maximum exploration factor
TEMPERATURE = 0.9  # Default temperature for softmax action selection

TOGGLE_PARAM_CLIP = True  # Toggle for parameter clipping
TOGGLE_SHORT_TERM_MEMORY = True  # Toggle for short-term memory
TOGGLE_PARAM_MODULO = True  # Toggle for parameter modulo wrapping ([-pi, pi])

# Entropy regularization coefficient for policy gradient loss.
# Higher values (e.g., 0.1) encourage more exploration by making the policy more random,
# which can help escape local optima but may slow convergence. Lower values (e.g., 0.01)
# make the policy more deterministic, focusing on exploitation, but risk premature convergence.
# Typical range: 0.01 (low exploration) to 0.1 (high exploration).
# Example: 0.05 is a balanced default.
ENTROPY_BETA = 0.05


class DynamicBrain(Brain):
    """
    Dynamic quantum brain architecture using parameterized quantum circuits.

    This implementation represents a flexible quantum brain designed for decision-making.
    It uses a dynamic number of qubits, specified during initialization, with parameterized RX, RY,
    and RZ gates to encode the agent's state. The circuit also includes CX gates to introduce
    entanglement between the qubits. The output of the circuit is measured and mapped to actions
    dynamically based on the number of qubits.

    Key Features:
    - Supports a dynamic number of qubits for scalability and experimentation.
    - Parameterized gates allow for dynamic updates based on the agent's state and learning.
    - Entanglement is introduced to model complex decision-making processes.
    - Designed to be flexible and suitable for simulators with varying resources.

    This architecture is ideal for testing and exploring advanced quantum reinforcement learning
    concepts.
    """

    def __init__(  # noqa: PLR0913, PLR0915
        self,
        device: str = "CPU",
        shots: int = 100,
        num_qubits: int = 5,
        num_layers: int = 3,
        learning_rate: DynamicLearningRate
        | AdamLearningRate
        | PerformanceBasedLearningRate
        | None = None,
        gradient_method: GradientCalculationMethod = GradientCalculationMethod.RAW,
        parameter_initializer: ZeroInitializer
        | RandomPiUniformInitializer
        | RandomSmallUniformInitializer
        | None = None,
    ) -> None:
        """
        Initialize the DynamicBrain with a dynamic number of qubits.

        Parameters
        ----------
        device : str
            The device to use for simulation (e.g., "CPU" or "GPU").
        shots : int
            The number of shots for the quantum simulation.
        num_qubits : int
            The number of qubits to use in the quantum circuit.
        num_layers : int
            The number of layers in the quantum circuit.
        learning_rate : DynamicLearningRate | AdamLearningRate | PerformanceBasedLearningRate | None
            The learning rate strategy for parameter updates, by default None.
            If None, a default dynamic learning rate will be used.
        gradient_method : GradientCalculationMethod
            The method to use for gradient calculation, by default RAW.
        parameter_initializer : ParameterInitializer, optional
            The initializer to use for parameter initialization, by default ZeroInitializer.
        """
        self.device = device.upper()
        self.shots = shots
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_actions = num_qubits**2

        # --- Parameter sharing: one parameter per gate per qubit, shared across all layers ---
        self.parameters = {
            "rx": [Parameter(f"θ_rx_{i}") for i in range(self.num_qubits)],
            "ry": [Parameter(f"θ_ry_{i}") for i in range(self.num_qubits)],
            "rz": [Parameter(f"θ_rz_{i}") for i in range(self.num_qubits)],
        }

        self.rng = np.random.default_rng()
        self.steps = 0
        self.satiety = 1.0
        self.gradient_method = gradient_method

        self.parameter_initializer = parameter_initializer or RandomSmallUniformInitializer()
        param_keys = (
            [f"θ_rx_{i}" for i in range(self.num_qubits)]
            + [f"θ_ry_{i}" for i in range(self.num_qubits)]
            + [f"θ_rz_{i}" for i in range(self.num_qubits)]
        )
        self.parameter_values = self.parameter_initializer.initialize(num_qubits, param_keys)

        self.latest_input_parameters = None
        self.latest_updated_parameters = None
        self.latest_counts = None
        self.latest_action: ActionData | None = None
        self.latest_gradients = None
        self.latest_learning_rate = None
        self.latest_exploration_factor = None
        self.latest_temperature = None

        self.history_params: list[BrainParams] = []
        self.history_input_parameters: list[dict[str, float]] = []
        self.history_updated_parameters: list[dict[str, float]] = []
        self.history_gradients: list[list[float]] = []
        self.history_gradient_strengths: list[float] = []
        self.history_gradient_directions: list[float] = []
        self.history_rewards: list[float] = []
        self.history_rewards_norm: list[float] = []
        self.history_learning_rates: list[float | dict[str, float]] = []
        self.history_exploration_factors: list[float] = []
        self.history_temperatures: list[float] = []

        self.learning_rate = learning_rate or DynamicLearningRate()
        if isinstance(self.learning_rate, DynamicLearningRate):
            logger.info(
                "Using dynamic learning rate strategy for parameter updates.",
            )
            logger.info(
                f"Initial learning rate: {self.learning_rate.initial_learning_rate}",
            )
            logger.info(
                f"Decay rate: {self.learning_rate.decay_rate}",
            )
        elif isinstance(self.learning_rate, AdamLearningRate):
            logger.info(
                "Using Adam learning rate strategy for parameter updates.",
            )
            logger.info(
                f"Initial learning rate: {self.learning_rate.initial_learning_rate}",
            )
            logger.info(
                f"Beta1: {self.learning_rate.beta1}",
            )
            logger.info(
                f"Beta2: {self.learning_rate.beta2}",
            )
            logger.info(
                f"Epsilon: {self.learning_rate.epsilon}",
            )
        elif isinstance(self.learning_rate, PerformanceBasedLearningRate):
            logger.info(
                "Using performance-based learning rate strategy for parameter updates.",
            )
            logger.info(
                f"Initial learning rate: {self.learning_rate.learning_rate}",
            )
            logger.info(
                f"Minimum learning rate: {self.learning_rate.min_learning_rate}",
            )
            logger.info(
                f"Maximum learning rate: {self.learning_rate.max_learning_rate}",
            )
            logger.info(
                f"Adjustment factor: {self.learning_rate.adjustment_factor}",
            )

        # Log parameter initialization range
        logger.debug(
            "Initializing parameters uniformly in the range [-pi, pi]: "
            f"{str(self.parameter_values).replace('θ', 'theta_')}",
        )

        # --- Reward normalization and baseline tracking ---
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 0
        self.reward_baseline = 0.0
        self.reward_alpha = 0.01  # For running average baseline

    def build_brain(self, input_data: list[float] | None = None) -> QuantumCircuit:
        """
        Build the quantum circuit for the dynamic brain.

        Each layer consists of data re-uploading (Rx with input),
        parameterized Rx, Ry, Rz on every qubit, followed by CZ between all pairs of qubits.

        Parameters
        ----------
        input_data : list[float] | None
            Input data to encode via data re-uploading at each layer (one float per qubit).
            If None, default to zeros.
            If provided, it should be a list of floats, one per qubit.

        Returns
        -------
        QuantumCircuit
            The quantum circuit representing the brain.
        """
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)

        # Default to zeros if no input_data provided
        if input_data is None:
            input_data = [0.0] * self.num_qubits

        for _ in range(self.num_layers):
            for i in range(self.num_qubits):
                # Data re-uploading: encode input at each layer
                qc.rx(input_data[i], i)
                qc.rx(self.parameters["rx"][i], i)
                qc.ry(self.parameters["ry"][i], i)
                qc.rz(self.parameters["rz"][i], i)
            # Entangle every unique pair of qubits with CZ (i < j)
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    qc.cz(i, j)

        qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc

    def run_brain(  # noqa: PLR0915
        self,
        params: BrainParams,
        reward: float | None = None,
        input_data: list[float] | None = None,
    ) -> dict[str, int]:
        """
        Run the quantum brain simulation, incorporating gradient information and data re-uploading.

        Parameters
        ----------
        params : BrainParams
            Parameters for the quantum brain.
        reward : float, optional
            Reward signal for learning, by default None.
        input_data : list[float], optional
            Input data to encode via data re-uploading at each layer (one float per qubit).

        Returns
        -------
        dict[str, int]
            Measurement counts from the quantum circuit.
        """
        gradient_strength = params.gradient_strength
        gradient_direction = params.gradient_direction
        agent_direction = params.agent_direction

        # Validate input parameters
        if gradient_strength is None or gradient_direction is None or agent_direction is None:
            error_msg = "Gradient strength, direction, and agent direction must be provided."
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.history_params.append(params)
        self.history_gradient_strengths.append(gradient_strength)  # Used for reporting only
        self.history_gradient_directions.append(gradient_direction)  # Used for reporting only

        qc = self.build_brain(input_data=input_data)

        # --- Reward normalization and baseline update ---
        # Update running mean and variance (Welford's algorithm)
        if reward is not None:
            self.reward_count += 1
            delta = reward - self.reward_mean
            self.reward_mean += delta / self.reward_count
            delta2 = reward - self.reward_mean
            self.reward_var += delta * delta2
            reward_std = max((self.reward_var / self.reward_count) ** 0.5, 1e-8)
            # Update running baseline (exponential moving average)
            self.reward_baseline = (
                1 - self.reward_alpha
            ) * self.reward_baseline + self.reward_alpha * reward
            # Normalize and baseline-subtract reward
            norm_reward = (reward - self.reward_baseline) / reward_std
        else:
            norm_reward = 0.0

        self.history_rewards.append(reward or 0.0)
        self.history_rewards_norm.append(norm_reward or 0.0)

        # Use normalized, baseline-subtracted reward for learning
        if norm_reward is not None and self.latest_counts is not None:
            gradients = self.compute_gradients(
                self.latest_counts,
                norm_reward,
                self.latest_action
                if self.latest_action is not None
                else ActionData(state="", action="", probability=0.0),
            )
            self.update_parameters(gradients, norm_reward)

        # Update input parameters to use parameter sharing (one per qubit per gate)
        input_params = {}
        rx_param = None
        ry_param = None
        rz_param = None

        if TOGGLE_SHORT_TERM_MEMORY:
            rx_param = self.history_params[-1].gradient_strength
            ry_param = (
                self.history_params[-2].gradient_strength
                if len(self.history_params) > 1
                else rx_param
            )
            rz_param = (
                self.history_params[-3].gradient_strength
                if len(self.history_params) > 2  # noqa: PLR2004
                else ry_param
            )
        else:
            rx_param = ry_param = rz_param = self.history_params[-1].gradient_strength

        for i in range(self.num_qubits):
            input_params[self.parameters["rx"][i]] = self.parameter_values[
                f"θ_rx_{i}"
            ] + rx_param * np.cos(i * np.pi / self.num_qubits)
            input_params[self.parameters["ry"][i]] = self.parameter_values[
                f"θ_ry_{i}"
            ] + ry_param * np.cos(i * np.pi / self.num_qubits)
            input_params[self.parameters["rz"][i]] = self.parameter_values[
                f"θ_rz_{i}"
            ] + rz_param * np.cos(i * np.pi / self.num_qubits)

        # Store latest input parameters for tracking
        self.latest_input_parameters = input_params
        self.history_input_parameters.append(input_params)

        logger.debug(
            "Input parameters with gradient information: "
            f"{str(input_params).replace('θ', 'theta_')}",
        )

        bound_qc = qc.assign_parameters(input_params, inplace=False)

        simulator = AerSimulator(device=self.device)
        transpiled = transpile(bound_qc, simulator)
        result = simulator.run(transpiled, shots=self.shots).result()
        counts = result.get_counts()

        # Decrease satiety at each step
        self.satiety = max(0.0, self.satiety - 0.01)  # Decrease satiety gradually

        logger.debug(f"Satiety after step {self.steps}: {self.satiety}, ")

        if self.satiety <= 0.0:
            logger.warning("Satiety is zero.")

        # Calculate exploration factor based on satiety
        # NOTE: Not used anymore, previously used for temperature calculation
        self.latest_exploration_factor = (
            EXPLORATION_MIN + (EXPLORATION_MAX - EXPLORATION_MIN) * self.satiety
        )
        self.latest_temperature = TEMPERATURE * self.latest_exploration_factor

        self.history_temperatures.append(self.latest_temperature)
        self.history_exploration_factors.append(self.latest_temperature)

        logger.debug(
            f"Exploration factor: {self.latest_exploration_factor}, "
            f"Temperature: {self.latest_temperature}",
        )

        self.latest_counts = counts
        return counts

    def compute_gradients(
        self,
        counts: dict[str, int],
        reward: float,
        action: ActionData,
    ) -> list[float]:
        """
        Compute gradients based on counts and reward, using the selected gradient method.

        Parameters
        ----------
        counts : dict[str, int]
            Measurement counts from the quantum circuit.
        reward : float
            Reward signal to guide gradient computation.

        Returns
        -------
        list[float]
            Gradients for each parameter.
        """
        total_shots = sum(counts.values())
        probabilities = {key: value / total_shots for key, value in counts.items()}
        gradients = []

        # Dynamically generate binary strings for the number of qubits
        binary_states = [f"{{:0{self.num_qubits}b}}".format(i) for i in range(2**self.num_qubits)]

        # --- Standard policy gradient: reward * grad(log(prob)) ---
        # Also compute entropy for regularization
        entropy = 0.0
        gradients = []
        for key in binary_states:
            probability = probabilities.get(key, 1e-8)  # Avoid log(0)
            log_prob = np.log(probability)
            entropy -= probability * log_prob
            gradient = reward * (1 - probability) if key == action.state else -reward * probability
            gradients.append(gradient)

        # Add entropy regularization to gradients
        gradients = [g + ENTROPY_BETA * entropy for g in gradients]

        logger.debug(f"Computed gradients (policy gradient + entropy): {gradients}")

        post_processed_gradients = compute_gradients(
            gradients,
            self.gradient_method,
            self.parameter_values,
        )

        # Store gradients for tracking
        self.latest_gradients = post_processed_gradients
        self.history_gradients.append(post_processed_gradients)

        logger.debug(
            f"{self.gradient_method.value.capitalize()} gradients: {post_processed_gradients}",
        )

        # Log a warning if any gradient is out of the acceptable range
        for i, gradient in enumerate(post_processed_gradients):
            if gradient >= 1.0 or gradient <= -1.0:
                warning_message = (
                    f"Gradient for parameter θ{i} is out of bounds: {gradient}. "
                    "Gradients should be in the range (-1.0, 1.0)."
                )
                logger.warning(
                    warning_message.replace("θ", "theta_"),
                )

        return post_processed_gradients

    def update_parameters(  # noqa: C901, PLR0912, PLR0915
        self,
        gradients: list[float],
        reward: float | None = None,
    ) -> None:
        """
        Update quantum circuit parameter values based on gradients with a dynamic learning rate.

        Parameters
        ----------
        gradients : list[float]
            Gradients for each parameter.
        initial_learning_rate : float, optional
            Initial learning rate for parameter updates.
        decay_rate : float, optional
            Rate at which the learning rate decays over time, by default 0.01.
        """
        if isinstance(self.learning_rate, DynamicLearningRate):
            learning_rate = self.learning_rate.get_learning_rate()

            if len(gradients) != len(self.parameter_values) / 3 != self.num_qubits:
                error_message = (
                    f"Gradients length {len(gradients)} does not match parameter values length "
                    f"{len(self.parameter_values) / 3} for {self.num_qubits} qubits."
                )
                logger.error(error_message)
                raise ValueError(error_message)

            if TOGGLE_SHORT_TERM_MEMORY and len(self.history_gradients) >= 3:  # noqa: PLR2004
                grads_rx, grads_ry, grads_rz = (
                    self.history_gradients[-1],
                    self.history_gradients[-2],
                    self.history_gradients[-3],
                )
            else:
                grads_rx = grads_ry = grads_rz = self.history_gradients[-1]

            for i in range(self.num_qubits):
                self.parameter_values[f"θ_rx_{i}"] -= learning_rate * grads_rx[i]
                self.parameter_values[f"θ_ry_{i}"] -= learning_rate * grads_ry[i]
                self.parameter_values[f"θ_rz_{i}"] -= learning_rate * grads_rz[i]

            # Store learning rate for tracking
            self.latest_learning_rate = learning_rate
            self.history_learning_rates.append(learning_rate)

            logger.debug(
                f"Updated parameters with dynamic learning rate {learning_rate}: "
                f"{str(self.parameter_values).replace('θ', 'theta_')}",
            )
        elif isinstance(self.learning_rate, AdamLearningRate):
            if TOGGLE_SHORT_TERM_MEMORY:
                logger.warning(
                    "Adam learning rate strategy is not compatible with short-term memory.",
                )

            effective_learning_rates = self.learning_rate.get_learning_rate(
                gradients,
                list(self.parameter_values.keys()),
            )
            for param_name, _grad in zip(self.parameter_values.keys(), gradients, strict=False):
                self.parameter_values[param_name] -= effective_learning_rates[param_name]

            # Store learning rate for tracking
            self.latest_learning_rate = effective_learning_rates
            self.history_learning_rates.append(effective_learning_rates)

            logger.debug(
                f"Updated parameters with Adam learning rate: "
                f"{str(self.parameter_values).replace('θ', 'theta_')}",
            )
        elif isinstance(self.learning_rate, PerformanceBasedLearningRate):
            # Use performance-based learning rate strategy
            current_performance = reward if reward is not None else 0.0
            learning_rate = self.learning_rate.get_learning_rate(current_performance)

            if len(gradients) != len(self.parameter_values) / 3 != self.num_qubits:
                error_message = (
                    f"Gradients length {len(gradients)} does not match parameter values length "
                    f"{len(self.parameter_values) / 3} for {self.num_qubits} qubits."
                )
                logger.error(error_message)
                raise ValueError(error_message)

            if TOGGLE_SHORT_TERM_MEMORY and len(self.history_gradients) >= 3:  # noqa: PLR2004
                grads_rx = self.history_gradients[-1]
                grads_ry = self.history_gradients[-2]
                grads_rz = self.history_gradients[-3]
            else:
                grads_rx = grads_ry = grads_rz = gradients

            for i in range(self.num_qubits):
                self.parameter_values[f"θ_rx_{i}"] -= learning_rate * grads_rx[i]
                self.parameter_values[f"θ_ry_{i}"] -= learning_rate * grads_ry[i]
                self.parameter_values[f"θ_rz_{i}"] -= learning_rate * grads_rz[i]

            # Store learning rate for tracking
            self.latest_learning_rate = learning_rate
            self.history_learning_rates.append(learning_rate)

            logger.debug(
                f"Updated parameters with performance-based learning rate {learning_rate}: "
                f"{str(self.parameter_values).replace('θ', 'theta_')}",
            )

        # Parameter clipping: keep all parameters in [-pi, pi]
        if TOGGLE_PARAM_CLIP:
            logger.debug(
                "Clipping parameters to the range [-pi, pi]",
            )
            for i in range(self.num_qubits):
                self.parameter_values[f"θ_rx_{i}"] = np.clip(
                    self.parameter_values[f"θ_rx_{i}"],
                    -np.pi,
                    np.pi,
                )
                self.parameter_values[f"θ_ry_{i}"] = np.clip(
                    self.parameter_values[f"θ_ry_{i}"],
                    -np.pi,
                    np.pi,
                )
                self.parameter_values[f"θ_rz_{i}"] = np.clip(
                    self.parameter_values[f"θ_rz_{i}"],
                    -np.pi,
                    np.pi,
                )

        # Parameter modulo wrapping: keep all parameters in [-pi, pi]
        if TOGGLE_PARAM_MODULO:
            logger.debug(
                "Applying parameter modulo wrapping to the range [-pi, pi]",
            )
            for i in range(self.num_qubits):
                self.parameter_values[f"θ_rx_{i}"] = (
                    (self.parameter_values[f"θ_rx_{i}"] + np.pi) % (2 * np.pi)
                ) - np.pi
                self.parameter_values[f"θ_ry_{i}"] = (
                    (self.parameter_values[f"θ_ry_{i}"] + np.pi) % (2 * np.pi)
                ) - np.pi
                self.parameter_values[f"θ_rz_{i}"] = (
                    (self.parameter_values[f"θ_rz_{i}"] + np.pi) % (2 * np.pi)
                ) - np.pi

        # Store latest updated parameters for tracking
        self.latest_updated_parameters = deepcopy(self.parameter_values)
        self.history_updated_parameters.append(deepcopy(self.parameter_values))

        # Increment the step count
        self.steps += 1

    def interpret_counts(
        self,
        counts: dict[str, int],
        *,
        top_only: bool = True,
        top_randomize: bool = True,
    ) -> list[ActionData] | ActionData:
        """
        Interpret the measurement counts and determine the action dynamically.

        Interpret the measurement counts and determine the action dynamically
        based on the number of qubits.

        Parameters
        ----------
        counts : dict[str, int]
            Measurement counts from the quantum circuit.
        top_only : bool, optional
            If True, return only the most probable action, by default True.
        top_randomize : bool, optional
            If True, select the most probable action randomly, by default True.

        Returns
        -------
        list[ActionData] | ActionData
            The most probable action or a list of actions with their probabilities.
        """
        logger.debug(f"Raw counts from quantum circuit: {counts}")

        # Generate all possible binary strings for the current number of qubits
        num_states = 2**self.num_qubits
        binary_states = [f"{{:0{self.num_qubits}b}}".format(i) for i in range(num_states)]

        # Define a pool of possible actions
        action_pool = ["forward", "left", "right", "stay"]

        # Map binary states to actions dynamically
        action_map = {
            state: action_pool[i % len(action_pool)] for i, state in enumerate(binary_states)
        }

        logger.debug(f"Dynamic action map: {action_map}")

        # Filter counts to include only valid binary states
        counts = {key: value for key, value in counts.items() if key in action_map}

        if not counts:
            error_msg = "No valid actions found in counts."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Analyze distribution of counts
        total_counts = sum(counts.values())
        distribution = {key: value / total_counts for key, value in counts.items()}
        logger.debug(f"Normalized distribution of counts: {distribution}")

        logger.debug(
            f"Most probable action: {max(counts, key=lambda k: int(counts.get(k, 0)))} "
            f"with count {max(counts.values())}",
        )

        # Add noise to probabilities to encourage exploration
        noise = self.rng.uniform(0, 0.05, len(counts))  # Add small random noise
        temperature = (
            self.latest_temperature or TEMPERATURE
        )  # Use last temperature or default to 0.5
        probabilities = {
            key: math.exp((value / total_counts) / temperature) + noise[i]
            for i, (key, value) in enumerate(counts.items())
        }
        total_prob = sum(probabilities.values())
        probabilities = {key: value / total_prob for key, value in probabilities.items()}

        logger.debug(f"Softmax probabilities with noise: {probabilities}")

        # Select an action based on the softmax probabilities
        sorted_actions: list[ActionData] = sorted(
            [
                ActionData(state=key, action=action_map[key], probability=prob)
                for key, prob in probabilities.items()
            ],
            key=lambda x: x.probability,
            reverse=True,
        )

        most_probable_action = sorted_actions[0]
        logger.debug(
            f"Most probable action: {most_probable_action} with probability {sorted_actions[0]}",
        )

        if top_only:
            if top_randomize:
                # Select the most probable action randomly
                actions = [action.state for action in sorted_actions]
                probabilities = [sorted_action.probability for sorted_action in sorted_actions]
                chosen_action_state = self.rng.choice(actions, p=probabilities)
                chosen_action = next(
                    action for action in sorted_actions if action.state == chosen_action_state
                )
                logger.debug(
                    f"Selected action: {chosen_action.action} "
                    "with probability {chosen_action.probability}",
                )
                self.latest_action = chosen_action
                return chosen_action
            # Return the most probable action directly
            self.latest_action = most_probable_action
            return most_probable_action

        self.latest_action = sorted_actions[0]
        return sorted_actions

    def update_memory(self, reward: float) -> None:
        """
        No-op method for updating memory.

        Parameters
        ----------
        reward : float
            Reward signal.
        """

    def inspect_circuit(self) -> QuantumCircuit:
        """
        Inspect the quantum circuit.

        Returns
        -------
        QuantumCircuit
            The quantum circuit representing the brain.
        """
        qc = self.build_brain()
        qc.draw("text")
        return qc

    def copy(self) -> "DynamicBrain":
        """
        Create a deep copy of the DynamicBrain instance.

        Returns
        -------
        DynamicBrain
            A new instance of DynamicBrain with the same state.
        """
        new_brain = DynamicBrain(
            device=self.device,
            shots=self.shots,
            num_qubits=self.num_qubits,
            num_layers=self.num_layers,
            learning_rate=self.learning_rate,
            gradient_method=self.gradient_method,
            parameter_initializer=self.parameter_initializer,
        )
        new_brain.parameter_values = deepcopy(self.parameter_values)

        new_brain.steps = self.steps
        new_brain.satiety = self.satiety
        new_brain.gradient_method = self.gradient_method

        new_brain.latest_input_parameters = deepcopy(self.latest_input_parameters)
        new_brain.latest_updated_parameters = deepcopy(self.latest_updated_parameters)
        new_brain.latest_counts = deepcopy(self.latest_counts)
        new_brain.latest_action = deepcopy(self.latest_action)
        new_brain.latest_gradients = deepcopy(self.latest_gradients)
        new_brain.latest_learning_rate = self.latest_learning_rate
        new_brain.latest_exploration_factor = self.latest_exploration_factor
        new_brain.latest_temperature = self.latest_temperature

        new_brain.learning_rate = deepcopy(self.learning_rate) if self.learning_rate else None

        new_brain.reward_mean = self.reward_mean
        new_brain.reward_var = self.reward_var
        new_brain.reward_count = self.reward_count
        new_brain.reward_baseline = self.reward_baseline
        new_brain.reward_alpha = self.reward_alpha
        return new_brain
