"""Dynamic Quantum Brain Architecture."""

import math
from copy import deepcopy

import numpy as np  # pyright: ignore[reportMissingImports]
from qiskit import QuantumCircuit, transpile  # pyright: ignore[reportMissingImports]
from qiskit.circuit import Parameter  # pyright: ignore[reportMissingImports]
from qiskit_aer import AerSimulator  # pyright: ignore[reportMissingImports]

from quantumnematode.brain._brain import Brain
from quantumnematode.logging_config import logger
from quantumnematode.optimizers.gradient_methods import GradientCalculationMethod, compute_gradients
from quantumnematode.optimizers.learning_rate import AdamLearningRate, DynamicLearningRate

EXPLORATION_MIN = 0.6  # Minimum exploration factor
EXPLORATION_MAX = 1.0  # Maximum exploration factor
TEMPERATURE = 0.9  # Default temperature for softmax action selection


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

    def __init__(
        self,
        device: str = "CPU",
        shots: int = 100,
        num_qubits: int = 5,
        learning_rate: DynamicLearningRate | AdamLearningRate | None = None,
        gradient_method: GradientCalculationMethod = GradientCalculationMethod.RAW,
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
        learning_rate : DynamicLearningRate | AdamLearningRate | None
            The learning rate strategy for parameter updates, by default None.
            If None, a default dynamic learning rate will be used.
        gradient_method : GradientCalculationMethod
            The method to use for gradient calculation, by default RAW.
        """
        self.device = device.upper()
        self.shots = shots
        self.num_qubits = num_qubits
        self.parameters = [Parameter(f"θ{i}") for i in range(num_qubits)]
        self.rng = np.random.default_rng()
        self.parameter_values = {
            f"θ{i}": self.rng.uniform(-np.pi, np.pi) for i in range(num_qubits)
        }
        self.steps = 0
        self.satiety = 1.0
        self.latest_input_parameters = None
        self.latest_updated_parameters = None
        self.latest_gradients = None
        self.latest_learning_rate = None
        self.latest_exploration_factor = None
        self.latest_temperature = None
        self.gradient_method = gradient_method

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

        # Log parameter initialization range
        logger.debug(
            "Initializing parameters uniformly in the range [-pi, pi]: "
            f"{str(self.parameter_values).replace('θ', 'theta_')}",
        )

    def build_brain(self) -> QuantumCircuit:
        """
        Build the quantum circuit for the dynamic brain.

        Build a dynamic quantum circuit based on the number of qubits
        and incorporate gradient information.

        Returns
        -------
        QuantumCircuit
            The quantum circuit representing the brain.
        """
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        for i, param in enumerate(self.parameters):
            qc.rx(param, i)
            qc.ry(param, i)
            qc.rz(param, i)
        for i in range(self.num_qubits - 1):
            qc.cx(i, i + 1)  # Entangle adjacent qubits
        qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc

    def run_brain(
        self,
        gradient_strength: float,
        gradient_direction: float,
        reward: float | None = None,
    ) -> dict[str, int]:
        """
        Run the quantum brain simulation, incorporating gradient information.

        Parameters
        ----------
        gradient_strength : float
            Intensity of the chemical gradient at the worm's position.
        gradient_direction : float
            Direction of the strongest gradient relative to the worm's orientation.
        reward : float, optional
            Reward signal for learning, by default None.

        Returns
        -------
        dict[str, int]
            Measurement counts from the quantum circuit.
        """
        qc = self.build_brain()

        # Incorporate gradient information into the parameters
        input_params = {
            f"θ{i}": self.parameter_values[f"θ{i}"]
            + gradient_strength * np.cos(gradient_direction + i * np.pi / self.num_qubits)
            for i in range(self.num_qubits)
        }

        # Store latest input parameters for tracking
        self.latest_input_parameters = input_params

        logger.debug(
            "Input parameters with gradient information: "
            f"{str(input_params).replace('θ', 'theta_')}",
        )

        bound_qc = qc.assign_parameters(input_params, inplace=False)

        simulator = AerSimulator(device=self.device)
        transpiled = transpile(bound_qc, simulator)
        result = simulator.run(transpiled, shots=self.shots).result()
        counts = result.get_counts()

        if reward is not None:
            gradients = self.compute_gradients(counts, reward)
            self.update_parameters(gradients)

        # Decrease satiety at each step
        self.satiety = max(0.0, self.satiety - 0.01)  # Decrease satiety gradually

        logger.debug(f"Satiety after step {self.steps}: {self.satiety}, ")

        if self.satiety <= 0.0:
            logger.warning("Satiety is zero.")

        # Calculate exploration factor based on satiety
        self.latest_exploration_factor = (
            EXPLORATION_MIN + (EXPLORATION_MAX - EXPLORATION_MIN) * self.satiety
        )
        self.latest_temperature = TEMPERATURE * self.latest_exploration_factor

        logger.debug(
            f"Exploration factor: {self.latest_exploration_factor}, "
            f"Temperature: {self.latest_temperature}",
        )

        return counts

    def compute_gradients(self, counts: dict[str, int], reward: float) -> list[float]:
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
        for key in ["00", "01", "10", "11"]:
            probability = probabilities.get(key, 0)
            gradient = reward * (1 - probability)
            gradients.append(gradient)

        logger.debug(f"Computed gradients: {gradients}")

        post_processed_gradients = compute_gradients(
            gradients,
            self.gradient_method,
        )

        self.latest_gradients = post_processed_gradients

        logger.debug(
            f"{self.gradient_method.value.capitalize()} gradients: {post_processed_gradients}",
        )
        return post_processed_gradients

    def update_parameters(
        self,
        gradients: list[float],
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
        # Use the selected learning rate strategy
        if isinstance(self.learning_rate, DynamicLearningRate):
            learning_rate = self.learning_rate.get_learning_rate()
            for param_name, grad in zip(self.parameter_values.keys(), gradients, strict=False):
                self.parameter_values[param_name] -= learning_rate * grad

            # Store learning rate for tracking
            self.latest_learning_rate = learning_rate

            logger.debug(
                f"Updated parameters with dynamic learning rate {learning_rate}: "
                f"{str(self.parameter_values).replace('θ', 'theta_')}",
            )
        elif isinstance(self.learning_rate, AdamLearningRate):
            effective_learning_rates = self.learning_rate.get_learning_rate(
                gradients,
                self.parameter_values.keys(),
            )
            for param_name, _grad in zip(self.parameter_values.keys(), gradients, strict=False):
                self.parameter_values[param_name] -= effective_learning_rates[param_name]

            # Store learning rate for tracking
            self.latest_learning_rate = effective_learning_rates

            logger.debug(
                f"Updated parameters with Adam learning rate: "
                f"{str(self.parameter_values).replace('θ', 'theta_')}",
            )

        # Store latest updated parameters for tracking
        self.latest_updated_parameters = deepcopy(self.parameter_values)

        # Increment the step count
        self.steps += 1

    def interpret_counts(
        self,
        counts: dict[str, int],
        *,
        best_only: bool = True,
    ) -> list[tuple[str, float]] | str:
        """
        Interpret the measurement counts and determine the action dynamically.

        Interpret the measurement counts and determine the action dynamically
        based on the number of qubits.

        Parameters
        ----------
        counts : dict[str, int]
            Measurement counts from the quantum circuit.

        Returns
        -------
        list[tuple[str, float]] | str
            List of tuples containing actions and their probabilities,
            or the single most probable action.
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
            logger.error("No valid actions found in counts. Defaulting to 'unknown'.")
            return "unknown"

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
        sorted_actions = sorted(
            [(action_map[key], prob) for key, prob in probabilities.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        most_probable_action = sorted_actions[0][0]
        logger.debug(
            f"Most probable action: {most_probable_action} with probability {sorted_actions[0][1]}",
        )

        if best_only:
            return most_probable_action

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
            learning_rate=self.learning_rate,
            gradient_method=self.gradient_method,
        )
        new_brain.parameter_values = deepcopy(self.parameter_values)
        new_brain.steps = self.steps
        new_brain.satiety = self.satiety
        new_brain.latest_input_parameters = deepcopy(self.latest_input_parameters)
        new_brain.latest_updated_parameters = deepcopy(self.latest_updated_parameters)
        new_brain.latest_gradients = deepcopy(self.latest_gradients)
        new_brain.latest_learning_rate = self.latest_learning_rate
        new_brain.latest_exploration_factor = self.latest_exploration_factor
        new_brain.latest_temperature = self.latest_temperature
        new_brain.learning_rate = deepcopy(self.learning_rate) if self.learning_rate else None
        return new_brain
