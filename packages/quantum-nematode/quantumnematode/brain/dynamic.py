"""Dynamic Quantum Brain Architecture."""

from copy import deepcopy
import math

import numpy as np  # pyright: ignore[reportMissingImports]
from qiskit import QuantumCircuit, transpile  # pyright: ignore[reportMissingImports]
from qiskit.circuit import Parameter  # pyright: ignore[reportMissingImports]
from qiskit_aer import AerSimulator  # pyright: ignore[reportMissingImports]

from quantumnematode.brain._brain import Brain
from quantumnematode.logging_config import logger

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

    def __init__(self, device: str = "CPU", shots: int = 100, num_qubits: int = 5) -> None:
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
        
        # Log parameter initialization range
        logger.debug(
            f"Initializing parameters uniformly in the range [-pi, pi]: {str(self.parameter_values).replace('θ', 'theta_')}",
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
        Compute gradients based on counts and reward.

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

        # Normalize gradients to prevent large updates
        gradients = [
            g / max(abs(g) for g in gradients) if max(abs(g) for g in gradients) > 0 else g
            for g in gradients
        ]

        logger.debug(f"Computed gradients: {gradients}")

        # Store gradients for tracking
        self.latest_gradients = gradients

        return gradients

    def update_parameters(
        self,
        gradients: list[float],
        initial_learning_rate: float = 0.1,
        decay_rate: float = 0.01,
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
        # Increase learning rate for faster convergence
        dynamic_learning_rate = initial_learning_rate / (1 + decay_rate * self.steps)
        dynamic_learning_rate *= 1.5  # Scale up the learning rate by 1.5x

        for param_name, grad in zip(
            self.parameter_values.keys(),
            gradients,
            strict=False,
        ):
            self.parameter_values[param_name] -= dynamic_learning_rate * grad

        logger.debug(
            f"Updated parameters with dynamic learning rate {dynamic_learning_rate}: "
            f"{str(self.parameter_values).replace('θ', 'theta_')}",
        )

        # Store latest updated parameters for tracking
        self.latest_updated_parameters = deepcopy(self.parameter_values)

        # Store learning rate for tracking
        self.latest_learning_rate = dynamic_learning_rate

        # Increment the step count
        self.steps += 1

    def interpret_counts(
        self,
        counts: dict[str, int],
    ) -> str:
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
        str
            Action to be taken by the agent.
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
        actions, probs = zip(
            *[(action_map.get(key, "unknown"), prob) for key, prob in probabilities.items()],
            strict=False,
        )
        selected_action = self.rng.choice(actions, p=probs)

        logger.debug(f"Selected action: {selected_action}")

        return selected_action

    def update_memory(self, reward: float) -> None:
        """
        No-op method for updating memory.

        Parameters
        ----------
        reward : float
            Reward signal.
        """
