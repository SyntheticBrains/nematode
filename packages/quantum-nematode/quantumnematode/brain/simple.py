"""Simple Quantum Brain Architecture."""

import numpy as np  # pyright: ignore[reportMissingImports]
from qiskit import QuantumCircuit, transpile  # pyright: ignore[reportMissingImports]
from qiskit.circuit import Parameter  # pyright: ignore[reportMissingImports]
from qiskit_aer import AerSimulator  # pyright: ignore[reportMissingImports]

from quantumnematode.brain._brain import Brain
from quantumnematode.logging_config import logger


class SimpleBrain(Brain):
    """
    Simple quantum brain architecture using parameterized quantum circuits.

    This implementation represents a lightweight quantum brain designed for basic decision-making.
    It uses a 2-qubit quantum circuit with parameterized RX, RY, and RZ gates to encode the agent's
    state. The circuit also includes a CX gate to introduce entanglement between the qubits. The
    output of the circuit is measured and mapped to one of four possible actions: up, down, left,
    or right.

    Key Features:
    - Uses 2 qubits for simplicity and efficiency.
    - Parameterized gates allow for dynamic updates based on the agent's state and learning.
    - Entanglement is introduced to model complex decision-making processes.
    - Designed to be lightweight and suitable for simulators with limited resources.

    This architecture is ideal for testing and exploring basic quantum reinforcement learning
    concepts.
    """

    def __init__(self, device: str = "CPU") -> None:
        self.device = device.upper()
        self.theta_x = Parameter("θx")
        self.theta_y = Parameter("θy")
        self.theta_z = Parameter("θz")
        self.theta_entangle = Parameter("θentangle")
        self.parameter_values = {
            "θx": 0.0,
            "θy": 0.0,
            "θz": 0.0,
            "θentangle": 0.0,
        }

    def build_brain(self) -> QuantumCircuit:
        """
        Build the quantum circuit for the simple brain.

        Returns
        -------
        QuantumCircuit
            The quantum circuit representing the brain.
        """
        qc = QuantumCircuit(2, 2)
        qc.rx(self.theta_x, 0)
        qc.ry(self.theta_y, 1)
        qc.rz(self.theta_z, 0)
        qc.cx(0, 1)
        qc.ry(self.theta_entangle, 1)
        qc.measure([0, 1], [0, 1])
        return qc

    def run_brain(
        self,
        dx: int,
        dy: int,
        grid_size: int,
        reward: float | None = None,
    ) -> dict[str, int]:
        """
        Run the quantum brain simulation.

        Parameters
        ----------
        dx : int
            Distance to the goal along the x-axis.
        dy : int
            Distance to the goal along the y-axis.
        grid_size : int
            Size of the grid environment.
        reward : float, optional
            Reward signal for learning, by default None.

        Returns
        -------
        dict[str, int]
            Measurement counts from the quantum circuit.
        """
        qc = self.build_brain()
        rng = np.random.default_rng()
        input_x = (
            self.parameter_values["θx"] + dx / (grid_size - 1) * np.pi + rng.uniform(-0.1, 0.1)
        )
        input_y = (
            self.parameter_values["θy"] + dy / (grid_size - 1) * np.pi + rng.uniform(-0.1, 0.1)
        )
        input_z = self.parameter_values["θz"] + rng.uniform(0, 2 * np.pi)
        input_entangle = self.parameter_values["θentangle"] + rng.uniform(0, 2 * np.pi)

        logger.debug(
            f"dx={dx}, dy={dy}, input_x={input_x}, input_y={input_y}, "
            f"input_z={input_z}, input_entangle={input_entangle}",
        )

        bound_qc = qc.assign_parameters(
            {
                self.theta_x: input_x,
                self.theta_y: input_y,
                self.theta_z: input_z,
                self.theta_entangle: input_entangle,
            },
            inplace=False,
        )

        simulator = AerSimulator(device=self.device)
        transpiled = transpile(bound_qc, simulator)
        result = simulator.run(transpiled, shots=1024).result()
        counts = result.get_counts()

        logger.debug(f"Counts: {counts}")

        if reward is not None:
            gradients = self.compute_gradients(counts, reward)
            self.update_parameters(gradients)

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
        return gradients

    def update_parameters(
        self,
        gradients: list[float],
        learning_rate: float = 0.1,
    ) -> None:
        """
        Update quantum circuit parameter values based on gradients.

        Parameters
        ----------
        gradients : list[float]
            Gradients for each parameter.
        learning_rate : float, optional
            Learning rate for parameter updates, by default 0.1.
        """
        for param_name, grad in zip(
            self.parameter_values.keys(),
            gradients,
            strict=False,
        ):
            self.parameter_values[param_name] -= learning_rate * grad

        logger.debug(f"Updated parameters: {self.parameter_values}")

    def interpret_counts(
        self,
        counts: dict[str, int],
        agent_pos: list[int],
        grid_size: int,
    ) -> str:
        """
        Interpret the quantum circuit's output counts into an action.

        Parameters
        ----------
        counts : dict[str, int]
            Measurement counts from the quantum circuit.
        agent_pos : list[int]
            Current position of the agent.
        grid_size : int
            Size of the grid environment.

        Returns
        -------
        str
            Action to be taken by the agent.
        """
        # Sort counts by frequency
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        logger.debug(f"Sorted counts: {sorted_counts}")

        # Map the quantum output to valid actions dynamically
        valid_action_map = {}
        if agent_pos[1] < grid_size - 1:  # Can move up
            valid_action_map["00"] = "up"
        if agent_pos[1] > 0:  # Can move down
            valid_action_map["01"] = "down"
        if agent_pos[0] < grid_size - 1:  # Can move right
            valid_action_map["11"] = "right"
        if agent_pos[0] > 0:  # Can move left
            valid_action_map["10"] = "left"

        # Select the most common result or randomly choose among ties
        top_results = [result for result, count in sorted_counts if count == sorted_counts[0][1]]
        rng = np.random.default_rng()
        most_common = rng.choice(top_results)

        # Map the result to an action
        return valid_action_map.get(most_common, "unknown")
