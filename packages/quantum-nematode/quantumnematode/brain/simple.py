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

    def __init__(self, device: str = "CPU", shots: int = 100) -> None:
        self.device = device.upper()
        self.shots = shots
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
        gradient_strength: float,
        gradient_direction: float,
        reward: float | None = None,
    ) -> dict[str, int]:
        """
        Run the quantum brain simulation.

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
        input_x = self.parameter_values["θx"] + gradient_strength * np.pi + self.rng.uniform(-1.0, 1.0)
        input_y = self.parameter_values["θy"] + gradient_direction * np.pi + self.rng.uniform(-1.0, 1.0)
        input_z = self.parameter_values["θz"] + self.rng.uniform(0, 2 * np.pi)
        input_entangle = self.parameter_values["θentangle"] + self.rng.uniform(0, 2 * np.pi)

        logger.debug(
            f"input_x={input_x}, input_y={input_y}, "
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
        result = simulator.run(transpiled, shots=self.shots).result()
        counts = result.get_counts()

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

        # Normalize gradients to prevent large updates
        gradients = [
            g / max(abs(g) for g in gradients) if max(abs(g) for g in gradients) > 0 else g
            for g in gradients
        ]

        logger.debug(f"Computed gradients: {gradients}")
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

        logger.debug(f"Updated parameters: {str(self.parameter_values).replace('θ', 'theta_')}")

    def interpret_counts(
        self,
        counts: dict[str, int],
    ) -> str:
        """
        Interpret the measurement counts and determine the action.

        Parameters
        ----------
        counts : dict[str, int]
            Measurement counts from the quantum circuit.

        Returns
        -------
        str
            Action to be taken by the agent.
        """
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        most_common = sorted_counts[0][0]  # Binary string of the most common result

        # Map binary string to actions
        action_map = {
            "00": "forward",
            "01": "left",
            "11": "right",
            "10": "stay",
        }

        return action_map.get(most_common[:2], "unknown")

    def update_memory(self, reward: float) -> None:
        """
        No-op method for updating memory in the SimpleBrain.

        Parameters
        ----------
        reward : float
            Reward signal (not used in SimpleBrain).
        """
