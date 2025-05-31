"""Reduced Quantum Brain Architecture."""

import numpy as np  # pyright: ignore[reportMissingImports]
from qiskit import QuantumCircuit  # pyright: ignore[reportMissingImports]
from qiskit.circuit import Parameter  # pyright: ignore[reportMissingImports]
from qiskit_aer import AerSimulator  # pyright: ignore[reportMissingImports]

from quantumnematode.brain import Brain, BrainParams
from quantumnematode.logging_config import logger
from quantumnematode.models import ActionData


class ReducedBrain(Brain):
    """
    Reduced quantum brain architecture using 30 qubits for simulation.

    This implementation is a scaled-down version of the `ComplexBrain`, designed to fit within
    the limitations of current simulators. It uses 30 qubits to represent a subset of the
    nematode's neurons. The quantum circuit includes parameterized RX gates for each qubit to
    encode the state of individual neurons. CX gates are used to entangle adjacent qubits,
    simulating synaptic connections between neurons. The circuit is designed to balance
    complexity and computational feasibility.

    Key Features:
    - Uses 30 qubits to represent a subset of the nematode's neurons.
    - Parameterized gates allow for dynamic updates based on the agent's state and learning.
    - Entanglement models the connectivity between neurons.
    - Optimized for simulators with limited resources.

    This architecture is ideal for testing and exploring quantum reinforcement learning concepts
    on simulators while maintaining a balance between complexity and feasibility.
    """

    def __init__(self, device: str = "CPU", shots: int = 100) -> None:
        self.satiety = 1.0  # NOTE: Not used in this implementation
        self.device = device.upper()
        self.shots = shots
        self.neurons = [Parameter(f"θ{i}") for i in range(30)]
        self.parameter_values = {f"θ{i}": 0.0 for i in range(30)}

    def build_brain(self) -> QuantumCircuit:
        """
        Build the quantum circuit for the reduced brain.

        Returns
        -------
        QuantumCircuit
            The quantum circuit representing the brain.
        """
        qc = QuantumCircuit(30, 30)
        for i, neuron in enumerate(self.neurons):
            qc.rx(neuron, i)
        for i in range(29):
            qc.cx(i, i + 1)  # Entangle adjacent neurons
        qc.measure(range(30), range(30))
        return qc

    def run_brain(
        self,
        params: BrainParams,  # noqa: ARG002
        reward: float | None = None,
        input_data: list[float] | None = None,  # noqa: ARG002
    ) -> dict[str, int]:
        """
        Run the quantum brain simulation.

        Parameters
        ----------
        params : BrainParams
            Parameters for the quantum brain.
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

        # Assign parameters based on the agent's state and random noise
        rng = np.random.default_rng()
        for i, _neuron in enumerate(self.neurons):
            self.parameter_values[f"θ{i}"] += rng.uniform(-0.1, 0.1)

        bound_qc = qc.assign_parameters(
            {neuron: self.parameter_values[f"θ{i}"] for i, neuron in enumerate(self.neurons)},
            inplace=False,
        )

        # Use AerSimulator for simulation
        simulator = AerSimulator(device=self.device)
        result = simulator.run(bound_qc, shots=self.shots).result()
        counts = result.get_counts()

        # Update parameters if reward is provided
        if reward is not None:
            gradients = self.compute_gradients(counts, reward)
            self.update_parameters(gradients)

        return counts

    def compute_gradients(self, counts: dict[str, int], reward: float) -> list[float]:
        """
        Compute gradients based on counts and reward, with normalization to prevent large updates.

        Parameters
        ----------
        counts : dict[str, int]
            Measurement counts from the quantum circuit.
        reward : float
            Reward signal to guide gradient computation.

        Returns
        -------
        list[float]
            Normalized gradients for each parameter.
        """
        total_shots = sum(counts.values())
        probabilities = {key: value / total_shots for key, value in counts.items()}
        gradients = []
        for i in range(30):
            probability = probabilities.get(
                f"{i:05b}",
                0,
            )  # Binary representation of neuron index
            gradient = reward * (1 - probability)
            gradients.append(gradient)

        # # Normalize gradients to prevent large updates
        gradients = [
            g / max(abs(g) for g in gradients) if max(abs(g) for g in gradients) > 0 else g
            for g in gradients
        ]

        logger.debug(f"Computed gradients: {gradients}")
        return gradients

    def update_parameters(self, gradients: list[float], learning_rate: float = 0.1) -> None:
        """
        Update quantum circuit parameter values based on gradients.

        Parameters
        ----------
        gradients : list[float]
            Gradients for each parameter.
        learning_rate : float, optional
            Learning rate for parameter updates, by default 0.1.
        """
        for i, grad in enumerate(gradients):
            self.parameter_values[f"θ{i}"] -= learning_rate * grad

    def interpret_counts(
        self,
        counts: dict[str, int],
        *,
        top_only: bool = True,  # noqa: ARG002
        top_randomize: bool = True,  # noqa: ARG002
    ) -> list[ActionData] | ActionData:
        """
        Interpret the measurement counts and determine the action.

        Parameters
        ----------
        counts : dict[str, int]
            Measurement counts from the quantum circuit.

        Returns
        -------
        list[ActionData] | ActionData
            The most common action and its probability.
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

        chosen_action = action_map.get(most_common[:2], "unknown")
        probability = sorted_counts[0][1] / self.shots
        return ActionData(
            state=most_common,
            action=chosen_action,
            probability=probability,
        )

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

    def copy(self) -> "ReducedBrain":
        """
        Create a copy of the brain.

        Returns
        -------
        ReducedBrain
            A new instance of the ReducedBrain class with the same parameters.
        """
        error_msg = "Copying ReducedBrain is not implemented. Please implement the copy method."
        logger.error(error_msg)
        raise NotImplementedError(error_msg)
