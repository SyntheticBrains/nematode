"""Complex Quantum Brain Architecture."""

import numpy as np  # pyright: ignore[reportMissingImports]
from qiskit import QuantumCircuit  # pyright: ignore[reportMissingImports]
from qiskit.circuit import Parameter  # pyright: ignore[reportMissingImports]
from qiskit.providers.fake_provider import (  # pyright: ignore[reportMissingImports]
    GenericBackendV2,
)

from quantumnematode.brain._brain import Brain

QUBIT_COUNT = 302


class ComplexBrain(Brain):
    """
    Complex quantum brain architecture simulating 302 neurons with memory and behavior.

    This implementation models the nematode's brain with 302 qubits, each representing a neuron.
    The quantum circuit includes parameterized RX gates for each qubit to encode the state of
    individual neurons. CX gates are used to entangle adjacent qubits, simulating synaptic
    connections between neurons. The circuit is designed to capture the complexity of the
    nematode's neural network.

    Key Features:
    - Uses 302 qubits to represent the nematode's neurons.
    - Parameterized gates allow for dynamic updates based on the agent's state and learning.
    - Entanglement models the connectivity between neurons.
    - Designed for real quantum hardware (QPU) testing but can be simulated with sufficient
      computational resources.

    This architecture is ideal for advanced simulations and testing on real quantum hardware.
    Due to its complexity, it requires significant computational resources and is not suitable
    for all simulators.
    """

    def __init__(self, device: str = "CPU", shots: int = 100) -> None:
        self.device = device.upper()
        self.shots = shots
        self.neurons = [Parameter(f"θ{i}") for i in range(QUBIT_COUNT)]
        self.parameter_values = {f"θ{i}": 0.0 for i in range(QUBIT_COUNT)}

    def build_brain(self) -> QuantumCircuit:
        """
        Build the quantum circuit for the complex brain.

        Returns
        -------
        QuantumCircuit
            The quantum circuit representing the brain.
        """
        qc = QuantumCircuit(QUBIT_COUNT, QUBIT_COUNT)
        for i, neuron in enumerate(self.neurons):
            qc.rx(neuron, i)
        for i in range(301):
            qc.cx(i, i + 1)  # Entangle adjacent neurons
        qc.measure(range(QUBIT_COUNT), range(QUBIT_COUNT))
        return qc

    def run_brain(
        self,
        dx: int,  # noqa: ARG002
        dy: int,  # noqa: ARG002
        grid_size: int,  # noqa: ARG002
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

        # Assign parameters based on the agent's state and random noise
        rng = np.random.default_rng()
        for i, _neuron in enumerate(self.neurons):
            self.parameter_values[f"θ{i}"] += rng.uniform(-0.1, 0.1)

        bound_qc = qc.assign_parameters(
            {neuron: self.parameter_values[f"θ{i}"] for i, neuron in enumerate(self.neurons)},
            inplace=False,
        )

        # Use a fake 302Q QPU for simulation
        backend = GenericBackendV2(
            num_qubits=QUBIT_COUNT,
            basis_gates=["id", "rz", "sx", "x", "cx"],
            coupling_map=None,  # Fully connected
        )

        job = backend.run(bound_qc, shots=self.shots)
        result = job.result()
        counts = result.get_counts()

        # Update parameters if reward is provided
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
        for i in range(QUBIT_COUNT):
            probability = probabilities.get(
                f"{i:010b}",
                0,
            )  # Binary representation of neuron index
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
        for i, grad in enumerate(gradients):
            self.parameter_values[f"θ{i}"] -= learning_rate * grad

    def interpret_counts(
        self,
        counts: dict[str, int],
        agent_pos: list[int],
        grid_size: int,
    ) -> str:
        """
        Interpret the measurement counts and determine the action.

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
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        most_common = sorted_counts[0][0]  # Binary string of the most common result

        # Map binary string to actions
        valid_action_map = {}
        if agent_pos[1] < grid_size - 1:  # Can move up
            valid_action_map["00"] = "up"
        if agent_pos[1] > 0:  # Can move down
            valid_action_map["01"] = "down"
        if agent_pos[0] < grid_size - 1:  # Can move right
            valid_action_map["11"] = "right"
        if agent_pos[0] > 0:  # Can move left
            valid_action_map["10"] = "left"

        return valid_action_map.get(most_common[:2], "unknown")
