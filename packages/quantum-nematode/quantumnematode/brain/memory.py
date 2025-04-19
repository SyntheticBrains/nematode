"""Memory-based Quantum Brain Architecture."""

import numpy as np  # pyright: ignore[reportMissingImports]
from qiskit import QuantumCircuit, transpile  # pyright: ignore[reportMissingImports]
from qiskit.circuit import (  # pyright: ignore[reportMissingImports]
    ClassicalRegister,
    QuantumRegister,
)
from qiskit_aer import AerSimulator  # pyright: ignore[reportMissingImports]

from quantumnematode.brain._brain import Brain
from quantumnematode.logging_config import logger


class MemoryBrain(Brain):
    """
    Quantum brain architecture implementing Quantum Memory Encoding.

    This architecture encodes the nematode's memory of past actions and visited positions into
    quantum states. It leverages quantum superposition and interference to explore multiple
    memory states simultaneously and reinforce paths that lead to food while suppressing
    inefficient paths.

    Key Features:
    - Uses quantum registers to store memory states.
    - Applies quantum interference to prioritize successful paths.
    - Dynamically updates memory states based on rewards or penalties.
    - Optimized for simulators with limited resources.

    This architecture is ideal for exploring the role of memory in quantum-enhanced decision-making.
    """

    def __init__(self) -> None:
        self.memory_register = QuantumRegister(
            5,
            "memory",
        )  # 5 qubits for memory states
        self.action_register = QuantumRegister(2, "action")  # 2 qubits for actions
        self.classical_register = ClassicalRegister(
            2,
            "classical",
        )  # 2 classical bits for measurement
        self.circuit = QuantumCircuit(
            self.memory_register,
            self.action_register,
            self.classical_register,
        )

    def build_brain(self) -> QuantumCircuit:
        """
        Build the quantum circuit for the memory brain.

        Returns
        -------
        QuantumCircuit
            The quantum circuit representing the brain.
        """
        # Initialize memory and action registers
        self.circuit.h(self.memory_register)  # Put memory qubits into superposition
        self.circuit.h(self.action_register)  # Put action qubits into superposition

        # Example: Apply controlled operations based on memory states
        for i in range(len(self.memory_register)):  # Entangle memory with actions
            self.circuit.cx(self.memory_register[i], self.action_register[i % 2])

        # Introduce quantum interference to prioritize successful memory states
        for i in range(len(self.memory_register) - 1):
            self.circuit.cz(
                self.memory_register[i],
                self.memory_register[i + 1],
            )  # Apply controlled-Z gates

        # Apply a global phase shift to amplify successful paths
        self.circuit.p(np.pi / 4, self.memory_register)

        # Quantum Amplitude Amplification for Action Selection
        # Define an oracle to mark the desired action states
        # Refine amplitude amplification to prioritize successful actions
        for i in range(len(self.action_register)):
            if i % 2 == 0:  # Example: prioritize even-indexed actions
                self.circuit.z(self.action_register[i])

        # Apply a more targeted diffusion operator
        self.circuit.h(self.action_register)
        for i in range(len(self.action_register)):
            self.circuit.x(self.action_register[i])
        self.circuit.h(self.action_register[-1])
        self.circuit.mcx(
            list(range(len(self.action_register) - 1)),
            self.action_register[-1],
        )
        self.circuit.h(self.action_register[-1])
        for i in range(len(self.action_register)):
            self.circuit.x(self.action_register[i])
        self.circuit.h(self.action_register)

        # Measure action qubits
        self.circuit.measure(self.action_register, self.classical_register)

        return self.circuit

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

        # Optimize the circuit before simulation
        simulator = AerSimulator()
        qc = transpile(qc, simulator, optimization_level=3)

        # Simulate the quantum circuit
        result = simulator.run(qc, shots=1024).result()
        counts = result.get_counts()

        # Update memory states based on reward
        if reward is not None:
            self.update_memory(reward)

        return counts

    def update_memory(self, reward: float) -> None:
        """
        Update the memory states based on the reward signal.

        Parameters
        ----------
        reward : float
            Reward signal to guide memory updates.
        """
        logger.debug(f"Updating memory with reward: {reward}")
        for qubit in self.memory_register:
            if reward > 0:
                self.circuit.rx(abs(reward) * np.pi / 4, qubit)  # Reinforce
            else:
                self.circuit.rx(-abs(reward) * np.pi / 4, qubit)  # Suppress

    def log_circuit_details(self) -> None:
        """Log details of the quantum circuit for debugging purposes."""
        logger.debug("Quantum Circuit Details:")
        logger.debug(self.circuit.draw(output="text"))

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

        # Handle ties in measurement counts
        top_results = [result for result, count in sorted_counts if count == sorted_counts[0][1]]
        rng = np.random.default_rng()
        most_common = rng.choice(top_results)  # Randomly select among ties

        # Map the result to an action
        return valid_action_map.get(most_common[:2], "unknown")
