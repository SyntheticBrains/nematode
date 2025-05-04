from typing import Protocol

from qiskit import QuantumCircuit  # pyright: ignore[reportMissingImports]


class Brain(Protocol):
    """
    Base class for quantum brain architectures.

    This class defines the interface for building and running quantum brains.
    Subclasses should implement the build_brain and run_brain methods.
    """

    satiety: float

    def build_brain(self) -> QuantumCircuit:
        """
        Build the quantum circuit for the brain.

        This method should be implemented by subclasses.
        """
        error_msg = (
            "Subclasses must implement the build_brain method to create the quantum circuit."
        )
        raise NotImplementedError(error_msg)

    def run_brain(
        self,
        gradient_strength: float,
        gradient_direction: float,
        reward: float | None = None,
    ) -> dict[str, int]:
        """
        Run the quantum brain simulation.

        This method should be implemented by subclasses.
        """
        error_msg = "Subclasses must implement the run_brain method to run the quantum circuit."
        raise NotImplementedError(error_msg)

    def interpret_counts(
        self,
        counts: dict[str, int],
        *,
        top_only: bool = True,
        top_randomize: bool = True,
    ) -> list[tuple[str, float]] | str:
        """
        Interpret the measurement counts and determine the action.

        This method should be implemented by subclasses.
        """
        error_msg = "Subclasses must implement the interpret_counts method to interpret the counts."
        raise NotImplementedError(error_msg)

    def update_memory(self, reward: float) -> None:
        """
        Update the memory states based on the reward signal.

        This method should be implemented by subclasses.
        """
        error_msg = "Subclasses must implement the update_memory method to update memory states."
        raise NotImplementedError(error_msg)

    def inspect_circuit(self) -> QuantumCircuit:
        """
        Inspect the quantum circuit.

        This method should be implemented by subclasses.
        """
        error_msg = (
            "Subclasses must implement the inspect_circuit method to inspect the quantum circuit."
        )
        raise NotImplementedError(error_msg)

    def copy(self) -> "Brain":
        """
        Create a copy of the brain.

        This method should be implemented by subclasses.
        """
        error_msg = "Subclasses must implement the copy method to create a copy of the brain."
        raise NotImplementedError(error_msg)
