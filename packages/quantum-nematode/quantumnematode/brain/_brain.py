from typing import Protocol

from qiskit import QuantumCircuit  # pyright: ignore[reportMissingImports]


class Brain(Protocol):
    """
    Base class for quantum brain architectures.

    This class defines the interface for building and running quantum brains.
    Subclasses should implement the build_brain and run_brain methods.
    """

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
        dx: int,
        dy: int,
        grid_size: int,
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
        agent_pos: list[int],
        grid_size: int,
    ) -> str:
        """
        Interpret the measurement counts and determine the action.

        This method should be implemented by subclasses.
        """
        error_msg = "Subclasses must implement the interpret_counts method to interpret the counts."
        raise NotImplementedError(error_msg)
