"""Base class for parameter initialization strategies."""

from typing import Protocol


class ParameterInitializer(Protocol):
    """Base class for parameter initialization strategies."""

    def initialize(self, num_qubits: int, parameters: list[str] | None) -> dict[str, float]:
        """
        Initialize parameters for a quantum circuit.

        Parameters
        ----------
        num_qubits : int
            Number of qubits in the quantum circuit.
        parameters : list[str] | None
            List of parameter names to initialize. If None, all parameters will be initialized.

        Returns
        -------
        dict[str, float]
            A dictionary mapping parameter names to their initial values.
        """
        error_msg = "Subclasses must implement the `initialize` method."
        raise NotImplementedError(error_msg)
