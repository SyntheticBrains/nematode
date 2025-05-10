"""Zeros initializer for quantum parameters."""

from quantumnematode.initializers.base import ParameterInitializer


class ZeroInitializer(ParameterInitializer):
    """Initialize parameters to zero."""

    def initialize(self, num_qubits: int) -> dict[str, float]:
        """
        Initialize parameters to zero.

        Parameters
        ----------
        num_qubits : int
            Number of qubits in the quantum circuit.

        Returns
        -------
        dict[str, float]
            A dictionary mapping parameter names to their initial values.
        """
        return {f"θ{i}": 0.0 for i in range(num_qubits)}
