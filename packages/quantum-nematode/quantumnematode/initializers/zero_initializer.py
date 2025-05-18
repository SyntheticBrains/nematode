"""Zeros initializer for quantum parameters."""

from quantumnematode.initializers._initializer import ParameterInitializer


class ZeroInitializer(ParameterInitializer):
    """Initialize parameters to zero."""

    def initialize(self, num_qubits: int, parameters: list[str] | None) -> dict[str, float]:
        """
        Initialize parameters to zero.

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
        if parameters is not None:
            initialized_parameters = {}
            for param in parameters:
                if param in initialized_parameters:
                    error_message = f"Parameter {param} is already initialized."
                    raise ValueError(error_message)
                initialized_parameters[param] = 0.0
            return initialized_parameters

        return {f"θ{i}": 0.0 for i in range(num_qubits)}
