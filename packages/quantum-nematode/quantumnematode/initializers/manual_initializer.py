"""Manual parameter initializer for quantum parameters."""

from quantumnematode.initializers._initializer import ParameterInitializer


class ManualParameterInitializer(ParameterInitializer):
    """Initialize parameters with manually specified values."""

    def __init__(self, parameter_values: dict[str, float]) -> None:
        """
        Initialize the ManualParameterInitializer.

        Parameters
        ----------
        parameter_values : dict[str, float]
            A dictionary mapping parameter names to their initial values.
        """
        self.parameter_values = parameter_values.copy()

    def __str__(self) -> str:
        """Return string representation of the initializer."""
        param_count = len(self.parameter_values)
        return f"ManualParameterInitializer({param_count} parameters manually specified)"

    def initialize(self, num_qubits: int, parameters: list[str] | None) -> dict[str, float]:
        """
        Initialize parameters with manually specified values.

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

        Raises
        ------
        ValueError
            If a requested parameter is not found in the manual parameter values,
            or if parameters is None but num_qubits doesn't match the manual parameters.
        """
        if parameters is not None:
            initialized_parameters = {}
            for param in parameters:
                if param in initialized_parameters:
                    error_message = f"Parameter {param} is already initialized."
                    raise ValueError(error_message)
                if param not in self.parameter_values:
                    error_message = (
                        f"Parameter {param} not found in manual parameter values. "
                        f"Available parameters: {list(self.parameter_values.keys())}"
                    )
                    raise ValueError(error_message)
                initialized_parameters[param] = self.parameter_values[param]
            return initialized_parameters

        # If parameters is None, return all manual parameters
        # Check if we have the expected parameter structure for the given num_qubits
        expected_params = set()
        for layer in range(1, 3):  # Assuming 2 layers based on current structure
            for axis in ["rx", "ry", "rz"]:
                for i in range(num_qubits):
                    expected_params.add(f"θ_{axis}{layer}_{i}")

        manual_params = set(self.parameter_values.keys())
        if not expected_params.issubset(manual_params):
            missing_params = expected_params - manual_params
            error_message = (
                f"Manual parameter values are missing required parameters for {num_qubits} qubits. "
                f"Missing parameters: {sorted(missing_params)}. "
                f"Available parameters: {sorted(manual_params)}"
            )
            raise ValueError(error_message)

        return {param: self.parameter_values[param] for param in expected_params}
