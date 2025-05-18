"""
Random uniform initializer for quantum parameters.

TODO: Merge the random uniform initializers into one class with a parameter for the range.
"""

import numpy as np  # pyright: ignore[reportMissingImports]

from quantumnematode.initializers._initializer import ParameterInitializer


class RandomPiUniformInitializer(ParameterInitializer):
    """Initialize parameters uniformly in the range [-pi, pi]."""

    def initialize(self, num_qubits: int, parameters: list[str] | None) -> dict[str, float]:
        """
        Initialize parameters uniformly in the range [-pi, pi].

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
        rng = np.random.default_rng()

        if parameters is not None:
            initialized_parameters = {}
            for param in parameters:
                if param in initialized_parameters:
                    error_message = f"Parameter {param} is already initialized."
                    raise ValueError(error_message)
                initialized_parameters[param] = rng.uniform(-np.pi, np.pi)
            return initialized_parameters

        return {f"θ{i}": rng.uniform(-np.pi, np.pi) for i in range(num_qubits)}


class RandomSmallUniformInitializer(ParameterInitializer):
    """Initialize parameters uniformly in the range [-0.1, 0.1]."""

    def initialize(self, num_qubits: int, parameters: list[str] | None) -> dict[str, float]:
        """
        Initialize parameters uniformly in the range [-0.1, 0.1].

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
        rng = np.random.default_rng()

        if parameters is not None:
            initialized_parameters = {}
            for param in parameters:
                if param in initialized_parameters:
                    error_message = f"Parameter {param} is already initialized."
                    raise ValueError(error_message)
                initialized_parameters[param] = rng.uniform(-0.1, 0.1)
            return initialized_parameters

        return {f"θ{i}": rng.uniform(-0.1, 0.1) for i in range(num_qubits)}
