"""
Random uniform initializer for quantum parameters.

TODO: Merge the random uniform initializers into one class with a parameter for the range.
"""

import numpy as np  # pyright: ignore[reportMissingImports]

from quantumnematode.initializers._initializer import ParameterInitializer


class RandomPiUniformInitializer(ParameterInitializer):
    """Initialize parameters uniformly in the range [-pi, pi]."""

    def initialize(self, num_qubits: int) -> dict[str, float]:
        """
        Initialize parameters uniformly in the range [-pi, pi].

        Parameters
        ----------
        num_qubits : int
            Number of qubits in the quantum circuit.

        Returns
        -------
        dict[str, float]
            A dictionary mapping parameter names to their initial values.
        """
        rng = np.random.default_rng()
        return {f"θ{i}": rng.uniform(-np.pi, np.pi) for i in range(num_qubits)}

class RandomSmallUniformInitializer(ParameterInitializer):
    """Initialize parameters uniformly in the range [-0.1, 0.1]."""

    def initialize(self, num_qubits: int) -> dict[str, float]:
        """
        Initialize parameters uniformly in the range [-0.1, 0.1].

        Parameters
        ----------
        num_qubits : int
            Number of qubits in the quantum circuit.

        Returns
        -------
        dict[str, float]
            A dictionary mapping parameter names to their initial values.
        """
        rng = np.random.default_rng()
        return {f"θ{i}": rng.uniform(-0.1, 0.1) for i in range(num_qubits)}