"""Random uniform initializer for quantum parameters."""

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
