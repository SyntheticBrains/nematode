"""Random uniform initializer for quantum parameters."""

import numpy as np

from quantumnematode.initializers._initializer import ParameterInitializer
from quantumnematode.utils.seeding import get_rng


class RandomUniformInitializer(ParameterInitializer):
    """Initialize parameters uniformly in a configurable range [low, high]."""

    def __init__(self, low: float = -np.pi, high: float = np.pi) -> None:
        self.low = low
        self.high = high

    def __str__(self) -> str:
        """Return string representation of the initializer."""
        return f"RandomUniformInitializer(range=[{self.low}, {self.high}])"

    def initialize(
        self,
        num_qubits: int,
        parameters: list[str] | None,
        seed: int | None = None,
    ) -> dict[str, float]:
        """
        Initialize parameters uniformly in the configured range.

        Parameters
        ----------
        num_qubits : int
            Number of qubits in the quantum circuit.
        parameters : list[str] | None
            List of parameter names to initialize. If None, all parameters will be initialized.
        seed : int | None
            Random seed for reproducibility. If None, uses unseeded RNG.

        Returns
        -------
        dict[str, float]
            A dictionary mapping parameter names to their initial values.
        """
        rng = get_rng(seed)
        if parameters is not None:
            initialized_parameters = {}
            for param in parameters:
                if param in initialized_parameters:
                    error_message = f"Parameter {param} is already initialized."
                    raise ValueError(error_message)
                initialized_parameters[param] = rng.uniform(self.low, self.high)
            return initialized_parameters
        return {f"θ{i}": rng.uniform(self.low, self.high) for i in range(num_qubits)}


class RandomPiUniformInitializer(RandomUniformInitializer):
    """Initialize parameters uniformly in the range [-π, π]."""

    def __init__(self) -> None:
        super().__init__(low=-np.pi, high=np.pi)

    def __str__(self) -> str:
        """Return string representation of the initializer."""
        return "RandomPiUniformInitializer(range=[-pi, pi])"


class RandomSmallUniformInitializer(RandomUniformInitializer):
    """Initialize parameters uniformly in a small range [-0.1, 0.1]."""

    def __init__(self) -> None:
        super().__init__(low=-0.1, high=0.1)

    def __str__(self) -> str:
        """Return string representation of the initializer."""
        return "RandomSmallUniformInitializer(range=[-0.1, 0.1])"
