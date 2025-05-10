from quantumnematode.initializers.base import ParameterInitializer


import numpy as np


class RandomUniformInitializer(ParameterInitializer):
    """
    Initialize parameters uniformly in the range [-pi, pi].
    """
    def initialize(self, num_qubits: int) -> dict[str, float]:
        rng = np.random.default_rng()
        return {f"θ{i}": rng.uniform(-np.pi, np.pi) for i in range(num_qubits)}