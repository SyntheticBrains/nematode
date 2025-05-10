from quantumnematode.initializers.base import ParameterInitializer


class ZeroInitializer(ParameterInitializer):
    """
    Initialize parameters to zero.
    """
    def initialize(self, num_qubits: int) -> dict[str, float]:
        return {f"θ{i}": 0.0 for i in range(num_qubits)}