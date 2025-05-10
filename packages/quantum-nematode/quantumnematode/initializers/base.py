from typing import Protocol

class ParameterInitializer(Protocol):
    """
    Base class for parameter initialization strategies.
    """
    def initialize(self, num_qubits: int) -> dict[str, float]:
        raise NotImplementedError("Subclasses must implement the `initialize` method.")

