from qiskit import QuantumCircuit
from quantumnematode.brain._brain import Brain


class ComplexBrain(Brain):
    def build_brain(self) -> QuantumCircuit:
        raise NotImplementedError()

    def run_brain(
        self, dx: int, dy: int, grid_size: int, reward: float | None = None
    ) -> dict[str, int]:
        raise NotImplementedError()

    def interpret_counts(
        self,
        counts: dict[str, int],
        agent_pos: list[int],
        grid_size: int,
    ) -> str:
        raise NotImplementedError()
