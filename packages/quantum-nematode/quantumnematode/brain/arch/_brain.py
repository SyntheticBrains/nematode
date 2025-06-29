from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, Field  # pyright: ignore[reportMissingImports]

from quantumnematode.models import ActionData  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    from qiskit import QuantumCircuit  # pyright: ignore[reportMissingImports]


class BrainParams(BaseModel):
    gradient_strength: float | None = Field(
        default=None,
        description="Strength of the chemical gradient.",
    )
    gradient_direction: float | None = Field(
        default=None,
        description="Direction of the chemical gradient.",
    )
    agent_position: tuple[float, float] | None = Field(
        default=None,
        description="Current position of the agent in the environment.",
    )
    agent_direction: str | None = Field(
        default=None,
        description="Current direction of the agent in the environment.",
    )
    action: ActionData | None = Field(
        default=None,
        description="Action taken by the agent.",
    )


@runtime_checkable
class Brain(Protocol):
    satiety: float

    def run_brain(
        self,
        params: BrainParams,
        reward: float | None,
        input_data: list[float] | None,
    ) -> dict: ...

    def interpret_counts(
        self,
        counts: dict,
        *,
        top_only: bool,
        top_randomize: bool,
    ) -> ActionData | list[ActionData]: ...

    def update_memory(self, reward: float | None) -> None: ...

    def copy(self) -> "Brain": ...


@runtime_checkable
class QuantumBrain(Brain, Protocol):
    def build_brain(
        self,
        input_params: dict[str, dict[str, float]] | None,
    ) -> "QuantumCircuit": ...

    def inspect_circuit(self) -> "QuantumCircuit": ...


@runtime_checkable
class ClassicalBrain(Brain, Protocol):
    @property
    def action_names(self) -> list[str]: ...

    def learn(
        self,
        params: BrainParams,
        action_idx: int,
        reward: float,
        episode_rewards: list[float] | None,
        gamma: float,
    ) -> None: ...
