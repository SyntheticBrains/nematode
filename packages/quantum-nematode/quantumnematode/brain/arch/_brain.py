from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from quantumnematode.brain.dtypes import ActionData

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


class BrainData(BaseModel):
    """Data for the brain's operation."""

    run: list[int] = Field(default_factory=list, description="Run number")
    input_parameters: list[dict[str, float]] = Field(
        default_factory=list,
        description="Input parameters for the brain",
    )
    computed_gradients: list[list[float]] = Field(
        default_factory=list,
        description="Computed gradients for the brain",
    )
    learning_rate: list[float] = Field(
        default_factory=list,
        description="Learning rate used in the brain",
    )
    updated_parameters: list[dict[str, float]] = Field(
        default_factory=list,
        description="Updated parameters after training",
    )
    temperature: list[float] = Field(
        default_factory=list,
        description="Temperature parameter for the brain",
    )


class BrainParams(BaseModel):
    """Parameters for the brain's operation."""

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
    num_qubits: int

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
