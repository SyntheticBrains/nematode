from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from quantumnematode.brain.actions import Action, ActionData
from quantumnematode.env import Direction

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


class BrainData(BaseModel):
    """Data for the brain's operation."""

    action: ActionData | None = Field(
        default=None,
        description="Action taken by the agent.",
    )
    computed_gradients: list[float] | None = Field(
        default=None,
        description="Computed gradients for the brain",
    )
    counts: dict[str, int] | None = Field(
        default=None,
        description="Counts of actions taken by the agent",
    )
    gradient_direction: float | None = Field(
        default=None,
        description="Direction of the chemical gradient",
    )
    gradient_strength: float | None = Field(
        default=None,
        description="Strength of the chemical gradient",
    )
    input_parameters: dict[str, float] | None = Field(
        default=None,
        description="Input parameters for the brain",
    )
    learning_rate: float | None = Field(
        default=None,
        description="Learning rate used in the brain",
    )
    loss: float | None = Field(
        default=None,
        description="Loss value during training",
    )
    probability: float | None = Field(
        default=None,
        description="Probability of action taken by the agent",
    )
    reward: float | None = Field(
        default=None,
        description="Reward received by the agent",
    )
    reward_norm: float | None = Field(
        default=None,
        description="Normalized reward received by the agent",
    )
    temperature: float | None = Field(
        default=None,
        description="Temperature parameter for the brain",
    )
    updated_parameters: dict[str, float] | None = Field(
        default=None,
        description="Updated parameters after training",
    )


class BrainHistoryData(BaseModel):
    """Historic data for the brain's operation."""

    actions: list[ActionData] = Field(
        default_factory=list,
        description="Actions taken by the agent during training",
    )
    computed_gradients: list[list[float]] = Field(
        default_factory=list,
        description="Computed gradients for the brain",
    )
    counts: list[dict[str, int]] = Field(
        default_factory=list,
        description="Counts of actions taken by the agent",
    )
    gradient_directions: list[float] = Field(
        default_factory=list,
        description="Direction of the chemical gradient",
    )
    gradient_strengths: list[float] = Field(
        default_factory=list,
        description="Strength of the chemical gradient",
    )
    input_parameters: list[dict[str, float]] = Field(
        default_factory=list,
        description="Input parameters for the brain",
    )
    learning_rates: list[float] = Field(
        default_factory=list,
        description="Learning rates used in the brain",
    )
    losses: list[float] = Field(
        default_factory=list,
        description="Loss values during training",
    )
    probabilities: list[float] = Field(
        default_factory=list,
        description="Probabilities of actions taken by the agent",
    )
    rewards: list[float] = Field(
        default_factory=list,
        description="Rewards received by the agent",
    )
    rewards_norm: list[float] = Field(
        default_factory=list,
        description="Normalized rewards received by the agent",
    )
    temperatures: list[float] = Field(
        default_factory=list,
        description="Temperature parameters for the brain",
    )
    updated_parameters: list[dict[str, float]] = Field(
        default_factory=list,
        description="Updated parameters after training",
    )


class BrainParams(BaseModel):
    """Parameters for the brain's operation.

    Sensory inputs and state information passed from the environment to the brain.
    All fields are optional with None defaults for backward compatibility.
    """

    # --- Agent state ---
    agent_position: tuple[float, float] | None = Field(
        default=None,
        description="Current position of the agent in the environment.",
    )
    agent_direction: Direction | None = Field(
        default=None,
        description="Current direction the agent is facing.",
    )
    action: ActionData | None = Field(
        default=None,
        description="Last action taken by the agent.",
    )

    # --- Chemotaxis (food/predator gradients) ---
    gradient_strength: float | None = Field(
        default=None,
        description="Combined gradient magnitude (food attraction + predator repulsion).",
    )
    gradient_direction: float | None = Field(
        default=None,
        description="Combined gradient direction (radians).",
    )
    food_gradient_strength: float | None = Field(
        default=None,
        description="Food gradient magnitude at agent's position.",
    )
    food_gradient_direction: float | None = Field(
        default=None,
        description="Food gradient direction at agent's position (radians).",
    )
    predator_gradient_strength: float | None = Field(
        default=None,
        description="Predator gradient magnitude at agent's position.",
    )
    predator_gradient_direction: float | None = Field(
        default=None,
        description="Predator gradient direction at agent's position (radians).",
    )

    # --- Thermotaxis (temperature sensing) ---
    temperature: float | None = Field(
        default=None,
        description="Current temperature at agent's position (°C).",
    )
    temperature_gradient_strength: float | None = Field(
        default=None,
        description="Temperature gradient magnitude (°C per cell).",
    )
    temperature_gradient_direction: float | None = Field(
        default=None,
        description="Temperature gradient direction (radians).",
    )
    cultivation_temperature: float | None = Field(
        default=None,
        description="Cultivation temperature (Tc) - the agent's preferred temperature.",
    )

    # --- Mechanosensation (touch/contact) ---
    boundary_contact: bool | None = Field(
        default=None,
        description="True if agent is touching grid boundary.",
    )
    predator_contact: bool | None = Field(
        default=None,
        description="True if agent is in physical contact with a predator.",
    )

    # --- Homeostasis (internal state) ---
    satiety: float | None = Field(
        default=None,
        description="Current satiety level (hunger state, decays over time).",
    )
    health: float | None = Field(
        default=None,
        description="Current HP (decreases from damage events).",
    )
    max_health: float | None = Field(
        default=None,
        description="Maximum HP the agent can have.",
    )


@runtime_checkable
class Brain(Protocol):
    history_data: BrainHistoryData
    latest_data: BrainData

    def run_brain(
        self,
        params: BrainParams,
        reward: float | None,
        input_data: list[float] | None,
        *,
        top_only: bool,
        top_randomize: bool,
    ) -> list[ActionData]: ...

    def update_memory(self, reward: float | None) -> None: ...

    def prepare_episode(self) -> None:
        """Prepare for a new episode (called at episode start)."""
        ...

    def post_process_episode(self, *, episode_success: bool | None = None) -> None: ...

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
    def action_set(self) -> list[Action]: ...

    def learn(
        self,
        params: BrainParams,
        reward: float,
        *,
        episode_done: bool,
    ) -> None: ...
