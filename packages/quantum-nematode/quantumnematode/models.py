"""Models for the quantum nematode simulation."""

from enum import Enum

from pydantic import BaseModel, Field


class SimulationResult(BaseModel):
    """
    A class to represent the result of a simulation run.

    Attributes
    ----------
    run : int
        The run number of the simulation.
    steps : int
        The number of steps taken in the simulation.
    path : list[tuple[int, int]]
        The path taken by the agent during the simulation.
    total_reward : float
        The total reward received during the simulation.
    last_total_reward : float
        The last total reward received during the simulation.
    efficiency_score : float
        The efficiency score of the simulation, calculated as the offset
        from the perfect travel to the goal.
    """

    run: int
    steps: int
    path: list[tuple[int, int]]
    total_reward: float
    last_total_reward: float
    efficiency_score: float


class Theme(str, Enum):
    """Enum for simulation themes."""

    ASCII = "ascii"
    EMOJI = "emoji"


class TrackingData(BaseModel):
    """Data structure for tracking agent's brain parameters."""

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
