"""Models for the quantum nematode simulation."""

from enum import Enum

from pydantic import BaseModel, Field


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
