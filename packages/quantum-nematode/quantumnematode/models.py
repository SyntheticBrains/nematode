"""Models for the quantum nematode simulation."""

from pydantic import BaseModel  # pyright: ignore[reportMissingImports]


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
    """

    run: int
    steps: int
    path: list[tuple[int, int]]
    total_reward: float
    last_total_reward: float
