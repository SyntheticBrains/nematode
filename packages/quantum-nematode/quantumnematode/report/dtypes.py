"""Data types for reporting in Quantum Nematode."""

from pydantic import BaseModel


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
