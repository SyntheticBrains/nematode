"""Data types for reporting in Quantum Nematode."""

from pydantic import BaseModel, Field

from quantumnematode.brain.arch._brain import BrainHistoryData


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


TrackingRunIndex = int


class TrackingData(BaseModel):
    """Data structure for tracking agent's brain parameters across runs.

    Attributes
    ----------
    data : dict[TrackingRunIndex, BrainHistoryData]
        A dictionary mapping run indices to their corresponding brain history data.
    """

    data: dict[TrackingRunIndex, BrainHistoryData] = Field(
        default_factory=dict,
        description="Tracking data for each run, indexed by run number",
    )


class PerformanceMetrics(BaseModel):
    """Performance metrics for the simulation.

    Attributes
    ----------
    success_rate : float
        The rate of successful runs.
    average_steps : float
        The average number of steps taken across all runs.
    average_reward : float
        The average reward received across all runs.
    foraging_efficiency : float | None
        Foods collected per step (dynamic environments only).
    average_distance_efficiency : float | None
        Average distance efficiency per food (dynamic environments only).
    average_foods_collected : float | None
        Average number of foods collected per run (dynamic environments only).
    """

    success_rate: float
    average_steps: float
    average_reward: float
    foraging_efficiency: float | None = None
    average_distance_efficiency: float | None = None
    average_foods_collected: float | None = None
