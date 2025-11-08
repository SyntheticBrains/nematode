"""Data types for reporting in Quantum Nematode."""

from enum import Enum

from pydantic import BaseModel, Field

from quantumnematode.brain.arch._brain import BrainHistoryData


class TerminationReason(str, Enum):
    """Reason why an episode terminated.

    Attributes
    ----------
    GOAL_REACHED : str
        Agent reached the goal (MazeEnvironment).
    COMPLETED_ALL_FOOD : str
        Agent collected all available food (DynamicForagingEnvironment - not yet implemented).
    STARVED : str
        Agent's satiety reached zero (DynamicForagingEnvironment).
    MAX_STEPS : str
        Agent reached maximum allowed steps.
    INTERRUPTED : str
        Episode was interrupted (e.g., by user).
    """

    GOAL_REACHED = "goal_reached"
    COMPLETED_ALL_FOOD = "completed_all_food"
    STARVED = "starved"
    MAX_STEPS = "max_steps"
    INTERRUPTED = "interrupted"


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
    termination_reason : TerminationReason
        The reason why the episode terminated.
    success : bool
        Whether the run was successful (goal_reached or completed_all_food).
    efficiency_score : float | None
        The efficiency score of the simulation, calculated as the offset
        from the perfect travel to the goal (MazeEnvironment only).
    foods_collected : int | None
        Number of foods collected (DynamicForagingEnvironment only).
    foods_available : int | None
        Total number of foods available in the environment (DynamicForagingEnvironment only).
    satiety_remaining : float | None
        Remaining satiety at the end of the run (DynamicForagingEnvironment only).
    average_distance_efficiency : float | None
        Average distance efficiency per food collected (DynamicForagingEnvironment only).
    satiety_history : list[float] | None
        Step-by-step satiety levels throughout the run (DynamicForagingEnvironment only).
    """

    run: int
    steps: int
    path: list[tuple[int, int]]
    total_reward: float
    last_total_reward: float
    termination_reason: TerminationReason
    success: bool
    efficiency_score: float | None = None
    foods_collected: int | None = None
    foods_available: int | None = None
    satiety_remaining: float | None = None
    average_distance_efficiency: float | None = None
    satiety_history: list[float] | None = None


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
    total_successes : int
        Total number of successful runs.
    total_starved : int
        Total number of runs that ended due to starvation.
    total_max_steps : int
        Total number of runs that hit maximum steps.
    total_interrupted : int
        Total number of runs that were interrupted.
    """

    success_rate: float
    average_steps: float
    average_reward: float
    foraging_efficiency: float | None = None
    average_distance_efficiency: float | None = None
    average_foods_collected: float | None = None
    total_successes: int = 0
    total_starved: int = 0
    total_max_steps: int = 0
    total_interrupted: int = 0
