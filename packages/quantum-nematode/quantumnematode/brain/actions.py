"""Data types for brains in Quantum Nematode."""

from enum import StrEnum

from pydantic import BaseModel


class Action(StrEnum):  # pragma: no cover
    """Actions that the agent can take."""

    FORWARD = "forward"
    LEFT = "left"
    RIGHT = "right"
    STAY = "stay"
    FORWARD_LEFT = "forward-left"
    FORWARD_RIGHT = "forward-right"


# Default 4-action set
DEFAULT_ACTIONS = [Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY]

# 6-action set
# NOTE: Not yet implemented.
SIX_ACTIONS = [
    Action.FORWARD,
    Action.FORWARD_LEFT,
    Action.LEFT,
    Action.FORWARD_RIGHT,
    Action.RIGHT,
    Action.STAY,
]


class ActionData(BaseModel):  # pragma: no cover
    """
    A class to represent the action taken by the agent.

    Attributes
    ----------
    state : str
        The current state of the agent.
    action : Action | None
        The discrete action taken by the agent (grid substrate), or ``None`` in
        continuous mode where the action is carried by ``continuous`` instead.
    probability : float
        The probability of taking the action in the current state.
    continuous : tuple[float, float] | None
        The continuous action ``(speed, turn)`` on the continuous-2D substrate,
        or ``None`` on the discrete grid substrate. On the continuous substrate the
        values are normalized (``speed ∈ [0, 1]``, ``turn ∈ [-1, 1]``); the
        environment rescales them to physical units.
    """

    state: str
    action: Action | None = None
    probability: float
    continuous: tuple[float, float] | None = None
