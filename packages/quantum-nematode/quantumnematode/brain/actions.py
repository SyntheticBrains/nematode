"""Data types for brains in Quantum Nematode."""

from enum import Enum

from pydantic import BaseModel


class Action(str, Enum):
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


class ActionData(BaseModel):
    """
    A class to represent the action taken by the agent.

    Attributes
    ----------
    state : str
        The current state of the agent.
    action : str
        The action taken by the agent.
    probability : float
        The probability of taking the action in the current state.
    """

    state: str
    action: Action
    probability: float
