"""Data types for brains in Quantum Nematode."""

from pydantic import BaseModel


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
    action: str
    probability: float
