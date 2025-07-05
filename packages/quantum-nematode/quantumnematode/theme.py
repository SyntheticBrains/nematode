"""Themes for the quantum nematode simulation."""

from enum import Enum

from pydantic import BaseModel


class Theme(str, Enum):
    """Enum for simulation themes."""

    ASCII = "ascii"
    EMOJI = "emoji"


class ThemeSymbolSet(BaseModel):
    """Symbol set for a specific theme.

    Attributes
    ----------
    goal : str
        Symbol representing the goal.
    body : str
        Symbol representing the body of the nematode.
    up : str
        Symbol for moving up.
    down : str
        Symbol for moving down.
    left : str
        Symbol for moving left.
    right : str
        Symbol for moving right.
    empty : str
        Symbol for an empty cell.
    """

    goal: str
    body: str
    up: str
    down: str
    left: str
    right: str
    empty: str
