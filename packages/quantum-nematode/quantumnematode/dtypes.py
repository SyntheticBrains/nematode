"""Data types for the quantum nematode simulation."""

from enum import Enum


class Theme(str, Enum):
    """Enum for simulation themes."""

    ASCII = "ascii"
    EMOJI = "emoji"
