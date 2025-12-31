"""Themes for the quantum nematode simulation."""

from enum import Enum

from pydantic import BaseModel


class Theme(str, Enum):
    """Simulation themes."""

    ASCII = "ascii"
    EMOJI = "emoji"
    UNICODE = "unicode"
    COLORED_ASCII = "colored_ascii"
    RICH = "rich"
    EMOJI_RICH = "emoji_rich"


DEFAULT_THEME = Theme.ASCII


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
    predator : str
        Symbol representing a predator (default/random type).
    predator_stationary : str
        Symbol for stationary predators (toxic zones).
    predator_pursuit : str
        Symbol for pursuit predators (actively chase agent).
    """

    goal: str
    body: str
    up: str
    down: str
    left: str
    right: str
    empty: str
    predator: str
    predator_stationary: str
    predator_pursuit: str


class DarkColorRichStyleConfig(BaseModel):
    """Rich styling configuration for colored on dark background Rich theme.

    Attributes
    ----------
    goal_style : str
        Rich style string for the goal (e.g., "bold red").
    body_style : str
        Rich style string for body segments.
    agent_style : str
        Rich style string for the agent.
    empty_style : str
        Rich style string for empty cells.
    grid_background : str
        Rich style string for grid cell backgrounds.
    """

    goal_style: str = "bold red"
    body_style: str = "bold blue"
    agent_style: str = "bold green"
    empty_style: str = "dim grey93"
    grid_background: str = "bold grey93"


THEME_SYMBOLS = {
    Theme.ASCII: ThemeSymbolSet(
        goal="*",
        body="O",
        up="^",
        down="v",
        left="<",
        right=">",
        empty=".",
        predator="#",
        predator_stationary="X",
        predator_pursuit="@",
    ),
    Theme.EMOJI: ThemeSymbolSet(
        goal="🦠",
        body="🔵",
        up="🔼",
        down="🔽",
        left="◀️ ",
        right="▶️ ",
        empty="⬜️",
        predator="🕷️ ",
        predator_stationary="☠️ ",
        predator_pursuit="🦂",
    ),
    Theme.UNICODE: ThemeSymbolSet(
        goal="◆",
        body="●",
        up="↑",
        down="↓",
        left="←",
        right="→",
        empty="·",
        predator="#",
        predator_stationary="⊗",
        predator_pursuit="⊛",
    ),
    Theme.COLORED_ASCII: ThemeSymbolSet(
        goal="\033[91m*\033[0m",
        body="\033[94mO\033[0m",
        up="\033[92m^\033[0m",
        down="\033[92mv\033[0m",
        left="\033[92m<\033[0m",
        right="\033[92m>\033[0m",
        empty="\033[90m.\033[0m",
        predator="\033[91m#\033[0m",
        predator_stationary="\033[95mX\033[0m",
        predator_pursuit="\033[93m@\033[0m",
    ),
    Theme.RICH: ThemeSymbolSet(
        goal="⬢",
        body="◉",
        up="▲",
        down="▼",
        left="◀",
        right="▶",
        empty="·",
        predator="#",
        predator_stationary="⊗",
        predator_pursuit="⊛",
    ),
    Theme.EMOJI_RICH: ThemeSymbolSet(
        goal=" 🦠",
        body=" 🔵",
        up=" 🔼",
        down=" 🔽",
        left="◀️",
        right="▶️",
        empty=" ",
        predator=" 🕷️",
        predator_stationary=" ☠️",
        predator_pursuit="🦂",
    ),
}
