"""Themes for the quantum nematode simulation."""

from enum import StrEnum

from pydantic import BaseModel


class Theme(StrEnum):
    """Simulation themes."""

    ASCII = "ascii"
    EMOJI = "emoji"
    UNICODE = "unicode"
    COLORED_ASCII = "colored_ascii"
    RICH = "rich"
    EMOJI_RICH = "emoji_rich"
    PIXEL = "pixel"
    HEADLESS = "headless"


DEFAULT_THEME = Theme.PIXEL


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
    zone_lethal_cold : str
        Symbol/style for lethal cold zone background.
    zone_danger_cold : str
        Symbol/style for danger cold zone background.
    zone_discomfort_cold : str
        Symbol/style for discomfort cold zone background.
    zone_comfort : str
        Symbol/style for comfort zone background (typically same as empty).
    zone_discomfort_hot : str
        Symbol/style for discomfort hot zone background.
    zone_danger_hot : str
        Symbol/style for danger hot zone background.
    zone_lethal_hot : str
        Symbol/style for lethal hot zone background.
    zone_toxic : str
        Symbol/style for toxic zone background (stationary predator radius).
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
    # Zone backgrounds (optional, default to empty for themes without zone support)
    zone_lethal_cold: str = ""
    zone_danger_cold: str = ""
    zone_discomfort_cold: str = ""
    zone_comfort: str = ""
    zone_discomfort_hot: str = ""
    zone_danger_hot: str = ""
    zone_lethal_hot: str = ""
    zone_toxic: str = ""


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
    predator_style : str
        Rich style string for random predators.
    predator_stationary_style : str
        Rich style string for stationary predators.
    predator_pursuit_style : str
        Rich style string for pursuit predators.
    zone_lethal_cold_bg : str
        Background style for lethal cold temperature zone.
    zone_danger_cold_bg : str
        Background style for danger cold temperature zone.
    zone_discomfort_cold_bg : str
        Background style for discomfort cold temperature zone.
    zone_comfort_bg : str
        Background style for comfort temperature zone (default: no background).
    zone_discomfort_hot_bg : str
        Background style for discomfort hot temperature zone.
    zone_danger_hot_bg : str
        Background style for danger hot temperature zone.
    zone_lethal_hot_bg : str
        Background style for lethal hot temperature zone.
    zone_toxic_bg : str
        Background style for toxic zones (stationary predator damage radius).
    """

    goal_style: str = "bold red"
    body_style: str = "bold blue"
    agent_style: str = "bold green"
    empty_style: str = "dim grey93"
    grid_background: str = "bold grey93"

    # Predator foreground styles
    predator_style: str = "bold magenta"
    predator_stationary_style: str = "bold dark_magenta"
    predator_pursuit_style: str = "bold yellow"

    # Temperature zone background styles (cold to hot gradient)
    zone_lethal_cold_bg: str = "on blue"
    zone_danger_cold_bg: str = "on cyan"
    zone_discomfort_cold_bg: str = "on light_cyan3"
    zone_comfort_bg: str = ""  # No background override for comfort zone
    zone_discomfort_hot_bg: str = "on light_goldenrod1"
    zone_danger_hot_bg: str = "on orange1"
    zone_lethal_hot_bg: str = "on red"

    # Toxic zone background (higher priority than temperature)
    zone_toxic_bg: str = "on medium_purple"


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
        body="🟤",
        up="🔼",
        down="🔽",
        left="◀️ ",
        right="▶️ ",
        empty="⬜️",
        predator="🕷️ ",
        predator_stationary="☠️ ",
        predator_pursuit="🦂",
        # Zone backgrounds using colored square emojis
        zone_lethal_cold="🟦",
        zone_danger_cold="🟩",
        zone_discomfort_cold="🟩",
        zone_comfort="",
        zone_discomfort_hot="🟨",
        zone_danger_hot="🟧",
        zone_lethal_hot="🟥",
        zone_toxic="🟪",
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
        # Zone symbols using colored dots (foreground colors)
        zone_lethal_cold="\033[94m.\033[0m",  # Blue dot
        zone_danger_cold="\033[96m.\033[0m",  # Cyan dot
        zone_discomfort_cold="\033[96m.\033[0m",  # Cyan dot
        zone_comfort="",
        zone_discomfort_hot="\033[93m.\033[0m",  # Yellow dot
        zone_danger_hot="\033[33m.\033[0m",  # Orange/dark yellow dot
        zone_lethal_hot="\033[91m.\033[0m",  # Red dot
        zone_toxic="\033[95m.\033[0m",  # Magenta/purple dot
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
        body=" 🟤",
        up=" 🔼",
        down=" 🔽",
        left="◀️",
        right="▶️",
        empty=" ",
        predator=" 🕷️",
        predator_stationary=" ☠️",
        predator_pursuit="🦂",
    ),
    # PIXEL theme uses Pygame renderer; symbols are placeholders for fallback only
    Theme.PIXEL: ThemeSymbolSet(
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
    # HEADLESS theme skips all rendering; symbols are placeholders only
    Theme.HEADLESS: ThemeSymbolSet(
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
}
