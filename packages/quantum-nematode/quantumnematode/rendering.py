"""Episode rendering for visualization of the quantum nematode agent."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantumnematode.env import BaseEnvironment


class EpisodeRenderer:
    """Handles visualization and rendering during episode execution.

    The episode renderer manages when and how to display the environment grid,
    including support for showing only the last frame, clearing the screen between
    frames, and headless mode for testing.

    Parameters
    ----------
    enabled : bool, optional
        Whether rendering is enabled. If False, all render calls are no-ops.
        Default is True.

    Attributes
    ----------
    enabled : bool
        Whether rendering is currently enabled.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        """Initialize the episode renderer.

        Parameters
        ----------
        enabled : bool, optional
            Whether rendering is enabled, by default True.
        """
        self.enabled = enabled

    def render_if_needed(
        self,
        env: BaseEnvironment,
        step: int,
        max_steps: int,
        *,
        show_last_frame_only: bool = False,
        text: str | None = None,
    ) -> None:
        """Render the environment if rendering conditions are met.

        Parameters
        ----------
        env : BaseEnvironment
            The environment to render.
        step : int
            Current step number (0-indexed).
        max_steps : int
            Maximum number of steps in the episode.
        show_last_frame_only : bool, optional
            If True, only render the final frame (step == max_steps - 1).
            If False, render every frame. Default is False.
        text : str | None, optional
            Additional text to display above the grid, by default None.
        """
        if not self.enabled:
            return

        is_last_frame = step >= max_steps - 1
        should_render = (not show_last_frame_only) or is_last_frame

        if should_render:
            self.render_frame(env, text=text, clear_screen=not show_last_frame_only)

    def render_frame(
        self,
        env: BaseEnvironment,
        *,
        text: str | None = None,
        clear_screen: bool = True,
    ) -> None:
        """Render a single frame of the environment.

        Parameters
        ----------
        env : BaseEnvironment
            The environment to render.
        text : str | None, optional
            Additional text to display above the grid, by default None.
        clear_screen : bool, optional
            Whether to clear the screen before rendering, by default True.
        """
        if not self.enabled:
            return

        if clear_screen:
            self.clear_screen()

        grid = env.render()
        print(grid)  # noqa: T201
        if text:
            print(text)  # noqa: T201

    @staticmethod
    def clear_screen() -> None:
        """Clear the terminal screen.

        Uses appropriate clear command for the operating system.
        """
        # Use ANSI escape sequence for better compatibility
        if sys.platform.startswith("win"):
            os.system("cls")  # noqa: S605, S607 - safe for terminal clearing
        else:
            # ANSI escape sequence: clear screen and move cursor to top-left
            print("\033[2J\033[H", end="")  # noqa: T201
