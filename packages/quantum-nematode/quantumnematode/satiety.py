"""Satiety (hunger) management system for the quantum nematode agent."""

from __future__ import annotations

from quantumnematode.agent import SatietyConfig  # noqa: TC001 - needed at runtime


class SatietyManager:
    """Manages the agent's satiety (hunger) level.

    The satiety manager handles hunger mechanics including satiety decay over time,
    restoration when food is consumed, and starvation detection. Satiety is clamped
    between 0.0 (starved) and the maximum satiety level (full).

    Parameters
    ----------
    config : SatietyConfig
        Configuration for satiety parameters (initial level, decay rate, etc.).

    Attributes
    ----------
    max_satiety : float
        Maximum satiety level (initial satiety value).
    decay_rate : float
        Amount of satiety lost per step.
    _current_satiety : float
        Current satiety level (private, use current_satiety property).
    """

    def __init__(self, config: SatietyConfig) -> None:
        """Initialize the satiety manager.

        Parameters
        ----------
        config : SatietyConfig
            Configuration for satiety parameters.
        """
        self.max_satiety = config.initial_satiety
        self.decay_rate = config.satiety_decay_rate
        self._current_satiety = config.initial_satiety

    @property
    def current_satiety(self) -> float:
        """Get the current satiety level (read-only).

        Returns
        -------
        float
            Current satiety level, clamped between 0.0 and max_satiety.
        """
        return self._current_satiety

    def decay_satiety(self) -> float:
        """Decay satiety by the configured decay rate.

        Satiety cannot go below 0.0.

        Returns
        -------
        float
            New satiety level after decay.
        """
        self._current_satiety = max(0.0, self._current_satiety - self.decay_rate)
        return self._current_satiety

    def restore_satiety(self, amount: float) -> float:
        """Restore satiety by the given amount.

        Satiety cannot exceed max_satiety.

        Parameters
        ----------
        amount : float
            Amount of satiety to restore.

        Returns
        -------
        float
            New satiety level after restoration.
        """
        self._current_satiety = min(self.max_satiety, self._current_satiety + amount)
        return self._current_satiety

    def is_starved(self) -> bool:
        """Check if the agent is starved.

        Returns
        -------
        bool
            True if current satiety is 0.0, False otherwise.
        """
        return self._current_satiety <= 0.0

    def reset(self) -> None:
        """Reset satiety to initial (maximum) level."""
        self._current_satiety = self.max_satiety
