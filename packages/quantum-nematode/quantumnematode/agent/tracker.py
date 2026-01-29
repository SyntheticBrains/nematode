"""Episode tracking in the Quantum Nematode agent."""

from quantumnematode.agent.runners import EpisodeData


class EpisodeTracker:
    """Tracks data across a single episode.

    Attributes
    ----------
    episode_data : EpisodeData
        Data for the completed episode.
    """

    def __init__(self) -> None:
        """Initialize the metrics tracker with zero counters."""
        self.data = EpisodeData(
            steps=0,
            rewards=0.0,
            foods_collected=0,
            distance_efficiencies=[],
            satiety_history=[],
            health_history=[],
            temperature_history=[],
            predator_encounters=0,
            successful_evasions=0,
            in_danger=False,
        )

    @property
    def distance_efficiencies(self) -> list[float]:
        """Get the distance efficiencies for the episode."""
        return self.data.distance_efficiencies

    @property
    def foods_collected(self) -> int:
        """Get the total foods collected for the episode."""
        return self.data.foods_collected

    @property
    def rewards(self) -> float:
        """Get the total rewards for the episode."""
        return self.data.rewards

    @property
    def steps(self) -> int:
        """Get the total steps for the episode."""
        return self.data.steps

    @property
    def satiety_history(self) -> list[float]:
        """Get the satiety history for the episode."""
        return self.data.satiety_history

    @property
    def health_history(self) -> list[float]:
        """Get the health history for the episode."""
        return self.data.health_history

    @property
    def temperature_history(self) -> list[float]:
        """Get the temperature history for the episode."""
        return self.data.temperature_history

    @property
    def predator_encounters(self) -> int:
        """Get the total predator encounters for the episode."""
        return self.data.predator_encounters

    @predator_encounters.setter
    def predator_encounters(self, value: int) -> None:
        """Set the total predator encounters for the episode."""
        self.data.predator_encounters = value

    @property
    def successful_evasions(self) -> int:
        """Get the total successful evasions for the episode."""
        return self.data.successful_evasions

    @successful_evasions.setter
    def successful_evasions(self, value: int) -> None:
        """Set the total successful evasions for the episode."""
        self.data.successful_evasions = value

    @property
    def in_danger(self) -> bool:
        """Get whether agent is currently in danger."""
        return self.data.in_danger

    @in_danger.setter
    def in_danger(self, value: bool) -> None:
        """Set whether agent is currently in danger."""
        self.data.in_danger = value

    def track_food_collection(self, distance_efficiency: float | None = None) -> None:
        """Track food collection event.

        Parameters
        ----------
        distance_efficiency : float | None, optional
            For dynamic environments, the ratio of optimal distance to actual
            distance traveled. None if no food collected.
        """
        self.data.foods_collected += 1
        if distance_efficiency is not None:
            self.data.distance_efficiencies.append(distance_efficiency)

    def track_reward(self, reward: float) -> None:
        """Track a single reward.

        Parameters
        ----------
        reward : float
            Reward received for this instance.
        """
        self.data.rewards += reward

    def track_satiety(self, satiety: float) -> None:
        """Track current satiety level.

        Parameters
        ----------
        satiety : float
            Current satiety level for this step.
        """
        self.data.satiety_history.append(satiety)

    def track_health(self, health: float) -> None:
        """Track current health level.

        Parameters
        ----------
        health : float
            Current health (HP) level for this step.
        """
        self.data.health_history.append(health)

    def track_temperature(self, temperature: float) -> None:
        """Track current temperature.

        Parameters
        ----------
        temperature : float
            Current temperature at agent position for this step.
        """
        self.data.temperature_history.append(temperature)

    def track_step(self, reward: float = 0.0, satiety: float | None = None) -> None:
        """Track a single step.

        Parameters
        ----------
        reward : float
            Reward received for this step.
        satiety : float | None, optional
            Current satiety level (for dynamic foraging environments).
        """
        self.data.steps += 1
        self.data.rewards += reward
        if satiety is not None:
            self.data.satiety_history.append(satiety)

    def reset(self) -> None:
        """Reset the episode data for a new episode."""
        self.data = EpisodeData(
            rewards=0.0,
            steps=0,
            foods_collected=0,
            distance_efficiencies=[],
            satiety_history=[],
            health_history=[],
            temperature_history=[],
            predator_encounters=0,
            successful_evasions=0,
            in_danger=False,
        )
