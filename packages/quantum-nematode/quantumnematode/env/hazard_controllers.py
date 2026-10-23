"""Non-learning controllers that drive the locomotor primitives through their start hazards.

Each implements the brain protocol and learns nothing, so it runs through the same runner as a
brain (satiety, lawn intake, movement cost, capture and episode endings) and emits the same five
numbers in ``[-1, 1]``: run speed, run curvature, and a start hazard for each of reversal, omega
and delta.

- :class:`ConstantHazards` holds its speed and hazards fixed, with straight runs: bout starts are
  then a Poisson process whatever the worm senses.
- :class:`PirouetteHazards` is the klinokinesis rule: its reversal and turn hazards are
  ``rate * exp(-beta * d)``, ``d`` the food derivative feature as a brain sees it
  (``tanh(dC/dt * derivative_scale)``), so they rise as the concentration falls. Runs are straight.
  With ``beta`` negative the rule is inverted, reorienting as the concentration rises.

Both record the hazards they emit on the steps they are in a run, so a matched constant-hazard
controller can be set from a pirouette controller's mean.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from quantumnematode.brain.actions import ActionData
from quantumnematode.brain.arch._brain import BrainData, BrainHistoryData
from quantumnematode.env.primitives import (
    BOUT_KINDS,
    PrimitiveKinematics,
    load_kinematics,
    raw_for_hazard,
    raw_for_speed,
)

if TYPE_CHECKING:
    from quantumnematode.brain.arch import BrainParams


class _HazardController:
    """The brain protocol around a rule that maps a step's senses onto speed and hazards."""

    def __init__(self, speed_mm_per_s: float, kinematics: PrimitiveKinematics | None) -> None:
        self.kinematics = kinematics if kinematics is not None else load_kinematics()
        self.speed_mm_per_s = speed_mm_per_s
        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()
        # The hazards emitted on each step the worm was in a run, per bout kind.
        self.run_hazards: list[tuple[float, float, float]] = []

    def hazards(self, params: BrainParams) -> tuple[float, float, float]:
        """Return this step's start hazards per second for reversal, omega and delta."""
        raise NotImplementedError

    def run_brain(
        self,
        params: BrainParams,
        reward: float | None,  # noqa: ARG002
        input_data: list[float] | None,  # noqa: ARG002
        *,
        top_only: bool,  # noqa: ARG002
        top_randomize: bool,  # noqa: ARG002
    ) -> list[ActionData]:
        """Return the step's five numbers: speed, zero curvature, and the three start hazards."""
        rates = self.hazards(params)
        if params.locomotor_state in (None, "run"):
            self.run_hazards.append(rates)
        raw = (
            raw_for_speed(self.speed_mm_per_s, self.kinematics),
            0.0,
            *(raw_for_hazard(rate, self.kinematics) for rate in rates),
        )
        action = ActionData(state="continuous", probability=1.0, continuous=raw)
        self.latest_data.action = action
        self.history_data.actions.append(action)
        return [action]

    def mean_run_hazards(self) -> tuple[float, float, float]:
        """Return the mean hazard per bout kind over the steps this controller spent in runs."""
        if not self.run_hazards:
            msg = "no run steps recorded"
            raise ValueError(msg)
        n = len(self.run_hazards)
        sums = [sum(rates[k] for rates in self.run_hazards) for k in range(len(BOUT_KINDS))]
        return (sums[0] / n, sums[1] / n, sums[2] / n)

    def update_memory(self, reward: float | None) -> None:
        """Learn nothing."""

    def prepare_episode(self) -> None:
        """Keep nothing across episodes but the record of run hazards."""

    def post_process_episode(self, *, episode_success: bool | None = None) -> None:
        """Learn nothing."""

    def copy(self) -> _HazardController:
        """Return this controller: it holds no state a copy would need apart."""
        return self


class ConstantHazards(_HazardController):
    """Fixed speed, straight runs and fixed start hazards: bout starts form a Poisson process."""

    def __init__(
        self,
        speed_mm_per_s: float,
        hazards_per_s: tuple[float, float, float],
        kinematics: PrimitiveKinematics | None = None,
    ) -> None:
        super().__init__(speed_mm_per_s, kinematics)
        self.hazards_per_s = hazards_per_s

    def hazards(self, params: BrainParams) -> tuple[float, float, float]:  # noqa: ARG002
        """Return the fixed hazards."""
        return self.hazards_per_s


class PirouetteHazards(_HazardController):
    """Klinokinesis: start hazards that rise as the sensed food concentration falls.

    Each bout kind's hazard is ``rate * exp(-beta * d)``, with ``d`` the food derivative feature a
    brain reads, ``tanh(dC/dt * derivative_scale)``.
    """

    def __init__(
        self,
        speed_mm_per_s: float,
        hazards_per_s: tuple[float, float, float],
        beta: float,
        kinematics: PrimitiveKinematics | None = None,
    ) -> None:
        super().__init__(speed_mm_per_s, kinematics)
        self.hazards_per_s = hazards_per_s
        self.beta = beta

    def hazards(self, params: BrainParams) -> tuple[float, float, float]:
        """Return each kind's hazard, scaled by ``exp(-beta * d)``."""
        derivative = params.food_dconcentration_dt or 0.0
        feature = math.tanh(derivative * params.derivative_scale)
        scale = math.exp(-self.beta * feature)
        rates = self.hazards_per_s
        return (rates[0] * scale, rates[1] * scale, rates[2] * scale)
