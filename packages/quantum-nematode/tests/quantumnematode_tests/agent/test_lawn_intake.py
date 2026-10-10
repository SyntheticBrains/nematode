"""Lawn intake in the episode runner, and intake in the run's summary.

Covers the patchy-lawns spec's scenarios "Eating depletes the cell under the worm" (through the
runner: reward and satiety follow intake times quality), "Off a lawn there is no intake", and
"Intake is recorded per episode".
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from quantumnematode.agent import SatietyConfig
from quantumnematode.agent.runners import StandardEpisodeRunner
from quantumnematode.agent.satiety import SatietyManager
from quantumnematode.agent.tracker import EpisodeTracker
from quantumnematode.env.continuous_2d import Continuous2DEnvironment, Continuous2DParams
from quantumnematode.env.env import ForagingParams
from quantumnematode.env.lawns import LawnParams
from quantumnematode.env.theme import Theme
from quantumnematode.report.csv_export import _simulation_result_to_row
from quantumnematode.report.dtypes import SimulationResult, TerminationReason

LAWNS = LawnParams(
    count=1,
    intake_fraction=0.2,
    reward_per_intake=3.0,
    satiety_per_intake=0.5,
    quality=(2.0, 2.0),
)


def _agent(position: tuple[float, float] | None) -> Any:
    env = Continuous2DEnvironment(
        continuous=Continuous2DParams(world_size_mm=20.0),
        foraging=ForagingParams(food_model="lawns", lawns=LAWNS),
        seed=5,
        theme=Theme.HEADLESS,
    )
    assert env.lawn_field is not None
    target = tuple(env.lawn_field.centres[0]) if position is None else position
    env.agents["default"].pos_continuous = (float(target[0]), float(target[1]))
    satiety = SatietyManager(SatietyConfig(initial_satiety=100.0))
    satiety.decay_satiety()  # below the maximum, so a restore is visible
    return SimpleNamespace(
        env=env,
        agent_id="default",
        _episode_tracker=EpisodeTracker(),
        _satiety_manager=satiety,
    )


def _runner() -> StandardEpisodeRunner:
    return StandardEpisodeRunner.__new__(StandardEpisodeRunner)


def test_eating_pays_reward_and_satiety_by_quality() -> None:
    """On a lawn, the cell loses a fifth of its density; reward and satiety follow it by quality."""
    agent = _agent(None)
    field = agent.env.lawn_field
    before = agent._satiety_manager.current_satiety
    reward = _runner()._handle_lawn_intake(agent, 0.5)
    value = 0.2 * 2.0  # intake_fraction of a full cell, times quality
    assert reward == pytest.approx(0.5 + 3.0 * value)
    assert agent._episode_tracker.intake == pytest.approx(value)
    restored = agent._satiety_manager.current_satiety - before
    assert restored == pytest.approx(min(100.0 * 0.5 * value, 100.0 - before))
    assert field.density.min() == pytest.approx(0.8)


def test_off_a_lawn_nothing_changes() -> None:
    """Off every lawn the reward, the intake and the lawns are untouched."""
    agent = _agent((0.5, 0.5))
    assert _runner()._handle_lawn_intake(agent, 0.5) == 0.5
    assert agent._episode_tracker.intake == 0.0
    assert np.all(agent.env.lawn_field.density == 1.0)


def test_point_food_is_untouched() -> None:
    """Without lawns the runner's lawn step returns the reward unchanged."""
    env = Continuous2DEnvironment(
        continuous=Continuous2DParams(world_size_mm=20.0),
        foraging=ForagingParams(foods_on_grid=3),
        seed=5,
        theme=Theme.HEADLESS,
    )
    agent: Any = SimpleNamespace(env=env, agent_id="default", _episode_tracker=EpisodeTracker())
    assert _runner()._handle_lawn_intake(agent, 0.25) == 0.25


def test_intake_is_recorded_per_episode() -> None:
    """A lawn run's intake reaches the summary CSV; a point-food run's column is empty."""
    base = {
        "run": 1,
        "steps": 10,
        "path": [],
        "total_reward": 1.0,
        "last_total_reward": 1.0,
        "termination_reason": TerminationReason.MAX_STEPS,
        "success": True,
    }
    row = _simulation_result_to_row(SimulationResult(**base, intake=1.25))
    assert row["intake"] == 1.25
    missing = _simulation_result_to_row(SimulationResult(**base))["intake"]
    assert isinstance(missing, float)
    assert np.isnan(missing)
