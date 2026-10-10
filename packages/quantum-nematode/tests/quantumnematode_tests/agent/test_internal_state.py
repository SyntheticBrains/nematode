"""The internal-state sensory module, and behaviour capture's lawn fields.

Covers the brain-architecture scenario "Satiety reaches the brain" and the realworm-behavioural-
validation scenario "A lawn step is recorded".
"""

from __future__ import annotations

import pytest
from quantumnematode.agent import QuantumNematodeAgent, RewardConfig, SatietyConfig
from quantumnematode.brain.arch import BrainParams, ConnectomePPOBrain, ConnectomePPOBrainConfig
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.modules import SENSORY_MODULES, ModuleName
from quantumnematode.env.continuous_2d import Continuous2DEnvironment, Continuous2DParams
from quantumnematode.env.env import ForagingParams
from quantumnematode.env.lawns import LawnParams
from quantumnematode.env.theme import Theme
from quantumnematode.report.behaviour_export import _record
from quantumnematode.utils.config_loader import SensingConfig

MODULE = SENSORY_MODULES[ModuleName.INTERNAL_STATE]


@pytest.mark.parametrize(
    ("satiety", "maximum", "expected"),
    [(50.0, 100.0, 0.5), (100.0, 100.0, 1.0), (150.0, 100.0, 1.0), (None, 100.0, 0.0)],
)
def test_satiety_reaches_the_brain(
    satiety: float | None,
    maximum: float,
    expected: float,
) -> None:
    """The module carries satiety as a fraction of its maximum, clamped, 0 when unknown."""
    features = MODULE.extract(BrainParams(satiety=satiety, max_satiety=maximum))
    assert features.strength == pytest.approx(expected)
    assert MODULE.classical_dim == 1


def _agent(*, lawns: bool) -> QuantumNematodeAgent:
    foraging = (
        ForagingParams(food_model="lawns", lawns=LawnParams(count=1, radius_mm=3.0))
        if lawns
        else ForagingParams(foods_on_grid=3)
    )
    env = Continuous2DEnvironment(
        continuous=Continuous2DParams(world_size_mm=20.0, max_step_mm=1.0),
        foraging=foraging,
        seed=0,
        theme=Theme.HEADLESS,
    )
    if env.lawn_field is not None:
        centre = env.lawn_field.centres[0]
        env.agents["default"].pos_continuous = (float(centre[0]), float(centre[1]))
    brain = ConnectomePPOBrain(
        config=ConnectomePPOBrainConfig(
            seed=0,
            action_mode="continuous",
            rollout_buffer_size=16,
            num_minibatches=2,
            num_epochs=2,
        ),
        device=DeviceType.CPU,
    )
    return QuantumNematodeAgent(
        brain=brain,
        env=env,
        satiety_config=SatietyConfig(initial_satiety=100.0),
        sensing_config=SensingConfig(capture_behaviour=True),
    )


def test_a_lawn_step_is_recorded() -> None:
    """Started on a lawn, the capture records satiety, a positive intake, and being on a lawn."""
    agent = _agent(lawns=True)
    reward = RewardConfig(
        penalty_stuck_position=0.0,
        reward_exploration=0.0,
        reward_distance_scale=0.0,
    )
    agent.run_episode(reward, max_steps=6)
    steps = agent.behaviour[1:]  # the first record precedes any move
    assert steps
    first = steps[0]
    assert first.satiety is not None
    assert first.on_lawn is True
    assert first.intake is not None
    assert first.intake > 0
    assert agent._episode_tracker.intake >= first.intake


def test_point_food_captures_leave_the_lawn_fields_out() -> None:
    """Without lawns the new fields stay None and the export omits them, as before."""
    agent = _agent(lawns=False)
    agent.run_episode(RewardConfig(), max_steps=4)
    record = _record(agent.behaviour[0])
    assert {"satiety", "intake", "on_lawn"}.isdisjoint(record)
