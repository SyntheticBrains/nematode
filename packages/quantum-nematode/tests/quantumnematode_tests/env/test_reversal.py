"""Signed speed and the step's duration in worm time on the continuous-2D environment.

Covers the continuous-2d-environment requirements "Signed speed" (reversal off by default; a
negative speed moves the worm backward; sensing follows the head and the displacement; capture
records the signed speed only under reversal) and "The step's duration in worm time".
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from quantumnematode.agent import QuantumNematodeAgent, RewardConfig, SatietyConfig
from quantumnematode.brain.arch import ConnectomePPOBrain, ConnectomePPOBrainConfig
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.env.continuous_2d import Continuous2DEnvironment, Continuous2DParams
from quantumnematode.env.env import DEFAULT_AGENT_ID
from quantumnematode.env.worm_time import (
    UNDULATION_PERIOD_S,
    step_worm_seconds,
    undulations_per_step,
)
from quantumnematode.report.behaviour_export import write_behaviour_capture
from quantumnematode.report.dtypes import BehaviourStep
from quantumnematode.utils.config_loader import SensingConfig, load_simulation_config

_REPO = Path(__file__).resolve().parents[5]
_CELLS = (
    _REPO / "configs/scenarios/foraging/"
    "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350.yml",
    _REPO / "configs/scenarios/thermal_foraging/"
    "connectomeppo_small_continuous2d_thermal_klinotaxis_t35.yml",
)


def _env(*, reversal: bool, max_step: float = 1.0) -> Continuous2DEnvironment:
    return Continuous2DEnvironment(
        continuous=Continuous2DParams(
            world_size_mm=20.0,
            max_step_mm=max_step,
            allow_reversal=reversal,
        ),
    )


def _pos(env: Continuous2DEnvironment) -> tuple[float, float]:
    pc = env.agents[DEFAULT_AGENT_ID].pos_continuous
    assert pc is not None
    return pc


class TestReversalOff:
    def test_off_is_the_default(self) -> None:
        assert Continuous2DParams().allow_reversal is False

    def test_a_negative_speed_still_stops_the_worm(self) -> None:
        env = _env(reversal=False)
        before = _pos(env)
        env.move_agent_normalized(speed_norm=-1.0, turn_norm=0.0)
        assert _pos(env) == pytest.approx(before)


class TestReversalOn:
    def test_a_negative_speed_backs_the_worm_up(self) -> None:
        env = _env(reversal=True)
        x0, y0 = _pos(env)
        env.move_agent_continuous(speed=-0.6, turn=0.0)  # heading 0 is +x
        x, y = _pos(env)
        assert x == pytest.approx(x0 - 0.6)
        assert y == pytest.approx(y0)

    def test_the_heading_does_not_flip(self) -> None:
        env = _env(reversal=True)
        env.move_agent_continuous(speed=-1.0, turn=0.3)
        assert env.agents[DEFAULT_AGENT_ID].heading_rad == pytest.approx(0.3)

    def test_backward_speed_is_clamped_at_max_step(self) -> None:
        env = _env(reversal=True, max_step=1.0)
        x0, _ = _pos(env)
        env.move_agent_continuous(speed=-5.0, turn=0.0)
        assert _pos(env)[0] == pytest.approx(x0 - 1.0)

    def test_normalized_minus_one_is_a_full_backward_step(self) -> None:
        env = _env(reversal=True, max_step=0.8)
        x0, _ = _pos(env)
        env.move_agent_normalized(speed_norm=-1.0, turn_norm=0.0)
        assert _pos(env)[0] == pytest.approx(x0 - 0.8)

    def test_sensing_reads_the_backed_up_position(self) -> None:
        """The sensors read the float position, so a backward step moves what they sample."""
        env = _env(reversal=True)
        env.move_agent_continuous(speed=-1.0, turn=0.0)
        agent = env.agents[DEFAULT_AGENT_ID]
        assert agent.pos_continuous == pytest.approx(_pos(env))
        assert agent.heading_rad == pytest.approx(0.0)


def _agent(*, reversal: bool) -> QuantumNematodeAgent:
    env = Continuous2DEnvironment(
        continuous=Continuous2DParams(world_size_mm=20.0, max_step_mm=1.0, allow_reversal=reversal),
        seed=0,
    )
    brain = ConnectomePPOBrain(
        config=ConnectomePPOBrainConfig(
            seed=0,
            action_mode="continuous",
            signed_speed=reversal,
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


class TestCapture:
    def test_signed_speed_is_recorded_under_reversal(self) -> None:
        agent = _agent(reversal=True)
        agent.run_episode(RewardConfig(), max_steps=40)
        steps = agent.behaviour
        assert steps[0].speed_signed == 0.0
        for prev, cur in zip(steps, steps[1:], strict=False):
            expected = (cur.x - prev.x) * math.cos(cur.heading_rad) + (cur.y - prev.y) * math.sin(
                cur.heading_rad,
            )
            assert cur.speed_signed == pytest.approx(expected)
            assert abs(cur.speed_signed) <= 1.0 + 1e-9

    def test_an_untrained_signed_policy_reverses_sometimes(self) -> None:
        agent = _agent(reversal=True)
        agent.run_episode(RewardConfig(), max_steps=60)
        assert any((b.speed_signed or 0.0) < 0.0 for b in agent.behaviour)

    def test_without_reversal_nothing_is_recorded_or_exported(self, tmp_path: Path) -> None:
        agent = _agent(reversal=False)
        agent.run_episode(RewardConfig(), max_steps=20)
        assert all(b.speed_signed is None for b in agent.behaviour)

        class _Run:
            run, seed, behaviour = 1, 0, agent.behaviour

        path = write_behaviour_capture([_Run()], tmp_path)  # type: ignore[list-item]
        assert path is not None
        step = json.loads(path.read_text())["runs"][0]["steps"][0]
        assert "speed_signed" not in step
        assert set(step) == {f for f in BehaviourStep.__dataclass_fields__ if f != "speed_signed"}


class TestStepConstant:
    def test_one_body_length_is_five_worm_seconds(self) -> None:
        assert step_worm_seconds(1.0) == pytest.approx(5.0)
        assert undulations_per_step(1.0) == pytest.approx(5.0 / UNDULATION_PERIOD_S)
        assert 3.0 <= undulations_per_step(1.0) < 3.2

    @pytest.mark.parametrize("config", _CELLS, ids=["hard350", "thermal_t35"])
    def test_block_v_cells_step_five_worm_seconds(self, config: Path) -> None:
        env = load_simulation_config(str(config)).environment
        assert env is not None
        assert env.continuous is not None
        assert step_worm_seconds(env.continuous.max_step_mm) == pytest.approx(5.0)

    def test_a_negative_step_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            step_worm_seconds(-1.0)
