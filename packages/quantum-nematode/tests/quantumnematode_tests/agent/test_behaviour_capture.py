"""Opt-in continuous behavioural-trajectory capture (real-worm chemotaxis validation, §1)."""

from __future__ import annotations

import math

from quantumnematode.agent import QuantumNematodeAgent, RewardConfig, SatietyConfig
from quantumnematode.brain.arch import ConnectomePPOBrain, ConnectomePPOBrainConfig
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.env.continuous_2d import Continuous2DEnvironment, Continuous2DParams
from quantumnematode.utils.config_loader import SensingConfig


def _agent(*, capture: bool) -> QuantumNematodeAgent:
    env = Continuous2DEnvironment(
        continuous=Continuous2DParams(world_size_mm=20.0, max_step_mm=1.0),
        seed=0,
    )
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
        sensing_config=SensingConfig(capture_behaviour=capture),
    )


def test_capture_off_records_nothing():
    """Default (capture off): no behavioural trajectory is recorded."""
    agent = _agent(capture=False)
    agent.run_episode(RewardConfig(), max_steps=20)
    assert agent.behaviour == []


def test_capture_on_records_finite_deduped_steps():
    """Capture on: one finite BehaviourStep per step, step indices deduped + monotonic."""
    agent = _agent(capture=True)
    agent.run_episode(RewardConfig(), max_steps=20)
    assert len(agent.behaviour) > 0
    steps = [b.step for b in agent.behaviour]
    assert steps == sorted(steps)  # monotonic
    assert len(steps) == len(set(steps))  # deduped (one record per step)
    for b in agent.behaviour:
        assert all(
            math.isfinite(v)
            for v in (
                b.x,
                b.y,
                b.heading_rad,
                b.concentration,
                b.dc_dt,
                b.grad_dir,
                b.grad_strength,
            )
        )


def test_capture_resets_each_episode():
    """Behaviour is per-episode — reset on reset_environment, not accumulated across episodes."""
    agent = _agent(capture=True)
    agent.run_episode(RewardConfig(), max_steps=20)
    assert len(agent.behaviour) > 0
    agent.reset_environment()
    assert agent.behaviour == []
