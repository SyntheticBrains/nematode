"""Opt-in continuous behavioural-trajectory capture for real-worm chemotaxis validation."""

from __future__ import annotations

import math
from typing import Literal

from quantumnematode.agent import QuantumNematodeAgent, RewardConfig, SatietyConfig
from quantumnematode.brain.arch import ConnectomePPOBrain, ConnectomePPOBrainConfig
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.env.continuous_2d import Continuous2DEnvironment, Continuous2DParams
from quantumnematode.env.env import ThermotaxisParams
from quantumnematode.utils.config_loader import SensingConfig, SensingMode


def _agent(
    *,
    capture: bool,
    chemotaxis_mode: SensingMode = SensingMode.ORACLE,
    capture_behaviour_modality: Literal["food", "thermotaxis"] = "food",
    thermotaxis: ThermotaxisParams | None = None,
) -> QuantumNematodeAgent:
    env = Continuous2DEnvironment(
        continuous=Continuous2DParams(world_size_mm=20.0, max_step_mm=1.0),
        seed=0,
        thermotaxis=thermotaxis,
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
        sensing_config=SensingConfig(
            capture_behaviour=capture,
            chemotaxis_mode=chemotaxis_mode,
            capture_behaviour_modality=capture_behaviour_modality,
        ),
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


def test_capture_is_byte_identical_to_capture_off():
    """The captured trajectory must not perturb the run: capture-on and -off paths are identical.

    The whole premise of behavioural validation is that the recorded worm IS the real (uncaptured)
    worm — so with a matched seed the agent path is identical whether capture is on or off.
    """
    off = _agent(capture=False)
    off.run_episode(RewardConfig(), max_steps=20)
    on = _agent(capture=True)
    on.run_episode(RewardConfig(), max_steps=20)
    assert on.path == off.path


def _thermotaxis(base_temperature: float) -> ThermotaxisParams:
    """Build a linear-gradient thermotaxis field (Tc=20, +x gradient); base sets the spawn error."""
    return ThermotaxisParams(
        enabled=True,
        cultivation_temperature=20.0,
        base_temperature=base_temperature,
        gradient_direction=0.0,
        gradient_strength=0.5,
    )


def test_thermotaxis_modality_captures_setpoint_drive():
    """Thermotaxis capture records the homeostatic thermal drive + toward-comfort direction.

    The drive is ``-|T - Tc|`` (<= 0, zero at the cultivation temperature) and the thermal gradient
    is live, so the same bias-curve metrics can grade thermotaxis against the setpoint.
    """
    # Cool base (15 < Tc) -> a real thermal error toward warmer.
    agent = _agent(
        capture=True,
        capture_behaviour_modality="thermotaxis",
        thermotaxis=_thermotaxis(15.0),
    )
    agent.run_episode(RewardConfig(), max_steps=20)
    assert len(agent.behaviour) > 0
    # The drive is the setpoint error, never positive (0 only exactly at Tc).
    assert all(b.concentration <= 1e-9 for b in agent.behaviour)
    # The linear thermal gradient is non-zero across the arena, so it is captured live.
    assert any(b.grad_strength > 0.0 for b in agent.behaviour)


def test_thermotaxis_toward_comfort_flips_above_setpoint():
    """When the worm is warmer than Tc, the toward-comfort direction flips away from warmer.

    Guards the ``else tgrad[1] + pi`` branch: with a base ABOVE Tc the spawn is too hot, so
    toward-comfort points *down* the thermal gradient (the increasing-temperature direction + pi).
    """
    agent = _agent(
        capture=True,
        capture_behaviour_modality="thermotaxis",
        thermotaxis=_thermotaxis(25.0),  # spawn (centre) is HOT -> T > Tc -> flip
    )
    env = agent.env
    agent.run_episode(RewardConfig(), max_steps=20)
    # For every captured step, the toward-comfort direction is 0 (up-gradient) when the worm is
    # colder than Tc and ~pi (down-gradient) when hotter — verify the hot steps take the flip.
    hot = [b for b in agent.behaviour if (env.get_temperature((b.x, b.y)) or 0.0) > 20.0]
    assert hot  # the worm starts hot, so some steps must be above Tc
    # gradient direction is 0 (toward +x/warmer), so the flipped toward-comfort is exactly pi.
    assert all(abs(b.grad_dir - math.pi) < 1e-6 for b in hot)


def test_derivative_sensing_captures_live_gradient():
    """Derivative sensing (which pops the food-gradient keys) still records a live nonzero gradient.

    Guards the popped-keys regression the live snapshot was built for: the capture reads the true
    gradient before the derivative-sensing pipeline pops those keys.
    """
    agent = _agent(capture=True, chemotaxis_mode=SensingMode.DERIVATIVE)
    agent.run_episode(RewardConfig(), max_steps=20)
    assert len(agent.behaviour) > 0
    # The true food gradient is non-zero across the arena, so at least one step must log it — a
    # regression that read grad after the pop would record grad_strength == 0.0 for every step.
    assert any(b.grad_strength > 0.0 for b in agent.behaviour)
