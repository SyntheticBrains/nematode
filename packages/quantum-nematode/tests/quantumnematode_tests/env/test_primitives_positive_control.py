"""The primitive layer's positive control: executed kinematics, and klinokinesis is expressible.

Covers the locomotor-primitives spec's scenarios "Executed bouts match their source" and
"Klinokinesis is expressible and timing alone is not". No learner is involved. Bouts are measured
from the trajectory alone, never from their labels; and three non-learning controllers on one lawn
show that hazards keyed to dC/dt find food where hazards at the same mean rate do not, while the
inverted rule finds it less often still.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from quantumnematode.agent import QuantumNematodeAgent
from quantumnematode.env.continuous_2d import Continuous2DEnvironment, Continuous2DParams
from quantumnematode.env.env import DEFAULT_AGENT_ID
from quantumnematode.env.hazard_controllers import ConstantHazards, PirouetteHazards
from quantumnematode.env.primitives import load_kinematics, raw_for_hazard, raw_for_speed
from quantumnematode.env.theme import Theme
from quantumnematode.utils.config_loader import (
    configure_environment,
    configure_reward,
    configure_satiety,
    create_env_from_config,
    load_simulation_config,
)
from scipy import stats

_REPO = Path(__file__).resolve().parents[5]
_CONFIG = (
    _REPO
    / "configs"
    / "scenarios"
    / "foraging"
    / "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_internal_s1_primitives.yml"
)
_K = load_kinematics()
_BOUTS = 2000
_RUN_SPEED = 0.1
_FIXTURE_SEEDS = range(5994, 6000)


def _forced_bouts(kind: str, seed: int) -> list[dict[str, float]]:
    """Force ``_BOUTS`` bouts of ``kind``, a run between each, and measure each from the path.

    The driver raises ``kind``'s hazard to the maximum only after a run step, so consecutive bouts
    are separated by a run. Each bout is then read from positions and headings alone: a reversal
    is a stretch of backward steps, a turn a stretch of steps that turn without moving. The other
    kinds keep the minimum hazard, so now and then one of them starts instead; the trajectory says
    which, and only stretches of the forced kind's shape are kept.
    """
    env = Continuous2DEnvironment(
        continuous=Continuous2DParams(
            world_size_mm=1e6,
            max_step_mm=0.2,
            step_seconds=1.0,
            locomotion="primitives",
        ),
        seed=seed,
    )
    index = {"reversal": 2, "omega": 3, "delta": 4}[kind]
    low = raw_for_hazard(_K.hazard_min_per_s, _K)
    state = env.agents[DEFAULT_AGENT_ID]
    state.pos_continuous = (5e5, 5e5)
    bouts: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    previous_run = False
    while len(bouts) < _BOUTS:
        action = [raw_for_speed(_RUN_SPEED, _K), 0.0, low, low, low]
        if previous_run:
            action[index] = 1.0
        before, heading = state.pos_continuous, state.heading_rad
        assert before is not None
        env.move_agent_primitives(tuple(action))
        after = state.pos_continuous
        assert after is not None
        along = (after[0] - before[0]) * math.cos(heading) + (after[1] - before[1]) * math.sin(
            heading,
        )
        turn = math.remainder(state.heading_rad - heading, math.tau)
        moving_back = along < -1e-12
        turning_still = abs(along) < 1e-12 and abs(turn) > 1e-12
        in_bout = moving_back or turning_still
        if in_bout:
            if current is None:
                current = {"steps": 0, "distance": 0.0, "angle": 0.0, "back": 0, "still": 0}
            current["steps"] += 1
            current["distance"] -= along
            current["angle"] += turn
            current["back" if moving_back else "still"] += 1
        elif current is not None:
            shape = "back" if kind == "reversal" else "still"
            if current[shape] == current["steps"]:
                bouts.append(current)
            current = None
        previous_run = not in_bout
    return bouts


def _ks_p(a: np.ndarray, b: np.ndarray) -> float:
    """Return the two-sample Kolmogorov-Smirnov p-value of ``a`` against ``b``."""
    result: Any = stats.ks_2samp(a, b)
    return float(result.pvalue)


def _rounded_durations(draw: object, rng: np.random.Generator, n: int) -> np.ndarray:
    return np.ceil([draw.sample(rng) - 1e-9 for _ in range(n)])  # type: ignore[attr-defined]


@pytest.mark.slow
def test_reversals_match_their_source() -> None:
    """Whole-step durations match rounded draws; distances match ``v_rev * D`` directly."""
    bouts = _forced_bouts("reversal", 5994)
    rng = np.random.default_rng(1)
    durations = [_K.reversal_duration.sample(rng) for _ in range(_BOUTS)]
    steps = np.array([b["steps"] for b in bouts])
    distances = np.array([b["distance"] for b in bouts])
    assert _ks_p(steps, np.ceil(np.array(durations) - 1e-9)) > 0.01
    expected = np.array(durations) * _K.reversal_speed_mm_per_s
    assert _ks_p(distances, expected) > 0.01
    assert all(abs(b["angle"]) < 1e-12 for b in bouts)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["omega", "delta"])
def test_turns_match_their_source(kind: str) -> None:
    """Whole-step durations match rounded draws; angles match the source directly, one sign each."""
    bouts = _forced_bouts(kind, 5995)
    angle_draw = _K.omega_angle if kind == "omega" else _K.delta_angle
    duration_draw = _K.omega_duration if kind == "omega" else _K.delta_duration
    rng = np.random.default_rng(2)
    angles = np.array([b["angle"] for b in bouts])
    assert len({math.copysign(1.0, a) for a in angles}) == 1
    expected = [angle_draw.sample(rng) for _ in range(_BOUTS)]
    assert _ks_p(np.abs(angles), np.array(expected)) > 0.01
    steps = np.array([b["steps"] for b in bouts])
    assert _ks_p(steps, _rounded_durations(duration_draw, rng, _BOUTS)) > 0.01


def _first_lawn_step(controller: object, seed: int, steps: int) -> int:
    """Run one episode from 6 mm off a central lawn; return the first step on it, or ``steps``."""
    config = load_simulation_config(str(_CONFIG))
    environment = configure_environment(config)
    lawns = environment.foraging.lawns.model_copy(update={"count": 1, "start_on_lawn": True})  # type: ignore[union-attr]
    environment = environment.model_copy(
        update={"foraging": environment.foraging.model_copy(update={"lawns": lawns})},  # type: ignore[union-attr]
    )
    env = create_env_from_config(environment, seed=seed, theme=Theme.HEADLESS)
    rng = np.random.default_rng([seed, 1])
    bearing = rng.uniform(-math.pi, math.pi)
    centre = env.lawn_field.centres[0]  # type: ignore[union-attr]
    state = env.agents[DEFAULT_AGENT_ID]
    state.pos_continuous = (
        float(centre[0]) + 8.5 * math.cos(bearing),
        float(centre[1]) + 8.5 * math.sin(bearing),
    )
    state.position = env._discretise(state.pos_continuous)  # type: ignore[attr-defined]
    state.heading_rad = float(rng.uniform(-math.pi, math.pi))
    sensing = environment.get_sensing_config().model_copy(update={"capture_behaviour": True})
    agent = QuantumNematodeAgent(
        brain=controller,  # type: ignore[arg-type]
        env=env,
        theme=Theme.HEADLESS,
        satiety_config=configure_satiety(config),
        sensing_config=sensing,
    )
    agent.run_episode(configure_reward(config), max_steps=steps)
    return next((b.step for b in agent.behaviour if b.on_lawn), steps)


def _reach(controller: object, steps: int = 1200) -> np.ndarray:
    seeds = [seed * 100 + episode for seed in _FIXTURE_SEEDS for episode in range(32)]
    return np.array([_first_lawn_step(controller, seed, steps) < steps for seed in seeds])


def _bootstrap_lower(a: np.ndarray, b: np.ndarray) -> float:
    """Return the 2.5th percentile of mean(a) - mean(b) over 2,000 resamples."""
    rng = np.random.default_rng(5994)
    diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(2000)]
    return float(np.percentile(diffs, 2.5))


@pytest.mark.slow
def test_klinokinesis_is_expressible_and_timing_alone_is_not() -> None:
    """Pirouettes find the lawn more often than matched constant hazards; inverted, less often."""
    base = (0.02, 0.01, 0.01)
    pirouette = PirouetteHazards(0.15, base, beta=3.0)
    reached = _reach(pirouette)
    matched = _reach(ConstantHazards(0.15, pirouette.mean_run_hazards()))
    assert _bootstrap_lower(reached, matched) > 0.0

    inverted = PirouetteHazards(0.15, base, beta=-3.0)
    reached_inverted = _reach(inverted)
    matched_inverted = _reach(ConstantHazards(0.15, inverted.mean_run_hazards()))
    assert _bootstrap_lower(matched_inverted, reached_inverted) > 0.0
