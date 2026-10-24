"""Food-dependent reversal kinematics: on a lawn a reversal is short and slow, off it as published.

Covers the locomotor-primitives spec's scenarios "A reversal on a lawn is short and slow", "A
reversal off the lawn keeps the published values", "A depleted cell is still a lawn", "A bout that
crosses the edge keeps its kinematics", "The off-food setting reproduces earlier records", "The option without
primitives is refused" and "A measured value without its record is refused". "Executed bouts match
their source" on a lawn is in ``test_primitives_positive_control.py``, and "Both runners draw the
same" in ``evaluation/test_batched_parity.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from quantumnematode.env.continuous_2d import Continuous2DEnvironment
from quantumnematode.env.env import DEFAULT_AGENT_ID
from quantumnematode.env.primitives import (
    KINEMATICS_PATH,
    PrimitiveState,
    _start_bout,
    load_kinematics,
    raw_for_hazard,
    raw_for_speed,
)
from quantumnematode.env.theme import Theme
from quantumnematode.utils.config_loader import (
    configure_environment,
    create_env_from_config,
    load_simulation_config,
)

_REPO = Path(__file__).resolve().parents[5]
_FORAGING = _REPO / "configs" / "scenarios" / "foraging"
_STEM = "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_internal_s1_primitives"
_FLAVELL = _FORAGING / f"{_STEM}_assay_uniform_lawn_flavell.yml"
_FREE = _FORAGING / f"{_STEM.removesuffix('_s1_primitives')}_s1.yml"
_K = load_kinematics()


def _lawn_env(seed: int = 5991, **continuous: Any) -> Continuous2DEnvironment:
    """Return Flavell's uniform-lawn environment with the worm at the lawn's centre."""
    config = load_simulation_config(str(_FLAVELL))
    environment = configure_environment(config)
    if continuous:
        environment = environment.model_copy(
            update={"continuous": environment.continuous.model_copy(update=continuous)},  # type: ignore[union-attr]
        )
    env = create_env_from_config(environment, seed=seed, theme=Theme.HEADLESS)
    assert isinstance(env, Continuous2DEnvironment)
    centre = env.lawn_field.centres[0]  # type: ignore[union-attr]
    _place(env, (float(centre[0]), float(centre[1])))
    return env


def _place(env: Continuous2DEnvironment, position: tuple[float, float]) -> None:
    state = env.agents[DEFAULT_AGENT_ID]
    state.pos_continuous = position
    state.position = env._discretise(position)  # type: ignore[attr-defined]


def _reversal_action() -> tuple[float, ...]:
    """Run slowly and start a reversal almost surely: the reversal hazard at its maximum."""
    low = raw_for_hazard(_K.hazard_min_per_s, _K)
    return (raw_for_speed(0.05, _K), 0.0, 1.0, low, low)


def test_the_on_food_entries_are_pinned() -> None:
    """The two measured entries are Scheer's on-lawn reversals, read from the committed record."""
    data = json.loads(KINEMATICS_PATH.read_text())
    assert data["on_food_reversal_speed_mm_per_s"]["value"] == 0.069
    duration = data["on_food_reversal_duration_s"]
    assert (duration["mean"], duration["sd"]) == (2.047, 0.899)
    for key in ("on_food_reversal_speed_mm_per_s", "on_food_reversal_duration_s"):
        assert data[key]["status"] == "measured"
        assert data[key]["condition"] == "on food"
    assert _K.on_food_reversal_speed_mm_per_s == 0.069
    assert (_K.on_food_reversal_duration.mean_s, _K.on_food_reversal_duration.sd_s) == (
        2.047,
        0.899,
    )
    record = json.loads((_REPO / duration["record"]).read_text())
    inside = record["filtered"]["reversal_duration_s"]["inside"]
    assert round(inside["value"], 3) == 2.047
    assert round(inside["sd"], 3) == 0.899
    speed = record["filtered"]["reversal_speed_mm_per_s"]["inside"]
    assert round(speed["value"], 3) == 0.069


@pytest.mark.parametrize("field", ["script", "record"])
def test_a_measured_value_without_its_record_is_refused(tmp_path: Path, field: str) -> None:
    """A measured entry must name the script and the record it came from, and both must exist."""
    data = json.loads(KINEMATICS_PATH.read_text())
    del data["on_food_reversal_duration_s"][field]
    path = tmp_path / "primitives.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=f"on_food_reversal_duration_s.*{field}"):
        load_kinematics(path)


def test_a_reversal_on_a_lawn_is_short_and_slow() -> None:
    """On food a reversal backs at 0.069 mm/s for a gamma(2.047, 0.899) draw; off food, published."""
    for on_food, speed, draw in (
        (True, _K.on_food_reversal_speed_mm_per_s, _K.on_food_reversal_duration),
        (False, _K.reversal_speed_mm_per_s, _K.reversal_duration),
    ):
        bout = _start_bout(
            "reversal",
            PrimitiveState(ventral_side=1),
            _K,
            np.random.default_rng(7),
            on_food=on_food,
        )
        assert bout.speed_mm_per_s == -speed
        assert bout.remaining_s == draw.sample(np.random.default_rng(7))


def test_turns_do_not_depend_on_food() -> None:
    """Omega and delta bouts draw the same on and off food."""
    for kind in ("omega", "delta"):
        on, off = (
            _start_bout(
                kind,
                PrimitiveState(ventral_side=1),
                _K,
                np.random.default_rng(3),
                on_food=f,
            )
            for f in (True, False)
        )
        assert on == off


def _first_reversal_speed(env: Continuous2DEnvironment) -> float:
    """Step until a reversal runs; return its speed (positive)."""
    for _ in range(50):
        env.move_agent_primitives(_reversal_action())
        bout = env.primitive_states[DEFAULT_AGENT_ID].bout
        if bout is not None and bout.kind == "reversal":
            return -bout.speed_mm_per_s
    msg = "no reversal started"
    raise AssertionError(msg)


def test_the_environment_keys_reversals_on_the_lawn() -> None:
    """On the lawn the on-food speed; off it, or in a phase without lawns, the off-food speed."""
    env = _lawn_env()
    assert _first_reversal_speed(env) == _K.on_food_reversal_speed_mm_per_s
    off = _lawn_env()
    centre = off.lawn_field.centres[0]  # type: ignore[union-attr]
    radius = off.lawn_field.radius_mm  # type: ignore[union-attr]
    _place(off, (float(centre[0]) + radius + 1.0, float(centre[1])))
    assert _first_reversal_speed(off) == _K.reversal_speed_mm_per_s
    bare = _lawn_env()
    bare.lawn_field = None  # what a phase without food leaves
    assert _first_reversal_speed(bare) == _K.reversal_speed_mm_per_s


def test_a_depleted_cell_is_still_a_lawn() -> None:
    """Lawn geometry, not density, decides: an eaten-out cell still draws the on-food kinematics."""
    env = _lawn_env()
    env.lawn_field.density[:] = 0.0  # type: ignore[union-attr]
    assert _first_reversal_speed(env) == _K.on_food_reversal_speed_mm_per_s


def test_a_bout_that_crosses_the_edge_keeps_its_kinematics() -> None:
    """A reversal that starts just inside the edge and backs out keeps its speed to its end."""
    env = _lawn_env()
    field = env.lawn_field
    assert field is not None
    centre = field.centres[0]
    # Just inside the edge, facing the centre, so a reversal backs out across the edge.
    _place(env, (float(centre[0]) + field.radius_mm - 0.01, float(centre[1])))
    env.agents[DEFAULT_AGENT_ID].heading_rad = math.pi
    speed = _first_reversal_speed(env)
    assert speed == _K.on_food_reversal_speed_mm_per_s
    state = env.agents[DEFAULT_AGENT_ID]
    crossed = False
    while env.primitive_states[DEFAULT_AGENT_ID].bout is not None:
        before = state.pos_continuous
        assert before is not None
        remaining = env.primitive_states[DEFAULT_AGENT_ID].bout.remaining_s  # type: ignore[union-attr]
        env.move_agent_primitives(_reversal_action())
        after = state.pos_continuous
        assert after is not None
        moved = math.hypot(after[0] - before[0], after[1] - before[1])
        assert moved == pytest.approx(speed * min(1.0, remaining), abs=1e-12)
        crossed = crossed or field.lawn_at(after) is None
    assert crossed


def _run(env: Continuous2DEnvironment, steps: int = 400) -> list[tuple[float, float]]:
    action = (raw_for_speed(0.05, _K), 0.0, raw_for_hazard(0.05, _K), *(2 * (-1.0,)))
    path = []
    for _ in range(steps):
        env.move_agent_primitives(action)
        position = env.agents[DEFAULT_AGENT_ID].pos_continuous
        assert position is not None
        path.append(position)
    return path


def test_the_off_food_setting_reproduces_earlier_records() -> None:
    """``off_food`` moves the worm, and draws, exactly as primitives did before the option."""
    off_food = _lawn_env(reversal_kinematics="off_food")
    before_option = _lawn_env()
    # Before the option, no reversal ever took the on-food kinematics.
    before_option._on_food_for_reversal = lambda _agent_id: False  # type: ignore[method-assign]
    assert _run(off_food) == _run(before_option)
    assert off_food.rng.bit_generator.state == before_option.rng.bit_generator.state
    assert _run(_lawn_env()) != _run(_lawn_env(reversal_kinematics="off_food"))


def test_the_option_without_primitives_is_refused(tmp_path: Path) -> None:
    """Setting ``reversal_kinematics`` with free locomotion names both settings."""
    data = yaml.safe_load(_FREE.read_text())
    data["environment"]["continuous"]["reversal_kinematics"] = "off_food"
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(ValueError, match=r"reversal_kinematics: off_food.*locomotion: free"):
        load_simulation_config(str(path))


def test_unset_means_food_dependent_and_free_locomotion_loads() -> None:
    """Unset, a primitives config is food-dependent; a free config loads with the field unset."""
    config = load_simulation_config(str(_FLAVELL))
    assert config.environment.continuous.reversal_kinematics is None  # type: ignore[union-attr]
    env = _lawn_env()
    assert env.continuous.reversal_kinematics == "food_dependent"
    free = load_simulation_config(str(_FREE))
    assert free.environment.continuous.reversal_kinematics is None  # type: ignore[union-attr]
