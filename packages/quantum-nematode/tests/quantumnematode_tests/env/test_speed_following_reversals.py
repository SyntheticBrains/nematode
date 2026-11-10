"""An on-food reversal speed that follows the run speed: a line on the worm's recent runs, clipped.

Covers the locomotor-primitives spec's scenarios "A reversal after steady runs follows their speed",
"The speed is clipped to the measured range", "Before the window fills, the pooled speed", "Bouts do
not enter the window", "A shorter step keeps a 3-s window", "Off the lawn nothing changes", "Unset,
nothing changes", "The option is refused where it cannot act" and "The coupling entry is measured
and checked". "Both runners give each reversal the same speed" is in
``evaluation/test_batched_parity.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from quantumnematode.env.continuous_2d import Continuous2DEnvironment
from quantumnematode.env.env import DEFAULT_AGENT_ID
from quantumnematode.env.primitives import (
    KINEMATICS_PATH,
    Bout,
    PrimitiveState,
    SpeedCoupling,
    _start_bout,
    load_kinematics,
    map_action,
    mean_run_speed,
    push_run_speed,
    raw_for_hazard,
    raw_for_speed,
    recent_run_speed,
    step_primitives,
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
_C = _K.on_food_reversal_speed_coupling
_LOW = raw_for_hazard(_K.hazard_min_per_s, _K)
_ENTRY = "on_food_reversal_speed_coupling"


def _run_action(speed: float) -> tuple[float, ...]:
    """Run at ``speed``, every start hazard at its minimum."""
    return (raw_for_speed(speed, _K), 0.0, _LOW, _LOW, _LOW)


def _reversal_action(speed: float) -> tuple[float, ...]:
    """Run at ``speed`` until a reversal starts, the reversal hazard at its maximum."""
    return (raw_for_speed(speed, _K), 0.0, 1.0, _LOW, _LOW)


def _commanded(speed: float) -> float:
    """Return the run speed the primitives move the worm at for a requested ``speed``."""
    return map_action(_run_action(speed), _K).speed_mm_per_s


def _lawn_env(seed: int = 9601, **continuous: Any) -> Continuous2DEnvironment:
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


# ----- the entry -----


def test_the_coupling_entry_is_pinned() -> None:
    """The measured entry is the record's held-out line, clipped to its deciles' range."""
    data = json.loads(KINEMATICS_PATH.read_text())
    entry = data[_ENTRY]
    assert entry["status"] == "measured"
    assert entry["condition"] == "on food"
    assert (
        SpeedCoupling(
            intercept_mm_per_s=0.0464,
            slope=0.606,
            low_mm_per_s=0.0484,
            high_mm_per_s=0.0839,
            window_s=3.0,
        )
        == _C
    )
    vendored = json.loads((_REPO / entry["record"]).read_text())["primitives_entry"]
    assert round(vendored["intercept_mm_per_s"], 4) == _C.intercept_mm_per_s
    assert round(vendored["slope"], 3) == _C.slope
    assert round(vendored["low_mm_per_s"], 4) == _C.low_mm_per_s
    assert round(vendored["high_mm_per_s"], 4) == _C.high_mm_per_s
    assert vendored["window_s"] == _C.window_s


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("intercept_mm_per_s", 0.0),
        ("slope", -0.1),
        ("low_mm_per_s", 0.09),
        ("window_s", 0.0),
    ],
)
def test_a_bad_coupling_is_refused(tmp_path: Path, field: str, value: float) -> None:
    """A non-positive intercept, a negative slope, an inverted range or no window is refused."""
    data = json.loads(KINEMATICS_PATH.read_text())
    data[_ENTRY][field] = value
    path = tmp_path / "primitives.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="speed coupling"):
        load_kinematics(path)


def test_a_coupling_without_its_record_is_refused(tmp_path: Path) -> None:
    """The coupling is measured, so it must name an existing record."""
    data = json.loads(KINEMATICS_PATH.read_text())
    del data[_ENTRY]["record"]
    path = tmp_path / "primitives.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=f"{_ENTRY}.*record"):
        load_kinematics(path)


# ----- the reading -----


def test_the_speed_is_the_line_clipped_to_the_measured_range() -> None:
    """Inside the range the line; a still worm the low; a fast one the high."""
    assert _C.speed(0.02) == _C.intercept_mm_per_s + _C.slope * 0.02
    assert _C.intercept_mm_per_s < _C.low_mm_per_s
    assert _C.speed(0.0) == _C.low_mm_per_s
    assert _C.speed(0.2) == _C.high_mm_per_s


def test_a_reversal_after_steady_runs_follows_their_speed() -> None:
    """Three run steps at 0.02 mm/s fill the window; a reversal on food then backs at the line."""
    state = PrimitiveState(ventral_side=1)
    rng = np.random.default_rng(9602)
    for _ in range(3):
        step = step_primitives(
            state,
            _run_action(0.02),
            1.0,
            _K,
            rng,
            on_food=True,
            follows_run_speed=True,
        )
        assert step.kind == "run"
    speed = _commanded(0.02)
    assert state.run_speeds == [speed] * 3
    mean = recent_run_speed(state.run_speeds, 3)
    assert mean is not None
    assert mean == mean_run_speed([speed] * 3)
    bout = _start_bout(
        "reversal",
        state,
        _K,
        np.random.default_rng(7),
        on_food=True,
        run_speed_mean=mean,
    )
    assert bout.speed_mm_per_s == -_C.speed(mean)
    assert bout.remaining_s == _K.on_food_reversal_duration.sample(np.random.default_rng(7))


def test_before_the_window_fills_the_pooled_speed() -> None:
    """With fewer than three run steps seen, a reversal on food backs at the pooled 0.069 mm/s."""
    assert recent_run_speed([0.02, 0.02], 3) is None
    bout = _start_bout(
        "reversal",
        PrimitiveState(ventral_side=1),
        _K,
        np.random.default_rng(7),
        on_food=True,
        run_speed_mean=None,
    )
    assert bout.speed_mm_per_s == -_K.on_food_reversal_speed_mm_per_s == -0.069


def test_bouts_do_not_enter_the_window() -> None:
    """Steps of a running omega leave the window as it was; the next run step enters it."""
    first = _commanded(0.02)
    state = PrimitiveState(
        ventral_side=1,
        bout=Bout(kind="omega", remaining_s=2.0, turn_rad_per_s=1.0),
        run_speeds=[first] * 3,
    )
    rng = np.random.default_rng(9603)
    for _ in range(2):
        step = step_primitives(state, _run_action(0.15), 1.0, _K, rng, follows_run_speed=True)
        assert step.kind == "omega"
    assert state.run_speeds == [first] * 3
    step_primitives(state, _run_action(0.15), 1.0, _K, rng, follows_run_speed=True)
    assert state.run_speeds == [first, first, _commanded(0.15)]


def test_a_shorter_step_keeps_a_3_s_window() -> None:
    """At a 0.5-s step the window holds the last six run steps."""
    assert _C.window_steps(0.5) == 6
    assert _C.window_steps(1.0) == 3
    state = PrimitiveState(ventral_side=1)
    rng = np.random.default_rng(9604)
    for _ in range(8):
        step_primitives(
            state,
            _run_action(0.02),
            0.5,
            _K,
            rng,
            on_food=True,
            follows_run_speed=True,
        )
    assert len(state.run_speeds) == 6
    window: list[float] = []
    for value in range(10):
        push_run_speed(window, float(value), 6)
    assert window == [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


def test_the_window_is_kept_only_when_the_speed_follows() -> None:
    """Without the option a run step leaves the window empty."""
    state = PrimitiveState(ventral_side=1)
    rng = np.random.default_rng(9605)
    for _ in range(3):
        step_primitives(state, _run_action(0.02), 1.0, _K, rng, on_food=True)
    assert state.run_speeds == []


def test_off_the_lawn_nothing_changes() -> None:
    """Off food a reversal keeps the published 0.14 mm/s and its gamma, whatever the runs were."""
    bout = _start_bout(
        "reversal",
        PrimitiveState(ventral_side=1),
        _K,
        np.random.default_rng(7),
        on_food=False,
        run_speed_mean=0.02,
    )
    assert bout.speed_mm_per_s == -_K.reversal_speed_mm_per_s == -0.14
    assert bout.remaining_s == _K.reversal_duration.sample(np.random.default_rng(7))


# ----- the environment -----


def _reversal_after_runs(env: Continuous2DEnvironment, speed: float) -> float:
    """Run three steps at ``speed``, then step until a reversal starts; return its speed."""
    for _ in range(3):
        assert env.move_agent_primitives(_run_action(speed)) == "run"
    for _ in range(50):
        env.move_agent_primitives(_reversal_action(speed))
        bout = env.primitive_states[DEFAULT_AGENT_ID].bout
        if bout is not None and bout.kind == "reversal":
            return -bout.speed_mm_per_s
    msg = "no reversal started"
    raise AssertionError(msg)


def test_the_environment_follows_the_run_speed_on_the_lawn() -> None:
    """On the lawn, the line on the runs; on a depleted cell too; off it and without food, 0.14."""
    expected = _C.speed(mean_run_speed([_commanded(0.02)] * 3))
    assert (
        _reversal_after_runs(_lawn_env(on_food_reversal_speed="follows_run_speed"), 0.02)
        == expected
    )
    depleted = _lawn_env(on_food_reversal_speed="follows_run_speed")
    depleted.lawn_field.density[:] = 0.0  # type: ignore[union-attr]
    assert _reversal_after_runs(depleted, 0.02) == expected
    off = _lawn_env(on_food_reversal_speed="follows_run_speed")
    field = off.lawn_field
    assert field is not None
    centre = field.centres[0]
    _place(off, (float(centre[0]) + field.radius_mm + 0.5, float(centre[1])))
    assert _reversal_after_runs(off, 0.02) == _K.reversal_speed_mm_per_s
    bare = _lawn_env(on_food_reversal_speed="follows_run_speed")
    bare.lawn_field = None  # what a phase without food leaves
    assert _reversal_after_runs(bare, 0.02) == _K.reversal_speed_mm_per_s
    assert _reversal_after_runs(_lawn_env(), 0.02) == _K.on_food_reversal_speed_mm_per_s


def _run(
    env: Continuous2DEnvironment,
    steps: int = 300,
) -> tuple[list[tuple[float, float]], list[str]]:
    """Run slowly with a modest reversal hazard; return the path and each step's primitive."""
    action = (raw_for_speed(0.05, _K), 0.0, raw_for_hazard(0.05, _K), _LOW, _LOW)
    field = env.lawn_field
    assert field is not None
    path, kinds = [], []
    for _ in range(steps):
        kinds.append(env.move_agent_primitives(action))
        position = env.agents[DEFAULT_AGENT_ID].pos_continuous
        assert position is not None
        assert field.lawn_at(position) is not None
        path.append(position)
    return path, kinds


def test_unset_and_pooled_change_nothing_and_the_option_changes_no_draw() -> None:
    """Unset and ``pooled`` run alike; on, the worm moves differently but draws exactly the same."""
    unset, pooled = _lawn_env(), _lawn_env(on_food_reversal_speed="pooled")
    assert _run(unset) == _run(pooled)
    assert unset.rng.bit_generator.state == pooled.rng.bit_generator.state
    following = _lawn_env(on_food_reversal_speed="follows_run_speed")
    path, kinds = _run(following)
    plain_path, plain_kinds = _run(_lawn_env())
    assert kinds == plain_kinds
    assert path != plain_path
    assert following.rng.bit_generator.state == unset.rng.bit_generator.state


# ----- the config -----


def test_the_option_without_primitives_is_refused(tmp_path: Path) -> None:
    """Setting ``on_food_reversal_speed`` with free locomotion names both settings."""
    data = yaml.safe_load(_FREE.read_text())
    data["environment"]["continuous"]["on_food_reversal_speed"] = "pooled"
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(ValueError, match=r"on_food_reversal_speed: pooled.*locomotion: free"):
        load_simulation_config(str(path))


def test_following_with_off_food_reversals_is_refused(tmp_path: Path) -> None:
    """``follows_run_speed`` needs reversals that take the on-food values."""
    data = yaml.safe_load(_FLAVELL.read_text())
    data["environment"]["continuous"]["reversal_kinematics"] = "off_food"
    data["environment"]["continuous"]["on_food_reversal_speed"] = "follows_run_speed"
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(ValueError, match=r"follows_run_speed.*reversal_kinematics: off_food"):
        load_simulation_config(str(path))


def test_unset_means_pooled() -> None:
    """Unset, a primitives config's on-food reversals take the pooled speed."""
    config = load_simulation_config(str(_FLAVELL))
    assert config.environment.continuous.on_food_reversal_speed is None  # type: ignore[union-attr]
    assert _lawn_env().continuous.on_food_reversal_speed == "pooled"
