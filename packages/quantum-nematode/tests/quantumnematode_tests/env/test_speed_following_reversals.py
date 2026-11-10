"""An on-food reversal speed that follows the run speed: knots on the worm's recent runs.

Covers the locomotor-primitives spec's scenarios "A reversal after steady runs follows their
speed", "Between the knots the speed is interpolated, beyond them flat", "Before the window fills,
the pooled speed", "Bouts do not enter the window", "A shorter step keeps a 3-s window", "Off the
lawn nothing changes", "Unset, nothing changes", "The option is refused where it cannot act" and
"The coupling entry is measured and checked". "Both runners give each reversal the same speed" is
in ``evaluation/test_batched_parity.py``.
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


_KNOTS_BEFORE = (0.0046, 0.0083, 0.0114, 0.0144, 0.0176, 0.0213, 0.0257, 0.0315, 0.0403, 0.0519)
_KNOTS_BEFORE_TOP = (0.0685, 0.1042)
_KNOTS_SPEED = (0.0484, 0.0501, 0.052, 0.054, 0.0572, 0.059, 0.0629, 0.0668, 0.0745, 0.0798)
_KNOTS_SPEED_TOP = (0.0858, 0.0961)


def test_the_coupling_entry_is_pinned() -> None:
    """The measured entry is the record's held-out knots: deciles 1 to 9 and a split top decile."""
    data = json.loads(KINEMATICS_PATH.read_text())
    entry = data[_ENTRY]
    assert entry["status"] == "measured"
    assert entry["condition"] == "on food"
    assert (
        SpeedCoupling(
            knots_before_mm_per_s=_KNOTS_BEFORE + _KNOTS_BEFORE_TOP,
            knots_speed_mm_per_s=_KNOTS_SPEED + _KNOTS_SPEED_TOP,
            window_s=3.0,
        )
        == _C
    )
    vendored = json.loads((_REPO / entry["record"]).read_text())["primitives_entry"]
    assert tuple(round(v, 4) for v in vendored["knots_before_mm_per_s"]) == _C.knots_before_mm_per_s
    assert tuple(round(v, 4) for v in vendored["knots_speed_mm_per_s"]) == _C.knots_speed_mm_per_s
    assert vendored["window_s"] == _C.window_s


def _with_knots(before: list[float], speed: list[float]) -> dict[str, Any]:
    data = json.loads(KINEMATICS_PATH.read_text())
    data[_ENTRY]["knots_before_mm_per_s"] = before
    data[_ENTRY]["knots_speed_mm_per_s"] = speed
    return data


@pytest.mark.parametrize(
    "data",
    [
        _with_knots([0.01, 0.02], [0.05]),  # unequal lengths
        _with_knots([0.01], [0.05]),  # one knot
        _with_knots([0.02, 0.01], [0.05, 0.06]),  # speeds before not increasing
        _with_knots([0.01, 0.01], [0.05, 0.06]),  # a repeated speed before
        _with_knots([-0.01, 0.02], [0.05, 0.06]),  # a negative speed before
        _with_knots([0.01, 0.02], [0.0, 0.06]),  # a reversal speed that is not positive
        _with_knots([0.01, float("inf")], [0.05, 0.06]),  # not finite
        {**json.loads(KINEMATICS_PATH.read_text())}
        | {_ENTRY: {**json.loads(KINEMATICS_PATH.read_text())[_ENTRY], "window_s": 0.0}},
    ],
    ids=["lengths", "one", "decreasing", "repeated", "negative", "speed", "infinite", "window"],
)
def test_a_bad_coupling_is_refused(tmp_path: Path, data: dict[str, Any]) -> None:
    """Knots of unequal length, too few, out of order, non-positive or infinite, or no window."""
    path = tmp_path / "primitives.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="speed coupling"):
        load_kinematics(path)


def test_a_missing_coupling_is_refused(tmp_path: Path) -> None:
    """Kinematics without the coupling entry are refused by name, not with a bare key error."""
    data = json.loads(KINEMATICS_PATH.read_text())
    del data[_ENTRY]
    path = tmp_path / "primitives.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=f"lack '{_ENTRY}'"):
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


def test_the_speed_is_interpolated_between_the_knots() -> None:
    """At a knot its speed; between two, on the line joining them; beyond the outermost, flat."""
    xs, ys = _C.knots_before_mm_per_s, _C.knots_speed_mm_per_s
    for x, y in zip(xs, ys, strict=True):
        assert _C.speed(x) == y
    middle = (xs[4] + xs[5]) / 2.0
    assert _C.speed(middle) == pytest.approx((ys[4] + ys[5]) / 2.0, abs=1e-15)
    assert _C.speed(0.0) == ys[0] == 0.0484
    assert _C.speed(0.2) == ys[-1] == 0.0961
    speeds = [_C.speed(v / 1000.0) for v in range(150)]
    assert speeds == sorted(speeds)  # never slower for a faster worm, with these knots


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
    """Unset and ``pooled`` run alike; on, the worm moves differently but draws exactly the same.

    A test cannot run the code from before the option, so byte-identity against it is checked
    outside the suite: ``scripts/analysis/speedup_identity_check.py`` against a baseline taken
    before the change, on the batched path.
    """
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


def test_the_option_without_lawns_cannot_load(tmp_path: Path) -> None:
    """Without lawns no reversal can start on one; a primitives config without lawns cannot load.

    Primitives need a step of at most one second, and a configured step is accepted only on lawns,
    so the option is never inert: the configuration is refused before it could act.
    """
    data = yaml.safe_load(_FLAVELL.read_text())
    data["environment"]["foraging"]["food_model"] = "points"
    data["environment"]["foraging"].pop("lawns")
    data["environment"]["continuous"]["on_food_reversal_speed"] = "follows_run_speed"
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(ValueError, match="food_model: lawns"):
        load_simulation_config(str(path))


def test_unset_means_pooled() -> None:
    """Unset, a primitives config's on-food reversals take the pooled speed."""
    config = load_simulation_config(str(_FLAVELL))
    assert config.environment.continuous.on_food_reversal_speed is None  # type: ignore[union-attr]
    assert _lawn_env().continuous.on_food_reversal_speed == "pooled"
