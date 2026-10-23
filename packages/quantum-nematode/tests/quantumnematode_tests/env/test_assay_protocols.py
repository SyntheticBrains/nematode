"""Assay protocols: phases, placements, the circular arena, the slow reserve and their scope.

Covers the assay-protocols spec's scenarios "Food is removed and the worm transferred", "An off-lawn
start", "The phases fix the episode's length", "No assay is unchanged", "Starvation would end an
assay", "The worm never leaves the well", "Scheer's lawn", "Twelve hours off food", "Off food the
reserve is linear", "Fresh food keeps the reserve full" and "An assay lawn outlasts the assay"; and
the realworm-behavioural-validation scenario "A food-removal capture is labelled".
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from quantumnematode.agent import QuantumNematodeAgent
from quantumnematode.agent.satiety import SatietyManager
from quantumnematode.env.continuous_2d import Continuous2DEnvironment
from quantumnematode.env.env import DEFAULT_AGENT_ID
from quantumnematode.env.hazard_controllers import ConstantHazards
from quantumnematode.env.primitives import Bout, PrimitiveState
from quantumnematode.env.theme import Theme
from quantumnematode.utils.config_loader import (
    configure_environment,
    configure_reward,
    configure_satiety,
    create_env_from_config,
    load_simulation_config,
)

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "campaigns"))
import generate_assay_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]

_FORAGING = _REPO / "configs" / "scenarios" / "foraging"


def _path(protocol: str, arm: str = "internal") -> Path:
    return _FORAGING / f"{gen.stem(arm, protocol)}.yml"


def _raw(protocol: str) -> dict[str, Any]:
    return yaml.safe_load(_path(protocol).read_text())


def _load(tmp_path: Path, data: dict[str, Any]) -> Any:
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return load_simulation_config(str(path))


def _env(protocol: str, seed: int = 5991) -> Continuous2DEnvironment:
    config = load_simulation_config(str(_path(protocol)))
    env = create_env_from_config(configure_environment(config), seed=seed)
    assert isinstance(env, Continuous2DEnvironment)
    return env


# ---------------------------------------------------------------------------------------------
# Phases and placements
# ---------------------------------------------------------------------------------------------


def test_food_is_removed_and_the_worm_transferred() -> None:
    """At the removal the lawn and its odour are gone, the worm is moved, its bout is kept."""
    env = _env("food_removal_calhoun")
    assert env.assay_phase_name == "fed"
    assert env.lawn_field is not None
    state = env.agents[DEFAULT_AGENT_ID]
    state.pos_continuous = (3.0, 4.0)
    state.heading_rad = 0.25
    bout = Bout(kind="reversal", remaining_s=2.0, speed_mm_per_s=-0.14)
    env.primitive_states[DEFAULT_AGENT_ID] = PrimitiveState(ventral_side=1, bout=bout)
    env.advance_assay(2699)
    assert env.assay_phase_name == "fed"
    env.advance_assay(2700)
    assert env.assay_phase_name == "assay"
    assert env.lawn_field is None
    assert state.pos_continuous == pytest.approx((12.7, 12.7))
    assert state.heading_rad != 0.25
    assert env.get_food_concentration(position=state.pos_continuous) == 0.0
    assert env.primitive_states[DEFAULT_AGENT_ID].bout is bout


@pytest.mark.parametrize("seed", range(5991, 6191))
def test_an_off_lawn_start(seed: int) -> None:
    """Scheer's worm starts on bare agar, clear of the lawn and inside the well."""
    env = _env("scheer_lawn", seed)
    x, y = env.agents[DEFAULT_AGENT_ID].pos_continuous  # type: ignore[misc]
    centre = 5.0
    assert math.hypot(x - centre, y - centre) <= 4.0 + 1e-9
    assert env.lawn_field is not None
    assert env.lawn_field.centres[0].tolist() == [5.0, 5.0]
    assert math.hypot(x - centre, y - centre) >= 1.5 + 0.5


def test_lawn_and_arena_centre_placements() -> None:
    """The fed phase starts on the lawn's centre, and Flavell's worm on the arena's."""
    env = _env("food_removal_lopezcruz")
    assert env.agents[DEFAULT_AGENT_ID].pos_continuous == pytest.approx((50.0, 50.0))
    flavell = _env("uniform_lawn_flavell")
    assert flavell.lawn_field is not None
    assert flavell.lawn_field.centres[0].tolist() == [17.5, 17.5]
    assert flavell.agents[DEFAULT_AGENT_ID].pos_continuous == pytest.approx((17.5, 17.5))


def test_the_worm_never_leaves_the_well() -> None:
    """10,000 random moves stay inside the circle, and pushing outward slides along the wall."""
    env = _env("scheer_lawn")
    rng = np.random.default_rng(5992)
    state = env.agents[DEFAULT_AGENT_ID]
    for _ in range(10_000):
        env._turn_then_move(state, float(rng.uniform(-3, 3)), float(rng.uniform(-math.pi, math.pi)))
        x, y = state.pos_continuous  # type: ignore[misc]
        assert math.hypot(x - 5.0, y - 5.0) <= 5.0 + 1e-9
    state.pos_continuous, state.heading_rad = (5.0, 5.0), 0.0
    env._turn_then_move(state, 20.0, 0.0)
    assert state.pos_continuous == pytest.approx((10.0, 5.0))


def test_the_capture_and_record_carry_the_protocol() -> None:
    """Phase labels change at the configured step, and the record gives the analysis window."""
    env = _env("food_removal_lopezcruz")
    record = env.assay_record()
    assert record is not None
    assert [p["start_step"] for p in record["phases"]] == [0, 2700]  # type: ignore[index]
    assert record["analysis_steps"] == [2700, 5400]
    scheer = _env("scheer_lawn").assay_record()
    assert scheer is not None
    assert scheer["analysis_steps"] == [1200, 3600]


# ---------------------------------------------------------------------------------------------
# The slow reserve and the assay lawn
# ---------------------------------------------------------------------------------------------


def _satiety(protocol: str = "scheer_lawn") -> SatietyManager:
    return SatietyManager(configure_satiety(load_simulation_config(str(_path(protocol)))))


def test_off_food_the_reserve_is_linear() -> None:
    """87.5% after 5,400 s, and empty at 43,200 s within one step."""
    satiety = _satiety()
    full = satiety.max_satiety
    for _ in range(5400):
        satiety.decay_satiety()
    assert satiety.current_satiety / full == pytest.approx(0.875, abs=1e-6)
    steps = 5400
    while satiety.current_satiety > 0.0:
        satiety.decay_satiety()
        steps += 1
    assert abs(steps - 43_200) <= 1


def test_fresh_food_keeps_the_reserve_full() -> None:
    """A fresh cell's bite outweighs a step's decay, so the reserve stays at its maximum."""
    config = load_simulation_config(str(_path("uniform_lawn_flavell")))
    lawns = configure_environment(config).foraging.lawns  # type: ignore[union-attr]
    assert lawns is not None
    satiety = _satiety("uniform_lawn_flavell")
    for _ in range(5400):
        satiety.decay_satiety()
        bite = lawns.intake_fraction * 1.0
        satiety.restore_satiety(satiety.max_satiety * lawns.satiety_per_intake * bite)
    assert satiety.current_satiety == satiety.max_satiety


def test_an_assay_lawn_outlasts_the_assay() -> None:
    """A still worm's cell keeps at least 40% of its density over 90 minutes."""
    env = _env("uniform_lawn_flavell")
    field = env.lawn_field
    lawns = env.foraging.lawns
    assert field is not None
    assert lawns is not None
    cell = (17.5, 17.5)
    for _ in range(5400):
        field.eat(cell, lawns.intake_fraction)
    index = field.cell_at(cell)
    assert index is not None
    assert field.density[index] >= 0.4


def test_an_empty_reserve_does_not_end_an_assay() -> None:
    """Through the runner, a worm whose reserve is gone runs to the end of its episode."""
    config = load_simulation_config(str(_path("scheer_lawn")))
    environment = configure_environment(config)
    agent = QuantumNematodeAgent(
        brain=ConstantHazards(0.1, (1e-4, 1e-4, 1e-4)),
        env=create_env_from_config(environment, seed=5993, theme=Theme.HEADLESS),
        theme=Theme.HEADLESS,
        satiety_config=configure_satiety(config),
        sensing_config=environment.get_sensing_config(),
    )
    agent._satiety_manager._current_satiety = 1e-6
    result = agent.run_episode(configure_reward(config), max_steps=50)
    assert agent._episode_tracker.steps == 50
    assert agent.current_satiety == 0.0
    del result


# ---------------------------------------------------------------------------------------------
# Scope and timing
# ---------------------------------------------------------------------------------------------


def test_the_phases_fix_the_episodes_length(tmp_path: Path) -> None:
    """``max_steps`` times the reference step must equal the phases' length."""
    data = _raw("scheer_lawn")
    data["max_steps"] = 700
    with pytest.raises(ValueError, match="lasts 3600 s but max_steps 700"):
        _load(tmp_path, data)


def test_a_duration_off_the_step_grid_is_refused(tmp_path: Path) -> None:
    """Every duration is a whole number of the reference step."""
    data = _raw("scheer_lawn")
    data["environment"]["assay"]["phases"][0]["duration_s"] = 3601.0
    data["max_steps"] = 720
    with pytest.raises(ValueError, match="whole multiples"):
        _load(tmp_path, data)


def test_starvation_would_end_an_assay(tmp_path: Path) -> None:
    """An assay needs a reserve that never ends the episode."""
    data = _raw("scheer_lawn")
    data["satiety"]["starvation_ends_episode"] = True
    with pytest.raises(ValueError, match="starvation_ends_episode: false"):
        _load(tmp_path, data)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (
            lambda d: d["environment"]["foraging"]["lawns"].update(start_on_lawn=True),
            "start_on_lawn",
        ),
        (lambda d: d.update(manyworlds_mode={}), "manyworlds_mode"),
        (
            lambda d: d["environment"]["assay"]["phases"][0].update(food="none"),
            "needs the phase's lawns",
        ),
        (
            lambda d: d["environment"]["foraging"]["lawns"].update(count=2),
            "places one lawn",
        ),
        (
            lambda d: d["environment"]["foraging"]["lawns"].update(placement="random"),
            "lawns.placement: centre",
        ),
        (
            lambda d: d["environment"]["assay"].update(assay_phase="missing"),
            "names assay_phase",
        ),
        (
            lambda d: d["environment"]["assay"].update(analysis_from_s=3600.0),
            "starts its analysis",
        ),
    ],
)
def test_an_assay_outside_its_scope_is_refused(
    tmp_path: Path,
    change: Any,
    match: str,
) -> None:
    """Every scope rule refuses at load, naming what it refuses."""
    data = _raw("scheer_lawn")
    change(data)
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, data)


def test_no_assay_is_unchanged() -> None:
    """Without an assay a lawn environment has no phase, no record, and places its lawns itself."""
    config = load_simulation_config(
        str(_FORAGING / "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_internal_s1.yml"),
    )
    env = create_env_from_config(configure_environment(config), seed=5991)
    assert isinstance(env, Continuous2DEnvironment)
    assert env.assay_phase_name is None
    assert env.assay_record() is None
    assert env.lawn_field is not None
    state = env.rng.bit_generator.state
    env.advance_assay(100)
    assert env.rng.bit_generator.state == state


# ---------------------------------------------------------------------------------------------
# The generated configs
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("arm", gen.ARMS)
@pytest.mark.parametrize("protocol", list(gen.PROTOCOLS))
def test_the_committed_assay_configs_are_the_generators(arm: str, protocol: str) -> None:
    """Each assay config on disk is what the generator writes, and it loads at one second."""
    path, text = gen.derive(arm, protocol)
    assert path.read_text() == text
    config = load_simulation_config(str(path))
    assert config.step_seconds == 1.0
    assay = config.environment.assay  # type: ignore[union-attr]
    assert assay is not None
    assert config.max_steps == round(assay.total_s())
    assert config.satiety is not None
    assert not config.satiety.starvation_ends_episode
    assert config.satiety.satiety_decay_rate == pytest.approx(300 / 43_200)
    lawns = config.environment.foraging.lawns  # type: ignore[union-attr]
    assert lawns is not None
    assert lawns.intake_fraction == pytest.approx(1e-4, rel=1e-3)
    assert lawns.movement_cost_per_mm == 0.0
