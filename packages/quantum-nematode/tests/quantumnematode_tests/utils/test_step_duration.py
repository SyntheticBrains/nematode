"""One configured step duration, and every step-coupled constant derived from it.

Covers the step-duration spec's scenarios: "Unset is byte-identical", "The reference step is the
config's own", "A ratio that is not whole is refused", "A 1-s lawn worm's derived values", "A
campaign run records its step", "Settling dynamics keep their time constant", "Deriving twice
changes nothing", "Physical quantities are preserved", "The positive control can fail", "Every field
is classified", "A thermal config at one second is refused" and "A frozen scenario at its default
still loads"; and the continuous-2d-environment spec's "A configured step is the environment's
step".
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from quantumnematode.agent.adaptive_sensor import AdaptiveChemosensor
from quantumnematode.agent.agent import RewardConfig, SatietyConfig
from quantumnematode.agent.satiety import SatietyManager
from quantumnematode.agent.stam import STAMBuffer
from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrainConfig
from quantumnematode.brain.arch.mlpppo import MLPPPOBrainConfig
from quantumnematode.env.continuous_2d import Continuous2DEnvironment
from quantumnematode.env.env import DEFAULT_AGENT_ID
from quantumnematode.env.worm_time import step_worm_seconds
from quantumnematode.utils.config_loader import (
    Continuous2DConfig,
    EnvironmentConfig,
    ForagingConfig,
    LawnConfig,
    SensingConfig,
    SimulationConfig,
    create_env_from_config,
    load_simulation_config,
)
from quantumnematode.utils.step_duration import (
    MAX_STEP_RATIO,
    derive_step_coupled,
    discount_per_step,
    fraction_per_step,
    step_ratio,
)

_REPO = Path(__file__).resolve().parents[5]
_FORAGING = _REPO / "configs" / "scenarios" / "foraging"
_LAWNS = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_lawns"
_CONNECTOME = _FORAGING / f"{_LAWNS}_modulated.yml"
_STAGE0 = _FORAGING / f"{_LAWNS}_connectome_e005.yml"
_MLP = _FORAGING / "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_internal.yml"
_BODY = _FORAGING / "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_body_internal.yml"
_THERMAL = _REPO / "configs" / "scenarios" / "thermal_foraging"


def _raw(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _load(tmp_path: Path, data: dict[str, Any], name: str = "config.yml") -> SimulationConfig:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return load_simulation_config(str(path))


def _at(data: dict[str, Any], step: float | None) -> dict[str, Any]:
    out = copy.deepcopy(data)
    if step is not None:
        out["step_seconds"] = step
    return out


def _pair(tmp_path: Path, path: Path = _CONNECTOME) -> tuple[SimulationConfig, SimulationConfig]:
    """Return a config at its 5-s reference step and the same config at 1 s."""
    data = _raw(path)
    return _load(tmp_path, data, "five.yml"), _load(tmp_path, _at(data, 1.0), "one.yml")


# ---------------------------------------------------------------------------------------------
# The key, the reference step and the ratio
# ---------------------------------------------------------------------------------------------


def test_unset_derives_nothing(tmp_path: Path) -> None:
    """Unset, the loaded config is the one the YAML constructs, with no record."""
    data = _raw(_CONNECTOME)
    loaded = _load(tmp_path, data)
    assert loaded.step_derivation is None
    assert loaded == SimulationConfig(**data)


def test_the_reference_step_is_the_config_s_own(tmp_path: Path) -> None:
    """At ``max_step_mm: 1.0`` the reference step is 5 s, so a 1-s step has ratio 5."""
    config = SimulationConfig(**_at(_raw(_CONNECTOME), 1.0))
    assert step_ratio(config) == 5
    derived = _load(tmp_path, _at(_raw(_CONNECTOME), 1.0))
    assert derived.step_derivation is not None
    assert derived.step_derivation.reference_seconds == 5.0
    assert derived.step_derivation.ratio == 5


@pytest.mark.parametrize("step", [2.0, 10.0, 5.0 / (MAX_STEP_RATIO + 1)])
def test_a_ratio_that_is_not_whole_is_refused(tmp_path: Path, step: float) -> None:
    """A step that does not divide the reference step a whole number of times is refused."""
    with pytest.raises(ValueError, match="reference step"):
        _load(tmp_path, _at(_raw(_CONNECTOME), step))


def test_the_reference_step_itself_derives_nothing(tmp_path: Path) -> None:
    """``step_seconds`` equal to the reference step converts nothing."""
    data = _raw(_CONNECTOME)
    loaded = _load(tmp_path, _at(data, 5.0))
    assert loaded.step_derivation is None
    assert loaded == SimulationConfig(**_at(data, 5.0))


def test_a_zero_step_length_has_no_reference(tmp_path: Path) -> None:
    """With ``max_step_mm: 0`` there is no full-speed step to take as the reference."""
    data = _at(_raw(_CONNECTOME), 1.0)
    data["environment"]["continuous"]["max_step_mm"] = 0.0
    with pytest.raises(ValueError, match="max_step_mm is 0"):
        _load(tmp_path, data)


def test_the_environment_s_step_is_never_written(tmp_path: Path) -> None:
    """The loader sets ``environment.continuous.step_seconds``; a config writing it is refused."""
    data = _raw(_CONNECTOME)
    data["environment"]["continuous"]["step_seconds"] = 1.0
    with pytest.raises(ValueError, match="set the top-level"):
        _load(tmp_path, data)


# ---------------------------------------------------------------------------------------------
# The derivation table
# ---------------------------------------------------------------------------------------------


def test_the_rules() -> None:
    """Each kind's rule, at k = 1/5."""
    k = 0.2
    assert fraction_per_step(0.02, k) == pytest.approx(1.0 - 0.98**0.2)
    assert (1.0 - fraction_per_step(0.02, k)) ** 5 == pytest.approx(0.98)
    assert discount_per_step(0.99, k) ** 5 == pytest.approx(0.99)


def test_a_one_second_lawn_worm_s_derived_values(tmp_path: Path) -> None:
    """Every converted constant on the connectome lawn worm, against its rule."""
    five, one = _pair(tmp_path)
    r, k = 5, 0.2
    e5, e1 = five.environment, one.environment
    assert e5 is not None
    assert e1 is not None
    c5, c1 = e5.get_continuous_config(), e1.get_continuous_config()
    assert c1.max_step_mm == pytest.approx(0.2)
    assert c1.max_turn_rad == pytest.approx(c5.max_turn_rad * k)
    assert c1.step_seconds == 1.0
    assert c5.step_seconds is None
    l5 = e5.foraging.lawns  # type: ignore[union-attr]
    l1 = e1.foraging.lawns  # type: ignore[union-attr]
    assert l5 is not None
    assert l1 is not None
    assert l1.intake_fraction == pytest.approx(fraction_per_step(l5.intake_fraction, k))
    assert l1.regrowth_per_step == l5.regrowth_per_step * k
    s5, s1 = e5.get_sensing_config(), e1.get_sensing_config()
    assert s1.stam_buffer_size == s5.stam_buffer_size * r
    assert s1.stam_decay_rate == pytest.approx(s5.stam_decay_rate * k)
    assert s1.adaptive_chemosensor_alpha == pytest.approx(
        fraction_per_step(s5.adaptive_chemosensor_alpha, k),
    )
    assert s1.derivative_scale == s5.derivative_scale * r
    assert one.satiety is not None
    assert five.satiety is not None
    assert one.satiety.satiety_decay_rate == pytest.approx(five.satiety.satiety_decay_rate * k)
    assert one.reward is not None
    assert five.reward is not None
    assert one.reward.penalty_step == pytest.approx(five.reward.penalty_step * k)
    assert one.reward.penalty_boundary_collision == pytest.approx(
        five.reward.penalty_boundary_collision * k,
    )
    assert one.reward.stuck_position_threshold == five.reward.stuck_position_threshold * r
    assert one.max_steps == five.max_steps * r  # type: ignore[operator]
    assert one.brain is not None
    b1 = one.brain.config
    assert isinstance(b1, ConnectomePPOBrainConfig)
    assert b1.gamma == pytest.approx(0.99**k)
    assert b1.gae_lambda == pytest.approx(0.95**k)
    assert b1.modulator_step_seconds == pytest.approx(1.0)
    # Event and per-unit terms are not step-coupled.
    assert one.reward.reward_goal == five.reward.reward_goal
    assert one.satiety.satiety_gain_per_food == five.satiety.satiety_gain_per_food
    assert l1.reward_per_intake == l5.reward_per_intake
    assert l1.movement_cost_per_mm == l5.movement_cost_per_mm


def test_settling_dynamics_keep_their_time_constant(tmp_path: Path) -> None:
    """Under settling the brain refuses any membrane constant but 1, and none is converted."""
    _, one = _pair(tmp_path)
    assert one.brain is not None
    assert isinstance(one.brain.config, ConnectomePPOBrainConfig)
    assert one.brain.config.dynamics == "settling"
    assert one.brain.config.membrane_tau_steps == 1.0


def test_leaky_dynamics_convert_the_membrane_time_constant(tmp_path: Path) -> None:
    """Under leaky dynamics the membrane time constant stays the same in worm-seconds."""
    data = _raw(_STAGE0)
    data["brain"]["config"].update({"dynamics": "leaky", "membrane_tau_steps": 2.0})
    one = _load(tmp_path, _at(data, 1.0))
    assert one.brain is not None
    assert isinstance(one.brain.config, ConnectomePPOBrainConfig)
    assert one.brain.config.dynamics == "leaky"
    assert one.brain.config.membrane_tau_steps == 10.0


def test_the_mlp_lawn_worm_converts_its_discounts(tmp_path: Path) -> None:
    """The MLP's step-coupled settings are its discounts."""
    five, one = _pair(tmp_path, _MLP)
    assert one.brain is not None
    assert five.brain is not None
    assert isinstance(one.brain.config, MLPPPOBrainConfig)
    assert one.brain.config.gamma == pytest.approx(five.brain.config.gamma**0.2)  # type: ignore[union-attr]
    assert one.brain.config.rollout_buffer_size == five.brain.config.rollout_buffer_size  # type: ignore[union-attr]


def test_an_unset_satiety_block_is_derived_from_its_defaults(tmp_path: Path) -> None:
    """A default is converted like a written value."""
    data = _raw(_CONNECTOME)
    del data["satiety"]
    one = _load(tmp_path, _at(data, 1.0))
    assert one.satiety is not None
    assert one.satiety.satiety_decay_rate == pytest.approx(SatietyConfig().satiety_decay_rate * 0.2)


def test_deriving_twice_changes_nothing(tmp_path: Path) -> None:
    """A derived config's reference step is its own step, so a second derivation is a no-op."""
    _, one = _pair(tmp_path)
    assert step_ratio(one) == 1
    assert derive_step_coupled(one) is one
    rebuilt = SimulationConfig.model_validate(one.model_dump())
    assert step_ratio(rebuilt) == 1
    assert derive_step_coupled(rebuilt) is rebuilt


def test_a_run_records_its_step(tmp_path: Path) -> None:
    """The record names the step, the reference and the ratio, then each derived constant."""
    _, one = _pair(tmp_path)
    assert one.step_derivation is not None
    lines = one.step_derivation.lines()
    assert lines[0] == "Step: 1 s (reference 5 s, x5)"
    assert any(line.startswith("  max_steps: 720 -> 3600") for line in lines)
    assert any("brain.config.gamma" in line for line in lines)


# ---------------------------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------------------------


def _refused(tmp_path: Path, data: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, _at(data, 1.0))


def test_the_grid_is_refused(tmp_path: Path) -> None:
    """The grid has no step duration."""
    data = _raw(_REPO / "configs" / "scenarios" / "foraging" / "mlpppo_small_oracle.yml")
    _refused(tmp_path, data, "not continuous_2d")


def test_point_food_is_refused(tmp_path: Path) -> None:
    """Only the lawn food model is converted."""
    data = _raw(
        _FORAGING / "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_ppo_w64.yml",
    )
    _refused(tmp_path, data, "food model is not lawns")


@pytest.mark.parametrize("block", ["thermotaxis", "aerotaxis", "predators", "social_feeding"])
def test_a_frozen_modality_is_refused(tmp_path: Path, block: str) -> None:
    """A thermal, oxygen, predator or social block has per-step constants with no rule."""
    data = _raw(_CONNECTOME)
    data["environment"][block] = {"enabled": True}
    _refused(tmp_path, data, f"{block} is enabled")


def test_a_thermal_config_at_one_second_is_refused(tmp_path: Path) -> None:
    """A committed thermotaxis config at one second names the scope that excludes it."""
    path = next(_THERMAL.glob("*continuous2d*.yml"))
    _refused(tmp_path, _raw(path), "step_seconds is set")


def test_the_kinematic_body_is_refused(tmp_path: Path) -> None:
    """The body's reversal and refractory counts have no rule."""
    _refused(tmp_path, _raw(_BODY), "kinematic")


def test_another_brain_is_refused(tmp_path: Path) -> None:
    """Only the connectome and MLP-PPO brains have their settings converted."""
    data = _raw(_MLP)
    data["brain"] = {
        "name": "lstmppo",
        "config": {"action_mode": "continuous", "signed_speed": True},
    }
    _refused(tmp_path, data, "the brain is 'lstmppo'")


def test_plasticity_is_refused(tmp_path: Path) -> None:
    """Within-lifetime plasticity's trace decay and rate are per step."""
    data = _raw(_MLP)
    data["brain"]["config"]["enable_activity_traces"] = True
    _refused(tmp_path, data, "within-lifetime plasticity")


def test_hyperparameter_evolution_is_refused(tmp_path: Path) -> None:
    """Hyperparameter evolution rewrites brain values after the conversion."""
    data = _raw(_MLP)
    data["hyperparam_schema"] = [
        {"name": "learning_rate", "type": "float", "bounds": [1e-4, 1e-3]},
    ]
    _refused(tmp_path, data, "hyperparam_schema")


def test_an_unset_episode_length_is_refused(tmp_path: Path) -> None:
    """An unset ``max_steps`` takes its default later, where nothing converts it."""
    data = _raw(_MLP)
    del data["max_steps"]
    _refused(tmp_path, data, "max_steps is unset")


def test_a_frozen_scenario_at_its_default_still_loads() -> None:
    """Every committed thermal config loads unchanged, with nothing derived."""
    for path in sorted(_THERMAL.glob("*.yml")):
        assert load_simulation_config(str(path)).step_derivation is None


# ---------------------------------------------------------------------------------------------
# Every field is classified
# ---------------------------------------------------------------------------------------------

# Every field of every model the lawn point worm reads that is converted, and the rule.
_CONVERTED: dict[str, set[str]] = {
    "SimulationConfig": {"max_steps"},
    "Continuous2DConfig": {"max_step_mm", "max_turn_rad"},
    "SensingConfig": {
        "stam_buffer_size",
        "stam_decay_rate",
        "adaptive_chemosensor_alpha",
        "derivative_scale",
    },
    "LawnConfig": {"intake_fraction", "regrowth_per_step"},
    "SatietyConfig": {"satiety_decay_rate"},
    "RewardConfig": {"penalty_step", "penalty_boundary_collision", "stuck_position_threshold"},
    "ConnectomePPOBrainConfig": {
        "gamma",
        "gae_lambda",
        "modulator_step_seconds",
        "membrane_tau_steps",
    },
    "MLPPPOBrainConfig": {"gamma", "gae_lambda"},
}

# Fields refused away from zero or off at a non-reference step, so never converted.
_REFUSED: dict[str, set[str]] = {
    "SimulationConfig": {"multi_agent", "evolution", "coevolution", "hyperparam_schema"},
    "EnvironmentConfig": {
        "thermotaxis",
        "aerotaxis",
        "predators",
        "pheromones",
        "social_feeding",
        "bit_memory_task",
        "associative_memory_task",
        "env_type",
    },
    "Continuous2DConfig": {"body_model", "body_substeps", "body_steering_gain"},
    "ForagingConfig": {"food_model"},
    "RewardConfig": {
        "penalty_stuck_position",
        "penalty_anti_dithering",
        "reward_exploration",
        "reward_distance_scale",
    },
    "ConnectomePPOBrainConfig": {"learning_rule", "enable_activity_traces"},
    "MLPPPOBrainConfig": {"learning_rule", "enable_activity_traces"},
}

# Why the remaining fields are not step-coupled, by kind. A new field lands in none of these sets
# and fails the walk until it is classified.
_NOT_STEP_COUPLED_REASONS = {
    "spatial": "a length, area or radius in mm or cells",
    "event": "paid or counted per event (a food, a lawn arrival, a death), not per step",
    "per unit": "per unit eaten or per mm moved",
    "training budget": "a learning hyperparameter chosen by registration, not a physical rate",
    "structure": "architecture, mode, seed, source or bookkeeping",
    "frozen-only": "read only by a scenario or mechanism refused at a non-reference step",
    "carrier": "the configured step itself",
}


def _not_step_coupled() -> dict[str, set[str]]:
    """Every remaining field, by model; each kind of reason is in ``_NOT_STEP_COUPLED_REASONS``."""
    return {
        "SimulationConfig": {
            "seed",
            "brain",
            "shots",
            "body_length",
            "qubits",
            "learning_rate",
            "gradient",
            "parameter_initializer",
            "reward",
            "satiety",
            "modules",
            "manyworlds_mode",
            "environment",
            "step_seconds",
        },
        "EnvironmentConfig": {
            "grid_size",
            "viewport_size",
            "continuous",
            "foraging",
            "health",
            "sensing",
            "assay",
        },
        "Continuous2DConfig": {
            "world_size_mm",
            "body_length_mm",
            "capture_radius_mm",
            "sweep_amplitude_mm",
            "predator_damage_radius_mm",
            "allow_reversal",
            "step_seconds",
            "locomotion",
            "arena_shape",
        },
        "ForagingConfig": set(ForagingConfig.model_fields) - {"food_model"},
        "LawnConfig": set(LawnConfig.model_fields) - _CONVERTED["LawnConfig"],
        "SensingConfig": set(SensingConfig.model_fields) - _CONVERTED["SensingConfig"],
        "SatietyConfig": {"initial_satiety", "satiety_gain_per_food", "starvation_ends_episode"},
        "RewardConfig": set(RewardConfig.model_fields)
        - _CONVERTED["RewardConfig"]
        - _REFUSED["RewardConfig"],
        "ConnectomePPOBrainConfig": set(ConnectomePPOBrainConfig.model_fields)
        - _CONVERTED["ConnectomePPOBrainConfig"]
        - _REFUSED["ConnectomePPOBrainConfig"],
        "MLPPPOBrainConfig": set(MLPPPOBrainConfig.model_fields)
        - _CONVERTED["MLPPPOBrainConfig"]
        - _REFUSED["MLPPPOBrainConfig"],
    }


# The fields of the broad models that are classified by subtraction above, pinned so that a field
# added to one of them fails here until it is classified on purpose.
_PINNED_FIELD_COUNTS = {
    "ForagingConfig": len(ForagingConfig.model_fields),
    "LawnConfig": len(LawnConfig.model_fields),
    "SensingConfig": len(SensingConfig.model_fields),
    "RewardConfig": len(RewardConfig.model_fields),
    "ConnectomePPOBrainConfig": len(ConnectomePPOBrainConfig.model_fields),
    "MLPPPOBrainConfig": len(MLPPPOBrainConfig.model_fields),
}
# LawnConfig.placement and the assay's durations in seconds were added on 2026-10-24, and are
# not step-coupled.
# Audited 2026-10-23. Under lawns: ForagingConfig's point-food and depletion fields are unread, and
# the Fick kernel is static; SensingConfig's modes are structure, ``lateral_scale`` spatial and the
# adaptive epsilon a concentration; RewardConfig's predator, health and temperature terms are
# frozen-only and ``penalty_starvation`` and ``reward_goal`` events; the brains' remaining fields
# are plasticity (refused), PPO's training budget, or structure. Audited 2026-10-24: the data
# neuromodulator layer's seven fields are structure (layer, sensitivity, sources, null, seed,
# knockout genes) or already in worm-seconds (``modulator_timescales_s``; the per-step decay comes
# from ``modulator_step_seconds``, which is converted).
_EXPECTED_FIELD_COUNTS = {
    "ForagingConfig": 22,
    "LawnConfig": 14,
    "SensingConfig": 20,
    "RewardConfig": 15,
    "ConnectomePPOBrainConfig": 93,
    "MLPPPOBrainConfig": 68,
}


def test_every_field_is_classified() -> None:
    """Every field the lawn point worm reads is converted, refused, or recorded as not coupled."""
    models = {
        "SimulationConfig": SimulationConfig,
        "EnvironmentConfig": EnvironmentConfig,
        "Continuous2DConfig": Continuous2DConfig,
        "ForagingConfig": ForagingConfig,
        "LawnConfig": LawnConfig,
        "SensingConfig": SensingConfig,
        "SatietyConfig": SatietyConfig,
        "RewardConfig": RewardConfig,
        "ConnectomePPOBrainConfig": ConnectomePPOBrainConfig,
        "MLPPPOBrainConfig": MLPPPOBrainConfig,
    }
    not_coupled = _not_step_coupled()
    for name, model in models.items():
        classified = (
            _CONVERTED.get(name, set()) | _REFUSED.get(name, set()) | not_coupled.get(name, set())
        )
        assert set(model.model_fields) == classified, (
            f"{name}: unclassified {sorted(set(model.model_fields) - classified)}, "
            f"stale {sorted(classified - set(model.model_fields))}"
        )
    assert _PINNED_FIELD_COUNTS == _EXPECTED_FIELD_COUNTS, (
        "a broad model gained or lost a field: classify it in _CONVERTED, _REFUSED or a "
        "not-step-coupled set, then update _EXPECTED_FIELD_COUNTS"
    )


# ---------------------------------------------------------------------------------------------
# The positive control: physical quantities are preserved, and the test can fail
# ---------------------------------------------------------------------------------------------


def _scripted_worm(config: SimulationConfig, seconds: float, turn: float = 0.0) -> Any:
    """Drive a worm at full speed for ``seconds`` from the arena's centre, heading east."""
    assert config.environment is not None
    env = create_env_from_config(config.environment, seed=5991)
    assert isinstance(env, Continuous2DEnvironment)
    state = env.agents[DEFAULT_AGENT_ID]
    centre = env.continuous.world_size_mm / 2.0
    state.pos_continuous = (centre / 2.0, centre)
    state.heading_rad = 0.0
    steps = round(seconds / env.step_seconds)
    path = 0.0
    for _ in range(steps):
        before = state.pos_continuous
        env.move_agent_normalized(0.0 if turn else 1.0, turn)
        after = state.pos_continuous
        path += math.dist(before, after)
    return env, state, path, steps


def test_a_configured_step_is_the_environment_s_step(tmp_path: Path) -> None:
    """The environment's step is the configured one, and its full-speed step 0.2 mm."""
    five, one = _pair(tmp_path)
    env5 = create_env_from_config(five.environment, seed=1)  # type: ignore[arg-type]
    env1 = create_env_from_config(one.environment, seed=1)  # type: ignore[arg-type]
    assert isinstance(env5, Continuous2DEnvironment)
    assert isinstance(env1, Continuous2DEnvironment)
    assert env5.step_seconds == 5.0
    assert env1.step_seconds == 1.0
    assert env1.body_params.step_seconds == 1.0
    assert step_worm_seconds(env1.continuous.max_step_mm) == pytest.approx(1.0)


def test_a_straight_run_is_preserved(tmp_path: Path) -> None:
    """At full speed for 30 s the worm reaches the same place by the same 6 mm path."""
    five, one = _pair(tmp_path)
    _, s5, path5, n5 = _scripted_worm(five, 30.0)
    _, s1, path1, n1 = _scripted_worm(one, 30.0)
    assert (n5, n1) == (6, 30)
    assert path5 == pytest.approx(6.0)
    assert path1 == pytest.approx(6.0)
    assert s1.pos_continuous == pytest.approx(s5.pos_continuous)


def test_turning_in_place_is_preserved(tmp_path: Path) -> None:
    """At full turn for 30 s the heading ends the same."""
    five, one = _pair(tmp_path, _MLP)
    _, s5, _, _ = _scripted_worm(five, 30.0, turn=0.1)
    _, s1, _, _ = _scripted_worm(one, 30.0, turn=0.1)
    assert math.cos(s1.heading_rad - s5.heading_rad) == pytest.approx(1.0)


def test_the_positive_control_can_fail(tmp_path: Path) -> None:
    """A 1-s worm with its per-step values unconverted covers five times the ground in 10 s."""
    data = _raw(_CONNECTOME)
    five = _load(tmp_path, data, "five.yml")
    assert five.environment is not None
    unconverted = five.environment.model_copy(
        update={
            "continuous": five.environment.get_continuous_config().model_copy(
                update={"step_seconds": 1.0},
            ),
        },
    )
    raw = five.model_copy(update={"environment": unconverted})
    _, _, path, steps = _scripted_worm(raw, 10.0)
    _, _, converted, _ = _scripted_worm(_load(tmp_path, _at(data, 1.0), "one.yml"), 10.0)
    assert steps == 10
    assert converted == pytest.approx(2.0)
    assert path == pytest.approx(10.0)


def test_satiety_is_preserved(tmp_path: Path) -> None:
    """Satiety after 30 s of basal decay is the same at both steps."""
    five, one = _pair(tmp_path)
    m5 = SatietyManager(five.satiety)  # type: ignore[arg-type]
    m1 = SatietyManager(one.satiety)  # type: ignore[arg-type]
    for _ in range(6):
        m5.decay_satiety()
    for _ in range(30):
        m1.decay_satiety()
    assert m1.current_satiety == pytest.approx(m5.current_satiety)


def test_a_grazed_cell_is_preserved(tmp_path: Path) -> None:
    """A cell grazed for 60 s keeps the same density at both steps."""
    five, one = _pair(tmp_path)
    remaining = []
    for config, steps in ((five, 12), (one, 60)):
        env = create_env_from_config(config.environment, seed=5992)  # type: ignore[arg-type]
        field, lawns = env.lawn_field, env.foraging.lawns
        assert field is not None
        assert lawns is not None
        cell = (float(field.centres[0][0]), float(field.centres[0][1]))
        for _ in range(steps):
            field.eat(cell, lawns.intake_fraction)
        remaining.append(field.density.copy())
    np.testing.assert_allclose(remaining[1], remaining[0], rtol=1e-12)
    assert remaining[0].min() < 1.0


def test_stam_weights_are_preserved_by_age_in_seconds(tmp_path: Path) -> None:
    """A sample's STAM weight is the same function of its age in seconds at both steps."""
    five, one = _pair(tmp_path)
    s5 = five.environment.get_sensing_config()  # type: ignore[union-attr]
    s1 = one.environment.get_sensing_config()  # type: ignore[union-attr]
    w5 = STAMBuffer(s5.stam_buffer_size, s5.stam_decay_rate)._weights
    w1 = STAMBuffer(s1.stam_buffer_size, s1.stam_decay_rate)._weights
    assert len(w1) == 5 * len(w5)
    np.testing.assert_allclose(w1[::5], w5, rtol=1e-12)


def test_the_adaptive_baseline_s_step_response_is_preserved(tmp_path: Path) -> None:
    """After a step in concentration the background is the same every 5 s at both steps."""
    five, one = _pair(tmp_path)
    responses = []
    for config, per_reference in ((five, 1), (one, 5)):
        sensing = config.environment.get_sensing_config()  # type: ignore[union-attr]
        sensor = AdaptiveChemosensor(alpha=sensing.adaptive_chemosensor_alpha)
        sensor.adapt(0.0, 0.0)
        trace = []
        for i in range(12 * per_reference):
            sensor.adapt(1.0, 0.0)
            if (i + 1) % per_reference == 0:
                trace.append(sensor.background)
        responses.append(trace)
    np.testing.assert_allclose(responses[1], responses[0], rtol=1e-12)
