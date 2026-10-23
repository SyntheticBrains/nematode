"""The locomotor primitives: kinematics, the hazard interface, bouts, the senses and their scope.

Covers the locomotor-primitives spec's scenarios "A reversal backs without turning", "Omega and
delta turn to opposite sides", "Free locomotion is unchanged", "Constant hazards are a Poisson
process", "No start inside a bout's last step", "Crossing the edge", "A reversal is felt" and "The
5-second worm is refused"; the continuous-action-policy scenarios "The action is five bounded
numbers" and "A brain and an environment that disagree are refused"; and the realworm-behavioural-
validation scenario "A primitive step is labelled".
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from quantumnematode.agent.agent import RewardConfig
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.modules import SENSORY_MODULES, ModuleName
from quantumnematode.env.continuous_2d import Continuous2DEnvironment, Continuous2DParams
from quantumnematode.env.env import DEFAULT_AGENT_ID
from quantumnematode.env.hazard_controllers import ConstantHazards
from quantumnematode.env.primitives import (
    KINEMATICS_PATH,
    PRIMITIVES_ACTION_DIM,
    Bout,
    PrimitiveState,
    load_kinematics,
    map_action,
    raw_for_hazard,
    raw_for_speed,
    step_primitives,
)
from quantumnematode.utils.config_loader import load_simulation_config

_REPO = Path(__file__).resolve().parents[5]
_FORAGING = _REPO / "configs" / "scenarios" / "foraging"
_LAWNS = "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_internal"
_MLP = _FORAGING / f"{_LAWNS}_s1_primitives.yml"
_CONNECTOME = (
    _FORAGING
    / "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_modulated_s1_primitives.yml"
)
_K = load_kinematics()


def _raw(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _load(tmp_path: Path, data: dict[str, Any]) -> Any:
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return load_simulation_config(str(path))


def _env(seed: int = 5991, world: float = 1000.0) -> Continuous2DEnvironment:
    return Continuous2DEnvironment(
        continuous=Continuous2DParams(
            world_size_mm=world,
            max_step_mm=0.2,
            step_seconds=1.0,
            locomotion="primitives",
        ),
        seed=seed,
    )


def _hazard_action(rates: tuple[float, float, float], speed: float = 0.1) -> tuple[float, ...]:
    return (raw_for_speed(speed, _K), 0.0, *(raw_for_hazard(r, _K) for r in rates))


# ---------------------------------------------------------------------------------------------
# Kinematics and the mapping
# ---------------------------------------------------------------------------------------------


def test_the_kinematics_are_their_sources() -> None:
    """Every value is the vendored one, and every entry names its source, condition and status."""
    data = json.loads(KINEMATICS_PATH.read_text())
    for key, entry in data.items():
        if key != "description":
            assert {"source", "condition", "status"} <= set(entry), key
    assert _K.v_max_mm_per_s == 0.2
    assert _K.omega_max_rad_per_s == pytest.approx(math.radians(50.0))
    assert _K.reversal_speed_mm_per_s == 0.14
    assert (_K.reversal_duration.mean_s, _K.reversal_duration.sd_s) == (2.69, 1.75)
    assert math.degrees(_K.omega_angle.mean_rad) == pytest.approx(155.0)
    assert (_K.omega_duration.mean_s, _K.omega_duration.sd_s) == (7.0, 3.0)
    assert math.degrees(_K.delta_angle.mean_rad) == pytest.approx(90.0)
    assert (_K.delta_duration.mean_s, _K.delta_duration.sd_s) == (6.0, 2.0)
    assert (_K.hazard_min_per_s, _K.hazard_max_per_s) == (1e-4, 1.0)


def test_the_mapping_spans_each_range() -> None:
    """Speed, curvature and hazards span their ranges, and the inverses round-trip."""
    low = map_action([-1.0] * 5, _K)
    high = map_action([1.0] * 5, _K)
    assert (low.speed_mm_per_s, high.speed_mm_per_s) == (0.0, pytest.approx(0.2))
    assert high.curvature_rad_per_s == pytest.approx(math.radians(50.0))
    assert low.hazards_per_s == pytest.approx((1e-4, 1e-4, 1e-4))
    assert high.hazards_per_s == pytest.approx((1.0,) * 3)
    middle = map_action([0.0] * 5, _K)
    assert middle.hazards_per_s == pytest.approx((0.01,) * 3)
    inverse = map_action([raw_for_speed(0.07, _K), 0, raw_for_hazard(0.05, _K), 0, 0], _K)
    assert inverse.speed_mm_per_s == pytest.approx(0.07)
    assert inverse.hazards_per_s[0] == pytest.approx(0.05)
    with pytest.raises(ValueError, match="5 entries"):
        map_action([0.0, 0.0], _K)


# ---------------------------------------------------------------------------------------------
# Bouts
# ---------------------------------------------------------------------------------------------


def test_a_reversal_backs_without_turning_and_nothing_starts_in_its_last_step() -> None:
    """A 2.4-s reversal backs 0.336 mm over three steps; the earliest next start is the fourth."""
    state = PrimitiveState(ventral_side=1)
    state.bout = Bout(kind="reversal", remaining_s=2.4, speed_mm_per_s=-0.14)
    rng = np.random.default_rng(0)
    every_hazard = _hazard_action((1.0, 1.0, 1.0))
    steps = [step_primitives(state, every_hazard, 1.0, _K, rng) for _ in range(3)]
    assert [s.kind for s in steps] == ["reversal"] * 3
    assert [s.heading_change_rad for s in steps] == [0.0] * 3
    assert sum(s.displacement_mm for s in steps) == pytest.approx(-0.14 * 2.4)
    assert steps[2].displacement_mm == pytest.approx(-0.14 * 0.4)
    assert state.bout is None
    draws_before = rng.bit_generator.state
    step_primitives(state, every_hazard, 1.0, _K, rng)
    assert rng.bit_generator.state != draws_before  # the fourth step draws a start


def test_omega_and_delta_turn_to_opposite_sides() -> None:
    """Within an episode every omega turns one way and every delta the other."""
    sides = set()
    for seed in range(5991, 5999):
        env = _env(seed)
        signs: dict[str, set[float]] = {"omega": set(), "delta": set()}
        for kind, rates in (("omega", (1e-4, 1.0, 1e-4)), ("delta", (1e-4, 1e-4, 1.0))):
            for _ in range(200):
                if env.move_agent_primitives(_hazard_action(rates)) == kind:
                    bout = env.primitive_states[DEFAULT_AGENT_ID].bout
                    if bout is not None:
                        signs[kind].add(math.copysign(1.0, bout.turn_rad_per_s))
        assert len(signs["omega"]) == 1
        assert len(signs["delta"]) == 1
        assert signs["omega"] != signs["delta"]
        sides |= signs["omega"]
    assert sides == {-1.0, 1.0}  # each side is ventral for some worm


def test_constant_hazards_are_a_poisson_process() -> None:
    """With constant hazards, the run steps before each start are geometric at the per-step rate."""
    rate = 0.05
    state = PrimitiveState(ventral_side=1)
    rng = np.random.default_rng(5991)
    action = _hazard_action((rate, 1e-4, 1e-4))
    waits: list[int] = []
    run = 0
    for _ in range(60_000):
        idle = state.bout is None
        step = step_primitives(state, action, 1.0, _K, rng)
        if idle and step.kind == "run":
            run += 1
        elif idle:
            waits.append(run)
            run = 0
    p = 1.0 - math.exp(-(rate + 2e-4))
    assert len(waits) > 500
    assert np.mean(waits) == pytest.approx((1.0 - p) / p, rel=0.08)


def test_free_locomotion_draws_nothing_and_moves_as_before() -> None:
    """Under free locomotion a move draws nothing from the RNG and keeps no primitive state."""
    env = Continuous2DEnvironment(continuous=Continuous2DParams(world_size_mm=50.0), seed=1)
    state = env.rng.bit_generator.state
    env.move_agent_normalized(0.5, 0.2)
    assert env.rng.bit_generator.state == state
    assert env.primitive_states == {}
    agent = env.agents[DEFAULT_AGENT_ID]
    assert agent.heading_rad == pytest.approx(0.1)
    assert agent.pos_continuous == pytest.approx(
        (25.0 + 0.5 * math.cos(0.1), 25.0 + 0.5 * math.sin(0.1)),
    )


def test_a_copy_keeps_the_running_bout() -> None:
    """Copying an environment carries each worm's bout and ventral side."""
    env = _env()
    while env.move_agent_primitives(_hazard_action((1.0, 1e-4, 1e-4))) != "reversal":
        pass
    clone = env.copy()
    assert clone.primitive_states == env.primitive_states
    assert clone.primitive_states is not env.primitive_states


# ---------------------------------------------------------------------------------------------
# The action space, the senses and the scope
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [_MLP, _CONNECTOME])
def test_a_primitives_brain_acts_with_five_bounded_numbers(path: Path) -> None:
    """Both brains emit five numbers in [-1, 1] through the standard runner, and update."""
    from quantumnematode.agent import QuantumNematodeAgent
    from quantumnematode.utils.config_loader import (
        configure_brain,
        configure_environment,
        configure_reward,
        configure_satiety,
        create_env_from_config,
    )

    sys_path = _REPO / "scripts" / "analysis"
    import sys

    sys.path.insert(0, str(sys_path))
    import body_kinematics_eval as harness  # pyright: ignore[reportMissingImports]

    brain, config, sensing = harness.build_brain(path, 5991)
    del configure_brain
    environment = configure_environment(config)
    agent = QuantumNematodeAgent(
        brain=brain,
        env=create_env_from_config(environment, seed=5991),
        satiety_config=configure_satiety(config),
        sensing_config=sensing.model_copy(update={"capture_behaviour": True}),
    )
    agent.run_episode(configure_reward(config), max_steps=40)
    actions = [a.continuous for a in brain.history_data.actions]
    assert actions
    assert all(a is not None and len(a) == PRIMITIVES_ACTION_DIM for a in actions)
    assert all(-1.0 <= v <= 1.0 for a in actions for v in a)  # type: ignore[union-attr]
    labels = {b.primitive for b in agent.behaviour}
    assert labels <= {"run", "reversal", "omega", "delta"}
    assert None not in labels


def _refused(tmp_path: Path, data: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, data)


def test_free_locomotion_with_a_primitives_brain_is_refused(tmp_path: Path) -> None:
    """The brain and the environment must agree."""
    data = _raw(_MLP)
    data["environment"]["continuous"]["locomotion"] = "free"
    data["brain"]["config"]["sensory_modules"].remove("locomotor_state")
    _refused(tmp_path, data, "locomotion is free but brain.config.action_space is primitives")


def test_primitive_locomotion_with_a_speed_turn_brain_is_refused(tmp_path: Path) -> None:
    """The brain and the environment must agree."""
    data = _raw(_MLP)
    data["brain"]["config"]["action_space"] = "speed_turn"
    _refused(tmp_path, data, "locomotion is primitives but brain.config.action_space is speed_turn")


def test_signed_speed_under_primitives_is_refused(tmp_path: Path) -> None:
    """Backing is the reversal primitive's."""
    data = _raw(_MLP)
    data["brain"]["config"]["signed_speed"] = True
    data["environment"]["continuous"]["allow_reversal"] = True
    _refused(tmp_path, data, "signed_speed is not read under action_space: primitives")


def test_a_state_dependent_std_under_primitives_is_refused(tmp_path: Path) -> None:
    """The state-dependent head is sized for two outputs."""
    data = _raw(_MLP)
    data["brain"]["config"]["continuous_std_mode"] = "state_dependent"
    _refused(tmp_path, data, "state_dependent")


def test_the_five_second_worm_is_refused(tmp_path: Path) -> None:
    """Bouts need second resolution."""
    data = _raw(_MLP)
    del data["step_seconds"]
    _refused(tmp_path, data, "at most 1 s")


def test_a_primitives_brain_on_another_architecture_is_refused(tmp_path: Path) -> None:
    """Only the connectome and MLP-PPO brains implement the primitives action space."""
    data = _raw(_MLP)
    data["brain"] = {
        "name": "lstmppo",
        "config": {"action_mode": "continuous", "action_space": "primitives"},
    }
    _refused(tmp_path, data, "lstmppo")


def test_locomotor_state_without_primitives_is_refused(tmp_path: Path) -> None:
    """The locomotor state reads the primitives."""
    data = _raw(_FORAGING / f"{_LAWNS}_s1.yml")
    data["brain"]["config"]["sensory_modules"].append("locomotor_state")
    _refused(tmp_path, data, "locomotor_state")


def test_food_contact_without_lawns_is_refused(tmp_path: Path) -> None:
    """Food contact reads the lawn under the worm."""
    data = _raw(
        _FORAGING / "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_ppo_w64.yml",
    )
    data["brain"]["config"]["sensory_modules"].append("food_contact")
    _refused(tmp_path, data, "food_contact")


def test_food_contact_falls_at_the_edge() -> None:
    """The feature is the cell's density on a lawn, 0 off it, and blind to quality."""
    module = SENSORY_MODULES[ModuleName.FOOD_CONTACT]
    assert module.to_classical(BrainParams(food_contact=1.0)).tolist() == [1.0]
    assert module.to_classical(BrainParams(food_contact=0.0)).tolist() == [0.0]
    assert module.to_classical(BrainParams()).tolist() == [0.0]


def test_food_contact_reads_the_cell_under_the_worm(tmp_path: Path) -> None:
    """On a fresh lawn cell the agent reports 1; moved off every lawn, 0."""
    from quantumnematode.agent import QuantumNematodeAgent
    from quantumnematode.utils.config_loader import configure_environment, create_env_from_config

    config = load_simulation_config(str(_MLP))
    env = create_env_from_config(configure_environment(config), seed=5992)
    agent = QuantumNematodeAgent(brain=ConstantHazards(0.1, (1e-4, 1e-4, 1e-4)), env=env)
    assert agent._food_contact() == 1.0
    state = env.agents[DEFAULT_AGENT_ID]
    centre = env.lawn_field.centres[0]  # type: ignore[union-attr]
    state.pos_continuous = (float(centre[0]) + 4.0, float(centre[1]))
    if env.lawn_field.lawn_at(state.pos_continuous) is None:  # type: ignore[union-attr]
        assert agent._food_contact() == 0.0
    del tmp_path


def test_a_reversal_is_felt() -> None:
    """The locomotor state reports reversal while the bout runs, and run the step after it ends."""
    env = _env()
    env.primitive_states[DEFAULT_AGENT_ID] = PrimitiveState(
        ventral_side=1,
        bout=Bout(kind="reversal", remaining_s=2.4, speed_mm_per_s=-0.14),
    )
    quiet = _hazard_action((1e-4, 1e-4, 1e-4))
    felt = []
    for _ in range(3):
        felt.append(env.running_primitive())
        env.move_agent_primitives(quiet)
    felt.append(env.running_primitive())
    assert felt == ["reversal", "reversal", "reversal", "run"]
    module = SENSORY_MODULES[ModuleName.LOCOMOTOR_STATE]
    assert module.to_classical(BrainParams(locomotor_state="reversal")).tolist() == [0, 1, 0, 0]
    assert module.to_classical(BrainParams(locomotor_state="run")).tolist() == [1, 0, 0, 0]
    assert module.classical_dim == 4


def test_a_primitive_step_is_labelled_and_backs_up() -> None:
    """Captured reversal steps carry their label and a negative signed speed."""
    from quantumnematode.agent import QuantumNematodeAgent
    from quantumnematode.utils.config_loader import configure_environment, create_env_from_config

    config = load_simulation_config(str(_MLP))
    environment = configure_environment(config)
    sensing = environment.get_sensing_config().model_copy(update={"capture_behaviour": True})
    agent = QuantumNematodeAgent(
        brain=ConstantHazards(0.1, (0.3, 1e-4, 1e-4)),
        env=create_env_from_config(environment, seed=5993),
        sensing_config=sensing,
    )
    agent.run_episode(RewardConfig(**copy.deepcopy(config.reward.model_dump())), max_steps=60)  # type: ignore[union-attr]
    reversals = [b for b in agent.behaviour if b.primitive == "reversal"]
    assert reversals
    assert all(b.speed_signed is not None and b.speed_signed < 0 for b in reversals)


def test_a_multi_agent_run_with_a_primitives_brain_is_refused(tmp_path: Path) -> None:
    """The primitives run one worm, so an agent's primitives brain is refused at load."""
    data = _raw(
        _REPO
        / "configs"
        / "scenarios"
        / "multi_agent_foraging"
        / "lstmppo_large_5agents_mixed_1F_4L_klinotaxis.yml",
    )
    data["multi_agent"]["agents"][0]["brain"] = {
        "name": "mlpppo",
        "config": {
            "action_mode": "continuous",
            "action_space": "primitives",
            "sensory_modules": ["food_chemotaxis"],
        },
    }
    _refused(tmp_path, data, "multi_agent must be disabled")
