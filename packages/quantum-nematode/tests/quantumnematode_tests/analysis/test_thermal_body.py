"""M.9's thermal body cell: its configs, its stage-1 rule, its reading, and sensing at the head.

Covers the continuous-2d-environment requirement "Thermotaxis through the kinematic body"
(temperature is read at the head) and the change's registration: the configs, the target rule and
the one registered reading.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any

import pytest
from quantumnematode.utils.config_loader import load_simulation_config

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))
sys.path.insert(0, str(_root / "scripts" / "campaigns"))

import gate_preflight as gp  # noqa: E402  # pyright: ignore[reportMissingImports]
import generate_thermal_body_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]
import thermal_body as tb  # noqa: E402  # pyright: ignore[reportMissingImports]

_ARMS = [
    (w, learner, t) for t in gen.TARGETS for w in (*gen.WIRINGS, "mlp") for learner in gen.LEARNERS
]


@functools.cache
def _loaded(stem: str) -> Any:
    return load_simulation_config(str(gen.THERMAL / f"{stem}.yml"))


class TestConfigs:
    @pytest.mark.parametrize(("wiring", "learner", "target"), _ARMS)
    def test_each_body_arm_is_the_generator_s_and_runs_the_body(
        self,
        wiring: str,
        learner: str,
        target: int,
    ) -> None:
        path, text = gen.derive(wiring, learner, target)
        assert path.read_text() == text
        config = _loaded(gen.stem(wiring, learner, target))
        assert config.environment.continuous.body_model == "kinematic"
        assert config.environment.continuous.allow_reversal is True
        assert config.environment.foraging.target_foods_to_collect == target
        assert config.max_steps == 500
        brain = config.brain.config
        assert brain.action_space == "body_drive"
        assert getattr(brain, "freeze_updates", False) is (learner == "frozen")
        if wiring == "mlp":
            assert brain.entropy_coef == gen.MLP_ENTROPY
        else:
            assert brain.entropy_coef == gen.CONNECTOME_ENTROPY
            assert brain.wiring == gen.WIRINGS[wiring]
            assert brain.connectome_source == "cook_2019_hermaphrodite"

    @pytest.mark.parametrize("learner", gen.LEARNERS)
    def test_the_anchor_check_is_the_point_worm_gap_only_null_at_20(self, learner: str) -> None:
        path, text = gen.derive_anchor(learner)
        assert path.read_text() == text
        config = _loaded(gen.anchor_stem(learner))
        assert config.environment.foraging.target_foods_to_collect == 20
        assert config.brain.config.wiring == "rewired_gap_junctions_only"
        assert getattr(config.environment.continuous, "body_model", "point") == "point"

    def test_the_table_fits_the_gate_preflight(self) -> None:
        for target in gen.TARGETS:
            for arms in tb.stems(target).values():
                assert set(arms) == set(gp.ARMS)

    def test_the_preflight_table_follows_the_chosen_target(self) -> None:
        """Decision G: the table the gate preflight reads names the arms at ``TARGET``."""
        assert tb.TARGET == 20
        assert tb.stems(tb.TARGET) == tb.STEMS
        for target in gen.TARGETS:
            for arms in tb.stems(target).values():
                assert all(stem.endswith(f"_t{target}") for stem in arms.values())


class TestRegistration:
    @pytest.mark.parametrize(
        ("verdicts", "expected"),
        [
            ({35: "passes", 20: "passes"}, 35),
            ({35: "fallback", 20: "passes"}, 20),
            ({35: "fails", 20: "fallback"}, None),
        ],
    )
    def test_the_target_rule(self, verdicts: dict[int, str], expected: int | None) -> None:
        assert tb.choose_target(verdicts) == expected

    @pytest.mark.parametrize(
        ("verdict", "expected"),
        [("passes", 20), ("fallback", None), ("fails", None)],
    )
    def test_the_retry_reads_target_20_alone(self, verdict: str, expected: int | None) -> None:
        """Decision G: the retry gates target 20 alone, on the same rule."""
        assert tb.RETRY_TARGETS == (20,)
        assert tb.choose_target({20: verdict}) == expected

    def test_the_retry_gate_reads_only_its_seeds_and_target(self, tmp_path: Path) -> None:
        """Decision G: ``mlp_gate`` on the retry's seeds and target 20 alone."""

        def write(stem: str, seed: int, successes: int) -> None:
            lines = [
                f"Run: {i + 1:5d} Status: {'SUCCESS' if i < successes else 'FAILED'} Eaten: 1/20"
                for i in range(40)
            ]
            (tmp_path / f"{stem}-seed{seed}.log").write_text("\n".join(lines) + "\n")

        for seed in tb.RETRY_SEEDS:
            # The last quarter (runs 31-40) all succeed for the learner, none for the floor.
            write(gen.stem("mlp", "ppo", 20), seed, 0)
            write(gen.stem("mlp", "frozen", 20), seed, 0)
            log = tmp_path / f"{gen.stem('mlp', 'ppo', 20)}-seed{seed}.log"
            log.write_text(log.read_text().replace("FAILED", "SUCCESS"))
        result = tb.mlp_gate([tmp_path], tb.RETRY_SEEDS, tb.RETRY_TARGETS)
        assert set(result["gates"]) == {20}
        assert result["gates"][20]["verdict"] == "passes"
        assert result["target"] == 20

    def test_the_minimum_is_m8_s(self) -> None:
        assert pytest.approx(2 / 3 * 0.08647916666666666) == tb.MINIMUM

    def test_the_seed_bands_are_disjoint(self) -> None:
        assert not set(tb.STAGE1_SEEDS) & set(tb.PILOT_SEEDS)
        assert not set(tb.RETRY_SEEDS) & (set(tb.STAGE1_SEEDS) | set(tb.PILOT_SEEDS))
        assert (
            max(tb.RETRY_SEEDS) < tb.PANEL_FIRST_SEED
            or min(tb.RETRY_SEEDS) > tb.PANEL_FIRST_SEED + 64
        )
        assert (
            max(tb.PILOT_SEEDS) < tb.PANEL_FIRST_SEED
            or min(tb.PILOT_SEEDS) > tb.PANEL_FIRST_SEED + 64
        )


_PASS = {"gate_passes": True, "saturated": False}
_FAIL = {"gate_passes": False, "saturated": False}


class TestReading:
    def test_a_failed_gap_only_level_is_unreadable(self) -> None:
        out = tb.read_panel({tb.GAP_ONLY: _FAIL, tb.CHEMICAL_ONLY: _PASS}, {})
        assert out["verdict"] == "unreadable"

    def test_a_failed_chemical_only_level_leaves_the_reading(self) -> None:
        primary = {"gap_mean": 0.09, "test": {"ci_lo": 0.07, "ci_hi": 0.11, "wilcoxon_p": 0.0001}}
        out = tb.read_panel({tb.GAP_ONLY: _PASS, tb.CHEMICAL_ONLY: _FAIL}, primary)
        assert out["verdict"] == "gap_advantage_survives"
        assert out["level_readable"][tb.CHEMICAL_ONLY] is False

    def test_no_lead_at_the_minimum(self) -> None:
        primary = {"gap_mean": 0.0, "test": {"ci_lo": -0.02, "ci_hi": 0.02, "wilcoxon_p": 0.5}}
        out = tb.read_panel({tb.GAP_ONLY: _PASS, tb.CHEMICAL_ONLY: _PASS}, primary)
        assert out["verdict"] == "no_lead_at_minimum"


def test_temperature_is_read_at_the_head() -> None:
    """Scenario "Temperature is read at the head": a thermal body agent steps and senses there."""
    import body_kinematics_eval as harness  # pyright: ignore[reportMissingImports]
    from quantumnematode.agent import QuantumNematodeAgent
    from quantumnematode.env.theme import Theme
    from quantumnematode.utils.config_loader import (
        configure_environment,
        configure_reward,
        configure_satiety,
        create_env_from_config,
    )

    path = gen.THERMAL / f"{gen.stem('wt', 'frozen', 35)}.yml"
    brain, config, sensing = harness.build_brain(path, 3)
    env = create_env_from_config(
        configure_environment(config),
        seed=3,
        max_body_length=1,
        theme=Theme.HEADLESS,
    )
    agent = QuantumNematodeAgent(
        brain=brain,
        env=env,
        theme=Theme.HEADLESS,
        satiety_config=configure_satiety(config),
        sensing_config=sensing,
    )
    agent.run_episode(configure_reward(config), max_steps=3)
    from quantumnematode.env.continuous_2d import Continuous2DEnvironment

    assert isinstance(env, Continuous2DEnvironment)
    head = env.body_midline()
    assert head is not None
    params = agent._create_brain_params()
    assert params.temperature == pytest.approx(env.get_temperature(position=tuple(head[0])))


def test_a_thermal_evaluation_records_the_thermal_drive() -> None:
    """Scenario "A thermal evaluation records the thermal drive": the setpoint error, not food."""
    import body_kinematics_eval as harness  # pyright: ignore[reportMissingImports]

    path = gen.THERMAL / f"{gen.stem('wt', 'frozen', 20)}.yml"
    capture = harness.run_capture(
        path,
        3,
        None,
        episodes=1,
        capture_behaviour=True,
        capture_modality="thermotaxis",
    )
    steps = capture.behaviour[0]
    assert steps
    centre = capture.world_size_mm / 2
    # The cell's field: 20 degrees C at the centre, 1.25 degrees C/mm along x; Tc is 20.
    for step in steps:
        temperature = 20.0 + 1.25 * (step.x - centre)
        assert step.concentration == pytest.approx(-abs(temperature - 20.0), abs=1e-6)


class TestThermotaxisComparisons:
    def _arms(self, offsets: dict[str, float]) -> dict[str, Any]:
        from quantumnematode.validation.datasets import load_bias_signatures

        signs = {k: r.sign for k, r in load_bias_signatures(modality="thermotaxis").items()}
        seeds = [str(s) for s in range(2301, 2309)]
        return {
            arm: {
                "bias_curves": {
                    "statistics": {
                        key: {
                            "per_seed": {
                                s: signs[key] * (offset + 0.01 * i) for i, s in enumerate(seeds)
                            },
                        }
                        for key in signs
                    },
                },
            }
            for arm, offset in offsets.items()
        }

    def test_learning_in_the_reference_direction_is_flagged(self) -> None:
        """A trained arm is paired with its floor, oriented so positive is the reference sign."""
        arms = self._arms(
            {"wt_ppo": 0.2, "wt_frozen": 0.0, "gaponly_ppo": 0.1, "gaponly_frozen": 0.1},
        )
        out = tb.thermal_comparisons(arms)
        wt = out["trained_minus_floor"]["wt_ppo"]["statistics"]
        null = out["trained_minus_floor"]["gaponly_ppo"]["statistics"]
        assert all(r["learned"] for r in wt.values())
        assert not any(r["learned"] for r in null.values())
        for test in out["wild_type_minus_gap_only"].values():
            assert test["mean_delta"] == pytest.approx(0.1)
            assert 0.0 <= test["wilcoxon_p_two_sided"] <= 1.0

    def test_a_missing_arm_leaves_its_comparison_empty(self) -> None:
        arms = self._arms({"wt_ppo": 0.2, "wt_frozen": 0.0})
        out = tb.thermal_comparisons(arms)
        assert all(
            r["arm_minus_floor"] is None
            for r in out["trained_minus_floor"]["gaponly_ppo"]["statistics"].values()
        )
        assert all(v is None for v in out["wild_type_minus_gap_only"].values())
