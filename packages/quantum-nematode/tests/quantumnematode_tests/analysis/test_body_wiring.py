"""C.1e's pilot: its configs, the minimum and its floor, the panel's seed count, and the gates.

Covers the add-body-wiring-contrast change's registered pilot rules: the minimum is 2/3 of the
pilot's reference, floored at 0.0367; the panel's n is the smallest whose MDE reaches the minimum,
capped at 64; a learner that fails its gates leaves the panel; the competent-seed frequency is an
exact McNemar description.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from quantumnematode.utils.config_loader import load_simulation_config

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
sys.path.insert(0, str(_REPO / "scripts" / "campaigns"))

import body_wiring as bw  # noqa: E402  # pyright: ignore[reportMissingImports]
import gate_preflight as gp  # noqa: E402  # pyright: ignore[reportMissingImports]
import generate_body_wiring_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]


class TestConfigs:
    @pytest.mark.parametrize("wiring", gen.PILOT_WIRINGS)
    @pytest.mark.parametrize("learner", gen.LEARNERS)
    def test_the_committed_pilot_configs_are_the_generators(
        self,
        wiring: str,
        learner: str,
    ) -> None:
        """Each pilot config on disk is what the generator writes, and it loads as intended."""
        path, text = gen.derive(wiring, learner)
        assert path.read_text() == text
        config = load_simulation_config(str(path))
        assert config.max_steps == 500
        assert config.environment is not None
        continuous = config.environment.continuous
        assert continuous is not None
        assert continuous.body_model == "kinematic"
        assert config.brain is not None
        brain = config.brain.config
        assert getattr(brain, "action_space", None) == "body_drive"
        assert getattr(brain, "entropy_coef", None) == 0.004
        assert getattr(brain, "wiring", None) == gen.WIRINGS[wiring]
        assert getattr(brain, "freeze_wiring", False) is (learner == "fw")
        assert getattr(brain, "freeze_updates", False) is (learner == "frozen")

    def test_the_committed_mlp_config_is_the_generators(self) -> None:
        """The MLP arm is C.1d's 500-step control at the body arms' entropy, and loads."""
        path, text = gen.derive_mlp()
        assert path.read_text() == text
        config = load_simulation_config(str(path))
        assert config.max_steps == 500
        assert config.brain is not None
        assert getattr(config.brain.config, "entropy_coef", None) == 0.004

    def test_the_table_fits_the_gate_preflight(self) -> None:
        """The pilot is one gate-preflight level of four arms: PPO alone."""
        stems = gp.panel_stems("body_wiring")
        assert set(stems) == {"ppo"}
        assert stems["ppo"]["wt_frozen"] == gen.stem("wt", "frozen")


class TestRules:
    def test_the_minimum_is_two_thirds_of_the_reference_with_a_floor(self) -> None:
        """2/3 of |reference|, never below the judged floor of 0.0367."""
        assert bw.minimum(0.09) == pytest.approx(0.06)
        assert bw.minimum(-0.09) == pytest.approx(0.06)
        assert bw.minimum(0.01) == pytest.approx(bw.MINIMUM_FLOOR)

    def test_the_panel_takes_the_smallest_n_that_reaches_the_minimum(self) -> None:
        """N is the smallest seed count whose MDE is at most the minimum, capped at 64."""
        plan = bw.panel_seeds(sd=0.1, minimum_effect=0.05)
        assert plan["n"] == 25
        assert not plan["capped"]
        assert bw.panel_seeds(sd=0.01, minimum_effect=0.05)["n"] == bw.MIN_PANEL_SEEDS
        assert bw.panel_seeds(sd=1.0, minimum_effect=0.05) == {
            "n": 64,
            "capped": True,
            "mde": pytest.approx(bw.MDE_Z / 8),
        }

    def test_the_floor_is_reported_as_a_share_of_the_wild_types_auc(self) -> None:
        """The judged floor's size on this cell is visible beside the minimum."""
        gates = {"gate_passes": True, "saturated": False}
        gap = {"gap_mean": 0.01, "per_seed": {1: 0.0, 2: 0.02}, "test": {}}
        reading = bw.read_learner(gates, gap, wt_auc=0.367)
        assert reading["minimum_floored"] is True
        assert reading["floor_share_of_wt_auc"] == pytest.approx(0.1)

    def test_the_per_seed_csv_has_a_row_per_seed(self, tmp_path: Path) -> None:
        """Each seed's plateaus, floors and gaps, under each learner."""
        per = {1701: {"learn": 50.0, "floor": 0.0}}
        result = {
            "seeds": [1701],
            "learners": {
                "ppo": {
                    "gates": {"wt": {"per_seed": per}, "rn": {"per_seed": per}},
                    "gaps": {
                        m: {"per_seed": {1701: 0.1}} for m in (bw.PRIMARY_METRIC, bw.BESIDE_METRIC)
                    },
                },
            },
        }
        rows = bw.write_csv(result, tmp_path / "per-seed.csv").read_text().splitlines()
        assert rows[0].startswith("seed,ppo_wt_plateau")
        assert rows[1].startswith("1701,50.000000,0.000000,50.000000")

    @pytest.mark.parametrize(
        ("reading", "verdict"),
        [
            ((0.06, 0.045, 0.075, 0.005), "wild_type_ahead"),
            ((0.02, 0.0, 0.049, 0.10), "unresolved_at_this_sensitivity"),
            ((0.001, -0.02, 0.022, 0.4), "no_wiring_effect_at_minimum"),
            ((-0.06, -0.075, -0.045, 0.005), "null_ahead"),
        ],
    )
    def test_the_panel_reading(self, reading: tuple[float, ...], verdict: str) -> None:
        """The registered reading's verdict at 0.0367, and only move_wt opens the boundary stage."""
        mean, lo, hi, p = reading
        gap = {
            "gap_mean": mean,
            "per_seed": {1: mean, 2: mean},
            "test": {"ci_lo": lo, "ci_hi": hi, "wilcoxon_p": p},
        }
        result = bw.read_panel({"gate_passes": True, "saturated": False}, gap)
        assert result["verdict"] == verdict
        assert result["boundary_stage_runs"] is (verdict == "wild_type_ahead")

    def test_the_mlp_is_read_beside(self, tmp_path: Path) -> None:
        """The MLP's plateaus come from its own logs; a missing seed is simply absent."""
        line = "Run: {} Status: SUCCESS Reason: goal Steps: 10 Eaten: 20/20\n"
        log = tmp_path / f"{gen.MLP_STEM}-seed1801.log"
        log.write_text("".join(line.format(r) for r in range(1, 41)))
        result = bw.mlp_plateaus([tmp_path], (1801, 1802))
        assert result["n_seeds"] == 1
        assert result["mean"] == pytest.approx(100.0)
        assert result["competent"] == 1

    def test_failed_gates_make_the_panel_unreadable(self) -> None:
        """No reading is classified when a learning arm fails its floor."""
        reading = bw.read_panel({"gate_passes": False, "saturated": False}, {})
        assert reading["verdict"] == "unreadable"

    def test_a_learner_failing_its_floor_leaves_the_panel(self) -> None:
        """An unreadable learner is recorded as leaving, with its reason."""
        gates = {"gate_passes": False, "saturated": False}
        gap = {"gap_mean": 0.05, "per_seed": {1: 0.0, 2: 0.1}, "test": {}}
        reading = bw.read_learner(gates, gap)
        assert reading["readable"] is False
        assert reading["leaves_panel"] == "a learning arm does not beat its floor"

    def test_the_competence_frequency_is_an_exact_mcnemar(self) -> None:
        """Only the seeds where the wirings disagree carry the test."""
        wt = {s: {"learn": v} for s, v in enumerate([80.0, 70.0, 60.0, 20.0, 50.0])}
        rn = {s: {"learn": v} for s, v in enumerate([20.0, 10.0, 15.0, 25.0, 55.0])}
        freq = bw.competence_frequency({"wt": {"per_seed": wt}, "rn": {"per_seed": rn}})
        assert (freq["only_wt"], freq["only_rn"]) == (3, 0)
        assert freq["mcnemar_exact_p"] == pytest.approx(0.25)
