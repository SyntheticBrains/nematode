"""C.3: the control's configs, the arms, the registered bands, the control's gate, and pooling.

Covers the add-body-validation change: the derivative-sensing control is C.1e's MLP arm with no
spatial head-sweep, learning and frozen; each reading is graded pass, partial or fail against its
fixed bands; a reading whose grade differs between 20 and 40 sub-steps is on an edge; a control that
does not beat its floor leaves the weathervane's specificity untested.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from quantumnematode.utils.config_loader import load_simulation_config

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
sys.path.insert(0, str(_REPO / "scripts" / "campaigns"))

import body_validation as bv  # noqa: E402  # pyright: ignore[reportMissingImports]
import generate_body_validation_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]
import generate_body_wiring_configs as wiring  # noqa: E402  # pyright: ignore[reportMissingImports]


class TestConfigs:
    @pytest.mark.parametrize("arm", ["learn", "frozen"])
    def test_the_committed_control_configs_are_the_generators(self, arm: str) -> None:
        """Each control config is the MLP arm in derivative sensing, and loads as intended."""
        path, text = gen.derive(arm)
        assert path.read_text() == text
        config = load_simulation_config(str(path))
        assert config.max_steps == 500
        assert config.environment is not None
        assert config.environment.sensing is not None
        assert config.environment.sensing.chemotaxis_mode == "derivative"
        assert config.brain is not None
        brain = config.brain.config
        assert getattr(brain, "entropy_coef", None) == 0.004
        assert getattr(brain, "freeze_updates", False) is (arm == "frozen")

    def test_every_arm_has_its_config(self) -> None:
        """Each arm's stem names a committed config."""
        for stem, _seeds, _graded in bv.ARMS.values():
            assert (wiring.FORAGING / f"{stem}.yml").is_file()

    def test_the_control_is_paired_with_the_mlp_arm(self) -> None:
        """The control's seeds are a subset of the panel's, so each pairs with an MLP seed."""
        assert set(bv.CONTROL_SEEDS) <= set(bv.PANEL_SEEDS)
        graded = {arm for arm, (_s, _seeds, g) in bv.ARMS.items() if g}
        assert graded == {"wild_type", "chemical_only_null", "mlp"}


class TestGrades:
    @pytest.mark.parametrize(
        ("name", "value", "expected"),
        [
            ("frequency_hz", 0.30, "pass"),
            ("frequency_hz", 0.15, "partial"),
            ("frequency_hz", 0.05, "fail"),
            ("speed_bl_per_s", 0.12, "pass"),
            ("speed_bl_per_s", 0.11, "partial"),
            ("wavelength_bl", 1.20, "fail"),
            ("eigenworm_variance", 0.99, "pass"),
            ("eigenworm_variance", 0.75, "partial"),
            ("forward_bout_share", 0.40, "fail"),
            ("forward_bout_share", None, None),
        ],
    )
    def test_a_reading_is_graded_against_its_bands(
        self,
        name: str,
        value: float | None,
        expected: str | None,
    ) -> None:
        """Pass inside the pass band, partial inside the partial band, else fail; None unread."""
        assert bv.grade(name, value) == expected

    def test_a_grade_that_flips_with_the_substeps_is_on_an_edge(self) -> None:
        """Speed passing at 20 sub-steps and partial at 40 sits on the band's edge."""
        base = {"frequency_hz": 0.3, "wavelength_bl": 0.65, "speed_bl_per_s": 0.121}
        doubled = {"frequency_hz": 0.31, "wavelength_bl": 0.66, "speed_bl_per_s": 0.118}
        assert bv.edges(base, doubled) == {
            "frequency_hz": False,
            "wavelength_bl": False,
            "speed_bl_per_s": True,
        }


def _curves(per_seed: dict[int, float]) -> dict[str, Any]:
    seeds = {str(s): v for s, v in per_seed.items()}
    return {
        "statistics": {
            "klinotaxis": {"per_seed": seeds},
            "klinotaxis_all": {"per_seed": seeds},
        },
    }


class TestControl:
    @staticmethod
    def _arms() -> dict[str, Any]:
        return {
            "mlp": {"bias_curves": _curves(dict.fromkeys(bv.CONTROL_SEEDS, 0.5))},
            "mlp_derivative": {"bias_curves": _curves(dict.fromkeys(bv.CONTROL_SEEDS, 0.1))},
        }

    def test_a_readable_control_pairs_the_weathervane_by_seed(self) -> None:
        """Beating its floor on every seed, the control's weathervane is paired with the MLP's."""
        learn = dict.fromkeys(bv.CONTROL_SEEDS, 60.0)
        frozen = dict.fromkeys(bv.CONTROL_SEEDS, 0.0)
        out = bv.control_comparison(self._arms(), learn, frozen)
        assert out["readable"]
        for key in ("klinotaxis", "klinotaxis_all"):
            reading = out["weathervane"][key]
            assert reading["n_seeds"] == len(bv.CONTROL_SEEDS)
            assert reading["mlp_minus_control"]["mean_delta"] == pytest.approx(0.4)

    def test_a_control_at_its_floor_leaves_the_specificity_untested(self) -> None:
        """No lift over the frozen floor: unreadable, and nothing is inferred."""
        flat = dict.fromkeys(bv.CONTROL_SEEDS, 0.0)
        out = bv.control_comparison(self._arms(), flat, flat)
        assert not out["readable"]
        assert out["weathervane_specificity"] == "untested"
        assert "weathervane" not in out

    def test_a_missing_control_seed_leaves_it_unreadable(self) -> None:
        """A seed without a plateau makes the gate incomplete."""
        learn = dict.fromkeys(bv.CONTROL_SEEDS[1:], 60.0)
        frozen = dict.fromkeys(bv.CONTROL_SEEDS, 0.0)
        out = bv.control_comparison(self._arms(), learn, frozen)
        assert out["gate"]["verdict"] == "incomplete"
        assert not out["readable"]

    def test_no_control_runs_leave_it_untested(self) -> None:
        """Without the control's runs there is no gate to read."""
        out = bv.control_comparison(self._arms(), {}, {})
        assert out["gate"] is None
        assert out["weathervane_specificity"] == "untested"


class TestPooling:
    def test_an_arm_pools_its_runs(self, tmp_path: Path) -> None:
        """Eigenworm variance pools sums of squares; omega rate is per worm-minute."""
        capture = tmp_path / "empty.json"
        capture.write_text(json.dumps({"runs": []}))
        kinematics = {
            "frequency_hz": 0.3,
            "wavelength_bl": 0.65,
            "speed_bl_per_s": 0.13,
            "reversal_fraction": 0.01,
            "steps_used": 100,
            "steps_near_wall": 0,
            "steps_undulating": 100,
        }
        run = {
            "world_size_mm": 20.0,
            "kinematics": kinematics,
            "half_step": kinematics,
            "amplitude_sample": [4.0, 5.0, 6.0],
            "forward_bout_share": 1.0,
            "capture": str(capture),
        }
        runs = [
            run
            | {
                "seed": 1,
                "eigenworm_captured_ss": 9.0,
                "eigenworm_total_ss": 10.0,
                "omega_turns": [3.0],
                "omega_turn_a3": [12.0],
                "worm_minutes": 1.0,
            },
            run
            | {
                "seed": 2,
                "eigenworm_captured_ss": 1.0,
                "eigenworm_total_ss": 10.0,
                "omega_turns": [],
                "omega_turn_a3": [],
                "worm_minutes": 3.0,
            },
            {"arm": "wild_type", "seed": 3, "missing": True},
        ]
        out = bv.summarise_arm(runs, graded=True)
        assert out["n_runs"] == 2
        assert out["missing_seeds"] == [3]
        assert out["readings"]["eigenworm_variance"] == pytest.approx(0.5)
        assert out["omega_turns"]["per_worm_minute"] == pytest.approx(0.25)
        assert out["omega_turns"]["omega_posture_share"] == pytest.approx(1.0)
        assert out["grades"]["frequency_hz"] == "pass"
        assert out["grades"]["eigenworm_variance"] == "fail"
        assert out["half_step"]["agrees"]
        assert out["bias_curves"] is None


def test_the_omega_posture_threshold_is_the_real_postures_tail() -> None:
    """The third eigenworm's 99th percentile over the 6,655 real postures, about 10.6."""
    assert bv.omega_posture_threshold() == pytest.approx(10.6, abs=0.1)
