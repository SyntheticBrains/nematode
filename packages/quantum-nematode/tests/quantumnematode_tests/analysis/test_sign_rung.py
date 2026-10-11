"""The per-connection signs rung: its configs, its constants and its readings.

Covers the connectome-ppo-brain requirement "Per-connection synapse signs" through the configs that
run it, and the registration's three readings, Holm-adjusted, and their verdicts.
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
import generate_sign_rung_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]
import sign_rung as sr  # noqa: E402  # pyright: ignore[reportMissingImports]

_THERMAL = _root / "configs" / "scenarios" / "thermal_foraging"


@functools.cache
def _loaded(stem: str) -> dict[str, Any]:
    return load_simulation_config(str(_THERMAL / f"{stem}.yml")).model_dump()


class TestConfigs:
    def test_every_arm_is_a_committed_config(self) -> None:
        for arms in sr.STEMS.values():
            for stem in arms.values():
                assert (_THERMAL / f"{stem}.yml").is_file(), stem

    @pytest.mark.parametrize("child", sorted(gen.CHILDREN))
    def test_each_arm_differs_from_its_parent_in_the_listed_keys_alone(self, child: str) -> None:
        parent, changes = gen.CHILDREN[child]
        got, expected = _loaded(child), dict(_loaded(parent))
        expected["brain"] = {**expected["brain"]}
        expected["brain"]["config"] = {**expected["brain"]["config"], **changes}
        assert got == expected

    @pytest.mark.parametrize("child", sorted(gen.CHILDREN))
    def test_each_committed_file_is_the_generator_s(self, child: str) -> None:
        path, text = gen.derive(child)
        assert path.read_text() == text

    def test_the_table_fits_the_gate_preflight(self) -> None:
        for arms in sr.STEMS.values():
            assert set(arms) == set(gp.ARMS)

    def test_the_random_wild_type_is_the_committed_t35_pair(self) -> None:
        stems = sr.STEMS[sr.RANDOM]
        assert stems["wt_learn"] == "connectomeppo_small_continuous2d_thermal_klinotaxis_t35"
        assert (
            stems["wt_frozen"] == "connectomeppo_small_continuous2d_thermal_klinotaxis_frozen_t35"
        )

    def test_grounded_levels_enforce_and_nulls_are_chemical_only(self) -> None:
        for level in (sr.ATLAS, sr.PER_CONNECTION):
            for arm, stem in sr.STEMS[level].items():
                brain = _loaded(stem)["brain"]["config"]
                assert brain["synapse_signs"] == level
                assert brain["enforce_synapse_signs"] is True
                expected = "rewired_chemical_only" if arm.startswith("rn") else "wild_type"
                assert brain["wiring"] == expected


class TestRegistration:
    def test_the_minimum_is_two_thirds_of_the_thermal_split(self) -> None:
        assert pytest.approx(2 / 3 * 0.08647916666666666) == sr.MINIMUM

    def test_the_pilot_seeds_are_disjoint_from_the_panel_s(self) -> None:
        assert max(sr.PILOT_SEEDS) < sr.PANEL_FIRST_SEED or min(sr.PILOT_SEEDS) > 2264


def _reading(mean: float, lo: float, hi: float, p: float) -> dict[str, Any]:
    return {"mean": mean, "test": {"ci_lo": lo, "ci_hi": hi, "wilcoxon_p": p}}


_PASS = {"gate_passes": True, "saturated": False}


class TestReadings:
    def test_holm_steps_down(self) -> None:
        assert sr.holm({"a": 0.01, "b": 0.02, "c": 0.04}) == pytest.approx(
            {"a": 0.03, "b": 0.04, "c": 0.04},
        )

    def test_a_failed_level_is_unreadable(self) -> None:
        gates = dict.fromkeys(sr.LEVELS, _PASS) | {
            sr.ATLAS: {"gate_passes": False, "saturated": False},
        }
        out = sr.read_panel(gates, {})
        assert out["verdict"] == "unreadable"

    def test_each_reading_takes_its_state_and_verdict(self) -> None:
        gates = dict.fromkeys(sr.LEVELS, _PASS)
        readings = {
            "lead": _reading(0.09, 0.07, 0.11, 0.0001),
            "grounding": _reading(0.07, 0.06, 0.09, 0.0001),
            "receptor": _reading(0.0, -0.02, 0.02, 0.5),
        }
        out = sr.read_panel(gates, readings)
        assert out["lead"]["verdict"] == "wild_type_leads"
        assert out["grounding"]["verdict"] == "lead_grows"
        assert out["receptor"]["verdict"] == "no_change_at_minimum"
        assert out["receptor"]["holm_q"] == pytest.approx(1.0)
