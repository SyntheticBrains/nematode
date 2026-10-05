"""A.3's boundary-preserving null panel: its configs, constants, verdict maps and manifest.

Covers the connectome-ppo-brain requirement "A boundary-preserving rewired null" through the configs
that run it, and the registration's readings.
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

import boundary_null as bn  # noqa: E402  # pyright: ignore[reportMissingImports]
import null_strength_control as nsc  # noqa: E402  # pyright: ignore[reportMissingImports]
import thermal_split as ts  # noqa: E402  # pyright: ignore[reportMissingImports]

_CONFIGS = _root / "configs" / "scenarios" / "foraging"


@functools.cache
def _loaded(stem: str) -> dict[str, Any]:
    """Load the whole simulation config as a run gets it, through the real loader."""
    return load_simulation_config(str(_CONFIGS / f"{stem}.yml")).model_dump()


class TestTheConfigs:
    def test_every_arm_is_a_committed_config(self) -> None:
        missing = [s for s in bn.LEVELS_BY_STEM if not (_CONFIGS / f"{s}.yml").is_file()]
        assert not missing, missing
        assert len(bn.LEVELS_BY_STEM) == 6

    @pytest.mark.parametrize("stem", sorted(bn.NEW_ARMS))
    def test_each_new_arm_differs_from_its_parent_in_wiring_alone(self, stem: str) -> None:
        _half, parent = bn.NEW_ARMS[stem]
        child, expected = _loaded(stem), dict(_loaded(parent))
        assert child["brain"]["config"]["wiring"] == "rewired_boundary_held"
        expected = {**expected, "brain": {**expected["brain"]}}
        expected["brain"]["config"] = {
            **expected["brain"]["config"],
            "wiring": "rewired_boundary_held",
        }
        assert child == expected

    def test_the_chemical_level_is_a6s(self) -> None:
        assert bn.STEMS[bn.CHEMICAL] == nsc.STEMS["ppo"][nsc.CHEMICAL]


class TestRegistration:
    def test_seeds_are_fresh(self) -> None:
        assert len(bn.SEEDS) == 128
        used = set(nsc.SEEDS_BY_HALF["ppo"]) | set(nsc.SEEDS_BY_HALF["reading"]) | set(ts.SEEDS)
        assert not used & set(bn.SEEDS)

    def test_the_minimum_is_two_thirds_of_a6s_chemical_only_lead(self) -> None:
        assert pytest.approx(0.02146875) == bn.REFERENCE_EFFECT
        assert pytest.approx(2 / 3 * 0.02146875) == bn.MINIMUM


def _gates(*, saturated: bool = False) -> dict[str, Any]:
    return {"gate_passes": True, "saturated": saturated}


def _t(lo: float, hi: float, q: float = 0.01) -> dict[str, float]:
    return {"ci_lo": lo, "ci_hi": hi, "bh_q": q}


class TestReading:
    def test_a_saturated_level_stops_the_reading(self) -> None:
        gates = {bn.CHEMICAL: _gates(), bn.BOUNDARY: _gates(saturated=True)}
        assert bn.read_panel(gates, {}, {}, _t(0.01, 0.03))["verdict"] == "unreadable"

    @pytest.mark.parametrize(
        ("mean", "test", "lead_lo", "verdict"),
        [
            (-0.03, _t(-0.04, -0.02), 0.01, "boundary"),
            (0.03, _t(0.02, 0.04), 0.01, "shortcuts_helped_null"),
            (0.001, _t(-0.005, 0.006, q=0.7), 0.01, "interior"),
            (0.001, _t(-0.005, 0.006, q=0.7), -0.01, "no_gap_to_attribute"),
            (0.008, _t(0.002, 0.012), 0.01, "below_minimum"),
        ],
    )
    def test_the_interaction_map(self, mean, test, lead_lo, verdict) -> None:
        gates = {level: _gates() for level in bn.LEVELS}
        lead = {"gap_mean": 0.02, "test": _t(lead_lo, 0.04)}
        got = bn.read_panel(gates, {"interaction_mean": mean, "test": test}, lead, _t(0.01, 0.03))
        assert got["interaction"]["verdict"] == verdict

    def test_the_lead_map(self) -> None:
        gates = {level: _gates() for level in bn.LEVELS}
        inter = {"interaction_mean": 0.0, "test": _t(-0.005, 0.005, q=0.9)}
        lead = {"gap_mean": 0.03, "test": _t(0.02, 0.04)}
        assert (
            bn.read_panel(gates, inter, lead, _t(0.01, 0.03))["lead"]["verdict"] == "lead_remains"
        )

    def test_no_base_effect_withholds_the_interaction_but_not_the_lead(self) -> None:
        gates = {level: _gates() for level in bn.LEVELS}
        inter = {"interaction_mean": -0.03, "test": _t(-0.04, -0.02)}
        lead = {"gap_mean": 0.03, "test": _t(0.02, 0.04)}
        got = bn.read_panel(gates, inter, lead, _t(-0.005, 0.03))
        assert got["interaction"]["verdict"] == "no_base_effect"
        assert got["lead"]["verdict"] == "lead_remains"


class TestManifest:
    def test_a_wild_type_run_serves_both_levels(self, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        for stem in bn.LEVELS_BY_STEM:
            (logs / f"{stem}-seed641.log").write_text("x")
        rows = (
            bn.build_manifest(tmp_path, tmp_path / "m.txt", seeds=(641,)).read_text().splitlines()
        )
        assert len(rows) == 4 * 2

    def test_a_foreign_log_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "other-seed641.log").write_text("x")
        with pytest.raises(bn.BoundaryNullError, match="does not have"):
            bn.build_manifest(tmp_path, tmp_path / "m.txt", seeds=(641,))
