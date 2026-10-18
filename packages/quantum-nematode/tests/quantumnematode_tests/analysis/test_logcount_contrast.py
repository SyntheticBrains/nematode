"""The log-count wiring contrast: its arms, every verdict-map row, and the manifest's refusals.

Covers `add-logcount-gap-contrast` (design Decisions A, B and F) and the
`architecture-comparison-protocol` scenario "The re-reading reuses the selection pilot's seeds": the
panel's seeds are disjoint from every committed band on this cell's gap contrasts.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))
sys.path.insert(0, str(_root / "scripts" / "campaigns"))

import gate_preflight as gp  # noqa: E402  # pyright: ignore[reportMissingImports]
import logcount_contrast as lc  # noqa: E402  # pyright: ignore[reportMissingImports]

_READABLE: dict[str, Any] = {"gate_passes": True, "saturated": False}


def _gap(mean: float, lo: float, hi: float, q: float) -> dict[str, Any]:
    return {"gap_mean": mean, "test": {"ci_lo": lo, "ci_hi": hi, "bh_q": q}}


class TestArms:
    def test_256_runs_on_the_reserved_band(self) -> None:
        assert tuple(range(3901, 3965)) == lc.SEEDS
        assert lc.plan()["runs"] == 256

    def test_the_seeds_are_disjoint_from_every_committed_band(self) -> None:
        assert lc.overlapping_bands() == []
        assert not set(lc.SEEDS) & set(lc.PILOT_SEEDS)

    def test_an_overlapping_band_is_named(self) -> None:
        assert lc.overlapping_bands((9871, 9999)) == [(9871, 9886)]

    def test_the_level_has_the_preflight_s_four_arms_on_existing_configs(self) -> None:
        assert set(lc.STEMS[lc.LEVEL]) == set(gp.ARMS)
        for config in lc.plan()["configs"]:
            assert (_root / config).is_file()
            assert "_g0a_log_" in config

    def test_the_minimum_is_the_carried_one(self) -> None:
        assert pytest.approx(0.0577, abs=5e-5) == lc.MINIMUM


class TestVerdictMap:
    def test_gates_failing_read_unreadable(self) -> None:
        out = lc.read({"gate_passes": False, "saturated": False}, _gap(0.1, 0.08, 0.12, 0.001))
        assert out["verdict"] == "unreadable"
        out = lc.read({"gate_passes": True, "saturated": True}, _gap(0.1, 0.08, 0.12, 0.001))
        assert out["verdict"] == "unreadable"

    def test_a_lead_at_the_minimum_reads_placement(self) -> None:
        out = lc.read(_READABLE, _gap(0.08, 0.06, 0.10, 0.001))
        assert out["verdict"] == "placement"
        assert out["mechanism_minimum"]["case"] == "carried"

    def test_a_lead_under_the_minimum_reads_below_with_its_direction(self) -> None:
        out = lc.read(_READABLE, _gap(0.03, 0.015, 0.045, 0.01))
        assert (out["verdict"], out["direction"]) == ("below_minimum", "wild_type")
        out = lc.read(_READABLE, _gap(-0.03, -0.045, -0.015, 0.01))
        assert (out["verdict"], out["direction"]) == ("below_minimum", "null")

    def test_an_interval_inside_the_minimum_across_zero_reads_no_wiring_effect(self) -> None:
        out = lc.read(_READABLE, _gap(0.004, -0.012, 0.02, 0.6))
        assert out["verdict"] == "no_wiring_effect"

    def test_a_null_ahead_by_the_minimum_reads_reversed(self) -> None:
        out = lc.read(_READABLE, _gap(-0.08, -0.10, -0.06, 0.001))
        assert out["verdict"] == "reversed"

    def test_the_band_between_the_interval_and_the_test_reads_unresolved(self) -> None:
        # The 80% interval excludes zero while the two-sided q does not reach 0.05.
        out = lc.read(_READABLE, _gap(0.02, 0.004, 0.036, 0.09))
        assert out["verdict"] == "unresolved"


class TestMechanismMinimum:
    def test_a_lead_inside_the_range_carries(self) -> None:
        assert lc.mechanism_minimum(0.09, 0.07, 0.11)["minimum"] == lc.MINIMUM

    def test_a_lead_above_the_range_refixes_at_two_thirds(self) -> None:
        rule = lc.mechanism_minimum(0.15, 0.13, 0.17)
        assert rule["case"] == "refixed"
        assert rule["minimum"] == pytest.approx(0.10)

    def test_a_lead_just_under_the_range_carries_rather_than_falling_below_the_bar(self) -> None:
        rule = lc.mechanism_minimum(0.0578, 0.045, 0.07)
        assert rule == {"case": "carried_below_range", "minimum": lc.MINIMUM}


class TestManifest:
    def _write(
        self,
        where: Path,
        *,
        skip: str | None = None,
        fail: str | None = None,
        mislabel: str | None = None,
    ) -> None:
        for stem in lc.STEMS[lc.LEVEL].values():
            for seed in lc.SEEDS:
                name = f"{stem}-seed{seed}"
                if name == skip:
                    continue
                ran = "other.yml" if name == mislabel else f"{stem}.yml"
                (where / f"{name}.log").write_text(f"  Config: /session/{ran}\n")
                (where / f"{name}.exit").write_text("1\n" if name == fail else "0\n")

    def test_a_complete_panel_passes(self, tmp_path: Path) -> None:
        self._write(tmp_path)
        lc.require_complete(lc.build_manifest([tmp_path], tmp_path / "m.txt"))

    def test_a_missing_run_is_refused(self, tmp_path: Path) -> None:
        stem = lc.STEMS[lc.LEVEL]["rn_frozen"]
        self._write(tmp_path, skip=f"{stem}-seed3964")
        manifest = lc.build_manifest([tmp_path], tmp_path / "m.txt")
        with pytest.raises(lc.ContrastError, match="rn_frozen: \\[3964\\]"):
            lc.require_complete(manifest)

    def test_a_failed_run_is_refused(self, tmp_path: Path) -> None:
        stem = lc.STEMS[lc.LEVEL]["wt_learn"]
        self._write(tmp_path, fail=f"{stem}-seed3901")
        manifest = lc.build_manifest([tmp_path], tmp_path / "m.txt")
        with pytest.raises(lc.ContrastError, match="wt_learn: \\[3901\\]"):
            lc.require_complete(manifest)

    def test_a_log_recording_another_config_is_refused(self, tmp_path: Path) -> None:
        stem = lc.STEMS[lc.LEVEL]["wt_learn"]
        self._write(tmp_path, mislabel=f"{stem}-seed3901")
        with pytest.raises(lc.ContrastError, match=r"other\.yml"):
            lc.build_manifest([tmp_path], tmp_path / "m.txt")

    def test_runs_off_the_band_are_not_read(self, tmp_path: Path) -> None:
        self._write(tmp_path)
        stem = lc.STEMS[lc.LEVEL]["wt_learn"]
        (tmp_path / f"{stem}-seed9871.log").write_text(f"  Config: /s/{stem}.yml\n")
        (tmp_path / f"{stem}-seed9871.exit").write_text("0\n")
        manifest = lc.build_manifest([tmp_path], tmp_path / "m.txt")
        assert " 9871 " not in manifest.read_text()
