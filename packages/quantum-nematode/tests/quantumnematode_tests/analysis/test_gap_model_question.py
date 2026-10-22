"""The gap-model question: its arms, each model's verdict, the joint outcome, and the refusals.

Covers `add-gap-model-question` (design Decisions D and E): every per-model verdict row, what each
pair of verdicts does to the STOP statement, the manifest's refusals, and seed disjointness.
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

import gap_model_question as gq  # noqa: E402  # pyright: ignore[reportMissingImports]
import gate_preflight as gp  # noqa: E402  # pyright: ignore[reportMissingImports]


def _gap(mean: float, lo: float, hi: float, q: float) -> dict[str, Any]:
    return {"gap_mean": mean, "test": {"ci_lo": lo, "ci_hi": hi, "bh_q": q}}


class TestArms:
    def test_512_runs_on_a_new_band(self) -> None:
        plan = gq.plan()
        assert plan["runs"] == 512
        assert plan["overlapping_bands"] == []
        assert not set(gq.SEEDS) & (set(gq.PILOT_SEEDS) | set(gq.SMOKE_SEEDS))
        assert not set(gq.PILOT_SEEDS) & set(gq.OHMIC_EVIDENCE_SEEDS)
        for config in plan["configs"]:
            assert (_root / config).is_file()

    def test_each_model_has_the_preflight_s_four_arms(self) -> None:
        for arms in gq.STEMS.values():
            assert set(arms) == set(gp.ARMS)
        assert all("_g0a_ohmic_" in s for s in gq.STEMS["ohmic"].values())
        assert all("_g0b_strength_" in s for s in gq.STEMS["strength"].values())


class TestVerdict:
    def test_failing_gates_read_unreadable(self) -> None:
        assert gq.verdict(False, _gap(0.1, 0.08, 0.12, 0.001))["verdict"] == "unreadable"  # noqa: FBT003

    @pytest.mark.parametrize(
        ("gap", "expected"),
        [
            (_gap(0.08, 0.06, 0.10, 0.001), "placement"),
            (_gap(0.03, 0.015, 0.045, 0.01), "below_minimum"),
            (_gap(0.004, -0.012, 0.02, 0.6), "no_wiring_effect"),
            (_gap(-0.08, -0.10, -0.06, 0.001), "reversed"),
            (_gap(0.02, 0.004, 0.036, 0.09), "unresolved"),
        ],
    )
    def test_every_state_maps_to_its_verdict(self, gap: dict[str, Any], expected: str) -> None:
        assert gq.verdict(True, gap)["verdict"] == expected  # noqa: FBT003

    def test_below_records_its_direction(self) -> None:
        out = gq.verdict(True, _gap(-0.03, -0.045, -0.015, 0.01))  # noqa: FBT003
        assert out["direction"] == "null"


class TestJoint:
    def test_either_placement_qualifies_the_stop_statement(self) -> None:
        out = gq.joint({"strength": "placement", "ohmic": "no_wiring_effect"})
        assert (out["stop_statement"], out["to_the_body"]) == ("qualified", ["strength"])

    def test_two_nulls_leave_it_standing_at_weak_ohmic_coupling(self) -> None:
        out = gq.joint({"strength": "no_wiring_effect", "ohmic": "no_wiring_effect"})
        assert out["stop_statement"] == "stands"
        assert "weak" in out["why"]

    @pytest.mark.parametrize("other", ["below_minimum", "unresolved", "reversed", "unreadable"])
    def test_anything_else_is_a_dated_decision(self, other: str) -> None:
        out = gq.joint({"strength": "no_wiring_effect", "ohmic": other})
        assert out["stop_statement"] == "dated_decision"


class TestManifest:
    def _write(self, where: Path, *, skip: str | None = None, mislabel: str | None = None) -> None:
        for arms in gq.STEMS.values():
            for stem in arms.values():
                for seed in gq.SEEDS:
                    name = f"{stem}-seed{seed}"
                    if name == skip:
                        continue
                    ran = "other.yml" if name == mislabel else f"{stem}.yml"
                    (where / f"{name}.log").write_text(f"  Config: /session/{ran}\n")
                    (where / f"{name}.exit").write_text("0\n")

    def test_a_complete_panel_passes(self, tmp_path: Path) -> None:
        self._write(tmp_path)
        gq.require_complete(gq.build_manifest([tmp_path], tmp_path / "m.txt"))

    def test_a_missing_run_is_refused(self, tmp_path: Path) -> None:
        stem = gq.STEMS["ohmic"]["rn_frozen"]
        self._write(tmp_path, skip=f"{stem}-seed4264")
        manifest = gq.build_manifest([tmp_path], tmp_path / "m.txt")
        with pytest.raises(gq.QuestionError, match=r"ohmic/rn_frozen: \[4264\]"):
            gq.require_complete(manifest)

    def test_a_log_recording_another_config_is_refused(self, tmp_path: Path) -> None:
        stem = gq.STEMS["strength"]["wt_learn"]
        self._write(tmp_path, mislabel=f"{stem}-seed4201")
        with pytest.raises(gq.QuestionError, match=r"other\.yml"):
            gq.build_manifest([tmp_path], tmp_path / "m.txt")
