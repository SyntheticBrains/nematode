"""Re-reading the committed tables.

The re-read's one job is to read the same numbers the record was scored on, under a second
statistic, without touching what was decided. So the anchor here is reproduction: the all-seeds
member must return panel 3's committed R1 and the frequency member its committed R2, because if it
does not, the re-read is reading something other than the committed table.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))

import l4_mixture_statistic as ms  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_reread as rr  # noqa: E402  # pyright: ignore[reportMissingImports]


class TestItReadsTheCommittedNumbers:
    def _panel3(self) -> dict:
        path = rr.SUPPORTING / "042-l4-panel3" / "panel3.json"
        return json.loads(path.read_text())["family"]

    def test_the_shift_member_reproduces_panel_threes_committed_r1(self) -> None:
        success, _ = rr.read_table("042-l4-panel3")
        out = ms.shift_contrast(success["wt_hebbian"], success["rn_hebbian"])
        committed = self._panel3()["R1"]
        assert out["n"] == committed["n"]
        assert out["effect"] == pytest.approx(committed["mean_delta"], abs=1e-6)
        assert out["p_improve"] == pytest.approx(committed["wilcoxon_p"], abs=1e-6)

    def test_the_frequency_member_reproduces_panel_threes_committed_r2(self) -> None:
        # R2 was the registered secondary; F is that test, generalised off its wt/rn naming. It was
        # computed on the replication seeds only, so the comparison is on those.
        from l4_panel3 import REPLICATION_SEEDS  # pyright: ignore[reportMissingImports]

        success, _ = rr.read_table("042-l4-panel3")
        restricted = {
            arm: {s: v for s, v in values.items() if s in REPLICATION_SEEDS}
            for arm, values in success.items()
        }
        out = ms.frequency_contrast(restricted["wt_hebbian"], restricted["rn_hebbian"])
        committed = self._panel3()["R2"]
        assert out["effect"] == pytest.approx(committed["mean_delta"])
        assert out["p_improve"] == pytest.approx(committed["p_value"], abs=1e-9)

    def test_every_registered_contrast_finds_its_arms(self) -> None:
        for contrast in rr.CONTRASTS:
            out = rr.reread(contrast)
            assert "error" not in out, f"{contrast.table}/{contrast.name}: {out.get('error')}"


class TestTheProtocolsAreKeptApart:
    def test_an_assay_is_read_against_its_own_per_seed_comparator(self) -> None:
        out = rr.reread(
            next(c for c in rr.CONTRASTS if c.table == "052-l4-endpoint-evaluation"),
        )
        assert out["protocol"] == "assay"
        assert rr.COMPARATOR_COLUMN in out["arm_b"]

    def test_a_panel_is_read_against_its_paired_arm(self) -> None:
        out = rr.reread(next(c for c in rr.CONTRASTS if c.table == "042-l4-panel3"))
        assert out["protocol"] == "panel"
        assert out["arm_b"] == "rn_hebbian"

    def test_the_assay_tables_have_no_graded_metric_and_say_so(self) -> None:
        # The committed assay tables carry no foods column. Reading fewer tables quietly would
        # hide that; the record states it instead.
        out = rr.reread(
            next(c for c in rr.CONTRASTS if c.table == "050-l4-perturbation-clone-assay"),
        )
        assert out["graded"]["available"] is False
        assert "foods" in out["graded"]["reason"]


class TestTheGradedReading:
    def test_a_panel_is_read_on_both_metrics(self) -> None:
        out = rr.reread(next(c for c in rr.CONTRASTS if c.table == "042-l4-panel3"))
        assert out["graded"]["available"] is True
        assert out["graded"]["W"]["defined"] is True

    def test_competence_comes_from_the_primary_metric(self) -> None:
        assert (
            "primary metric"
            in rr._graded(
                {1: 9.0, 2: 1.0},
                {1: 8.0, 2: 1.0},
                {1: 50.0, 2: 1.0},
                {1: 50.0, 2: 1.0},
            )["competence_from"]
        )

    def test_progress_short_of_a_clear_is_visible(self) -> None:
        # Both arms at the full-clear floor, one reaching far more food: the cliff metric sees
        # nothing and the graded reading sees the difference.
        a_success = dict.fromkeys(range(1, 9), 0.0)
        b_success = dict.fromkeys(range(1, 9), 0.0)
        a_foods = dict.fromkeys(range(1, 9), 8.0)
        b_foods = dict.fromkeys(range(1, 9), 2.0)
        assert ms.shift_contrast(a_success, b_success)["defined"] is False
        graded = rr._graded(a_foods, b_foods, a_success, b_success)
        assert graded["W"]["effect"] == pytest.approx(6.0)
        # No seed is competent on the primary metric, so the level contrast has no upper mode.
        assert graded["L"]["defined"] is False


class TestTheRecord:
    def test_it_is_strict_json(self) -> None:
        rows = [rr.reread(c) for c in rr.CONTRASTS[:3]]
        text = json.dumps(rr._jsonable({"contrasts": rows}), allow_nan=False)
        json.loads(text, parse_constant=lambda c: pytest.fail(f"bare {c} in the record"))
