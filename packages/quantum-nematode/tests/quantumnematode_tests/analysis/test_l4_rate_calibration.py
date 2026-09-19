"""L.1b's reading: L.1's interaction at the rate-matched 0.0001, with a minimum that decides.

* **The scanner reads only the registered arm.** `_r1e4` pooled logs are cells; `_r1e2` (L.0's
  rate check) is refused; an untagged pooled log is L.1's arm at 0.001 and is skipped, and L.1's
  own regex rejects the tagged names, so neither scanner can mis-read the other's cells.
* **The minimum is half of L.1's interaction**, and a significant result below it reads
  rate-specific with *shrunk below half* named.
* **The three-way is the committed interaction minus the matched one**, per seed.
* **Eight tests, one family.**
* **The instruments are read-only.** Two of them produced the cells this 2x2 reuses.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))

import l4_rate_calibration as rc  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_readout_width as rw  # noqa: E402  # pyright: ignore[reportMissingImports]

_READ_ONLY = (
    "l4_readout_width.py",
    "l4_feature_ablations.py",
    "connectome_structure_efficiency.py",
    "wiring_premise.py",
)


def _git(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["git", *args],  # noqa: S607
        cwd=_root,
        check=False,
        capture_output=True,
    )


def _require_revision(rev: str) -> None:
    if _git(["cat-file", "-e", f"{rev}^{{commit}}"]).returncode != 0:
        pytest.skip(f"{rev} is not in this clone (shallow checkout); the guard cannot run here")


def _cells(
    interaction_at: float,
    interaction_committed: float = 0.28,
    n: int = 12,
) -> dict[str, dict[str, dict[int, float]]]:
    """Two 2x2s whose per-seed interactions are the requested constants plus a small alternation."""
    seeds = range(1, n + 1)

    def panel(interaction: float) -> dict[str, dict[int, float]]:
        jitter = {s: 0.01 * (1 if s % 2 else -1) for s in seeds}
        return {
            "wt_pooled": dict.fromkeys(seeds, 0.4),
            "rn_pooled": dict.fromkeys(seeds, 0.5),
            "wt_wide": {s: 0.40 + interaction + jitter[s] for s in seeds},
            "rn_wide": dict.fromkeys(seeds, 0.5),
        }

    return {rc.MATCHED: panel(interaction_at), rc.COMMITTED: panel(interaction_committed)}


def _gates(*, passing: bool = True) -> dict[str, Any]:
    q = 0.001 if passing else 0.5
    out: dict[str, Any] = {
        f"{a}_gate": {"effect": 10.0, "q": q, "p_improve": q, "n": 12, "positive_seeds": 12}
        for a in rc.LEARNING_ARMS
    }
    out["gates_pass"] = passing
    return out


class TestTheScanner:
    @pytest.mark.parametrize(
        ("name", "arm"),
        [
            ("..._readout_only_r1e4-seed3.log", "wt_pooled"),
            ("..._readout_only_r1e4_rewired_null-seed3.log", "rn_pooled"),
        ],
    )
    def test_the_registered_labels_resolve_to_the_pooled_cells(self, name: str, arm: str) -> None:
        m = rc._LABEL.match(name.replace("...", rc._STEM))
        assert m is not None
        assert m.group("rate") == rc._RATE_TAG
        assert ("rn_pooled" if m.group("rewired") else "wt_pooled") == arm

    def test_l1s_regex_rejects_the_tagged_names_and_this_one_rejects_l1s(self) -> None:
        tagged = f"{rc._STEM}_readout_only_r1e4-seed3.log"
        untagged = f"{rc._STEM}_readout_only-seed3.log"
        wide = f"{rc._STEM}_readout_only_wide_r1e4-seed3.log"
        assert rw._LABEL.match(tagged) is None
        assert rc._LABEL.match(untagged) is None
        assert rc._LABEL.match(wide) is None

    def test_scan_refuses_the_rate_check_tag(self, tmp_path: Path) -> None:
        (tmp_path / f"{rc._STEM}_readout_only_r1e2-seed3.log").write_text("")
        with pytest.raises(ValueError, match="_r1e2"):
            rc.scan(tmp_path)

    def test_scan_skips_an_untagged_pooled_log(self, tmp_path: Path) -> None:
        (tmp_path / f"{rc._STEM}_readout_only-seed3.log").write_text("")
        scanned = rc.scan(tmp_path)
        assert scanned["runs"] == {"wt_pooled": {}, "rn_pooled": {}}


class TestAssembly:
    def test_matched_panel_takes_pooled_here_wide_from_l4_floors_from_l1(self) -> None:
        pooled = {"runs": {"wt_pooled": {1: "p_wt"}, "rn_pooled": {1: "p_rn"}}, "logs": {}}
        wide = {"runs": {"wt_wide": {1: "w_wt"}, "rn_wide": {1: "w_rn"}}, "logs": {}}
        committed = {"runs": {n: {1: f"l1_{n}"} for n in rc.CELLS}, "logs": {}}
        panels = rc.assemble(pooled, wide, committed)
        at = panels[rc.MATCHED]["runs"]
        assert at["wt_pooled"] == {1: "p_wt"}
        assert at["rn_wide"] == {1: "w_rn"}
        assert at["wt_wide_frozen"] == {1: "l1_wt_wide_frozen"}
        assert panels[rc.COMMITTED] is committed

    def test_strict_completeness_covers_every_cell_at_both_rates(self) -> None:
        seeds = (1, 2, 3)
        full = {"runs": {n: dict.fromkeys(seeds, "r") for n in rc.CELLS}}
        short = {"runs": {n: dict.fromkeys((1, 2), "r") for n in rc.CELLS}}
        rc.require_complete({rc.MATCHED: full, rc.COMMITTED: full}, seeds)
        with pytest.raises(ValueError, match=r"1e-3 wt_pooled seeds \[3\]"):
            rc.require_complete({rc.MATCHED: full, rc.COMMITTED: short}, seeds)
        assert rc.common_seeds({rc.MATCHED: full, rc.COMMITTED: short}, seeds) == (1, 2)


class TestTheMinimumIsHalfOfL1:
    def test_the_registered_minimum(self) -> None:
        assert pytest.approx(rc.L1_INTERACTION / 2, abs=0.0002) == rc.MIN_RETAIN
        assert rc.REGISTERED["minimum_retained"] == rc.MIN_RETAIN


class TestTheReadings:
    def _read(
        self,
        delta: float,
        q: float,
        *,
        gates_pass: bool = True,
        n: int = 96,
    ) -> dict[str, Any]:
        primary = {"interaction": {"mean_delta": delta, "q": q, "p_two_sided": q}}
        return rc.reading(_gates(passing=gates_pass), primary, n_common=n)

    def test_survives_needs_significance_and_the_minimum(self) -> None:
        out = self._read(0.20, 0.001)
        assert out["reading"] == "pool_effect_survives_the_rate"
        assert out["clears_minimum"] is True
        assert out["shrunk_below_half"] is False
        assert out["sign_flip_reproduced"] is False

    def test_significant_below_the_minimum_is_rate_specific_and_named_shrunk(self) -> None:
        out = self._read(0.08, 0.001)
        assert out["reading"] == "pool_effect_is_rate_specific"
        assert out["shrunk_below_half"] is True
        expected = 0.08 / rc.L1_INTERACTION
        assert out["fraction_of_l1_interaction_retained"] == pytest.approx(expected)

    def test_not_significant_is_rate_specific_as_a_failure_to_detect(self) -> None:
        out = self._read(0.20, 0.3)
        assert out["reading"] == "pool_effect_is_rate_specific"
        assert out["shrunk_below_half"] is False
        assert out["clears_minimum"] is True  # size carried, not the verdict

    def test_significant_negative_is_the_reverse_direction(self) -> None:
        assert self._read(-0.20, 0.001)["reading"] == "width_favours_the_shuffle_at_this_rate"

    def test_gates_are_read_first(self) -> None:
        assert self._read(0.20, 0.001, gates_pass=False)["reading"] == "no_learning"

    def test_the_seed_floor(self) -> None:
        assert self._read(0.20, 0.001, n=4)["reading"] == "insufficient_seeds"

    def test_vocabulary_is_derived_from_source(self) -> None:
        assert set(rc.READINGS) == {
            "pool_effect_survives_the_rate",
            "pool_effect_is_rate_specific",
            "width_favours_the_shuffle_at_this_rate",
            "no_learning",
            "insufficient_seeds",
        }


class TestTheContrasts:
    def test_three_way_is_committed_minus_matched_per_seed(self) -> None:
        cells = _cells(interaction_at=0.10, interaction_committed=0.30)
        out = rc.contrasts(cells, tuple(range(1, 13)))
        assert out["interaction"]["mean_delta"] == pytest.approx(0.10, abs=1e-9)
        committed = out["committed_rate_reference"]["interaction"]["mean_delta"]
        assert committed == pytest.approx(0.30, abs=1e-9)
        assert out["three_way"]["mean_delta"] == pytest.approx(0.20, abs=1e-9)
        assert out["wide_wiring_effect_at_matched_rate"]["mean_delta"] == pytest.approx(
            out["per_width_wiring_effect"]["wide"],
        )

    def test_eight_tests_in_one_family(self) -> None:
        cells = _cells(interaction_at=0.10)
        primary = rc.contrasts(cells, tuple(range(1, 13)))
        gate_result = {f"{a}_gate": {"p_improve": 0.001, "effect": 10.0} for a in rc.LEARNING_ARMS}
        fam = rc.adjust_family(primary, gate_result)
        assert fam["n_tests"] == 8
        for label in rc._FAMILY_CONTRASTS:
            assert "q" in primary[label]
        assert gate_result["gates_pass"] is True

    def test_orientation_before_agreement(self) -> None:
        primary = {"n_common": 5, "interaction": {"mean_delta": +0.2}}
        censored = {"n_common": 5, "interaction": {"mean_delta": -50.0}}
        # Lower-is-better on the censored axis: opposite raw signs AGREE.
        assert rc.metrics_agree(primary, censored, censored_hib=False) is True
        assert rc.metrics_agree(primary, censored, censored_hib=True) is False


class TestTheInstrumentsAreReadOnly:
    @pytest.mark.parametrize("module", _READ_ONLY)
    def test_unmodified_against_main(self, module: str) -> None:
        _require_revision("origin/main")
        diff = _git(["diff", "--quiet", "origin/main", "--", f"scripts/analysis/{module}"])
        assert diff.returncode in (0, 1), f"git could not compare {module}: {diff.stderr!r}"
        assert diff.returncode == 0, f"{module} differs from main; the instrument must not change"

    def test_the_helpers_are_imported_not_copied(self) -> None:
        src = (_root / "scripts" / "analysis" / "l4_rate_calibration.py").read_text()
        calls = (
            "rw.write_manifest(",
            "rw.cell_values(",
            "rw.contrasts(",
            "rw._two_sided(",
            "rw.censoring(",
            "fa.scan_rate_matched_wide(",
        )
        for call in calls:
            assert call in src, call
        names = (
            "write_manifest",
            "cell_values",
            "censoring",
            "_two_sided",
            "scan_rate_matched_wide",
        )
        for name in names:
            assert not re.search(rf"^def {name}\(", src, re.MULTILINE), name
