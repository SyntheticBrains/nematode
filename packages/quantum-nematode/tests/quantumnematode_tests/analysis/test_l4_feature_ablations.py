"""L.4 + L.5's reading: an interaction per ablation, a minimum effect that decides, read separately.

* **The minimum effect is a decision rule.** A significant interaction that removes less than
  two-thirds of the wide wiring effect reads `inconclusive_at_this_sensitivity`, never
  `carries_the_effect`. That is the gap L.1's review recorded, closed here.
* **`survives_without_it` is a failure to detect**, carried with the interaction's size and CI, and
  it also needs the ablated wiring effect itself to be significant and positive.
* **The atlas diagnostic qualifies, never rescues.** A `carries` on atlas with moved floors is
  carries-or-saturates.
* **Read per ablation.** Two different readings stay two readings.
* **The instruments are read-only.** L.1's harness produced the baseline this change reuses.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))

import l4_feature_ablations as fa  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_structural_probe as sp  # noqa: E402  # pyright: ignore[reportMissingImports]

_SEEDS = tuple(range(1, 97))
_READ_ONLY = ("l4_readout_width.py", "connectome_structure_efficiency.py", "wiring_premise.py")


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


def _cells(  # noqa: PLR0913 -- a fixture with six knobs, all named at the call site
    wt_abl: float,
    rn_abl: float,
    wt_base: float = 0.54,
    rn_base: float = 0.36,
    jitter: float = 0.01,
    ablation: str = "atlas",
) -> dict[str, dict[int, float]]:
    """Four cells with per-seed jitter that differs by cell, so contrasts have spread."""
    means = {
        f"wt_{ablation}": wt_abl,
        f"rn_{ablation}": rn_abl,
        f"wt_baseline_{ablation}": wt_base,
        f"rn_baseline_{ablation}": rn_base,
    }
    offs = {k: i + 1 for i, k in enumerate(means)}
    return {k: {s: v + jitter * ((s * offs[k]) % 7 - 3) for s in _SEEDS} for k, v in means.items()}


def _entry(  # noqa: PLR0913 -- a fixture with six knobs, all named at the call site
    inter_q: float,
    delta: float,
    abl_q: float,
    abl_delta: float,
    *,
    gates: bool = True,
    fires: bool = False,
    gains_fire: bool = False,
) -> dict[str, Any]:
    return {
        "contrasts": {
            "n_common": 96,
            "interaction": {
                "mean_delta": delta,
                "q": inter_q,
                "p_two_sided": inter_q,
                "ci_lo": delta - 0.05,
                "ci_hi": delta + 0.05,
            },
            "ablated_wiring_effect": {"mean_delta": abl_delta, "q": abl_q, "p_two_sided": abl_q},
        },
        "gates": {
            "gates_pass": gates,
            "floors_diagnostic": {"fires": fires},
            "gains_diagnostic": {"fires": gains_fire},
        },
    }


class TestTheArms:
    def test_eight_arms_four_learning_two_ablations(self) -> None:
        assert len(fa.ARMS) == 8
        assert len(fa.LEARNING_ARMS) == 4
        assert fa.ABLATIONS == ("atlas", "nogap")

    @pytest.mark.parametrize(
        ("name", "arm"),
        [
            ("..._readout_only_wide_atlas-seed3.log", "wt_atlas"),
            ("..._readout_only_wide_atlas_rewired_null-seed3.log", "rn_atlas"),
            ("..._frozen_wide_atlas-seed3.log", "wt_atlas_frozen"),
            ("..._frozen_wide_nogap_rewired_null-seed3.log", "rn_nogap_frozen"),
            ("..._readout_only_wide_nogap-seed3.log", "wt_nogap"),
        ],
    )
    def test_every_label_resolves_to_its_own_cell(self, name: str, arm: str) -> None:
        m = fa._LABEL.match(name.replace("...", fa._STEM))
        assert m is not None, name
        wiring = "rn" if m.group("rewired") else "wt"
        suffix = "" if m.group("arm") == "readout_only" else "_frozen"
        assert f"{wiring}_{m.group('abl')}{suffix}" == arm

    def test_an_l1_wide_log_is_not_an_ablation_arm(self) -> None:
        assert fa._LABEL.match(f"{fa._STEM}_readout_only_wide-seed3.log") is None

    @pytest.mark.parametrize(
        ("name", "arm"),
        [
            ("..._readout_only_wide_atlas_r1e4-seed3.log", "wt_atlas"),
            ("..._readout_only_wide_atlas_r1e4_rewired_null-seed3.log", "rn_atlas"),
        ],
    )
    def test_a_rate_tagged_atlas_arm_resolves_to_the_same_cell(self, name: str, arm: str) -> None:
        # Outcome B: the atlas learning arms run at 0.0001 under a rate tag -- same cell.
        m = fa._LABEL.match(name.replace("...", fa._STEM))
        assert m is not None
        assert m.group("rate") == "_r1e4"
        wiring = "rn" if m.group("rewired") else "wt"
        assert f"{wiring}_{m.group('abl')}" == arm

    def test_the_rate_matched_wide_label_and_nothing_else(self) -> None:
        stem = fa._STEM
        assert fa._WIDE_RATE_LABEL.match(f"{stem}_readout_only_wide_r1e4-seed3.log")
        assert fa._WIDE_RATE_LABEL.match(f"{stem}_readout_only_wide_r1e4_rewired_null-seed3.log")
        assert fa._WIDE_RATE_LABEL.match(f"{stem}_readout_only_wide-seed3.log") is None
        assert fa._WIDE_RATE_LABEL.match(f"{stem}_frozen_wide_r1e4-seed3.log") is None

    @pytest.mark.parametrize(
        "name",
        [
            "..._readout_only_wide_atlas-seed3.log",  # pre-amendment atlas learning run at 0.001
            "..._readout_only_wide_atlas_r1e2-seed3.log",  # a rate-check log
            "..._readout_only_wide_atlas_r1e2_rewired_null-seed3.log",
            "..._frozen_wide_atlas_r1e4-seed3.log",  # a tag on a floor
            "..._readout_only_wide_nogap_r1e4_rewired_null-seed3.log",  # a tag on a nogap arm
        ],
    )
    def test_scan_refuses_a_log_whose_rate_tag_does_not_fit_its_arm(
        self,
        tmp_path: Path,
        name: str,
    ) -> None:
        (tmp_path / name.replace("...", fa._STEM)).write_text("")
        with pytest.raises(ValueError, match="rate tag"):
            fa.scan(tmp_path)

    def test_scan_accepts_the_registered_tags(self, tmp_path: Path) -> None:
        # Correctly tagged files pass the tag check; an empty log is then dropped as unparseable.
        for name in (
            "_readout_only_wide_atlas_r1e4-seed3.log",
            "_readout_only_wide_atlas_r1e4_rewired_null-seed3.log",
            "_frozen_wide_atlas-seed3.log",
            "_readout_only_wide_nogap-seed3.log",
        ):
            (tmp_path / f"{fa._STEM}{name}").write_text("")
        scanned = fa.scan(tmp_path)
        assert all(runs == {} for runs in scanned["runs"].values())

    def test_strict_mode_refuses_a_baseline_missing_a_seed(self) -> None:
        seeds = (1, 2, 3)
        scanned = {"runs": {n: dict.fromkeys(seeds, "r") for n in fa.ARMS}}
        full = {"runs": {n: dict.fromkeys(seeds, "r") for n in fa.BASELINE_CELLS}}
        short = {"runs": {n: dict.fromkeys((1, 2), "r") for n in fa.BASELINE_CELLS}}
        fa.require_common_complete(scanned, {"atlas": full, "nogap": full}, seeds)
        with pytest.raises(ValueError, match="atlas baseline wt_wide seeds \\[3\\]"):
            fa.require_common_complete(scanned, {"atlas": short, "nogap": full}, seeds)
        assert fa.common_seeds(scanned, {"atlas": short, "nogap": full}, seeds) == (1, 2)

    def test_merge_baseline_takes_learning_from_the_rate_run_and_floors_from_l1(self) -> None:
        main = {
            "runs": {
                k: {1: f"main_{k}"}
                for k in ("wt_wide", "rn_wide", "wt_wide_frozen", "rn_wide_frozen")
            },
            "logs": {
                k: {1: Path(f"main_{k}")}
                for k in ("wt_wide", "rn_wide", "wt_wide_frozen", "rn_wide_frozen")
            },
        }
        learning = {"runs": {"wt_wide": {1: "rate_wt"}, "rn_wide": {1: "rate_rn"}}, "logs": {}}
        merged = fa.merge_baseline(main, learning)
        assert merged["runs"]["wt_wide"] == {1: "rate_wt"}
        assert merged["runs"]["rn_wide"] == {1: "rate_rn"}
        assert merged["runs"]["wt_wide_frozen"] == {1: "main_wt_wide_frozen"}
        assert merged["runs"]["rn_wide_frozen"] == {1: "main_rn_wide_frozen"}


class TestTheMinimumIsAgainstTheRightEffect:
    def test_two_thirds_of_the_wide_wiring_effect(self) -> None:
        assert pytest.approx(0.1852) == fa.L1_WIDE_WIRING_EFFECT
        assert pytest.approx(2 / 3 * 0.1852, abs=0.001) == fa.MIN_CARRY
        # And NOT half of L.1's +0.2818 interaction, which was the first draft's error.
        assert pytest.approx(0.14, abs=0.005) != fa.MIN_CARRY
        assert "pooled cells" in fa.REGISTERED["why_not_l1_interaction"]

    def test_the_power_at_the_minimum_is_about_eighty_percent(self) -> None:
        se = fa.L1_INTERACTION_SD / np.sqrt(96)
        assert 0.75 <= fa._power(fa.MIN_CARRY, se) <= 0.85
        # and half would not have been resolvable at this n
        assert fa._power(0.093, se) < 0.65


class TestTheReadings:
    def test_a_large_negative_interaction_carries(self) -> None:
        out = fa.reading("nogap", _entry(0.001, -0.15, 0.5, 0.03))
        assert out["reading"] == "carries_the_effect"
        assert out["clears_minimum"] is True
        assert out["qualified_carries_or_saturates"] is False
        assert out["fraction_of_wide_effect_removed"] == pytest.approx(0.15 / 0.1852)

    def test_significant_but_below_the_minimum_does_not_carry(self) -> None:
        out = fa.reading("nogap", _entry(0.001, -0.08, 0.001, 0.10))
        assert out["reading"] == "inconclusive_at_this_sensitivity"
        assert out["clears_minimum"] is False

    def test_survives_needs_the_ablated_effect_itself(self) -> None:
        out = fa.reading("nogap", _entry(0.6, -0.02, 0.001, 0.17))
        assert out["reading"] == "survives_without_it"
        # same non-significant interaction but no ablated effect either: not survives
        out = fa.reading("nogap", _entry(0.6, -0.02, 0.4, 0.03))
        assert out["reading"] == "inconclusive_at_this_sensitivity"

    def test_a_positive_interaction_amplifies(self) -> None:
        assert fa.reading("atlas", _entry(0.001, +0.12, 0.001, 0.30))["reading"] == "amplifies"

    def test_the_atlas_diagnostic_qualifies_a_carries(self) -> None:
        out = fa.reading("atlas", _entry(0.001, -0.15, 0.5, 0.03, fires=True))
        assert out["reading"] == "carries_the_effect"
        assert out["qualified_carries_or_saturates"] is True

    def test_the_diagnostic_does_not_qualify_nogap(self) -> None:
        out = fa.reading("nogap", _entry(0.001, -0.15, 0.5, 0.03, fires=True))
        assert out["qualified_carries_or_saturates"] is False

    def test_the_diagnostic_never_rescues(self) -> None:
        # fires + not carrying -> still not carrying
        assert (
            fa.reading("atlas", _entry(0.6, -0.02, 0.4, 0.03, fires=True))["reading"]
            == "inconclusive_at_this_sensitivity"
        )

    def test_the_gains_diagnostic_qualifies_a_carries_on_atlas(self) -> None:
        # Registered after the pilot: atlas arms gained +2.2/+2.8 foods over their floors against
        # the wide arms' +13.7/+7.8, bimodally, while the floors themselves did not move. A negative
        # interaction there is the wild type having more to lose, not the feature carrying it.
        out = fa.reading("atlas", _entry(0.001, -0.15, 0.5, 0.03, gains_fire=True))
        assert out["reading"] == "carries_the_effect"
        assert out["qualified_carries_or_unlearnable"] is True
        assert out["qualified_carries_or_saturates"] is False

    def test_the_gains_diagnostic_does_not_qualify_nogap(self) -> None:
        out = fa.reading("nogap", _entry(0.001, -0.15, 0.5, 0.03, gains_fire=True))
        assert out["qualified_carries_or_unlearnable"] is False

    def test_the_gains_diagnostic_never_rescues(self) -> None:
        out = fa.reading("atlas", _entry(0.6, -0.02, 0.4, 0.03, gains_fire=True))
        assert out["reading"] == "inconclusive_at_this_sensitivity"
        assert out["qualified_carries_or_unlearnable"] is False

    def test_gates_come_first(self) -> None:
        assert (
            fa.reading("atlas", _entry(0.001, -0.15, 0.5, 0.03, gates=False))["reading"]
            == "no_learning"
        )

    def test_the_reachability_floor(self) -> None:
        e = _entry(0.001, -0.15, 0.5, 0.03)
        e["contrasts"]["n_common"] = 4
        assert fa.reading("atlas", e)["reading"] == "insufficient_seeds"
        assert fa.MIN_SEEDS == 5  # 2**-5 = 0.031 <= 0.05; 2**-4 = 0.0625 is not

    def test_every_reading_has_prose(self) -> None:
        assert set(fa.READINGS) == {
            "carries_the_effect",
            "survives_without_it",
            "amplifies",
            "inconclusive_at_this_sensitivity",
            "no_learning",
            "insufficient_seeds",
        }


class TestTheInteraction:
    def test_removing_the_whole_effect_gives_minus_the_wide_effect(self) -> None:
        cells = _cells(0.45, 0.45, wt_base=0.54, rn_base=0.36, jitter=0.0)
        out = fa.contrasts(cells, "atlas", _SEEDS)
        assert out["interaction"]["mean_delta"] == pytest.approx(-(0.54 - 0.36))
        assert out["ablated_wiring_effect"]["mean_delta"] == pytest.approx(0.0)

    def test_an_intact_effect_gives_zero_interaction(self) -> None:
        cells = _cells(0.64, 0.46, wt_base=0.54, rn_base=0.36, jitter=0.0)
        assert fa.contrasts(cells, "atlas", _SEEDS)["interaction"]["mean_delta"] == pytest.approx(
            0.0,
        )

    def test_two_sided_is_two_times_min_capped(self) -> None:
        out = fa.contrasts(_cells(0.45, 0.45), "atlas", _SEEDS)["interaction"]
        expected = min(1.0, 2.0 * min(out["p_wild_type_gains_more"], out["p_null_gains_more"]))
        assert out["p_two_sided"] == pytest.approx(expected)


class TestOneFamilyReadPerAblation:
    @staticmethod
    def _per() -> dict[str, dict[str, Any]]:
        def e(inter_p: float, abl_p: float) -> dict[str, Any]:
            return {
                "contrasts": {
                    "interaction": {"p_two_sided": inter_p, "mean_delta": -0.15},
                    "ablated_wiring_effect": {"p_two_sided": abl_p, "mean_delta": 0.03},
                },
                "gates": {
                    "wt_gate": {"p_improve": 1e-6},
                    "rn_gate": {"p_improve": 1e-6},
                    "prior": {"p_two_sided": 0.4},
                    "floors_diagnostic": {"fires": False},
                    "gains_diagnostic": {"fires": False},
                },
            }

        return {"atlas": e(0.001, 0.5), "nogap": e(0.8, 0.001)}

    def test_ten_tests_get_a_q_each(self) -> None:
        per = self._per()
        fam = fa.adjust_family(per)
        assert fam["n_tests"] == 10
        for ab in fa.ABLATIONS:
            assert "q" in per[ab]["contrasts"]["interaction"]
            assert "q" in per[ab]["gates"]["prior"]
            assert per[ab]["gates"]["gates_pass"] is True

    def test_a_split_stays_a_split(self) -> None:
        per = self._per()
        fa.adjust_family(per)
        for ab in fa.ABLATIONS:
            per[ab]["contrasts"]["n_common"] = 96
            per[ab]["contrasts"]["interaction"].update({"ci_lo": -0.2, "ci_hi": -0.1})
        readings = {ab: fa.reading(ab, per[ab])["reading"] for ab in fa.ABLATIONS}
        assert readings == {"atlas": "carries_the_effect", "nogap": "survives_without_it"}


class TestTheInstrumentsAreReadOnly:
    @pytest.mark.parametrize("module", _READ_ONLY)
    def test_unmodified_against_main(self, module: str) -> None:
        _require_revision("origin/main")
        diff = _git(["diff", "--quiet", "origin/main", "--", f"scripts/analysis/{module}"])
        assert diff.returncode in (0, 1), f"git could not compare {module}: {diff.stderr!r}"
        assert diff.returncode == 0, f"{module} differs from main; the instrument must not change"

    def test_the_helpers_are_imported_not_copied(self) -> None:
        src = (_root / "scripts" / "analysis" / "l4_feature_ablations.py").read_text()
        assert "rw._two_sided(" in src
        assert "rw.censoring(" in src
        assert "rw.cell_values(" in src
        assert not re.search(r"^def _two_sided\(", src, re.MULTILINE)
        assert not re.search(r"^def censoring\(", src, re.MULTILINE)


class TestTheProbe:
    def test_jaccard_on_a_hand_built_mask(self) -> None:
        # 6 pre neurons, 4 post (pool) neurons in two classes of two.
        mask = np.zeros((6, 4), dtype=bool)
        mask[[0, 1, 2], 0] = True  # post 0 <- {0,1,2}
        mask[[0, 1, 3], 1] = True  # post 1 <- {0,1,3}   J(0,1) = 2/4 = 0.5
        mask[[4], 2] = True  # post 2 <- {4}
        mask[[5], 3] = True  # post 3 <- {5}       J(2,3) = 0
        flat = np.array([0, 1, 2, 3])
        assert sp.within_class_jaccard(mask, flat, [(0, 2), (2, 4)]) == pytest.approx([0.5, 0.0])

    def test_the_join_refuses_a_one_sided_seed(self) -> None:
        with pytest.raises(ValueError, match="one side only"):
            sp.test_probe({1: 0.1, 2: 0.2, 3: 0.3}, {1: -0.1, 2: -0.2})

    def test_the_minimum_is_applied(self) -> None:
        # a perfect but tiny-n positive relationship: significant, rho = 1 clears the minimum
        j = {s: s / 10 for s in range(1, 13)}
        d = {s: -1.0 + s / 10 for s in range(1, 13)}
        out = sp.test_probe(j, d)
        assert out["reading"] == "probe_supported"
        assert out["rho"] == pytest.approx(1.0)
        # reversed direction: one-sided positive p is ~1, so null
        assert sp.test_probe(j, {s: 1.0 - s / 10 for s in range(1, 13)})["reading"] == "probe_null"
        assert sp.MIN_RHO == 0.3

    def test_per_seed_drop_reads_l1s_committed_file(self) -> None:
        path = (
            _root
            / "docs"
            / "experiments"
            / "logbooks"
            / "supporting"
            / "066-l4-readout-width"
            / "per-seed.csv"
        )
        drop = sp.per_seed_drop(path)
        assert len(drop) == 96
        # L.1's headline: the null got WORSE at the per-neuron width, so the mean drop is negative
        assert np.mean(list(drop.values())) < 0
