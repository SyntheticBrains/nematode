"""L.0's reading, and the four ways it could say more than it measured.

* **The gates are read before the contrast.** A contrast against a null presupposes both arms
  learned, and a prior that separates between wirings would mean the rewiring changed the substrate
  before any learning did. Either makes the primary uninterpretable, so either voids it.
* **The registered bar is a fraction, not significance.** Block V's minimum is 20% off
  time-to-competence; a significant advantage below it is `below_bar` and is reported as suggestive,
  which is what V.1 and V.3 would have been held to.
* **No result here satisfies D2's primary**, which needs the wiring itself to be plastic. The
  harness carries that as a field so a positive cannot be read as converting Phase 7's SPLIT.
* **The comparator is context, not a delta.** V.3's +23.5% ran under PPO; treating the difference
  quantitatively is the cross-regime comparison the project's own commensurability rule forbids.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))

import l4_frozen_features as ff  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_mixture_statistic as ms  # noqa: E402  # pyright: ignore[reportMissingImports]


@dataclass
class _Record:
    """The fields of a scanned run this harness reads."""

    success: float
    foods: float


def _arm(foods: float, *, spread: float = 0.1) -> dict[int, _Record]:
    return {s: _Record(0.0, foods + spread * i) for i, s in enumerate(ff.SEEDS)}


def _scanned(**overrides: dict[int, _Record]) -> dict[str, Any]:
    runs = {
        "wt_learning": _arm(17.5),
        "rn_learning": _arm(17.4),
        "wt_frozen": _arm(2.3),
        "rn_frozen": _arm(2.3),
    }
    runs.update(overrides)
    return {"runs": runs, "logs": {}}


def _efficiency(delta: float, q: float, rewired_mean: float = 1165.0) -> dict[str, Any]:
    """Build a synthetic efficiency report in the committed script's shape."""
    metrics = {
        name: {
            "higher_is_better": higher,
            "wild_mean": rewired_mean - delta,
            "rewired_mean": rewired_mean,
            "mean_delta": delta,
            "bh_fdr_q": q,
            "wild_better_seeds": 24,
        }
        for name, higher in (
            ("auc_success", True),
            ("auc_foods", True),
            ("episodes_to_30pct_success", False),
            ("episodes_to_90pct_foods_plateau", False),
        )
    }
    return {"metrics": metrics, "n_paired_seeds": len(ff.SEEDS), "verdict": "x", "per_seed": {}}


def _analyse(
    scanned: dict[str, Any],
    efficiency: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Score a synthetic campaign: the committed efficiency script and the drift reader stubbed.

    Drift needs real checkpoints on disk and the substrate-frozen check has its own test. What
    these exercise is the reading -- the gates, the bar and the refusals.
    """
    monkeypatch.setattr(ff.eff, "analyse", lambda _m: efficiency)
    monkeypatch.setattr(
        ff,
        "drift",
        lambda *_a, **_k: {
            "wt_learning": {"mean_relative": 0.0, "n_read": 32, "available": True},
            "rn_learning": {"mean_relative": 0.0, "n_read": 32, "available": True},
            "drift_evidence_complete": True,
            "substrate_frozen": True,
        },
    )
    return ff.analyse(scanned, Path("unused-manifest.txt"))


class TestTheGatesAreReadFirst:
    def test_a_failed_learning_gate_voids_the_contrast(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Both arms at their floor: a contrast between two arms that did not learn is not a wiring
        # result, whatever the efficiency metrics say.
        out = _analyse(
            _scanned(wt_learning=_arm(2.3), rn_learning=_arm(2.3)),
            _efficiency(delta=300.0, q=0.001),
            monkeypatch,
        )
        assert out["verdict"] == "void"
        assert "did not learn" in out["why"]

    def test_a_separated_prior_voids_the_contrast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The rewiring changing the substrate before any learning makes the primary uninterpretable.
        out = _analyse(
            _scanned(rn_frozen=_arm(0.2)),
            _efficiency(delta=300.0, q=0.001),
            monkeypatch,
        )
        assert out["verdict"] == "void"
        assert "before learning" in out["why"]

    def test_clean_gates_let_the_contrast_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out = _analyse(_scanned(), _efficiency(delta=300.0, q=0.001), monkeypatch)
        assert out["gates"]["gates_pass"]
        assert not out["gates"]["prior_separates"]
        assert out["verdict"] != "void"

    def test_the_prior_reference_is_context_not_a_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out = _analyse(_scanned(), _efficiency(delta=300.0, q=0.001), monkeypatch)
        assert set(out["gates"]["prior_reference"]) == {"V.1 thermal", "V.3 hard_food"}
        assert "not inherited" in out["gates"]["prior_reference_note"]


class TestTheRegisteredBarIsAFraction:
    def test_clearing_it_reads_legible(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 273 of 1165 episodes is 23.4%, above the 20% minimum.
        out = _analyse(_scanned(), _efficiency(delta=273.0, q=0.003), monkeypatch)
        assert out["verdict"] == "wiring_is_legible"
        assert out["clears_registered_bar"]
        assert out["gain_fraction"] == pytest.approx(273.0 / 1165.0)

    def test_significant_but_under_the_bar_reads_below_bar(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 100 of 1165 is 8.6%: real, and not what block V registered.
        out = _analyse(_scanned(), _efficiency(delta=100.0, q=0.01), monkeypatch)
        assert out["verdict"] == "below_bar"
        assert not out["clears_registered_bar"]
        assert "suggestive" in out["why"]

    def test_no_significant_advantage_reads_inert(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out = _analyse(_scanned(), _efficiency(delta=40.0, q=0.6), monkeypatch)
        assert out["verdict"] == "wiring_is_inert_as_features"
        # Three regimes now: PPO gradient learning (034), local rules that write the wiring
        # (063), and a readout learning on frozen features (this).
        assert "third learning regime" in out["why"]

    def test_a_significant_advantage_the_wrong_way_is_not_legible(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The script orients its delta so positive means the wild type is faster; a significant
        # NEGATIVE delta is the null being faster, which is not a wiring result for the wild type.
        out = _analyse(_scanned(), _efficiency(delta=-300.0, q=0.001), monkeypatch)
        assert out["verdict"] == "wiring_is_inert_as_features"


class TestWhatTheHarnessRefusesToClaim:
    def test_d2s_primary_is_never_satisfied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for efficiency in (_efficiency(delta=400.0, q=0.001), _efficiency(delta=0.0, q=0.9)):
            out = _analyse(_scanned(), efficiency, monkeypatch)
            assert out["satisfies_d2_primary"] is False
            assert "leaves the wiring frozen" in out["d2_note"]

    def test_the_comparator_is_marked_as_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out = _analyse(_scanned(), _efficiency(delta=273.0, q=0.003), monkeypatch)
        assert out["comparator"]["gain_fraction"] == pytest.approx(0.235)
        assert "never a quantitative delta" in out["comparator"]["note"]

    def test_a_positive_is_named_a_performance_claim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out = _analyse(_scanned(), _efficiency(delta=273.0, q=0.003), monkeypatch)
        assert "PERFORMANCE claim" in out["why"]


class TestThePowerArithmetic:
    def test_it_reproduces_the_registered_figures(self) -> None:
        # The numbers the registration rests on: 12/16 needed at 16 pairs, 22/32 at 32.
        assert ff.power(16)["k_needed"] == 12
        assert ff.power(32)["k_needed"] == 22

    def test_thirty_two_pairs_beat_sixteen_against_the_comparator(self) -> None:
        # The whole reason for 32. The midpoint of V.3's observed 21/32-26/32 is 73.4%, NOT the
        # round 73% it prints as -- at 16 pairs that leaves 57.3% power, so an effect of the
        # comparator's size would have been missed more often than not caught.
        midpoint = f"{sum(ff.V3_WIN_RATE_RANGE) / 2:.0%}"
        assert ff.power(16)["power_against_comparator"][midpoint] == pytest.approx(0.573, abs=5e-3)
        assert ff.power(32)["power_against_comparator"][midpoint] == pytest.approx(0.792, abs=5e-3)

    def test_the_range_endpoints_are_the_comparator_s_own(self) -> None:
        # 21 and 26 of 32, from V.3's record; a rounded stand-in for either would move every figure
        # above, which is how the registered 63%/85% came to be wrong before this test existed.
        low, high = ff.V3_WIN_RATE_RANGE
        assert low == pytest.approx(21 / 32)
        assert high == pytest.approx(26 / 32)

    def test_it_travels_with_the_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            ff,
            "drift",
            lambda *_a, **_k: {
                "substrate_frozen": True,
                "drift_evidence_complete": True,
                "wt_learning": {},
                "rn_learning": {},
            },
        )
        out = _analyse(_scanned(), _efficiency(delta=273.0, q=0.003), monkeypatch)
        assert out["power"]["n_pairs"] == len(ff.SEEDS)
        assert "planning figures" in out["power"]["note"]


class TestTheManifest:
    def test_it_uses_the_arm_names_the_script_requires(self, tmp_path: Path) -> None:
        # `analyse` requires exactly `wild_type` and `rewired_null` and fails fast otherwise.
        logs = {
            "wt_learning": {s: _root / f"wt-{s}.log" for s in (1, 2)},
            "rn_learning": {s: _root / f"rn-{s}.log" for s in (1, 2)},
        }
        path = ff.write_manifest({"logs": logs}, tmp_path / "m.txt", seeds=(1, 2))
        arms = {line.split()[0] for line in path.read_text().splitlines()}
        assert arms == {ff.eff._WILD, ff.eff._REWIRED}

    def test_the_floors_are_not_in_it(self, tmp_path: Path) -> None:
        # The floors are the gates' business; the efficiency contrast scores the learning pair.
        logs = {
            "wt_learning": {1: _root / "wt.log"},
            "rn_learning": {1: _root / "rn.log"},
            "wt_frozen": {1: _root / "wtf.log"},
            "rn_frozen": {1: _root / "rnf.log"},
        }
        path = ff.write_manifest({"logs": logs}, tmp_path / "m.txt", seeds=(1,))
        assert len(path.read_text().strip().splitlines()) == 2

    def test_a_missing_run_refuses_to_write_an_unpaired_manifest(self, tmp_path: Path) -> None:
        logs = {"wt_learning": {1: _root / "wt.log"}, "rn_learning": {}}
        with pytest.raises(ValueError, match="unpaired"):
            ff.write_manifest({"logs": logs}, tmp_path / "m.txt", seeds=(1,))


class TestAnIncompleteCampaignIsNotScored:
    def test_a_complete_campaign_passes(self) -> None:
        ff.require_complete(_scanned())

    def test_a_missing_run_refuses_a_verdict(self) -> None:
        scanned = _scanned()
        del scanned["runs"]["rn_learning"][ff.SEEDS[-1]]
        with pytest.raises(ValueError, match="campaign is incomplete"):
            ff.require_complete(scanned)

    def test_the_registered_seed_count_is_thirty_two(self) -> None:
        assert tuple(range(1, 33)) == ff.SEEDS


class TestTheLabelReadsOneArmEach:
    @pytest.mark.parametrize(
        ("suffix", "expected"),
        [
            ("readout_only", "wt_learning"),
            ("readout_only_rewired_null", "rn_learning"),
            ("frozen", "wt_frozen"),
            ("frozen_rewired_null", "rn_frozen"),
        ],
    )
    def test_each_arm_label_matches_itself(self, suffix: str, expected: str) -> None:
        match = ff._LABEL.match(f"{ff._STEM}_{suffix}-seed7.log")
        assert match is not None
        prefix = "rn" if match.group("rewired") else "wt"
        kind = "learning" if match.group("arm") == "readout_only" else "frozen"
        assert f"{prefix}_{kind}" == expected
        assert match.group("seed") == "7"

    def test_r2s_other_routings_are_not_this_comparison(self) -> None:
        # R.2's campaign directory carries symmetric, random, scalar and plastic_readout; none of
        # them is an arm here.
        for other in ("symmetric", "random_motor", "random", "scalar", "plastic_readout"):
            assert ff._LABEL.match(f"{ff._STEM}_{other}-seed1.log") is None


class TestAPilotIsNotAVerdict:
    """Below about five pairs no gate can pass, so no gate failing is informative."""

    @staticmethod
    def _four_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ff, "SEEDS", (101, 102, 103, 104))

    def test_the_verdict_is_withheld(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._four_seeds(monkeypatch)
        monkeypatch.setattr(
            ff,
            "drift",
            lambda *_a, **_k: {
                "substrate_frozen": True,
                "drift_evidence_complete": True,
                "wt_learning": {},
                "rn_learning": {},
            },
        )
        monkeypatch.setattr(ff.eff, "analyse", lambda _m: _efficiency(delta=400.0, q=0.001))
        out = ff.analyse(_scanned(), Path("m.txt"), seeds=(101, 102, 103, 104))
        assert out["verdict"] == "pilot"
        assert "not a registered verdict" in out["why"]

    def test_it_does_not_report_a_gate_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The defect this fixes: every gate reads as failed because none COULD pass, and the harness
        # announced `void - a learning gate failed` from that. R.2's harness had the same bug in a
        # more dangerous form, printing `does_not_learn - the programme stops`.
        self._four_seeds(monkeypatch)
        monkeypatch.setattr(
            ff,
            "drift",
            lambda *_a, **_k: {
                "substrate_frozen": True,
                "drift_evidence_complete": True,
                "wt_learning": {},
                "rn_learning": {},
            },
        )
        monkeypatch.setattr(ff.eff, "analyse", lambda _m: _efficiency(delta=400.0, q=0.001))
        out = ff.analyse(_scanned(), Path("m.txt"), seeds=(101, 102, 103, 104))
        assert out["verdict"] != "void"
        assert "a learning gate failed" not in out["why"]

    def test_the_power_floor_is_reported_rather_than_raised(self) -> None:
        # `next()` over an empty range raised StopIteration before this: at four pairs the smallest
        # one-sided p is 2**-4 = 0.0625, above the gate, so no k qualifies.
        floor = ff.power(4)
        assert floor["gate_reachable"] is False
        assert floor["k_needed"] is None
        assert floor["smallest_reachable_p"] == pytest.approx(0.0625)

    def test_the_registered_count_reaches_the_gate(self) -> None:
        assert ff.power(len(ff.SEEDS))["gate_reachable"] is True


class TestTheSubstrateMustStayFrozen:
    """If the substrate moved, this is not a fixed-features contrast at all."""

    @staticmethod
    def _drift(monkeypatch: pytest.MonkeyPatch, **over: object) -> None:
        base = {
            "wt_learning": {"mean_relative": 0.0, "n_read": 32, "available": True},
            "rn_learning": {"mean_relative": 0.0, "n_read": 32, "available": True},
            "drift_evidence_complete": True,
            "substrate_frozen": True,
        }
        base.update(over)
        monkeypatch.setattr(ff, "drift", lambda *_a, **_k: base)

    def test_nonzero_drift_voids_the_contrast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._drift(monkeypatch, substrate_frozen=False)
        monkeypatch.setattr(ff.eff, "analyse", lambda _m: _efficiency(delta=400.0, q=0.001))
        out = ff.analyse(_scanned(), Path("m.txt"))
        assert out["verdict"] == "void"
        assert "did not stay frozen" in out["why"]

    def test_incomplete_evidence_voids_and_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Distinguished from actual drift: "we could not check" is not "it stayed put".
        self._drift(monkeypatch, substrate_frozen=False, drift_evidence_complete=False)
        monkeypatch.setattr(ff.eff, "analyse", lambda _m: _efficiency(delta=400.0, q=0.001))
        out = ff.analyse(_scanned(), Path("m.txt"))
        assert out["verdict"] == "void"
        assert "incomplete" in out["why"]

    def test_completeness_needs_every_requested_seed(self) -> None:
        # With one arm's checkpoints missing, `all()` over a short list would have reported the
        # substrate frozen on the strength of whatever happened to be on disk.
        out = ff.drift({"logs": {}}, seeds=(1, 2, 3))
        assert out["drift_evidence_complete"] is False
        assert out["substrate_frozen"] is False


class TestThePriorIsHeldToTheFamily:
    """The prior can void the campaign, so its threshold has to be a real corrected q."""

    @staticmethod
    def _prior_q(wt_frozen: float, rn_frozen: float) -> dict[str, Any]:
        scanned = _scanned(wt_frozen=_arm(wt_frozen), rn_frozen=_arm(rn_frozen))
        return ff.gates(scanned)

    @staticmethod
    def _prior_split(wild_better: int) -> dict[str, Any]:
        """Build a prior where exactly ``wild_better`` of the 32 seeds favour the wild type.

        ``_arm`` cannot produce a marginal case -- its deltas are perfectly consistent in sign, so
        the test is either p = 1 or p ~ 0. Mixing the signs is what reaches the band where the raw
        one-sided minimum and the corrected two-sided q disagree.
        """
        n = len(ff.SEEDS)
        deltas = [1.0] * wild_better + [-1.0] * (n - wild_better)
        wt = {s: _Record(0.0, 3.0) for s in ff.SEEDS}
        rn = {s: _Record(0.0, 3.0 - d) for s, d in zip(ff.SEEDS, deltas, strict=True)}
        return ff.gates(_scanned(wt_frozen=wt, rn_frozen=rn))

    def test_a_two_sided_p_is_recorded_not_a_bare_minimum(self) -> None:
        # `min(p_improve, p_degrade)` is not a p-value: it doubles the type-I rate. The recorded
        # two-sided p is 2 * min, capped at 1.
        checks = self._prior_q(2.3, 2.3)
        prior = checks["prior"]
        expected = min(1.0, 2.0 * min(prior["p_improve"], prior["p_degrade"]))
        assert prior["p_two_sided"] == pytest.approx(expected)

    def test_separation_is_read_off_the_corrected_q(self) -> None:
        checks = self._prior_q(2.3, 2.3)
        assert checks["prior_separates"] is (checks["prior"]["q_improve"] <= ms.SIG_Q)

    def test_a_marginal_prior_the_raw_check_would_have_flagged_is_not_flagged(self) -> None:
        # The fix's effect, on the exact band where it bites: at 21 of 32 the raw one-sided minimum
        # is 0.0385 and clears the gate, while the corrected two-sided q is 0.0771 and does not.
        # Under the old `min(...)` check this VOIDED the campaign.
        checks = self._prior_split(wild_better=21)
        raw = min(checks["prior"]["p_improve"], checks["prior"]["p_degrade"])
        assert raw <= ms.SIG_Q, "fixture no longer exercises the case it was built for"
        assert checks["prior"]["q_improve"] > ms.SIG_Q
        assert checks["prior_separates"] is False

    def test_a_genuinely_separating_prior_still_separates(self) -> None:
        # The fix must not blunt the check it corrects: one seed further on, it fires.
        checks = self._prior_split(wild_better=22)
        assert checks["prior"]["q_improve"] <= ms.SIG_Q
        assert checks["prior_separates"] is True

    def test_a_wholly_separated_prior_separates(self) -> None:
        assert self._prior_q(2.3, 12.0)["prior_separates"] is True


class TestThePowerFiguresAreLabelledAsPlanning:
    def test_the_note_disclaims_the_registered_procedure(self) -> None:
        # They size the seed count. They are not the power of a paired rank test corrected across
        # four metrics, which differs in both directions and needs an explicit alternative.
        note = ff.power(32)["note"]
        assert "planning" in note
        assert "BH-FDR" in note
