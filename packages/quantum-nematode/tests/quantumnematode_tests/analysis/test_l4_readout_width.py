"""L.1's reading: an interaction, gates before it, and a metric chosen for the contrast.

* **The primary is the INTERACTION, not a main effect.** A per-neuron readout has 78 parameters
  against 8, so a width gain read alone cannot separate "the pool hid the wiring's features" from
  "more parameters learn faster". Only `(wt_wide - wt_pooled) - (rn_wide - rn_pooled)` separates
  them, and the main effects must never be reported in its place.
* **Gates before the interaction.** An interaction between arms that did not learn is
  uninterpretable, so a failed gate returns `no_learning` whatever the interaction says.
* **`auc_success` is the primary, not the censored metric.** L.0 met asymmetric censoring on this
  cell; a difference of differences cannot be read on a metric whose censoring rate varies across
  the cells being differenced.
* **A null states its sensitivity.** The interaction's spread depends on a correlation the
  registration could not know, so the realised figure is computed and the correlation reported.
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

import connectome_structure_efficiency as eff  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_readout_width as rw  # noqa: E402  # pyright: ignore[reportMissingImports]

_SEEDS = tuple(range(1, 97))


def _cells(wt_pooled: float, rn_pooled: float, wt_wide: float, rn_wide: float, jitter: float = 0.0):
    """Four cells with a per-seed jitter, so the contrasts have a spread to test.

    Each cell gets its OWN jitter pattern: a shared one cancels in every difference, leaving the
    wiring differences constant across seeds and their correlation undefined.
    """
    offsets = {"wt_pooled": 1, "rn_pooled": 2, "wt_wide": 3, "rn_wide": 4}
    means = {
        "wt_pooled": wt_pooled,
        "rn_pooled": rn_pooled,
        "wt_wide": wt_wide,
        "rn_wide": rn_wide,
    }
    return {
        cell: {s: means[cell] + jitter * ((s * offsets[cell]) % 7 - 3) for s in _SEEDS}
        for cell in means
    }


class TestTheArmsAndTheirLabels:
    def test_there_are_eight_arms_four_of_them_floors(self) -> None:
        assert len(rw.ARMS) == 8
        assert len(rw.LEARNING_ARMS) == 4
        assert sum(1 for m in rw.ARMS.values() if not m["learns"]) == 4

    def test_each_width_has_its_own_floors(self) -> None:
        # Not shared across widths: the two widths are the same policy but not the same run.
        for width in rw.WIDTHS:
            for wiring in rw.WIRINGS:
                assert f"{wiring}_{width}_frozen" in rw.ARMS

    def test_only_the_learning_arms_enter_the_efficiency_contrast(self) -> None:
        for name, meta in rw.ARMS.items():
            assert (meta["eff"] is not None) == meta["learns"], name

    @pytest.mark.parametrize(
        ("name", "arm"),
        [
            ("...readout_only-seed7.log", "wt_pooled"),
            ("...readout_only_rewired_null-seed7.log", "rn_pooled"),
            ("...readout_only_wide-seed7.log", "wt_wide"),
            ("...readout_only_wide_rewired_null-seed7.log", "rn_wide"),
            ("...frozen-seed7.log", "wt_pooled_frozen"),
            ("...frozen_rewired_null-seed7.log", "rn_pooled_frozen"),
            ("...frozen_wide-seed7.log", "wt_wide_frozen"),
            ("...frozen_wide_rewired_null-seed7.log", "rn_wide_frozen"),
        ],
    )
    def test_every_arm_label_resolves_to_its_own_cell(self, name: str, arm: str) -> None:
        # A mis-keyed label would silently put a run in the wrong cell of the 2x2, which moves the
        # difference of differences without anything saying so.
        match = rw._LABEL.match(name.replace("...", f"{rw._STEM}_"))
        assert match is not None, name
        wiring = "rn" if match.group("rewired") else "wt"
        width = "wide" if match.group("wide") else "pooled"
        suffix = "" if match.group("arm") == "readout_only" else "_frozen"
        assert f"{wiring}_{width}{suffix}" == arm


class TestTheInteractionIsThePrimary:
    def test_a_pure_width_effect_produces_no_interaction(self) -> None:
        # Widening helps both wirings equally: more parameters help, wiring-blind.
        result = rw.contrasts(_cells(0.40, 0.40, 0.60, 0.60), _SEEDS)
        assert result["interaction"]["mean_delta"] == pytest.approx(0.0)
        assert result["width_main_effect"]["mean_delta"] == pytest.approx(0.20)

    def test_widening_helping_the_wild_type_more_is_a_positive_interaction(self) -> None:
        result = rw.contrasts(_cells(0.40, 0.40, 0.70, 0.50), _SEEDS)
        assert result["interaction"]["mean_delta"] == pytest.approx(0.20)
        assert (
            result["interaction"]["p_wild_type_gains_more"]
            < result["interaction"]["p_null_gains_more"]
        )

    def test_widening_helping_the_null_more_is_a_negative_interaction(self) -> None:
        result = rw.contrasts(_cells(0.40, 0.40, 0.50, 0.70), _SEEDS)
        assert result["interaction"]["mean_delta"] == pytest.approx(-0.20)

    def test_a_pure_wiring_effect_at_both_widths_produces_no_interaction(self) -> None:
        # L.0's situation carried to both widths: a wiring gap that widening does not change.
        result = rw.contrasts(_cells(0.50, 0.40, 0.70, 0.60), _SEEDS)
        assert result["interaction"]["mean_delta"] == pytest.approx(0.0)
        assert result["wiring_main_effect"]["mean_delta"] == pytest.approx(0.10)

    def test_the_main_effects_are_labelled_secondary(self) -> None:
        result = rw.contrasts(_cells(0.40, 0.40, 0.70, 0.50), _SEEDS)
        assert result["interaction"]["role"] == "primary"
        assert result["width_main_effect"]["role"] == "secondary"
        assert result["wiring_main_effect"]["role"] == "secondary"


class TestTheTwoSidedP:
    def test_it_is_twice_the_smaller_one_sided_and_never_exceeds_one(self) -> None:
        # `min(p_up, p_down)` is not a p-value and doubles the type-I rate -- the defect PR #375
        # caught in L.0's prior check, which the two registered directions here would repeat.
        for cells in (_cells(0.4, 0.4, 0.7, 0.5, 0.01), _cells(0.4, 0.4, 0.4, 0.4, 0.01)):
            inter = rw.contrasts(cells, _SEEDS)["interaction"]
            expected = min(
                1.0,
                2.0 * min(inter["p_wild_type_gains_more"], inter["p_null_gains_more"]),
            )
            assert inter["p_two_sided"] == pytest.approx(expected)
            assert 0.0 <= inter["p_two_sided"] <= 1.0


class TestTheGatesAreReadFirst:
    @staticmethod
    def _gates(*, pass_gates: bool) -> dict[str, Any]:
        return {"gates_pass": pass_gates}

    def test_a_failed_gate_returns_no_learning_whatever_the_interaction_says(self) -> None:
        loud = rw.contrasts(_cells(0.40, 0.40, 0.90, 0.40), _SEEDS)
        out = rw.reading(self._gates(pass_gates=False), loud, n_common=96)
        assert out["reading"] == "no_learning"
        assert out["reopens_l4_l5"] is False

    def test_too_few_seeds_outranks_even_a_failed_gate(self) -> None:
        loud = rw.contrasts(_cells(0.40, 0.40, 0.90, 0.40), _SEEDS)
        out = rw.reading(self._gates(pass_gates=False), loud, n_common=2)
        assert out["reading"] == "insufficient_seeds"


class TestTheReadings:
    @staticmethod
    def _read(cells: dict[str, dict[int, float]]) -> dict[str, Any]:
        return rw.reading(
            {"gates_pass": True},
            rw.contrasts(cells, _SEEDS),
            n_common=len(_SEEDS),
        )

    def test_a_clear_wild_type_gain_reopens_l4_and_l5(self) -> None:
        out = self._read(_cells(0.40, 0.40, 0.70, 0.50, 0.005))
        assert out["reading"] == "pooling_hid_structure"
        assert out["reopens_l4_l5"] is True

    def test_a_clear_null_gain_does_not(self) -> None:
        out = self._read(_cells(0.40, 0.40, 0.50, 0.70, 0.005))
        assert out["reading"] == "width_favours_the_shuffle"
        assert out["reopens_l4_l5"] is False
        assert "not explained" in out["why"]

    def test_no_interaction_retires_the_objection_at_a_stated_sensitivity(self) -> None:
        out = self._read(_cells(0.40, 0.40, 0.60, 0.60, 0.005))
        assert out["reading"] == "pooling_was_not_the_limit"
        assert out["reopens_l4_l5"] is False
        assert "sensitivity" in out["why"]

    def test_every_reading_has_registered_prose(self) -> None:
        assert set(rw.READINGS) == {
            "pooling_hid_structure",
            "width_favours_the_shuffle",
            "pooling_was_not_the_limit",
            "no_learning",
            "insufficient_seeds",
        }


class TestTheMetricChoice:
    def test_the_primary_is_not_the_censored_metric(self) -> None:
        assert rw.PRIMARY_METRIC == "auc_success"
        assert rw.CENSORED_METRIC == "episodes_to_30pct_success"
        assert rw.PRIMARY_METRIC != rw.CENSORED_METRIC

    def test_the_reason_travels_with_the_choice(self) -> None:
        # A departure from a committed instrument's metric that does not carry its reason is a
        # metric change nobody can audit later.
        assert "censoring" in rw.METRIC_NOTE
        assert "per cell" in rw.METRIC_NOTE.lower()

    def test_the_two_metrics_point_opposite_ways(self) -> None:
        assert eff._METRICS[rw.PRIMARY_METRIC] is True
        assert eff._METRICS[rw.CENSORED_METRIC] is False

    @staticmethod
    def _analysed(
        censored_wt_wide: float,
        monkeypatch: pytest.MonkeyPatch,
    ) -> dict[str, Any]:
        """Run `analyse` end to end with a controllable censored-metric interaction.

        The primary is held at a POSITIVE interaction (the wild type gains from widening) while the
        censored metric's `wt_wide` cell moves, which is what flips its interaction's raw sign.
        """
        primary = _cells(0.40, 0.40, 0.70, 0.50)
        censored = _cells(1000.0, 1000.0, censored_wt_wide, 900.0)
        seeds = [str(s) for s in _SEEDS]

        def fake(_cells_in: object, out: dict, _manifest: object = None) -> None:
            out["verdicts"] = {}
            out["efficiency"] = {}
            for width in rw.WIDTHS:
                out["efficiency"][width] = {
                    "horizon_episodes": 3000,
                    "n_paired_seeds": len(_SEEDS),
                    "metrics": {
                        rw.PRIMARY_METRIC: {"higher_is_better": True},
                        rw.CENSORED_METRIC: {"higher_is_better": False},
                    },
                    "per_seed": {
                        arm: {
                            s: {
                                rw.PRIMARY_METRIC: primary[f"{w}_{width}"][int(s)],
                                rw.CENSORED_METRIC: censored[f"{w}_{width}"][int(s)],
                            }
                            for s in seeds
                        }
                        for w, arm in (("wt", eff._WILD), ("rn", eff._REWIRED))
                    },
                }

        monkeypatch.setattr(rw, "scan", lambda *_a, **_k: {"runs": {}, "logs": {}})
        monkeypatch.setattr(rw, "common_seeds", lambda *_a, **_k: _SEEDS)
        monkeypatch.setattr(
            rw,
            "efficiency",
            lambda *_a, **_k: (fake(None, (d := {})) or d)["efficiency"],
        )
        monkeypatch.setattr(
            rw,
            "gates",
            lambda *_a, **_k: {
                "gates_pass": True,
                **{f"{a}_gate": {"p_improve": 0.001, "effect": 5.0} for a in rw.LEARNING_ARMS},
                **{f"prior_{w}": {"p_two_sided": 0.5, "effect": 0.0} for w in rw.WIDTHS},
            },
        )
        return rw.analyse(Path("unused"))

    def test_opposite_raw_signs_agree_once_oriented(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Primary interaction positive; censored `wt_wide` drops far, so its interaction is
        # negative -- and negative on a lower-is-better metric means the SAME thing.
        out = self._analysed(200.0, monkeypatch)
        assert out["primary"]["interaction"]["mean_delta"] > 0
        assert out["censored_axis"]["interaction"]["mean_delta"] < 0
        assert out["metrics_agree_in_direction"] is True
        assert out["metric_disagreement_note"] is None

    def test_equal_raw_signs_disagree_once_oriented(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Censored `wt_wide` rises, so its interaction is positive -- on a lower-is-better metric
        # that means the wild type got WORSE, opposing the primary.
        out = self._analysed(1400.0, monkeypatch)
        assert out["primary"]["interaction"]["mean_delta"] > 0
        assert out["censored_axis"]["interaction"]["mean_delta"] > 0
        assert out["metrics_agree_in_direction"] is False
        assert "disagree" in out["metric_disagreement_note"]

    def test_censoring_is_counted_per_cell_and_not_pooled(self) -> None:
        reports = {
            "pooled": {
                "horizon_episodes": 3000,
                "per_seed": {
                    eff._WILD: {str(s): {"episodes_to_30pct_success": 3000} for s in range(8)},
                    eff._REWIRED: {str(s): {"episodes_to_30pct_success": 500} for s in range(8)},
                },
            },
            "wide": {
                "horizon_episodes": 3000,
                "per_seed": {
                    arm: {str(s): {"episodes_to_30pct_success": 400} for s in range(8)}
                    for arm in (eff._WILD, eff._REWIRED)
                },
            },
        }
        out = rw.censoring(reports)
        assert out["wt_pooled"]["n_censored"] == 8
        assert out["rn_pooled"]["n_censored"] == 0
        assert out["wt_wide"]["n_censored"] == 0
        assert out["asymmetric"] is True
        assert out["max_rate_difference"] == pytest.approx(1.0)


class TestTheSensitivity:
    def test_the_panel_is_sized_to_the_sign_flip_threshold(self) -> None:
        # Not an arbitrary target: L.0 found the null ahead by 0.1076, so an interaction must exceed
        # that to flip the sign and mean the pool hid wiring structure. A smaller one
        # changes no verdict and reopens nothing.
        registered = rw.REGISTERED_SENSITIVITY
        assert registered["n_registered"] == 96
        # 0.1077 against 0.1076: the panel SITS AT the threshold rather than clearing it, so it is
        # at 80% power for exactly the sign-flipping effect and below that for anything smaller.
        assert registered["detectable_at_80_at_n96"] == pytest.approx(
            registered["minimum_interesting_interaction"],
            rel=0.01,
        )
        assert registered["sits_at_threshold_not_below"] is True
        assert registered["detectable_at_80_at_n32"] > registered["minimum_interesting_interaction"]

    def test_the_sizing_came_from_the_pilot_not_an_assumption(self) -> None:
        # The first registration assumed the widths' wiring differences would correlate. The pilot
        # measured rho = +0.02, close to independent, which is why the panel grew from 32 to 96.
        registered = rw.REGISTERED_SENSITIVITY
        assert registered["pilot_measured_rho"] == pytest.approx(0.02)
        assert registered["pilot_realised_interaction_sd"] == pytest.approx(0.3770)
        assert "pilot" in registered["note"]

    def test_the_realised_figure_comes_from_the_panels_own_deltas(self) -> None:
        cells = _cells(0.40, 0.40, 0.70, 0.50, 0.01)
        result = rw.contrasts(cells, _SEEDS)
        out = rw.sensitivity(result["interaction"]["per_seed_deltas"], result["n_common"], cells)
        assert out["available"] is True
        assert out["n_pairs"] == len(_SEEDS)
        assert out["detectable_at_80_percent"] > 0.0
        assert out["registered"]["detectable_at_80_at_n96"] == pytest.approx(0.1077)

    def test_the_width_correlation_is_reported_beside_it(self) -> None:
        # The registered bound assumed zero; the realised figure depends on it, so it is reported
        # rather than left for a reader to infer from the numbers.
        cells = _cells(0.40, 0.40, 0.70, 0.50, 0.01)
        result = rw.contrasts(cells, _SEEDS)
        out = rw.sensitivity(result["interaction"]["per_seed_deltas"], result["n_common"], cells)
        assert "width_difference_correlation" in out
        assert -1.0 <= out["width_difference_correlation"] <= 1.0

    def test_a_panel_too_small_for_a_spread_says_so(self) -> None:
        out = rw.sensitivity([], 0)
        assert out["available"] is False
        assert out["registered"]["n_registered"] == 96


class TestCompleteness:
    def test_an_incomplete_panel_refuses_a_reading(self) -> None:
        scanned = {"runs": {name: dict.fromkeys(range(1, 90), object()) for name in rw.ARMS}}
        with pytest.raises(ValueError, match="incomplete"):
            rw.require_complete(scanned, _SEEDS)

    def test_a_complete_panel_passes(self) -> None:
        scanned = {"runs": {name: dict.fromkeys(_SEEDS, object()) for name in rw.ARMS}}
        rw.require_complete(scanned, _SEEDS)
