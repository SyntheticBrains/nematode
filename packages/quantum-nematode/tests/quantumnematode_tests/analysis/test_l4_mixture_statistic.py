"""A contrast family matched to a bimodal outcome.

The failure these pin is a statistic that reads one component of a mixture and calls the other no
effect: five panels ran a paired rank test against an outcome where arms can differ in how often
they reach a competent policy or in how good it is when they do. The central risk in the family
itself is the reverse -- a level contrast that is really a frequency contrast in disguise, which
would make the map's central distinction unreachable -- so that is tested directly.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))

import l4_mixture_statistic as ms  # noqa: E402  # pyright: ignore[reportMissingImports]

_HIGH = 40.0  # comfortably competent
_LOW = 3.0  # comfortably dead


def _arm(competent: dict[int, float], dead: tuple[int, ...]) -> dict[int, float]:
    """Build an arm from its competent seeds and its dead ones."""
    return {**competent, **dict.fromkeys(dead, _LOW)}


class TestTheCommittedThresholdIsUsed:
    def test_it_is_imported_not_restated(self) -> None:
        from l4_panel2 import COMPETENT_THRESHOLD  # pyright: ignore[reportMissingImports]

        assert ms.COMPETENT_THRESHOLD is COMPETENT_THRESHOLD
        assert ms.COMPETENT_THRESHOLD == 20.0

    def test_the_significance_level_matches_the_registered_family(self) -> None:
        assert ms.SIG_Q == 0.05


class TestTheLevelContrastIsNotFrequencyInDisguise:
    """The defect that would collapse the map: L firing because B had fewer competent seeds."""

    def test_a_seed_competent_in_one_arm_enters_only_that_arms_level(self) -> None:
        a = _arm({1: 50.0, 2: 50.0}, (3, 4))
        b = _arm({1: 50.0}, (2, 3, 4))
        out = ms.level_contrast(a, b)
        assert out["a_n"] == 2
        assert out["b_n"] == 1
        # Seed 2 raised A's count, not B's level: the means are equal, so L sees nothing.
        assert out["effect"] == pytest.approx(0.0)

    def test_a_pure_frequency_difference_leaves_the_level_contrast_flat(self) -> None:
        # Every competent seed scores the same; A simply has more of them. A paired contrast over
        # the union of competent seeds would report a large positive effect here.
        a = _arm(dict.fromkeys(range(1, 9), _HIGH), tuple(range(9, 17)))
        b = _arm(dict.fromkeys(range(1, 3), _HIGH), tuple(range(3, 17)))
        assert ms.level_contrast(a, b)["effect"] == pytest.approx(0.0)

    def test_a_pure_level_difference_is_reported(self) -> None:
        a = _arm({1: 70.0, 2: 80.0, 3: 60.0}, (4, 5, 6, 7, 8))
        b = _arm({1: 30.0, 2: 25.0, 3: 35.0}, (4, 5, 6, 7, 8))
        out = ms.level_contrast(a, b)
        assert out["effect"] == pytest.approx(70.0 - 30.0)
        assert out["defined"] is True


class TestAnUndefinedLevelContrastIsNotANull:
    def test_it_is_undefined_when_one_arm_has_no_competent_seed(self) -> None:
        a = _arm({1: 50.0}, (2, 3, 4))
        b = _arm({}, (1, 2, 3, 4))
        out = ms.level_contrast(a, b)
        assert out["defined"] is False
        assert "no competent seed" in out["reason"]

    def test_an_undefined_member_does_not_enter_the_correction(self) -> None:
        # Entering it as p = 1.0 would shrink every other member's q by a member that measured
        # nothing.
        a = _arm({1: 50.0}, (2, 3, 4))
        b = _arm({}, (1, 2, 3, 4))
        out = ms.family(a, b)
        assert out["members"]["L"]["defined"] is False
        assert out["directions"]["L"] == "0"

    def test_it_reads_as_zero_in_the_map(self) -> None:
        a = _arm({1: 50.0}, (2, 3, 4))
        b = _arm({}, (1, 2, 3, 4))
        assert ms.read(a, b)["verdict"] in {"no_effect", "mixed_response", "frequency_only"}


class TestEveryCellOfTheDirectionTableIsNamed:
    def test_the_map_covers_every_combination(self) -> None:
        combinations = {(f, level) for f in "+-0" for level in "+-0"}
        # (0, 0) is resolved by the split test rather than the table.
        assert set(ms._MAP) | {("0", "0")} == combinations

    @pytest.mark.parametrize(
        ("directions", "expected"),
        [
            (("+", "+"), "shift"),
            (("-", "-"), "degrades"),
            (("+", "-"), "mixed_response"),
            (("-", "+"), "mixed_response"),
            (("0", "+"), "level_only"),
            (("+", "0"), "frequency_only"),
            (("0", "-"), "degrades"),
            (("-", "0"), "degrades"),
        ],
    )
    def test_each_cell_maps_as_registered(
        self,
        directions: tuple[str, str],
        expected: str,
    ) -> None:
        f, level = directions
        result = {
            "directions": {"F": f, "L": level, "W": "0"},
            "split": {"is_split": False},
        }
        assert ms.verdict(result) == expected

    def test_opposed_significant_contrasts_are_mixed_not_degrades(self) -> None:
        # Fewer seeds competent, the competent ones better: the strongest form of a split, and
        # the cell that would otherwise fall through to degradation.
        result = {"directions": {"F": "-", "L": "+", "W": "0"}, "split": {"is_split": False}}
        assert ms.verdict(result) == "mixed_response"

    def test_every_verdict_has_a_licence_recorded(self) -> None:
        assert set(ms.LICENSES) >= set(ms._MAP.values()) | {"no_effect"}
        assert ms.LICENSES["mixed_response"].startswith("nothing")


class TestTheSplitIsDescriptiveOnly:
    """It counts seeds moved each way; it does not decide anything."""

    def test_a_doubly_non_significant_panel_is_no_effect_however_it_is_split(self) -> None:
        # The null here is itself bimodal, so two arms drawn from one law almost always put some
        # seed high in one and low in the other. Treating that as a finding would name noise.
        for is_split in (True, False):
            result = {"directions": {"F": "0", "L": "0", "W": "0"}, "split": {"is_split": is_split}}
            assert ms.verdict(result) == "no_effect"

    def test_the_split_statistic_is_not_specific_under_a_bimodal_null(self) -> None:
        # The measurement that demoted it, at the size it was registered for.
        import numpy as np

        rng = np.random.default_rng(0)

        def draw(n: int) -> dict[int, float]:
            return {
                s: (rng.uniform(30, 70) if rng.random() < 0.4 else rng.uniform(0, 8))
                for s in range(n)
            }

        fires = sum(ms._split(draw(8), draw(8))["is_split"] for _ in range(400))
        # Both arms come from the same law, so every one of these is a false positive.
        assert fires / 400 > 0.5

    def test_missing_significance_alone_is_no_effect(self) -> None:
        result = {"directions": {"F": "0", "L": "0", "W": "0"}, "split": {"is_split": False}}
        assert ms.verdict(result) == "no_effect"

    def test_a_split_needs_both_an_improved_and_a_degraded_seed(self) -> None:
        improved_only = ms._split({1: 50.0, 2: 5.0}, {1: 5.0, 2: 5.0})
        assert improved_only["is_split"] is False
        both = ms._split({1: 50.0, 2: 5.0}, {1: 5.0, 2: 50.0})
        assert both["is_split"] is True

    def test_movement_inside_the_band_is_not_a_split(self) -> None:
        # An arm jittering around its comparator must not read as improving and degrading.
        jitter = ms._split({1: 45.0, 2: 35.0}, {1: 40.0, 2: 40.0})
        assert jitter["is_split"] is False


class TestTheFamilyOnConstructedPanels:
    def test_a_level_only_panel_is_not_no_effect(self) -> None:
        # Five competent seeds a side: at three a side the pool has only twenty splits, so no
        # distribution-free test can reach significance and the verdict is unreachable by
        # arithmetic rather than by evidence.
        a = _arm({1: 60.0, 2: 70.0, 3: 80.0, 4: 65.0, 5: 75.0}, (6, 7, 8))
        b = _arm({1: 30.0, 2: 25.0, 3: 35.0, 4: 28.0, 5: 32.0}, (6, 7, 8))
        assert ms.read(a, b)["verdict"] == "level_only"

    def test_the_same_panel_at_three_a_side_cannot_reach_it(self) -> None:
        a = _arm({1: 60.0, 2: 70.0, 3: 80.0}, (4, 5, 6, 7, 8))
        b = _arm({1: 30.0, 2: 25.0, 3: 35.0}, (4, 5, 6, 7, 8))
        out = ms.read(a, b)
        assert out["members"]["L"]["effect"] == pytest.approx(40.0)
        assert out["verdict"] == "no_effect"

    def test_a_frequency_only_panel_is_not_level_only(self) -> None:
        # Six discordant pairs: what the registered exact binomial needs to reach q <= 0.05.
        a = _arm(dict.fromkeys(range(1, 9), _HIGH), tuple(range(9, 17)))
        b = _arm(dict.fromkeys(range(1, 3), _HIGH), tuple(range(3, 17)))
        out = ms.read(a, b)
        assert out["verdict"] == "frequency_only"
        assert out["directions"]["L"] == "0"

    def test_frequency_only_is_reachable_on_eight_seeds(self) -> None:
        # Six discordant pairs fit inside eight seeds, so the verdict is not a sixteen-seed one:
        # it needs a lopsided split, not a larger panel.
        a = dict.fromkeys(range(1, 9), _HIGH)
        b = {1: _HIGH, 2: _HIGH, **dict.fromkeys(range(3, 9), 0.0)}
        out = ms.read(a, b)
        assert out["members"]["F"]["only_a"] == 6
        assert out["directions"]["F"] == "+"
        assert out["directions"]["L"] == "0"
        assert out["verdict"] == "frequency_only"

    def test_an_all_dead_panel_is_no_effect(self) -> None:
        a = _arm({}, tuple(range(1, 9)))
        b = _arm({}, tuple(range(1, 9)))
        assert ms.read(a, b)["verdict"] == "no_effect"

    def test_the_verdict_carries_what_it_licenses(self) -> None:
        a = _arm({1: 60.0, 2: 70.0, 3: 80.0}, (4, 5, 6, 7, 8))
        b = _arm({1: 30.0, 2: 25.0, 3: 35.0}, (4, 5, 6, 7, 8))
        out = ms.read(a, b)
        assert out["licenses"] == ms.LICENSES[out["verdict"]]


class TestReproducibility:
    def test_the_bootstrap_is_seeded(self) -> None:
        a = _arm({1: 60.0, 2: 70.0, 3: 80.0}, (4, 5, 6, 7, 8))
        b = _arm({1: 30.0, 2: 25.0, 3: 35.0}, (4, 5, 6, 7, 8))
        first, second = ms.level_contrast(a, b), ms.level_contrast(a, b)
        assert first["p_improve"] == second["p_improve"]
        assert first["ci80"] == second["ci80"]

    def test_the_rank_test_is_reported_beside_the_permutation(self) -> None:
        # It cannot reach significance at three-a-side, which is why it is not the test; it is
        # recorded so a reader can see the distribution-free number.
        a = _arm({1: 60.0, 2: 70.0, 3: 80.0}, (4, 5, 6, 7, 8))
        b = _arm({1: 30.0, 2: 25.0, 3: 35.0}, (4, 5, 6, 7, 8))
        out = ms.level_contrast(a, b)
        assert out["rank_p_improve"] == pytest.approx(0.05)


class TestTheLevelPValueComesFromANull:
    """The p-value must say how often chance produces this, not where the effect is."""

    def test_identical_arms_are_not_significant(self) -> None:
        same = {1: 40.0, 2: 50.0, 3: 60.0, 4: 70.0}
        out = ms.level_contrast(same, dict(same))
        assert out["effect"] == pytest.approx(0.0)
        assert out["p_improve"] > ms.SIG_Q
        assert out["p_degrade"] > ms.SIG_Q

    def test_the_null_is_the_pooled_split_not_the_observed_draws(self) -> None:
        # Resampling each arm as observed centres the draws on the observed difference, so the
        # share of them below zero shrinks as the arms grow tighter even when the two levels
        # barely differ. A tiny but very tight separation must not read as significant.
        a = {1: 40.1, 2: 40.2, 3: 40.1, 4: 40.2}
        b = {1: 40.0, 2: 39.9, 3: 40.0, 4: 39.9}
        out = ms.level_contrast(a, b)
        assert out["effect"] == pytest.approx(0.2, abs=0.05)
        # Under the old uncentred estimator every resampled difference was positive, giving the
        # smallest p the method can return.
        assert out["p_improve"] > 0.01

    def test_a_large_clean_separation_is_significant(self) -> None:
        a = dict(zip(range(1, 7), (70.0, 75.0, 80.0, 72.0, 78.0, 74.0), strict=True))
        b = dict(zip(range(1, 7), (25.0, 30.0, 22.0, 28.0, 24.0, 26.0), strict=True))
        assert ms.level_contrast(a, b)["p_improve"] <= ms.SIG_Q

    def test_the_interval_still_comes_from_the_observed_arms(self) -> None:
        a = _arm({1: 60.0, 2: 70.0, 3: 80.0}, (4, 5))
        b = _arm({1: 30.0, 2: 25.0, 3: 35.0}, (4, 5))
        out = ms.level_contrast(a, b)
        low, high = out["ci80"]
        assert low < out["effect"] < high
