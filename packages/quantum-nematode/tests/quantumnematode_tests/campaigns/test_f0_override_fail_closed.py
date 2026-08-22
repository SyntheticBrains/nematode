"""Regression tests for #279 — the F0 override used to fall back per-seed.

When ``--campaign-root`` is given, each ``(arm, seed)``'s F0 baseline is loaded
from ``per_gen_elites.jsonl``. Any seed whose file was missing or unreadable was
silently dropped from the override map and fell back to the **post-hoc retention
F0** — the baseline the operator had just asked to replace, and which the code
itself calls "the post-hoc F0" as distinct from "the biologically-correct" one.

The result was a single gate evaluation mixing two different baselines, with no
error and (in two of the three aggregators) no output at all. Because the gate is
a ratio of F0, that can flip an arm verdict.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.campaigns._common import (  # noqa: E402
    GATE_F1_RATIO,
    aggregate_per_arm_verdict,
    evaluate_decision_gate_one_seed,
    require_complete_f0_override,
)


def _retention(seeds: tuple[int, ...], f0: float, f1: float) -> dict[tuple[str, int, int], float]:
    """Build a survival table where every seed has identical, gate-shaped data."""
    return {
        ("A", seed, gen): value
        for seed in seeds
        for gen, value in ((0, f0), (1, f1), (2, f1), (3, f1))
    }


class TestPerSeedFallbackIsRefused:
    """The enforcement point: scoring a seed the override does not cover."""

    def test_missing_pair_raises_rather_than_falling_back(self) -> None:
        retention = _retention((1,), f0=0.80, f1=0.25)

        with pytest.raises(ValueError, match="override is missing"):
            evaluate_decision_gate_one_seed(
                retention=retention,
                arm="A",
                seed=1,
                f0_baseline_override={("A", 99): 0.40},
            )

    def test_covered_pair_uses_the_override(self) -> None:
        retention = _retention((1,), f0=0.80, f1=0.25)

        result = evaluate_decision_gate_one_seed(
            retention=retention,
            arm="A",
            seed=1,
            f0_baseline_override={("A", 1): 0.40},
        )

        assert result["f0"] == pytest.approx(0.40)

    def test_no_override_still_uses_the_post_hoc_baseline(self) -> None:
        """``None`` means "post-hoc for everything", which is consistent."""
        retention = _retention((1,), f0=0.80, f1=0.25)

        result = evaluate_decision_gate_one_seed(retention=retention, arm="A", seed=1)

        assert result["f0"] == pytest.approx(0.80)


class TestUpFrontCoverageCheck:
    """The usability layer: report every missing pair before doing any work."""

    def test_lists_all_missing_pairs_not_just_the_first(self) -> None:
        override = {("A", 1): 0.4}

        with pytest.raises(ValueError, match=r"A/seed-2.*A/seed-3") as excinfo:
            require_complete_f0_override(override, [("A", 1), ("A", 2), ("A", 3)])

        assert "covers 1 of 3" in str(excinfo.value)

    def test_complete_override_passes(self) -> None:
        override = {("A", 1): 0.4, ("A", 2): 0.5}

        require_complete_f0_override(override, [("A", 1), ("A", 2)])

    def test_none_override_is_not_an_error(self) -> None:
        require_complete_f0_override(None, [("A", 1), ("A", 2)])

    def test_only_the_gated_pairs_are_required(self) -> None:
        """Arms need not share a seed set, so a cartesian product would over-claim."""
        override = {("A", 1): 0.4, ("B", 2): 0.5}

        require_complete_f0_override(override, [("A", 1), ("B", 2)])


class TestTheVerdictFlipThisPrevents:
    """Why per-seed fallback mattered: it could change the arm verdict."""

    def test_mixed_baselines_would_have_flipped_go_to_pivot(self) -> None:
        """Reproduces the issue's worked example, with the fallback simulated.

        Two seeds with identical retention data. Seed 1's F0 is readable
        (training-time 0.40); seed 2's is not, so pre-fix it silently used the
        post-hoc 0.80. The gate is ``f1 >= GATE_F1_RATIO * f0``, so seed 1 clears
        a 0.16 bar and seed 2 faces a 0.32 bar on the same f1 = 0.25.
        """
        retention = _retention((1, 2), f0=0.80, f1=0.25)
        override = {("A", 1): 0.40, ("A", 2): 0.40}

        # Intended: both seeds on the training-time baseline.
        intended = [
            evaluate_decision_gate_one_seed(
                retention=retention,
                arm="A",
                seed=seed,
                f0_baseline_override=override,
            )
            for seed in (1, 2)
        ]
        assert aggregate_per_arm_verdict(intended) == "GO"

        # Pre-fix behaviour: seed 2 falls back to the post-hoc F0.
        mixed = [
            intended[0],
            evaluate_decision_gate_one_seed(retention=retention, arm="A", seed=2),
        ]
        assert aggregate_per_arm_verdict(mixed) == "PIVOT"

        # And that mixture is now unreachable: the missing pair raises.
        with pytest.raises(ValueError, match="override is missing"):
            evaluate_decision_gate_one_seed(
                retention=retention,
                arm="A",
                seed=2,
                f0_baseline_override={("A", 1): 0.40},
            )

    def test_the_gate_really_is_a_ratio_of_f0(self) -> None:
        """Pins the mechanism the flip depends on."""
        assert GATE_F1_RATIO * 0.40 < 0.25 < GATE_F1_RATIO * 0.80
