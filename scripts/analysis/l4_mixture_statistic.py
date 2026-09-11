"""A contrast family matched to a bimodal outcome, and the map that reads it.

Every panel in this phase applied a paired rank test to an outcome that is not unimodal: a seed
reaches a competent policy or a dead one, and the two arms can differ in how *often* they reach one
or in how *good* it is when they do. A test of either alone reports no effect when only the other
moves. Logbook 042 stated the shape -- "the wiring's mark is the level of the good fixed points,
not their frequency" -- named the contrast that follows from it, and carried the requirement
forward: on a bimodal outcome, register a statistic matched to the shape before the data exist.

The family has one member per component, plus the test the committed record was scored with:

* ``F`` -- competent-fraction discordance, an exact binomial on the discordant pairs at the
  committed threshold. Generalised from the registered ``l4_panel3.discordance``.
* ``L`` -- the difference in mean level among competent seeds, each arm over its OWN competent
  subset. Per arm and not paired: a pair with one arm competent and the other dead would otherwise
  contribute the whole competent value to L, which is a frequency event wearing level's name.
  Its test is a seeded label permutation: the null is that the two arms' competent seeds come
  from one level, built by pooling them and re-splitting at the observed sizes. The interval is a
  separate bootstrap of the arms as observed, which is what an interval is for. Resampling the
  arms as observed is *not* a null -- those draws are centred on the observed difference, so the
  share of them below zero asks where the effect is, not how often chance produces one this
  large. The rank test is reported beside both as a check.
* ``W`` -- the existing paired one-sided Wilcoxon with an 80% bootstrap CI, retained so a re-read
  stays comparable with the record it re-reads.

Both directions are tested, because the outcome map must name a degrading result as well as an
improving one: each direction is its own BH-FDR family over the three members, and a member reads
``+``, ``-`` or ``0``. One-sided tests in opposite directions cannot both fire.

What the members can and cannot detect at these panel sizes is a property of the tests, not of any
result, and is recorded here so it is read with them rather than discovered afterwards:

* ``F`` is an exact binomial on the discordant pairs alone, so it needs **six** pairs all falling
  one way to reach q <= 0.05 across three members (p = 0.0156); five reach only q = 0.094. Six is
  attainable on an eight-seed panel -- an arm competent on six seeds where the other is competent
  on none -- so ``frequency_only`` is reachable there, but only for a lopsided split: an
  eight-seed panel can show a real frequency difference F cannot call. This is the registered
  test and its strictness is left alone; loosening it because it is strict would be the move this
  family exists to prevent.
* ``L`` needs both arms to have a competent seed, and its resolution is set by how many. The floor
  is combinatorial and shared by every distribution-free test here: at three competent seeds a side
  the pool has only twenty splits, so the smallest reachable p is about 0.06 and the smallest q
  across three members about 0.19 -- ``level_only`` cannot fire. Four a side reaches q = 0.054,
  five a side 0.027. That is a limit of these panel sizes rather than of the test, and is left as
  one: a statistic returning small p-values at three a side is not measuring the null.
* Neither is a mixture model. Two marginal readings are what a panel of eight or sixteen seeds can
  actually support; fitting component means and a mixing weight is not.
* A per-seed "some up, some down" split is **not** evidence of a mixed response and is not used as
  a verdict: simulated against a null where both arms are drawn from one bimodal law it fires on
  81-100% of draws. It is recorded as a descriptive annotation only. ``mixed_response`` therefore
  requires the two contrasts to be significant against each other.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, mannwhitneyu, wilcoxon

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from l4_panel2 import COMPETENT_THRESHOLD  # noqa: E402  # pyright: ignore[reportMissingImports]
from weight_search_architecture_ranking import (  # noqa: E402
    bh_fdr,  # pyright: ignore[reportMissingImports]
)

SIG_Q = 0.05  # the level panel 2's family used
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 42
# A seed counts as improved or degraded for the split branch only if it moves by more than the
# assay's own per-seed hold band, so an arm that jitters around its comparator is not a split.
SPLIT_BAND = 10.0

MEMBERS: tuple[str, ...] = ("F", "L", "W")
# Enumerate the null exactly up to this many splits; every committed panel is far below it
# (the largest is sixteen competent seeds split evenly, 12,870 ways).
ENUMERATION_CAP = 200_000
_TOL = 1e-9


def _paired(a: dict[int, float], b: dict[int, float]) -> list[int]:
    """Seeds present in both arms, sorted."""
    return sorted(set(a) & set(b))


def frequency_contrast(
    a: dict[int, float],
    b: dict[int, float],
    threshold: float = COMPETENT_THRESHOLD,
) -> dict[str, Any]:
    """Test whether ``a`` reaches competence on more seeds than ``b``.

    The registered test, generalised from ``l4_panel3.discordance`` off its wild-type/rewired
    naming. Only the discordant pairs carry information about a difference in frequency, so the
    exact binomial is on those.
    """
    common = _paired(a, b)
    a_c = {s for s in common if a[s] >= threshold}
    b_c = {s for s in common if b[s] >= threshold}
    only_a = len(a_c - b_c)
    only_b = len(b_c - a_c)
    discordant = only_a + only_b
    return {
        "n": len(common),
        "threshold": threshold,
        "only_a": only_a,
        "only_b": only_b,
        "both": len(a_c & b_c),
        "a_competent": len(a_c),
        "b_competent": len(b_c),
        "a_fraction": (len(a_c) / len(common)) if common else float("nan"),
        "b_fraction": (len(b_c) / len(common)) if common else float("nan"),
        "effect": float(only_a - only_b),
        "p_improve": (
            1.0
            if discordant == 0
            else float(binomtest(only_a, discordant, 0.5, alternative="greater").pvalue)
        ),
        "p_degrade": (
            1.0
            if discordant == 0
            else float(binomtest(only_b, discordant, 0.5, alternative="greater").pvalue)
        ),
        "defined": bool(common),
    }


def _permutation_p(null: np.ndarray, observed: float, *, improve: bool, exact: bool) -> float:
    """Return the share of the null at least as extreme as the observed difference."""
    # A hair of tolerance so a split that equals the observed one is counted as extreme rather
    # than lost to floating-point drift in the summation.
    count = int(
        np.sum(null >= observed - _TOL) if improve else np.sum(null <= observed + _TOL),
    )
    if exact:
        return float(count / null.size)
    return float((count + 1) / (BOOTSTRAP_DRAWS + 1))


def level_contrast(
    a: dict[int, float],
    b: dict[int, float],
    threshold: float = COMPETENT_THRESHOLD,
) -> dict[str, Any]:
    """Where each arm's upper mode sits: its mean over its own competent seeds.

    Per arm, not paired. A seed competent in one arm and dead in the other contributes to that
    arm's level and to nothing else -- pairing over the union would let a frequency difference
    masquerade as a level one. Undefined where either arm has no competent seed: that is a panel
    with no upper mode to compare, not a null result.
    """
    common = _paired(a, b)
    a_vals = [a[s] for s in common if a[s] >= threshold]
    b_vals = [b[s] for s in common if b[s] >= threshold]
    if not a_vals or not b_vals:
        return {
            "defined": False,
            "threshold": threshold,
            "a_n": len(a_vals),
            "b_n": len(b_vals),
            "reason": "one arm has no competent seed, so there is no upper mode to compare",
            "effect": float("nan"),
            "p_improve": float("nan"),
            "p_degrade": float("nan"),
        }
    a_arr, b_arr = np.array(a_vals, dtype=float), np.array(b_vals, dtype=float)
    observed = float(a_arr.mean() - b_arr.mean())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # The interval comes from resampling each arm as observed: it describes where the difference
    # is, which is what an interval is for.
    draws = np.array(
        [
            rng.choice(a_arr, a_arr.size, replace=True).mean()
            - rng.choice(b_arr, b_arr.size, replace=True).mean()
            for _ in range(BOOTSTRAP_DRAWS)
        ],
    )
    # The p-value needs a distribution under the NULL, which those draws are not: they are
    # centred on the observed difference, so reading the share of them below zero asks where the
    # effect is rather than how often chance produces one this large. The null here is that the
    # two arms' competent seeds come from one level, so it is built by re-splitting the pool at
    # the observed sizes.
    null, exact = _null_differences(a_arr, b_arr)
    return {
        "defined": True,
        "threshold": threshold,
        "a_n": int(a_arr.size),
        "b_n": int(b_arr.size),
        "a_mean": float(a_arr.mean()),
        "b_mean": float(b_arr.mean()),
        "effect": float(a_arr.mean() - b_arr.mean()),
        "ci80": [float(np.quantile(draws, 0.1)), float(np.quantile(draws, 0.9))],
        # Enumerated: the share of splits at least as extreme, which already counts the observed
        # one, so it is exact and needs no correction. Sampled: the standard (count + 1) /
        # (draws + 1), never exactly zero.
        "exact": exact,
        "p_improve": _permutation_p(null, observed, improve=True, exact=exact),
        "p_degrade": _permutation_p(null, observed, improve=False, exact=exact),
        # Reported beside it, not used: it cannot reach significance at small subset sizes.
        "rank_p_improve": float(
            getattr(mannwhitneyu(a_arr, b_arr, alternative="greater"), "pvalue", 1.0),
        ),
        "rank_p_degrade": float(
            getattr(mannwhitneyu(a_arr, b_arr, alternative="less"), "pvalue", 1.0),
        ),
    }


def _null_differences(a_arr: np.ndarray, b_arr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Differences in mean under the null that both arms' seeds come from one level.

    Enumerated exactly where the pool is small enough, which every committed panel is: the
    largest is sixteen competent seeds split eight and eight, 12,870 ways. Exactness matters here
    for a reason beyond precision -- a sampled null makes the p-value depend on the order the
    values happen to arrive in, since a seeded generator applies the same index permutation to
    whatever array it is given, and seed-to-value assignment is arbitrary. Enumerating makes the
    result a function of the two multisets and nothing else.

    Above the cap the pool is re-split at random, from a canonically sorted copy so the sampled
    null stays a function of the multisets too.
    """
    pooled = np.concatenate([a_arr, b_arr])
    take = a_arr.size
    total = pooled.size
    if math.comb(total, take) <= ENUMERATION_CAP:
        whole = pooled.sum()
        splits = np.array(
            [sum(subset) for subset in itertools.combinations(pooled, take)],
            dtype=float,
        )
        return splits / take - (whole - splits) / (total - take), True
    ordered = np.sort(pooled)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    null = np.empty(BOOTSTRAP_DRAWS)
    for i in range(BOOTSTRAP_DRAWS):
        shuffled = rng.permutation(ordered)
        null[i] = shuffled[:take].mean() - shuffled[take:].mean()
    return null, False


def shift_contrast(a: dict[int, float], b: dict[int, float]) -> dict[str, Any]:
    """Run the all-seeds paired rank test the committed record was scored with."""
    common = _paired(a, b)
    deltas = [a[s] - b[s] for s in common]
    non_zero = [d for d in deltas if d != 0.0]
    if not non_zero:
        return {
            "defined": False,
            "n": len(common),
            "effect": 0.0,
            "p_improve": 1.0,
            "p_degrade": 1.0,
            "positive_seeds": 0,
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    arr = np.array(deltas, dtype=float)
    draws = np.array(
        [rng.choice(arr, arr.size, replace=True).mean() for _ in range(BOOTSTRAP_DRAWS)],
    )
    return {
        "defined": True,
        "n": len(common),
        "effect": float(arr.mean()),
        "ci80": [float(np.quantile(draws, 0.1)), float(np.quantile(draws, 0.9))],
        "p_improve": float(getattr(wilcoxon(deltas, alternative="greater"), "pvalue", 1.0)),
        "p_degrade": float(getattr(wilcoxon(deltas, alternative="less"), "pvalue", 1.0)),
        "positive_seeds": int(sum(1 for d in deltas if d > 0)),
    }


def _split(a: dict[int, float], b: dict[int, float], band: float = SPLIT_BAND) -> dict[str, Any]:
    """Count the seeds moved beyond the band each way; descriptive, never a verdict input.

    This was registered as the tie-breaker for the doubly-non-significant cell and is not used as
    one, for a reason that is structural rather than empirical: the null here is itself bimodal, so
    two arms drawn from the *same* law almost always put some seed high in one and low in the other.
    Simulating that null, "at least one seed each way beyond the band" fires on 81-100% of draws at
    every band from 5 to 25, and requiring three each way still fires on 77% at sixteen seeds. No
    threshold on this statistic is specific, because the quantity it counts is what bimodality
    produces by itself. The counts are kept because the shape is worth seeing; the verdict does not
    rest on them.
    """
    common = _paired(a, b)
    improved = [s for s in common if a[s] - b[s] > band]
    degraded = [s for s in common if b[s] - a[s] > band]
    return {
        "band": band,
        "improved_seeds": improved,
        "degraded_seeds": degraded,
        "is_split": bool(improved and degraded),
    }


def _direction(member: dict[str, Any], q_improve: float, q_degrade: float) -> str:
    """Return ``+``, ``-`` or ``0`` for one member, from its corrected p-values."""
    if not member.get("defined", True):
        return "0"
    if q_improve <= SIG_Q:
        return "+"
    if q_degrade <= SIG_Q:
        return "-"
    return "0"


def family(
    a: dict[int, float],
    b: dict[int, float],
    threshold: float = COMPETENT_THRESHOLD,
    band: float = SPLIT_BAND,
) -> dict[str, Any]:
    """Compute the three members, corrected in each direction, with each member's direction."""
    members = {
        "F": frequency_contrast(a, b, threshold),
        "L": level_contrast(a, b, threshold),
        "W": shift_contrast(a, b),
    }
    # An undefined member carries no p-value into the correction: it is excluded rather than
    # entered as 1.0, which would otherwise shrink every other member's q.
    defined = [m for m in MEMBERS if members[m].get("defined", True)]
    for direction in ("improve", "degrade"):
        qs = bh_fdr([members[m][f"p_{direction}"] for m in defined])
        for member, q in zip(defined, qs, strict=True):
            members[member][f"q_{direction}"] = float(q)
        for member in MEMBERS:
            if member not in defined:
                members[member][f"q_{direction}"] = float("nan")
    directions = {
        m: _direction(members[m], members[m]["q_improve"], members[m]["q_degrade"]) for m in MEMBERS
    }
    return {
        "members": members,
        "directions": directions,
        "split": _split(a, b, band),
        "alpha": SIG_Q,
        "n_pairs": len(_paired(a, b)),
    }


# The registered direction table: every combination of F and L is named, so a two-directional
# result cannot be discovered after the fact. ``mixed_response`` licenses nothing.
_MAP: dict[tuple[str, str], str] = {
    ("+", "+"): "shift",
    ("-", "-"): "degrades",
    ("+", "-"): "mixed_response",
    ("-", "+"): "mixed_response",
    ("0", "+"): "level_only",
    ("+", "0"): "frequency_only",
    ("0", "-"): "degrades",
    ("-", "0"): "degrades",
}

LICENSES: dict[str, str] = {
    "shift": "the effect is unambiguous",
    "level_only": "better fixed points, not more of them",
    "frequency_only": "more competent seeds, no better",
    "mixed_response": "nothing: it requires its own registration to act on",
    "no_effect": "the contrast is closed on this evidence",
    "degrades": "nothing",
}


def verdict(result: dict[str, Any]) -> str:
    """Read the direction table. ``L`` undefined reads as ``0``.

    A doubly-non-significant panel is ``no_effect``, whatever its per-seed spread: at these panel
    sizes "some seeds up, some seeds down" is what a bimodal null produces on its own, so treating
    it as a finding would name noise. ``mixed_response`` is reserved for the case the data can
    actually support -- the two contrasts significant *against each other*.
    """
    f, level = result["directions"]["F"], result["directions"]["L"]
    if (f, level) == ("0", "0"):
        return "no_effect"
    return _MAP[(f, level)]


def read(
    a: dict[int, float],
    b: dict[int, float],
    threshold: float = COMPETENT_THRESHOLD,
    band: float = SPLIT_BAND,
) -> dict[str, Any]:
    """Return the family, its verdict and what that verdict licenses."""
    result = family(a, b, threshold, band)
    result["verdict"] = verdict(result)
    result["licenses"] = LICENSES[result["verdict"]]
    return result
