#!/usr/bin/env python
"""V.4: whether block V's wiring advantage holds on rewirings that have never been used.

After [L.0](../../docs/experiments/logbooks/064-l4-frozen-features.md), block V's learning-speed
advantage is the only surviving wiring result in this project -- endpoint-inert under gradient
learning (034), actively harmful under local rules that write the wiring (R.2), indistinguishable
from a degree-matched shuffle as fixed features (L.0). What stands is V.1's **+35.4%** and V.3's
**+23.5%** off time-to-competence under PPO.

It has one open caveat. `rewire_seed` is unset in every wiring config, so each seed's rewired graph
derives from its run seed -- and **V.3's rewirings at seeds 1-32 are a subset of V.1's at 1-64**. The
two positives do not corroborate each other on independent nulls; they share them. Seeds 65-96 give
graphs no panel has used.

**This module is a manifest builder and a branch reporter, and deliberately nothing more.**
``wiring_premise`` already drives ``connectome_structure_efficiency`` itself and already owns the
registered 20% minimum (``MIN_EFFICIENCY_GAIN``), the per-cell verdicts with the gates read before the
contrast (``verdict``), the censoring guard (``CROSSING_FLOOR``) and the efficiency-arm mapping. A
replication exists to vary the evidence and hold the reading fixed: re-implementing any of that here
would let "the instrument changed" compete with "the effect is not there", and after the fact those are
not separable. So both committed harnesses are used unmodified and a test asserts it.

The verdict vocabulary is therefore the harness's, with V.1's prose branches mapped onto it rather than
run in parallel -- a parallel vocabulary is how two records come to disagree about the same run. Three
of the harness's verdicts have no prose branch and are registered anyway, because they are live rather
than hypothetical: ``saturated`` is what the klinotaxis cell returned in V.1's own pilot, ``no_learning``
means a gate failed, and a contrast the harness flags **materially censored** is the case L.0 met on
``hard350`` with five non-crossing seeds of 32. **None of those is evidence against the original
result.**
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import wiring_premise as wp  # noqa: E402  # pyright: ignore[reportMissingImports]

# Fresh in rewiring, task and initialisation together: V.1 ran 1-64 on the thermal cell and V.3 ran
# 1-32 on hard_food, so these are disjoint from both.
SEEDS = tuple(range(65, 97))
PRIOR_SEEDS = {"thermal": tuple(range(1, 65)), "hard_food": tuple(range(1, 33))}

_THERMAL = "connectomeppo_small_continuous2d_thermal_klinotaxis"
_HARD = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350"

# Config stem -> (cell, arm), stated explicitly rather than derived by pattern: the thermal arms carry
# their `_t20` suffix AFTER the arm part, so a regex over suffixes is easy to get subtly wrong and a
# mis-keyed arm would silently drop a side of a paired test.
ARM_BY_STEM: dict[str, tuple[str, str]] = {
    f"{_THERMAL}_t20": ("thermal", "wt_ppo"),
    f"{_THERMAL}_rewired_null_t20": ("thermal", "rn_ppo"),
    f"{_THERMAL}_frozen_t20": ("thermal", "wt_frozen"),
    f"{_THERMAL}_rewired_null_frozen_t20": ("thermal", "rn_frozen"),
    _HARD: ("hard_food", "wt_ppo"),
    f"{_HARD}_rewired_null": ("hard_food", "rn_ppo"),
    f"{_HARD}_frozen": ("hard_food", "wt_frozen"),
    f"{_HARD}_rewired_null_frozen": ("hard_food", "rn_frozen"),
}
CELLS = ("thermal", "hard_food")

# V.1's prose branches, mapped onto the harness's verdict names. The prose is what 057 registered and
# what a reader of it will look for; the names are what the record reports.
PROSE_BRANCH = {
    "specific_wiring": "replicates",
    "below_min_effect": "same direction, below the minimum",
    "degree_statistics": "does not replicate",
}
# Verdicts that are NOT a failure to replicate, and are reported as themselves.
NOT_A_FAILURE = {
    "saturated": (
        "the cell cannot answer on this axis -- what the klinotaxis cell returned in V.1's own pilot, "
        "both wirings at 100% and a contrast of exactly zero. Not a replication failure"
    ),
    "no_learning": "a learning gate failed, so the contrast is uninterpretable",
    "insufficient_seeds": "too few paired seeds survived to score the panel",
}

# The committed comparators, carried so a near-miss is read against the original's own variability.
COMPARATORS = {
    "thermal": {
        "gain_fraction": 0.354,
        "n_seeds": 64,
        "per_panel_gains": [0.464, 0.326, 0.318],
        "source": "V.1, pooled over 64 seeds; its panels of 16, 16 and 32 spread 15 points",
    },
    "hard_food": {
        "gain_fraction": 0.235,
        "n_seeds": 32,
        "per_panel_gains": None,
        "source": "V.3, 32 paired seeds, three of four efficiency metrics significant",
    },
}


def build_manifest(campaign_dir: Path, path: Path, seeds: tuple[int, ...] = SEEDS) -> Path:
    """Write the ``<cell> <arm> <seed> <out>`` manifest the committed harness reads.

    Raises on an unrecognised log name rather than skipping it: a mis-keyed arm would silently drop
    one side of a paired test, which is the failure the harness's own ``ManifestError`` exists for.
    """
    log_dir = campaign_dir / "logs" if (campaign_dir / "logs").is_dir() else campaign_dir
    lines: list[str] = []
    seen: set[tuple[str, str, int]] = set()
    for log in sorted(log_dir.glob("*.log")):
        stem, _, seed_part = log.stem.rpartition("-seed")
        if not seed_part.isdigit():
            msg = f"log name has no `-seedN` suffix: {log.name}"
            raise ValueError(msg)
        if stem not in ARM_BY_STEM:
            msg = f"log names a config this panel does not have: {log.name}"
            raise ValueError(msg)
        seed = int(seed_part)
        if seed not in seeds:
            continue
        cell, arm = ARM_BY_STEM[stem]
        key = (cell, arm, seed)
        if key in seen:
            msg = f"two logs for {cell}/{arm} seed {seed}"
            raise ValueError(msg)
        seen.add(key)
        # Repo-relative where possible, since that is how every committed manifest reads. A campaign
        # outside the repo falls back to the absolute path, which the harness also resolves correctly:
        # `REPO / absolute` returns the absolute operand.
        resolved = log.resolve()
        try:
            entry = resolved.relative_to(wp.REPO)
        except ValueError:
            entry = resolved
        lines.append(f"{cell} {arm} {seed} {entry}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def require_complete(manifest: Path, seeds: tuple[int, ...] = SEEDS) -> None:
    """Refuse to score a panel missing any registered cell.

    The harness reports ``missing_arms`` and drops unparseable runs with a warning, which surfaces a
    gap but does not stop the scoring. A replication whose consequence is withdrawing a committed
    result should not be assigned on a partial panel.
    """
    present: dict[tuple[str, str], set[int]] = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        cell, arm, seed, _ = line.split()
        present.setdefault((cell, arm), set()).add(int(seed))
    missing = [
        f"{cell}/{arm} seeds {sorted(set(seeds) - present.get((cell, arm), set()))}"
        for cell in CELLS
        for arm in wp.TESTED_ARMS
        if set(seeds) - present.get((cell, arm), set())
    ]
    if missing:
        msg = "panel is incomplete, so no branch is available: " + "; ".join(missing)
        raise ValueError(msg)


def branch(
    cell: str,
    entry: dict[str, Any],
    report: dict[str, Any] | None,
    *,
    verdict_reachable: bool = True,
) -> dict[str, Any]:
    """Map one cell's harness verdict onto V.1's registered prose branch.

    Read per cell and never pooled: a split is evidence about the scope of block V's generalisation,
    which it only remains if each cell is reported on its own.

    With ``verdict_reachable=False`` -- too few pairs for the smallest achievable exact p to reach
    the significance level -- **no branch is assigned at all**. At n pairs that p is 2**-n, so at 4
    pairs it is 0.0625 and nothing can clear q = 0.05: the harness then returns `no_learning` on
    gates of +14.95 and +77.37 full-clear points at 4/4, and `degree_statistics` on efficiency gains
    ABOVE the registered minimum. Reading either as a result would be reading the seed count.
    """
    name = entry.get("verdict")
    # The harness computes this for a printed flag and does not put it in the report, so it is
    # recomputed HERE THROUGH ITS OWN functions -- `crossing_rate` and `CROSSING_FLOOR` -- rather than
    # reimplemented. Reading a `materially_censored` key would silently always be False.
    rates = (
        {
            arm: wp.crossing_rate(report, arm)
            for arm in (wp.efficiency._WILD, wp.efficiency._REWIRED)
        }
        if report
        else {}
    )
    censored = bool(rates) and min(rates.values()) < wp.CROSSING_FLOOR
    out: dict[str, Any] = {
        "cell": cell,
        "harness_verdict": name,
        "verdict_reachable": verdict_reachable,
        "axis": entry.get("axis", "peak"),
        "peak_verdict": entry.get("peak_verdict"),
        "crossing_rates": rates,
        "crossing_floor": wp.CROSSING_FLOOR,
        "materially_censored": censored,
        "comparator": COMPARATORS[cell],
        "prior_seeds": list(PRIOR_SEEDS[cell]),
    }
    if not verdict_reachable:
        out["prose_branch"] = None
        out["is_replication_failure"] = False
        out["why"] = (
            "WITHHELD: too few pairs for any verdict. The harness's name above is what its rule "
            "returns at this seed count and is not a reading of the wiring"
        )
        return out
    if name in PROSE_BRANCH:
        out["prose_branch"] = PROSE_BRANCH[name]
        out["is_replication_failure"] = name == "degree_statistics"
        out["why"] = {
            "replicates": (
                "the advantage holds at the registered minimum on rewirings never used before: the "
                "caveat closes and block V's positive is independent in rewiring"
            ),
            "same direction, below the minimum": (
                "directionally consistent and under the bar -- a real but smaller effect, with the "
                "shrinkage named. It licenses the follow-up, not the claim"
            ),
            "does not replicate": (
                "the advantage does not hold on fresh rewirings. The registered panel is reported as "
                "not holding and the first positive is WITHDRAWN on the record rather than defended"
            ),
        }[out["prose_branch"]]
    else:
        out["prose_branch"] = None
        out["is_replication_failure"] = False
        out["why"] = NOT_A_FAILURE.get(str(name), f"unrecognised harness verdict {name!r}")
    if censored and not out["is_replication_failure"]:
        out["why"] += (
            "; and the contrast is materially censored below the harness's crossing floor, which is "
            "not evidence against the original result either"
        )
    return out


def analyse(manifest: Path, seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """Score the panel through the committed harness and report each cell's branch."""
    cells = wp.load(manifest)
    harness: dict[str, Any] = {}
    wp.analyse(cells, harness, manifest)
    # Read BEFORE the branches, because it can withhold all of them.
    power = _power(len(seeds))
    branches = {
        cell: branch(
            cell,
            harness["verdicts"].get(cell, {}),
            harness.get("efficiency", {}).get(cell),
            verdict_reachable=power["gate_reachable"],
        )
        for cell in CELLS
    }
    failures = [c for c, b in branches.items() if b["is_replication_failure"]]
    replicating = [c for c, b in branches.items() if b["prose_branch"] == "replicates"]
    split = bool(failures and replicating)
    return {
        "seeds": list(seeds),
        "fresh_in": (
            "rewiring, task and initialisation together -- `rewire_seed` is unset, so these vary as "
            "one. Isolating the rewiring alone would need `rewire_seed` pinned and is a different "
            "experiment from the one 058 registered"
        ),
        "branches": branches,
        "replicating_cells": replicating,
        "failing_cells": failures,
        "split": split,
        "pooled_reading_withheld": split,
        "split_note": (
            "one cell replicating and the other not is reported as a split, with the pooled reading "
            "withheld. Block V's claim is that the effect generalises from a foraging cell under "
            "thermal pressure to a foraging cell hard enough to discriminate, so a split is evidence "
            "about the SCOPE of that generalisation -- never resolved toward whichever cell supports "
            "the original"
        )
        if split
        else None,
        "verdicts_reachable": power["gate_reachable"],
        "power": power,
        "harness": harness,
    }


def _power(n_pairs: int) -> dict[str, Any]:
    """Carry L.0's registered arithmetic, because a non-replication costs committed records."""
    from math import comb

    def p_at(k: int) -> float:
        return sum(comb(n_pairs, i) for i in range(k, n_pairs + 1)) / 2**n_pairs

    k = next((k for k in range(n_pairs + 1) if p_at(k) <= wp.SIG_Q), None)
    if k is None:
        return {"n_pairs": n_pairs, "k_needed": None, "gate_reachable": False}
    low, high = 21 / 32, 26 / 32
    return {
        "n_pairs": n_pairs,
        "k_needed": k,
        "gate_reachable": True,
        "power_against_comparator": {
            f"{q:.0%}": sum(
                comb(n_pairs, i) * q**i * (1 - q) ** (n_pairs - i) for i in range(k, n_pairs + 1)
            )
            for q in (low, (low + high) / 2, high)
        },
        "note": (
            "sign-test planning figures, not the registered procedure's power: that is a paired rank "
            "test under BH-FDR, which differs in both directions"
        ),
    }


def _print(result: dict[str, Any]) -> None:
    """Print each cell's branch beside the comparator it is replicating."""
    print("\n" + "=" * 78)
    print(f"V.4 - fresh rewirings, seeds {result['seeds'][0]}-{result['seeds'][-1]}")
    print("=" * 78)
    print(f"  fresh in: {result['fresh_in']}")
    if not result["verdicts_reachable"]:
        n = result["power"]["n_pairs"]
        print(
            f"\n  !! NO VERDICT IS REACHABLE AT {n} PAIRS: the smallest achievable one-sided exact\n"
            f"     p is 2**-{n} = {2.0**-n:.4f} > q = {wp.SIG_Q}, so every gate and every contrast\n"
            "     fails on the seed count alone. The harness verdicts below are ITS RULE AT THIS n,\n"
            "     not readings of the wiring. Branches withheld.",
        )
    for cell, b in result["branches"].items():
        comp = b["comparator"]
        print(f"\n  {cell}:")
        print(f"    harness verdict : {b['harness_verdict']}  (axis {b['axis']})")
        print(f"    prose branch    : {b['prose_branch'] or '-- (not a replication branch)'}")
        print(f"    comparator      : {comp['gain_fraction']:.1%} over {comp['n_seeds']} seeds")
        if comp["per_panel_gains"]:
            spread = ", ".join(f"{g:.1%}" for g in comp["per_panel_gains"])
            print(f"                      its own panels: {spread}")
        if b["crossing_rates"]:
            rates = ", ".join(f"{a} {r:.0%}" for a, r in b["crossing_rates"].items())
            mark = (
                f"  <-- below the {b['crossing_floor']:.0%} floor: materially censored"
                if b["materially_censored"]
                else ""
            )
            print(f"    crossed 30%     : {rates}{mark}")
        print(f"    reading         : {b['why']}")
    if result["split"]:
        print(f"\n  SPLIT: {result['split_note']}")
    print(f"\n  replicating: {result['replicating_cells'] or 'none'}")
    print(f"  not holding: {result['failing_cells'] or 'none'}")


def main(argv: list[str] | None = None) -> int:
    """Build the manifest, score it through the committed harness, report each cell's branch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default="65-96")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="score what is present; for a pilot, never for a registered panel",
    )
    args = parser.parse_args(argv)

    low, _, high = args.seeds.partition("-")
    seeds = tuple(range(int(low), int(high or low) + 1))
    manifest = build_manifest(
        args.campaign,
        args.manifest or (args.campaign / "wiring-manifest.txt"),
        seeds,
    )
    if not args.allow_incomplete:
        require_complete(manifest, seeds)
    result = analyse(manifest, seeds)
    _print(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
