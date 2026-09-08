#!/usr/bin/env python
"""The warm-start panel: what a local rule does from a cloned competent policy.

Twelve arms on paired seeds 1-8, every warm-started arm loading a clone of one teacher's
policy made at its own seed. The metric and the statistics layer are the committed ones,
read through the first panel's log reader; panel 2's committed per-seed table supplies the
random-initialisation frozen floor the clone gate is read against.

What is confirmatory is fixed here, in code, before any data exist:

- six one-sided paired tests corrected together as one BH-FDR family -- W1 the wild-type
  plastic-set frozen clone over the random-initialisation wild-type frozen floor (the gate);
  W2 the wild-type over the rewired plastic-set frozen clone; W3 the wild-type over the
  rewired three-factor arm from the plastic-set clone (the primary); W4 that arm over its
  frozen clone; W5 that arm over its Hebbian clone; W6 the wild-type PPO arm from the
  full-set clone over low-noise PPO from random weights;
- an ordered verdict map: ``insufficient_seeds``, ``clone_fail`` (W1 fails),
  ``sanity_floor_fail`` (W4 or W5 fails), ``rewired_beats_wild_type``, ``specific_wiring``,
  ``degree_statistics``, ``inconclusive``; W2 and W6 annotate, and ``rule_destroys_clone``
  is recorded when W4's interval lies entirely below zero.

Everything else -- the remaining pairs, the teacher ceiling, clone fits, curves -- is
descriptive and labelled so.

Usage::

    uv run python scripts/analysis/l4_warm_start.py --campaign-dir campaigns/l4-warm-start-panel
        --teacher-json campaigns/l4-warm-start/teacher.json
        --clones-json campaigns/l4-warm-start/clones.json --out panel.json --csv per-seed.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from l4_panel import _LABEL as LABEL
from l4_panel import EXPERIMENTS, REPO, SeedRecord, paired, read_log
from l4_panel2 import distribution, restrict
from weight_search_architecture_ranking import bh_fdr

_STEM = "connectomeppo_small_continuous2d_combined_klinotaxis"
PANEL2_CSV = (
    REPO / "docs" / "experiments" / "logbooks" / "supporting" / "041-l4-panel2" / "per-seed.csv"
)

ARMS: dict[str, str] = {
    f"{_STEM}_plastic_frozen_clone": "wt_clone_frozen",
    f"{_STEM}_plastic_frozen_rewired_null_clone": "rn_clone_frozen",
    f"{_STEM}_plastic_hebbian_clone": "wt_clone_hebbian",
    f"{_STEM}_plastic_hebbian_rewired_null_clone": "rn_clone_hebbian",
    f"{_STEM}_plastic_clone": "wt_clone_plastic",
    f"{_STEM}_plastic_rewired_null_clone": "rn_clone_plastic",
    f"{_STEM}_plastic_frozen_fullclone": "wt_fullclone_frozen",
    f"{_STEM}_plastic_frozen_rewired_null_fullclone": "rn_fullclone_frozen",
    f"{_STEM}_lowstd_fullclone": "wt_fullclone_ppo",
    f"{_STEM}_rewired_null_lowstd_fullclone": "rn_fullclone_ppo",
    f"{_STEM}_lowstd": "wt_ppo",
    f"{_STEM}_rewired_null_lowstd": "rn_ppo",
}
ARM_KEYS: tuple[str, ...] = tuple(ARMS.values())
STEM_OF: dict[str, str] = {arm: stem for stem, arm in ARMS.items()}
PANEL_SEEDS: tuple[int, ...] = tuple(range(1, 9))
BUDGETS: dict[str, int] = {
    **dict.fromkeys(
        ("wt_clone_frozen", "rn_clone_frozen", "wt_fullclone_frozen", "rn_fullclone_frozen"),
        600,
    ),
    **dict.fromkeys(
        ("wt_clone_hebbian", "rn_clone_hebbian", "wt_clone_plastic", "rn_clone_plastic"),
        2000,
    ),
    **dict.fromkeys(("wt_fullclone_ppo", "rn_fullclone_ppo", "wt_ppo", "rn_ppo"), 3000),
}
EXTENSION = 1.5
SIG_Q = 0.05
CURVE_WINDOW = 250
MIN_PAIRED_SEEDS = 2
RANDOM_FLOOR_ARM = "wt_frozen"  # panel 2's random-initialisation wild-type frozen floor
FAMILY: tuple[str, ...] = ("W1", "W2", "W3", "W4", "W5", "W6")
FAMILY_PAIRS: dict[str, tuple[str, str]] = {
    "W1": ("wt_clone_frozen", "random_frozen"),
    "W2": ("wt_clone_frozen", "rn_clone_frozen"),
    "W3": ("wt_clone_plastic", "rn_clone_plastic"),
    "W4": ("wt_clone_plastic", "wt_clone_frozen"),
    "W5": ("wt_clone_plastic", "wt_clone_hebbian"),
    "W6": ("wt_fullclone_ppo", "wt_ppo"),
}
READS: dict[str, str] = {
    "W1": "plastic-set clone > random-initialisation frozen floor (the gate)",
    "W2": "wild-type holds the teacher's policy better than its rewired null (frozen clones)",
    "W3": "wild-type > rewired-null three-factor from the plastic-set clone (primary)",
    "W4": "three-factor from the clone > its frozen clone (the rule improves a competent start)",
    "W5": "three-factor from the clone > its Hebbian clone (reward matters from a competent start)",
    "W6": "PPO from the full-set clone > low-noise PPO from random weights (the warm start helps PPO)",
}
GATE_TESTS: tuple[str, ...] = ("W1", "W3", "W4", "W5")

Scanned = list[tuple[str, int, SeedRecord]]


# --- reading ----------------------------------------------------------------------------


def scan_campaign(campaign_dir: Path, experiments: Path = EXPERIMENTS) -> Scanned:
    """Read every registered run log under ``<campaign_dir>/logs`` (or the dir itself)."""
    log_dir = campaign_dir / "logs" if (campaign_dir / "logs").is_dir() else campaign_dir
    found: Scanned = []
    for log in sorted(log_dir.glob("*.log")):
        match = LABEL.match(log.name)
        if match is None or match.group("rate"):
            print(f"  WARN: skipping log with an unrecognised label: {log.name}")
            continue
        stem = match.group("stem")
        if stem not in ARMS:
            print(f"  WARN: skipping log whose config stem is not a registered arm: {log.name}")
            continue
        record = read_log(log, experiments)
        if record is None:
            print(f"  WARN: no parseable run lines in {log.name} - dropped")
            continue
        found.append((ARMS[stem], int(match.group("seed")), record))
    return found


def read_manifest(manifest: Path, experiments: Path = EXPERIMENTS) -> Scanned:
    """``<arm> <seed> <log>`` per line; blank and ``#`` lines skipped."""
    found: Scanned = []
    for raw in manifest.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3 or not parts[1].isdigit() or parts[0] not in ARM_KEYS:
            print(f"  WARN: skipping manifest line: {raw!r}")
            continue
        record = read_log(REPO / parts[2], experiments)
        if record is None:
            print(f"  WARN {parts[0]} seed {parts[1]}: no parseable plateau - dropped")
            continue
        found.append((parts[0], int(parts[1]), record))
    return found


def group_panel(scanned: Scanned) -> dict[str, dict[int, SeedRecord]]:
    """Group by arm and seed; seeds 1-8 only.

    A run's length must be the arm's budget or its registered extension, and of a
    duplicate the extension wins.
    """
    panel: dict[str, dict[int, SeedRecord]] = {}
    for arm, seed, record in scanned:
        if seed not in PANEL_SEEDS:
            msg = (
                f"seed {seed} ({arm}) is outside the panel seeds {PANEL_SEEDS[0]}-{PANEL_SEEDS[-1]}"
            )
            raise ValueError(msg)
        allowed = {BUDGETS[arm], int(BUDGETS[arm] * EXTENSION)}
        if record.episodes not in allowed:
            msg = (
                f"{arm} seed {seed}: a run of {record.episodes} episodes is neither the budget "
                f"nor its registered extension ({sorted(allowed)})"
            )
            raise ValueError(msg)
        seeds = panel.setdefault(arm, {})
        held = seeds.get(seed)
        keep = record
        if held is not None:
            if held.episodes == record.episodes:
                msg = (
                    f"{arm} seed {seed}: two runs of {record.episodes} episodes - a duplicate "
                    "that is not an extension"
                )
                raise ValueError(msg)
            keep = max(held, record, key=lambda r: r.episodes)
        seeds[seed] = keep
    return panel


def read_random_floor(path: Path = PANEL2_CSV) -> dict[int, float]:
    """Panel 2's random-initialisation wild-type frozen floor on the panel seeds."""
    values: dict[int, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["arm"] == RANDOM_FLOOR_ARM and int(row["seed"]) in PANEL_SEEDS:
                values[int(row["seed"])] = float(row["success"])
    if set(values) != set(PANEL_SEEDS):
        msg = f"panel 2's table lacks {RANDOM_FLOOR_ARM} on seeds {sorted(set(PANEL_SEEDS) - set(values))}"
        raise ValueError(msg)
    return values


def successes(panel: dict[str, dict[int, SeedRecord]]) -> dict[str, dict[int, float]]:
    """Reduce records to the ranked metric."""
    return {arm: {s: r.success for s, r in seeds.items()} for arm, seeds in panel.items()}


# --- confirmatory -----------------------------------------------------------------------


def family_tests(
    values: dict[str, dict[int, float]],
    random_floor: dict[int, float],
) -> dict[str, dict]:
    """Compute W1-W6 on the panel seeds, corrected together."""
    pool = {**values, "random_frozen": restrict(random_floor, PANEL_SEEDS)}
    raw = {
        t: paired(restrict(pool.get(a, {}), PANEL_SEEDS), restrict(pool.get(b, {}), PANEL_SEEDS))
        for t, (a, b) in FAMILY_PAIRS.items()
    }
    qs = bh_fdr([raw[t]["wilcoxon_p"] for t in FAMILY])
    for test, q in zip(FAMILY, qs, strict=True):
        stats = raw[test]
        stats["bh_q"] = q
        stats["sufficient"] = stats["n"] >= MIN_PAIRED_SEEDS
        stats["complete"] = tuple(stats["seeds"]) == PANEL_SEEDS
        stats["pass"] = bool(stats["sufficient"] and q < SIG_Q and stats["mean_delta"] > 0)
        stats["reverse"] = bool(stats["sufficient"] and stats["ci_hi"] < 0.0)
        stats["reads"] = READS[test]
    return raw


def verdict(tests: dict[str, dict]) -> str:
    """Assign the verdict by the ordered map: the first condition that holds names it."""
    w3 = tests["W3"]
    ordered: tuple[tuple[str, bool], ...] = (
        ("insufficient_seeds", not all(tests[t]["sufficient"] for t in GATE_TESTS)),
        ("clone_fail", not tests["W1"]["pass"]),
        ("sanity_floor_fail", not (tests["W4"]["pass"] and tests["W5"]["pass"])),
        ("rewired_beats_wild_type", w3["ci_hi"] < 0.0),
        ("specific_wiring", bool(w3["pass"])),
        ("degree_statistics", w3["ci_lo"] <= 0.0 <= w3["ci_hi"]),
    )
    return next((name for name, holds in ordered if holds), "inconclusive")


def annotate(tests: dict[str, dict]) -> dict[str, bool]:
    """Describe what W2, W6 and the W4 interval say; none changes the verdict."""
    return {
        "wild_type_holds_better": bool(tests["W2"]["pass"]),
        "warm_start_helps_ppo": bool(tests["W6"]["pass"]),
        "rule_destroys_clone": bool(tests["W4"]["sufficient"] and tests["W4"]["ci_hi"] < 0.0),
    }


# --- descriptive ------------------------------------------------------------------------


DESCRIPTIVE_PAIRS: tuple[tuple[str, str], ...] = (
    ("wt_fullclone_frozen", "rn_fullclone_frozen"),
    ("wt_fullclone_ppo", "rn_fullclone_ppo"),
    ("wt_ppo", "rn_ppo"),
    ("wt_fullclone_frozen", "wt_clone_frozen"),
    ("rn_clone_plastic", "rn_clone_frozen"),
    ("rn_clone_plastic", "rn_clone_hebbian"),
    ("rn_fullclone_ppo", "rn_ppo"),
)


def descriptive_pairs(values: dict[str, dict[int, float]]) -> list[dict]:
    """List the named pairs first, then every remaining pair, all uncorrected and labelled."""
    rows = []
    family = set(FAMILY_PAIRS.values())
    seen: set[tuple[str, str]] = set()
    ordered = list(DESCRIPTIVE_PAIRS) + [
        p for p in combinations(ARM_KEYS, 2) if p not in DESCRIPTIVE_PAIRS
    ]
    for a, b in ordered:
        key = (a, b)
        if (
            key in seen
            or key[::-1] in seen
            or key in family
            or key[::-1] in family
            or a not in values
            or b not in values
        ):
            continue
        seen.add(key)
        stats = paired(restrict(values[a], PANEL_SEEDS), restrict(values[b], PANEL_SEEDS))
        stats.update({"a": a, "b": b, "descriptive": True, "named": (a, b) in DESCRIPTIVE_PAIRS})
        rows.append(stats)
    return rows


def against_ceiling(values: dict[str, dict[int, float]], ceiling: float | None) -> dict[str, dict]:
    """Every arm's mean as a fraction of the teacher's frozen plateau tail, descriptive."""
    out: dict[str, dict] = {}
    for arm in ARM_KEYS:
        seeds = restrict(values.get(arm, {}), PANEL_SEEDS)
        mean = float(np.mean(list(seeds.values()))) if seeds else math.nan
        out[arm] = {"mean": mean, "fraction_of_ceiling": (mean / ceiling) if ceiling else None}
    return out


def clone_fits(records: list[dict[str, Any]]) -> dict[str, dict]:
    """Per clone set and wiring: held-out losses and the weak flags, descriptive."""
    out: dict[str, dict] = {}
    for r in records:
        key = f"{r['parameter_set']}_{r['wiring']}"
        slot = out.setdefault(key, {"held_out_losses": {}, "weak_seeds": [], "failed_seeds": []})
        if "failed" in r:
            slot["failed_seeds"].append(r["seed"])
            continue
        slot["held_out_losses"][r["seed"]] = r.get("held_out_loss")
        if r.get("weak"):
            slot["weak_seeds"].append(r["seed"])
    return out


def extensions_needed(panel: dict[str, dict[int, SeedRecord]]) -> list[dict]:
    """List the runs the plateau detector marks non-converged at their budget; one fresh run at 1.5x each."""
    rows = []
    for arm in ARM_KEYS:
        limit = int(BUDGETS[arm] * EXTENSION)
        for seed, record in sorted(panel.get(arm, {}).items()):
            if record.converged is False and record.episodes < limit:
                rows.append(
                    {"arm": arm, "seed": seed, "episodes": record.episodes, "extend_to": limit},
                )
    return rows


def analyse(
    panel: dict[str, dict[int, SeedRecord]],
    random_floor: dict[int, float],
    out: dict,
    *,
    ceiling: float | None = None,
    clones: list[dict[str, Any]] | None = None,
) -> dict:
    """Family, verdict, annotations, descriptive layers."""
    values = successes(panel)
    tests = family_tests(values, random_floor)
    result = verdict(tests)
    out["family"] = {t: tests[t] for t in FAMILY}
    out["verdict"] = {
        "verdict": result,
        "annotations": annotate(tests),
        "ensemble_invariance": {
            t: {"positive_seeds": tests[t]["positive_seeds"], "n": tests[t]["n"]}
            for t in ("W1", "W3", "W6")
        },
    }
    out["per_arm"] = {
        arm: {
            **distribution(restrict(values.get(arm, {}), PANEL_SEEDS)),
            "per_seed": dict(sorted(values.get(arm, {}).items())),
        }
        for arm in ARM_KEYS
    }
    out["random_floor"] = dict(sorted(random_floor.items()))
    out["teacher_ceiling"] = ceiling
    out["against_ceiling"] = against_ceiling(values, ceiling)
    out["clone_fits"] = clone_fits(clones or [])
    out["descriptive_pairs"] = descriptive_pairs(values)
    out["extensions_needed"] = extensions_needed(panel)
    return out


# --- output -----------------------------------------------------------------------------


def _print_panel(out: dict) -> None:
    print("\n" + "=" * 78)
    print("L4 WARM-START PANEL - plateau-tail full-clear success, paired seeds 1-8")
    print("=" * 78)
    for arm in ARM_KEYS:
        d = out["per_arm"][arm]
        if d["n"]:
            frac = out["against_ceiling"][arm]["fraction_of_ceiling"]
            frac_text = f"  of ceiling={frac:.2f}" if frac is not None else ""
            print(
                f"  {arm:20} n={d['n']}  mean={d['mean']:5.1f}  median={d['median']:5.1f}  competent={d['competent_fraction']:.2f}{frac_text}",
            )
    print("\n  Confirmatory family (BH-FDR, alpha 0.05):")
    for test in FAMILY:
        t = out["family"][test]
        flag = "PASS" if t["pass"] else ("REVERSE" if t["reverse"] else "fail")
        print(
            f"    {test}  d={t['mean_delta']:+6.2f}  CI[{t['ci_lo']:+6.2f},{t['ci_hi']:+6.2f}]  "
            f"p={t['wilcoxon_p']:.3f}  q={t['bh_q']:.3f}  +seeds={t['positive_seeds']}/{t['n']}  {flag}   {t['reads']}",
        )
        if not t["complete"]:
            print(f"        INCOMPLETE: {t['n']} of the registered seeds present")
    v = out["verdict"]
    print("-" * 78)
    print(f"  VERDICT: {v['verdict']}   annotations: {v['annotations']}")
    if out["extensions_needed"]:
        print(f"  EXTENSION NEEDED: {out['extensions_needed']}")


_CSV_FIELDS = [
    "arm",
    "seed",
    "success",
    "foods",
    "episodes",
    "converged",
    "onset",
    "evasion_rate",
    "temp_comfort",
    "peak_action_density",
]


def write_per_seed_csv(panel: dict[str, dict[int, SeedRecord]], path: Path) -> None:
    """One row per arm and seed."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for arm in ARM_KEYS:
            for seed, record in sorted(panel.get(arm, {}).items()):
                row = {k: v for k, v in asdict(record).items() if k in _CSV_FIELDS}
                writer.writerow({"arm": arm, "seed": seed, **row})


def write_curves_csv(panel: dict[str, dict[int, SeedRecord]], path: Path) -> None:
    """Write the per-seed learning curves, one row per window."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["arm", "seed", "window_end", "success"])
        for arm in ARM_KEYS:
            for seed, record in sorted(panel.get(arm, {}).items()):
                for index, value in enumerate(record.curve, 1):
                    writer.writerow([arm, seed, index * CURVE_WINDOW, f"{value:.2f}"])


def main(argv: list[str] | None = None) -> int:
    """Run the warm-start panel analysis from the command line; return the exit code."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--campaign-dir",
        type=Path,
        action="append",
        help="campaign dir(s) (reads logs/*.log); repeatable",
    )
    source.add_argument("--manifest", type=Path, help="<arm> <seed> <log> per line")
    ap.add_argument("--panel2-csv", type=Path, default=PANEL2_CSV)
    ap.add_argument("--teacher-json", type=Path, default=None)
    ap.add_argument("--clones-json", type=Path, default=None)
    ap.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--curves", type=Path, default=None)
    args = ap.parse_args(argv)

    scanned: Scanned = []
    if args.campaign_dir:
        for directory in args.campaign_dir:
            scanned += scan_campaign(directory, args.experiments_dir)
    else:
        scanned = read_manifest(args.manifest, args.experiments_dir)
    ceiling = (
        json.loads(args.teacher_json.read_text()).get("ceiling_plateau_tail")
        if args.teacher_json
        else None
    )
    clones = json.loads(args.clones_json.read_text()) if args.clones_json else None
    out: dict = {}
    try:
        panel = group_panel(scanned)
        analyse(panel, read_random_floor(args.panel2_csv), out, ceiling=ceiling, clones=clones)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print_panel(out)
    if args.csv:
        write_per_seed_csv(panel, args.csv)
    if args.curves:
        write_curves_csv(panel, args.curves)
    if args.out:
        args.out.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
