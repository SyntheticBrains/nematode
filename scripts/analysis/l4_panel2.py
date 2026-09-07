#!/usr/bin/env python
"""L4 panel 2: the Hebbian wiring contrast, the prior over policies, and count-scaled init.

Eight arms on one cell: the wild-type connectome and its degree-preserving rewired-null,
under degree-scaled and under synapse-count-scaled initialisation, each frozen and each
under the unmodulated Hebbian rule. The per-seed ranked metric and the statistics layer
are the committed ones (``t7_continuous_ranking.plateau_tail``;
``weight_search_architecture_ranking``), read through the first panel's log reader, so
the two panels are measured identically.

Two campaigns feed one analysis. The four Hebbian arms run on paired seeds 1-16; the four
frozen arms run on paired seeds 1-64 as a prior sweep, whose first sixteen seeds double
as the learning-gain floors. A frozen arm is a fixed policy, so its plateau tail is a
success rate, and the sweep reports for every arm the distribution of tails and the
competent fraction: the share of seeds whose tail is at least ``COMPETENT_THRESHOLD``
with no learning at all.

What is confirmatory is fixed here, in code, before any data exist:

- four one-sided paired tests corrected together as one BH-FDR family -- P1 wild-type
  Hebbian > rewired Hebbian under degree-scaled initialisation (the primary); P2 the same
  contrast under count-scaled initialisation; P3 count-scaled > degree-scaled on the
  wild-type Hebbian arm; P4 wild-type frozen > rewired frozen over the sweep seeds;
- a verdict assigned from P1 alone, in the rewired-null control's vocabulary:
  ``insufficient_seeds``, ``specific_wiring``, ``rewired_beats_wild_type``,
  ``degree_statistics``, ``inconclusive``. P2-P4 annotate the verdict and never change it.

Everything else -- the remaining pairwise deltas, the learning gains, the sweep
distributions, curves -- is descriptive and labelled so.

Usage::

    uv run python scripts/analysis/l4_panel2.py --campaign-dir campaigns/l4-panel2
        --sweep-dir campaigns/l4-panel2-sweep --out panel2.json --csv per-seed.csv
        --curves curves.csv
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

import numpy as np
from l4_panel import _LABEL as LABEL

# The first panel's reader (metric, experiment record, curve) and the committed
# statistics layer, reused verbatim.
from l4_panel import EXPERIMENTS, REPO, SeedRecord, paired, read_log
from weight_search_architecture_ranking import bh_fdr

_STEM = "connectomeppo_small_continuous2d_combined_klinotaxis_plastic"

# Config stem -> arm key. The registry is the only place a log is tied to an arm.
ARMS: dict[str, str] = {
    f"{_STEM}_frozen": "wt_frozen",
    f"{_STEM}_frozen_rewired_null": "rn_frozen",
    f"{_STEM}_frozen_countinit": "wt_frozen_count",
    f"{_STEM}_frozen_rewired_null_countinit": "rn_frozen_count",
    f"{_STEM}_hebbian": "wt_hebbian",
    f"{_STEM}_hebbian_rewired_null": "rn_hebbian",
    f"{_STEM}_hebbian_countinit": "wt_hebbian_count",
    f"{_STEM}_hebbian_rewired_null_countinit": "rn_hebbian_count",
}
ARM_KEYS: tuple[str, ...] = tuple(ARMS.values())
STEM_OF: dict[str, str] = {arm: stem for stem, arm in ARMS.items()}
FROZEN_ARMS: tuple[str, ...] = ("wt_frozen", "rn_frozen", "wt_frozen_count", "rn_frozen_count")
HEBBIAN_ARMS: tuple[str, ...] = ("wt_hebbian", "rn_hebbian", "wt_hebbian_count", "rn_hebbian_count")
FLOOR_OF: dict[str, str] = {
    "wt_hebbian": "wt_frozen",
    "rn_hebbian": "rn_frozen",
    "wt_hebbian_count": "wt_frozen_count",
    "rn_hebbian_count": "rn_frozen_count",
}

HEBBIAN_SEEDS: tuple[int, ...] = tuple(range(1, 17))
SWEEP_SEEDS: tuple[int, ...] = tuple(range(1, 65))
SEEDS_OF: dict[str, tuple[int, ...]] = {
    **dict.fromkeys(HEBBIAN_ARMS, HEBBIAN_SEEDS),
    **dict.fromkeys(FROZEN_ARMS, SWEEP_SEEDS),
}
HEBBIAN_BUDGET = 1000
SWEEP_BUDGET = 600
EXTENSION_BUDGET = 1500
COMPETENT_THRESHOLD = 20.0  # plateau-tail % at or above which an untrained policy is competent
SIG_Q = 0.05
CURVE_WINDOW = 250
MIN_PAIRED_SEEDS = 2  # a paired Wilcoxon needs at least two common seeds
FAMILY: tuple[str, ...] = ("P1", "P2", "P3", "P4")
FAMILY_PAIRS: dict[str, tuple[str, str]] = {
    "P1": ("wt_hebbian", "rn_hebbian"),
    "P2": ("wt_hebbian_count", "rn_hebbian_count"),
    "P3": ("wt_hebbian_count", "wt_hebbian"),
    "P4": ("wt_frozen", "rn_frozen"),
}
READS: dict[str, str] = {
    "P1": "wild-type Hebbian > rewired-null Hebbian, degree-scaled init (primary)",
    "P2": "wild-type Hebbian > rewired-null Hebbian, count-scaled init",
    "P3": "count-scaled > degree-scaled on the wild-type Hebbian arm",
    "P4": "wild-type frozen > rewired-null frozen over the sweep seeds (the prior)",
}

# One scanned run: arm key, seed, record.
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
        if len(parts) != 3 or not parts[1].isdigit():
            print(f"  WARN: skipping malformed manifest line: {raw!r}")
            continue
        arm, seed, log = parts[0], int(parts[1]), Path(parts[2])
        if arm not in ARM_KEYS:
            print(f"  WARN: skipping manifest line with unknown arm {arm!r}: {raw!r}")
            continue
        record = read_log(REPO / log, experiments)
        if record is None:
            print(f"  WARN {arm} seed {seed}: no parseable plateau in {log} - dropped")
            continue
        found.append((arm, seed, record))
    return found


def group_panel(scanned: Scanned) -> dict[str, dict[int, SeedRecord]]:
    """``{arm: {seed: record}}``; refuses any seed outside the arm's registered range."""
    panel: dict[str, dict[int, SeedRecord]] = {}
    for arm, seed, record in scanned:
        allowed = SEEDS_OF[arm]
        if seed not in allowed:
            msg = f"seed {seed} ({arm}) is outside the arm's registered seeds {allowed[0]}-{allowed[-1]}"
            raise ValueError(msg)
        seeds = panel.setdefault(arm, {})
        if seed in seeds:
            print(f"  WARN {arm} seed {seed}: duplicate run - the later log overwrites the earlier")
        seeds[seed] = record
    return panel


def successes(panel: dict[str, dict[int, SeedRecord]]) -> dict[str, dict[int, float]]:
    """Reduce records to the ranked metric."""
    return {arm: {s: r.success for s, r in seeds.items()} for arm, seeds in panel.items()}


def restrict(values: dict[int, float], seeds: tuple[int, ...]) -> dict[int, float]:
    """Keep only the given seeds."""
    return {s: v for s, v in values.items() if s in seeds}


# --- confirmatory -----------------------------------------------------------------------


def family_tests(values: dict[str, dict[int, float]]) -> dict[str, dict]:
    """Compute the four pre-registered tests, corrected together, each with its pass verdict.

    P1-P3 read the Hebbian seeds only, so a frozen arm's extra sweep seeds never enter
    a Hebbian contrast; P4 reads the full sweep.
    """
    raw: dict[str, dict] = {}
    for test, (a, b) in FAMILY_PAIRS.items():
        seeds = SWEEP_SEEDS if test == "P4" else HEBBIAN_SEEDS
        raw[test] = paired(
            restrict(values.get(a, {}), seeds),
            restrict(values.get(b, {}), seeds),
        )
    qs = bh_fdr([raw[t]["wilcoxon_p"] for t in FAMILY])
    for test, q in zip(FAMILY, qs, strict=True):
        stats = raw[test]
        stats["bh_q"] = q
        # The registered sufficiency rule (a paired Wilcoxon needs two common seeds) decides
        # the verdict; ``complete`` is descriptive and says whether every registered seed of
        # the pair is present, so a partial campaign is visible beside its result.
        stats["sufficient"] = stats["n"] >= MIN_PAIRED_SEEDS
        registered = SWEEP_SEEDS if test == "P4" else HEBBIAN_SEEDS
        stats["complete"] = tuple(stats["seeds"]) == registered
        stats["pass"] = bool(stats["sufficient"] and q < SIG_Q and stats["mean_delta"] > 0)
        stats["reverse"] = bool(stats["sufficient"] and stats["ci_hi"] < 0.0)
        stats["reads"] = READS[test]
    return raw


def verdict(tests: dict[str, dict]) -> str:
    """Assign the verdict from P1 alone, by the rewired-null control's ordered map."""
    p1 = tests["P1"]
    if not p1["sufficient"]:
        return "insufficient_seeds"
    if p1["pass"]:
        return "specific_wiring"
    if p1["ci_hi"] < 0.0:
        return "rewired_beats_wild_type"
    if p1["ci_lo"] <= 0.0 <= p1["ci_hi"]:
        return "degree_statistics"
    return "inconclusive"


def annotate(result: str, tests: dict[str, dict]) -> dict[str, bool | str]:
    """Describe what P2-P4 say about the verdict; they never change it."""
    out: dict[str, bool | str] = {
        "count_preserves_contrast": bool(tests["P2"]["pass"]),
        "count_improves_wild_type": bool(tests["P3"]["pass"]),
        "prior_differs": bool(tests["P4"]["pass"]),
    }
    if result == "specific_wiring":
        out["reading"] = (
            "the wild-type advantage is already present in the untrained prior (P4 passes)"
            if tests["P4"]["pass"]
            else "the wild-type advantage is created by Hebbian alignment, not present in the "
            "untrained prior (P4 does not pass)"
        )
    else:
        out["reading"] = (
            "no confirmed wild-type Hebbian advantage; P2-P4 are reported descriptively"
        )
    return out


# --- descriptive ------------------------------------------------------------------------


def distribution(values: dict[int, float]) -> dict:
    """Per-arm summary of the plateau tails: moments, quantiles and the competent fraction."""
    if not values:
        return {"n": 0}
    arr = np.array(sorted(values.values()), dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "competent_fraction": float(np.mean(arr >= COMPETENT_THRESHOLD)),
        "sorted": [float(v) for v in arr],
    }


def prior_sweep(values: dict[str, dict[int, float]]) -> dict:
    """Summarise the frozen arms over the sweep seeds: distributions and descriptive pairs."""
    out: dict = {"threshold": COMPETENT_THRESHOLD, "arms": {}, "pairs": []}
    for arm in FROZEN_ARMS:
        out["arms"][arm] = distribution(values.get(arm, {}))
    for a, b in (("wt_frozen_count", "wt_frozen"), ("rn_frozen_count", "rn_frozen")):
        stats = paired(values.get(a, {}), values.get(b, {}))
        stats.update({"a": a, "b": b, "descriptive": True})
        out["pairs"].append(stats)
    return out


def learning_gains(values: dict[str, dict[int, float]]) -> dict[str, dict]:
    """Each Hebbian arm minus its own frozen arm on the Hebbian seeds, per seed."""
    out: dict[str, dict] = {}
    for hebbian, frozen in FLOOR_OF.items():
        h = restrict(values.get(hebbian, {}), HEBBIAN_SEEDS)
        f = restrict(values.get(frozen, {}), HEBBIAN_SEEDS)
        per_seed = {s: h[s] - f[s] for s in sorted(set(h) & set(f))}
        out[hebbian] = {
            "floor": frozen,
            "per_seed": per_seed,
            "mean": float(np.mean(list(per_seed.values()))) if per_seed else math.nan,
            "positive_seeds": int(sum(1 for d in per_seed.values() if d > 0)),
            "n": len(per_seed),
        }
    return out


def descriptive_pairs(values: dict[str, dict[int, float]]) -> list[dict]:
    """Every pair of arms outside the family, uncorrected, on the seeds both arms share."""
    rows = []
    family = set(FAMILY_PAIRS.values())
    for a, b in combinations(ARM_KEYS, 2):
        if a not in values or b not in values or (a, b) in family or (b, a) in family:
            continue
        seeds = SWEEP_SEEDS if a in FROZEN_ARMS and b in FROZEN_ARMS else HEBBIAN_SEEDS
        stats = paired(restrict(values[a], seeds), restrict(values[b], seeds))
        stats.update({"a": a, "b": b, "descriptive": True})
        rows.append(stats)
    return rows


def extensions_needed(panel: dict[str, dict[int, SeedRecord]]) -> list[dict]:
    """Hebbian runs the plateau detector marks non-converged: each gets one fresh run at 1500."""
    rows = []
    for arm in HEBBIAN_ARMS:
        for seed, record in sorted(panel.get(arm, {}).items()):
            if record.converged is False and record.episodes < EXTENSION_BUDGET:
                rows.append({"arm": arm, "seed": seed, "episodes": record.episodes})
    return rows


def analyse(panel: dict[str, dict[int, SeedRecord]], out: dict) -> dict:
    """Family, verdict, annotations, sweep, gains and descriptive pairs, from the ranked metric."""
    values = successes(panel)
    tests = family_tests(values)
    result = verdict(tests)
    out["family"] = {t: tests[t] for t in FAMILY}
    out["verdict"] = {
        "verdict": result,
        "annotations": annotate(result, tests),
        "ensemble_invariance": {
            t: {"positive_seeds": tests[t]["positive_seeds"], "n": tests[t]["n"]}
            for t in ("P1", "P4")
        },
    }
    out["per_arm"] = {
        arm: {
            "mean": float(np.mean(list(seeds.values()))) if seeds else math.nan,
            "n": len(seeds),
            "per_seed": dict(sorted(seeds.items())),
        }
        for arm, seeds in values.items()
    }
    out["prior_sweep"] = prior_sweep(values)
    out["learning_gains"] = learning_gains(values)
    out["descriptive_pairs"] = descriptive_pairs(values)
    out["extensions_needed"] = extensions_needed(panel)
    return out


# --- output -----------------------------------------------------------------------------


def _print_panel(out: dict) -> None:
    print("\n" + "=" * 78)
    print("L4 PANEL 2 - plateau-tail full-clear success, paired seeds")
    print("=" * 78)
    for arm in ARM_KEYS:
        row = out["per_arm"].get(arm)
        if row is None:
            continue
        print(f"  {arm:17} {row['mean']:6.2f}   n={row['n']}")
    print("\n  Confirmatory family (BH-FDR, alpha 0.05):")
    for test in FAMILY:
        t = out["family"][test]
        flag = "PASS" if t["pass"] else ("REVERSE" if t["reverse"] else "fail")
        print(
            f"    {test}  d={t['mean_delta']:+6.2f}  CI[{t['ci_lo']:+6.2f},{t['ci_hi']:+6.2f}]  "
            f"p={t['wilcoxon_p']:.3f}  q={t['bh_q']:.3f}  +seeds={t['positive_seeds']}/{t['n']}  "
            f"{flag}   {t['reads']}",
        )
        if not t["complete"]:
            print(f"        INCOMPLETE: {t['n']} of the registered seeds present")
    print(f"\n  Prior sweep (competent fraction at >= {out['prior_sweep']['threshold']:.0f}%):")
    for arm in FROZEN_ARMS:
        d = out["prior_sweep"]["arms"][arm]
        if d["n"] == 0:
            continue
        print(
            f"    {arm:17} n={d['n']:3d}  mean={d['mean']:5.1f}  median={d['median']:5.1f}  "
            f"[{d['min']:.1f}, {d['max']:.1f}]  competent={d['competent_fraction']:.2f}",
        )
    print("\n  Learning gains (Hebbian minus own frozen, seeds 1-16):")
    for arm, g in out["learning_gains"].items():
        if g["n"]:
            print(f"    {arm:17} mean={g['mean']:+6.2f}  +seeds={g['positive_seeds']}/{g['n']}")
    v = out["verdict"]
    print("-" * 78)
    print(f"  VERDICT: {v['verdict']}   -- {v['annotations']['reading']}")
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
    """One row per arm and seed with the ranked metric and the descriptive sub-metrics."""
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
    """Run the panel-2 analysis from the command line; return the exit code."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--campaign-dir", type=Path, help="Hebbian campaign dir (reads logs/*.log)")
    source.add_argument("--manifest", type=Path, help="<arm> <seed> <log> per line")
    ap.add_argument("--sweep-dir", type=Path, default=None, help="prior-sweep campaign dir")
    ap.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS)
    ap.add_argument("--out", type=Path, default=None, help="write the summary JSON here")
    ap.add_argument("--csv", type=Path, default=None, help="write the per-seed table here")
    ap.add_argument("--curves", type=Path, default=None, help="write the learning curves here")
    args = ap.parse_args(argv)

    scanned = (
        scan_campaign(args.campaign_dir, args.experiments_dir)
        if args.campaign_dir
        else read_manifest(args.manifest, args.experiments_dir)
    )
    if args.sweep_dir:
        scanned += scan_campaign(args.sweep_dir, args.experiments_dir)
    out: dict = {}
    try:
        panel = group_panel(scanned)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    analyse(panel, out)
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
