#!/usr/bin/env python
"""The decorrelation test: does a decorrelating term recover what grounding the signs cost.

The sign-grounding test found that grounding the substrate's synapse signs made reward-free
Hebbian learning substantially worse (wild-type 31.5 to 14.0, rewired null 17.4 to 9.1) and
predicted that a rule with an anti-Hebbian or decorrelating term would recover the loss. This
re-runs that test's own Hebbian protocol under each variant and asks whether it did.

What is confirmatory is fixed here, in code, before any data exist:

- four one-sided paired tests corrected together as one BH-FDR family -- **D1** the wild-type
  anti-Hebbian arm over the committed wild-type grounded Hebbian values (the prediction),
  **D2** the wild-type Oja arm over the same, **D3** wild-type over rewired null under the
  anti-Hebbian variant, **D4** the same under Oja;
- an ordered verdict map: ``insufficient_seeds``, ``no_recovery`` (neither D1 nor D2 confirms --
  the outcome in which the prediction fails), ``recovery_specific`` (D1 only),
  ``recovery_general`` (D2 only), ``recovery_both``. D3 and D4 annotate and never decide.

Two annotations are computed and reported without touching the verdict: ``full_recovery``, whether
a recovered arm's 80% bootstrap interval reaches the committed RANDOM-sign mean for its wiring --
the difference between a term that helps and one that restores what grounding cost -- and the mean
decorrelation share from the rule's own telemetry, so a recovery whose share is near zero is
recorded as attributable to something other than the term.

Usage::

    uv run python scripts/analysis/l4_decorrelation.py --campaign-dir campaigns/l4-decorrelation
        --out panel.json --csv per-seed.csv --curves curves.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from l4_panel import _LABEL as LABEL
from l4_panel import EXPERIMENTS, REPO, SeedRecord, _experiment_json, paired, read_log
from weight_search_architecture_ranking import bh_fdr

_STEM = "connectomeppo_small_continuous2d_combined_klinotaxis_plastic_hebbian"

ARMS: dict[str, str] = {
    f"{_STEM}_atlassigns_antihebb": "wt_antihebb",
    f"{_STEM}_rewired_null_atlassigns_antihebb": "rn_antihebb",
    f"{_STEM}_atlassigns_oja": "wt_oja",
    f"{_STEM}_rewired_null_atlassigns_oja": "rn_oja",
}
ARM_KEYS = tuple(ARMS.values())

SEEDS = tuple(range(1, 17))
BUDGET = 1000
EXTENSION = 1.5

# The sign-grounding test's committed per-seed table: the grounded Hebbian arms these variants
# are read against, and never re-run.
GROUNDED_CSV = (
    REPO
    / "docs"
    / "experiments"
    / "logbooks"
    / "supporting"
    / "044-l4-atlas-signs"
    / "per-seed.csv"
)
GROUNDED_OF = {"wt_antihebb": "wt_hebbian_atlas", "rn_antihebb": "rn_hebbian_atlas"}
GROUNDED_OF.update({"wt_oja": "wt_hebbian_atlas", "rn_oja": "rn_hebbian_atlas"})
# Panel 2's committed random-sign means, the level grounding cost. Descriptive: the target
# `full_recovery` asks about, never a test.
RANDOM_SIGN_MEAN = {"wt": 31.5, "rn": 17.4}

ALPHA = 0.05

Scanned = list[tuple[str, int, SeedRecord, Path]]
LogOf = dict[str, dict[int, Path]]


def read_grounded(path: Path = GROUNDED_CSV) -> dict[str, dict[int, float]]:
    """Read the committed grounded Hebbian per-seed values on this test's seeds."""
    values: dict[str, dict[int, float]] = {"wt_hebbian_atlas": {}, "rn_hebbian_atlas": {}}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["arm"] in values and int(row["seed"]) in SEEDS:
                values[row["arm"]][int(row["seed"])] = float(row["success"])
    for arm, seeds in values.items():
        if set(seeds) != set(SEEDS):
            msg = f"the committed table lacks {arm} on seeds {sorted(set(SEEDS) - set(seeds))}"
            raise ValueError(msg)
    return values


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
        found.append((ARMS[stem], int(match.group("seed")), record, log))
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
        log = REPO / parts[2]
        record = read_log(log, experiments)
        if record is None:
            print(f"  WARN {parts[0]} seed {parts[1]}: no parseable plateau - dropped")
            continue
        found.append((parts[0], int(parts[1]), record, log))
    return found


def group_panel(scanned: Scanned) -> tuple[dict[str, dict[int, SeedRecord]], LogOf]:
    """Group by arm and seed; of a duplicate the registered extension wins."""
    panel: dict[str, dict[int, SeedRecord]] = {}
    logs: LogOf = {}
    allowed = {BUDGET, int(BUDGET * EXTENSION)}
    for arm, seed, record, log in scanned:
        if seed not in SEEDS:
            msg = f"seed {seed} ({arm}) is outside the test's seeds {SEEDS[0]}-{SEEDS[-1]}"
            raise ValueError(msg)
        if record.episodes not in allowed:
            msg = (
                f"{arm} seed {seed}: a run of {record.episodes} episodes is neither the budget "
                f"nor its registered extension ({sorted(allowed)})"
            )
            raise ValueError(msg)
        held = panel.setdefault(arm, {}).get(seed)
        if held is not None and held.episodes == record.episodes:
            msg = f"{arm} seed {seed}: two runs of {record.episodes} episodes - a duplicate"
            raise ValueError(msg)
        if held is None or record.episodes > held.episodes:
            panel.setdefault(arm, {})[seed] = record
            logs.setdefault(arm, {})[seed] = log
    return panel, logs


def successes(panel: dict[str, dict[int, SeedRecord]]) -> dict[str, dict[int, float]]:
    """Reduce records to the ranked metric."""
    return {arm: {s: r.success for s, r in seeds.items()} for arm, seeds in panel.items()}


def family_tests(
    values: dict[str, dict[int, float]],
    grounded: dict[str, dict[int, float]],
) -> dict[str, dict]:
    """Compute the four registered tests, corrected together."""
    definitions = {
        "D1": (
            "wt_antihebb",
            grounded["wt_hebbian_atlas"],
            "wt anti-Hebbian - wt grounded Hebbian",
        ),
        "D2": ("wt_oja", grounded["wt_hebbian_atlas"], "wt Oja - wt grounded Hebbian"),
        "D3": ("wt_antihebb", values.get("rn_antihebb", {}), "wt - rn under anti-Hebbian"),
        "D4": ("wt_oja", values.get("rn_oja", {}), "wt - rn under Oja"),
    }
    tests: dict[str, dict] = {}
    for name, (arm, comparator, label) in definitions.items():
        result = paired(values.get(arm, {}), comparator)
        result["contrast"] = label
        result["complete"] = len(result.get("seeds", [])) == len(SEEDS)
        tests[name] = result
    qs = bh_fdr([tests[name]["wilcoxon_p"] for name in ("D1", "D2", "D3", "D4")])
    for name, q in zip(("D1", "D2", "D3", "D4"), qs, strict=True):
        tests[name]["q"] = q
        tests[name]["confirms"] = bool(tests[name]["complete"] and q < ALPHA)
    return tests


def verdict(tests: dict[str, dict]) -> str:
    """Assign the ordered verdict, whose map is fixed before any data exist."""
    if not (tests["D1"]["complete"] and tests["D2"]["complete"]):
        return "insufficient_seeds"
    d1, d2 = tests["D1"]["confirms"], tests["D2"]["confirms"]
    if not d1 and not d2:
        return "no_recovery"
    if d1 and not d2:
        return "recovery_specific"
    if d2 and not d1:
        return "recovery_general"
    return "recovery_both"


def full_recovery(values: dict[str, dict[int, float]]) -> dict[str, dict]:
    """Whether each arm's 80% bootstrap interval reaches the committed random-sign mean.

    "The term helps" and "the term restores what grounding cost" are different claims, and
    only the second is what the sign-grounding test's prediction was about.
    """
    out: dict[str, dict] = {}
    rng = np.random.default_rng(20260909)
    for arm in ARM_KEYS:
        seeds = values.get(arm, {})
        target = RANDOM_SIGN_MEAN["wt" if arm.startswith("wt") else "rn"]
        if not seeds:
            out[arm] = {"mean": float("nan"), "target": target, "reaches": False, "ci": None}
            continue
        sample = np.array(sorted(seeds.values()), dtype=float)
        draws = rng.choice(sample, size=(10000, sample.size), replace=True).mean(axis=1)
        low, high = (float(np.quantile(draws, 0.1)), float(np.quantile(draws, 0.9)))
        out[arm] = {
            "mean": float(sample.mean()),
            "target": target,
            "ci": [low, high],
            "reaches": bool(high >= target),
        }
    return out


def decorrelation_shares(logs: LogOf, experiments: Path = EXPERIMENTS) -> dict[str, dict]:
    """Each arm's mean decorrelation share, from the runs' own exported telemetry.

    A recovery with a near-zero share was not produced by the term, whatever else it was.
    """
    out: dict[str, dict] = {}
    for arm in ARM_KEYS:
        values: list[float] = []
        for log in logs.get(arm, {}).values():
            experiment = _experiment_json(log.read_text(), experiments)
            exports = experiment.get("exports_path") if experiment else None
            if not exports:
                continue
            series = (
                REPO / exports / "session" / "data" / "tracking_plasticity_decorrelation_share.csv"
            )
            if not series.is_file():
                continue
            with series.open(newline="") as handle:
                run = [
                    float(row["plasticity_decorrelation_share"])
                    for row in csv.DictReader(handle)
                    if row.get("plasticity_decorrelation_share")
                ]
            if run:
                values.append(float(np.mean(run)))
        out[arm] = {
            "n_read": len(values),
            "mean": float(np.mean(values)) if values else float("nan"),
        }
    return out


def extensions_needed(panel: dict[str, dict[int, SeedRecord]]) -> list[dict]:
    """List the runs the plateau detector marks non-converged: the one registered extension."""
    needed: list[dict] = []
    for arm in ARM_KEYS:
        for seed, record in sorted(panel.get(arm, {}).items()):
            if record.episodes == BUDGET and record.converged is False:
                needed.append({"arm": arm, "seed": seed, "extend_to": int(BUDGET * EXTENSION)})
    return needed


def analyse(
    panel: dict[str, dict[int, SeedRecord]],
    logs: LogOf,
    experiments: Path = EXPERIMENTS,
    grounded_csv: Path = GROUNDED_CSV,
) -> dict:
    """Score the four arms against the committed table and assign the registered verdict."""
    values = successes(panel)
    grounded = read_grounded(grounded_csv)
    tests = family_tests(values, grounded)
    return {
        "seeds": list(SEEDS),
        "budget": BUDGET,
        "alpha": ALPHA,
        "comparator": {
            "grounded": {
                arm: {"mean": float(np.mean(list(seeds.values())))}
                for arm, seeds in grounded.items()
            },
            "random_sign_mean": RANDOM_SIGN_MEAN,
            "source": "the sign-grounding test's committed per-seed table; no arm re-run",
        },
        "arms": {
            arm: {
                "per_seed": {str(s): v for s, v in sorted(values.get(arm, {}).items())},
                "n": len(values.get(arm, {})),
                "mean": float(np.mean(list(values[arm].values())))
                if values.get(arm)
                else float("nan"),
            }
            for arm in ARM_KEYS
        },
        "tests": tests,
        "verdict": verdict(tests),
        "annotations": {
            "full_recovery": full_recovery(values),
            "decorrelation_share": decorrelation_shares(logs, experiments),
            "wiring_contrast_antihebb": tests["D3"]["confirms"],
            "wiring_contrast_oja": tests["D4"]["confirms"],
        },
        "extensions_needed": extensions_needed(panel),
    }


def _print_tests(out: dict) -> None:
    """Print the registered family."""
    print("\n  Registered family (BH-FDR, alpha = 0.05):")
    for name in ("D1", "D2", "D3", "D4"):
        test = out["tests"][name]
        mark = (
            "CONFIRMS" if test["confirms"] else ("incomplete" if not test["complete"] else "fail")
        )
        print(
            f"    {name} {test['contrast']:34} {test['mean_delta']:+6.2f}  "
            f"[{test['ci_lo']:+6.2f}, {test['ci_hi']:+6.2f}]  q={test['q']:.3f}  "
            f"{test.get('positive_seeds', 0)}/{len(test.get('seeds', []))}  {mark}",
        )


def _print_panel(out: dict) -> None:
    """Print the arms, the family and the annotations."""
    print(f"\nDecorrelation test (seeds 1-{max(out['seeds'])}, {out['budget']} episodes)")
    grounded = out["comparator"]["grounded"]
    print(
        f"  Comparators (committed, not re-run): wt grounded Hebbian "
        f"{grounded['wt_hebbian_atlas']['mean']:.1f}, rn {grounded['rn_hebbian_atlas']['mean']:.1f};"
        f" random-sign means wt {RANDOM_SIGN_MEAN['wt']}, rn {RANDOM_SIGN_MEAN['rn']}",
    )
    for arm in ARM_KEYS:
        row = out["arms"][arm]
        share = out["annotations"]["decorrelation_share"][arm]
        full = out["annotations"]["full_recovery"][arm]
        reaches = "reaches" if full["reaches"] else "short of"
        print(
            f"    {arm:12} mean {row['mean']:5.1f} (n={row['n']})  "
            f"share {share['mean']:.2f}  {reaches} the random-sign level ({full['target']})",
        )
    _print_tests(out)
    print(f"\n  VERDICT: {out['verdict']}")
    if out["extensions_needed"]:
        print(f"  Extensions needed: {out['extensions_needed']}")


def write_per_seed_csv(panel: dict[str, dict[int, SeedRecord]], path: Path) -> None:
    """One row per arm and seed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "seed", "success", "foods", "episodes", "converged", "onset"])
        for arm in ARM_KEYS:
            for seed, record in sorted(panel.get(arm, {}).items()):
                writer.writerow(
                    [
                        arm,
                        seed,
                        record.success,
                        record.foods,
                        record.episodes,
                        record.converged,
                        record.onset,
                    ],
                )


def write_curves_csv(panel: dict[str, dict[int, SeedRecord]], path: Path) -> None:
    """Learning curves, one row per arm, seed and block."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "seed", "block", "success"])
        for arm in ARM_KEYS:
            for seed, record in sorted(panel.get(arm, {}).items()):
                for block, value in enumerate(record.curve):
                    writer.writerow([arm, seed, block, f"{value:.4f}"])


def main(argv: list[str] | None = None) -> int:
    """Read the test's runs, apply the registered family and write the records."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--campaign-dir", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--curves", type=Path)
    args = parser.parse_args(argv)

    scanned = (
        read_manifest(args.manifest, args.experiments)
        if args.manifest
        else scan_campaign(args.campaign_dir, args.experiments)
    )
    if not scanned:
        print("no runs read", file=sys.stderr)
        return 1
    panel, logs = group_panel(scanned)
    out: dict[str, Any] = analyse(panel, logs, args.experiments)
    _print_panel(out)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    if args.csv:
        write_per_seed_csv(panel, args.csv)
    if args.curves:
        write_curves_csv(panel, args.curves)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
