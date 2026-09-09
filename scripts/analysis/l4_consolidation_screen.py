#!/usr/bin/env python
"""The clone assay: does a consolidation mechanism hold a competent policy.

A screen, not a confirmatory test. It reuses the seeds the warm-start panel already
reported, so it declares no multiple-comparisons family and assigns no verdict: a pass
licenses running the registered panel and nothing more.

One arm per mechanism, each the wild-type plastic clone arm with the mechanism's rule keys
and nothing else changed, started from that seed's plastic-set clone. Seeds 1-8 paired,
2000 episodes, no extension, the committed plateau-tail full-clear metric, read against the
warm-start panel's published frozen-clone values on the same seeds.

    holds     mean within HOLD_MEAN points of the frozen clone's mean, and at least
              HOLD_SEEDS of 8 seeds no more than HOLD_SEED points below their own
    improves  mean above the frozen clone's, and at least HOLD_SEEDS of 8 above their own
    pass      holds or improves

The endpoint cosine to the clone is reported beside the metric because a mechanism can pass
on behaviour while having rewritten the policy, which is what the diagnostic found the
unbraked rules do. The mean rate multiplier is reported beside it because a mechanism can
also hold a policy by not moving at all, and those are different results.

Usage::

    uv run python scripts/analysis/l4_consolidation_screen.py
        --campaign-dir campaigns/l4-consolidation --out screen.json --csv per-seed.csv
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
from l4_panel import EXPERIMENTS, REPO, SeedRecord, _experiment_json, read_log

_STEM = "connectomeppo_small_continuous2d_combined_klinotaxis"

# One arm per mechanism. The comparator is not re-run: it is the published table.
ARMS: dict[str, str] = {
    f"{_STEM}_plastic_clone_anchor": "anchor",
    f"{_STEM}_plastic_clone_rigidity": "rigidity",
    f"{_STEM}_plastic_clone_oracle": "oracle",
}
ARM_KEYS = tuple(ARMS.values())

SEEDS = tuple(range(1, 9))
BUDGET = 2000
# The warm-start panel's committed wild-type plastic-set frozen-clone values, seed by seed.
# Quoted rather than recomputed: the assay compares against what was published.
FROZEN_CLONE: dict[int, float] = {
    1: 39.3,
    2: 44.0,
    3: 40.0,
    4: 21.3,
    5: 47.1,
    6: 33.3,
    7: 61.3,
    8: 23.3,
}
FROZEN_MEAN = float(np.mean(list(FROZEN_CLONE.values())))

# The pass rule, fixed here before any run.
HOLD_MEAN = 5.0  # points the mean may sit below the frozen clone's mean
HOLD_SEED = 10.0  # points a seed may sit below its own frozen clone
HOLD_SEEDS = 6  # seeds of 8 that must clear that

CLONES = REPO / "campaigns" / "l4-warm-start" / "clones"

Scanned = list[tuple[str, int, SeedRecord, Path]]


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
            print(f"  WARN: skipping log whose config stem is not a screened arm: {log.name}")
            continue
        record = read_log(log, experiments)
        if record is None:
            print(f"  WARN: no parseable run lines in {log.name} - dropped")
            continue
        found.append((ARMS[stem], int(match.group("seed")), record, log))
    return found


def group(scanned: Scanned) -> tuple[dict[str, dict[int, SeedRecord]], dict[str, dict[int, Path]]]:
    """Group by arm and seed. The budget is exact: the assay registers no extension."""
    panel: dict[str, dict[int, SeedRecord]] = {}
    logs: dict[str, dict[int, Path]] = {}
    for arm, seed, record, log in scanned:
        if seed not in SEEDS:
            msg = f"seed {seed} ({arm}) is outside the assay's seeds {SEEDS[0]}-{SEEDS[-1]}"
            raise ValueError(msg)
        if record.episodes != BUDGET:
            msg = (
                f"{arm} seed {seed}: a run of {record.episodes} episodes is not the assay's "
                f"{BUDGET}, and the assay registers no extension"
            )
            raise ValueError(msg)
        if seed in panel.get(arm, {}):
            msg = f"{arm} seed {seed}: two runs at the budget - a duplicate"
            raise ValueError(msg)
        panel.setdefault(arm, {})[seed] = record
        logs.setdefault(arm, {})[seed] = log
    return panel, logs


def endpoint_cosine(log: Path, seed: int, experiments: Path = EXPERIMENTS) -> float | None:
    """Cosine between a run's final chemical weights and the clone it started from.

    A mechanism can hold the metric while having replaced the policy underneath it, which is
    what the unbraked rules were shown to do; the two are different results and this
    separates them. ``None`` when either endpoint is not on disk.
    """
    import torch

    experiment = _experiment_json(log.read_text(), experiments)
    exports = experiment.get("exports_path") if experiment else None
    final = (REPO / exports / "weights" / "final.pt") if exports else None
    clone = CLONES / f"plastic_wt_seed{seed}.pt"
    if final is None or not final.is_file() or not clone.is_file():
        return None
    started = torch.load(clone, weights_only=True)["topology"]["w_chem"].reshape(-1)
    ended = torch.load(final, weights_only=True)["topology"]["w_chem"].reshape(-1)
    return float(torch.nn.functional.cosine_similarity(started, ended, dim=0).item())


def rate_multiplier(log: Path, experiments: Path = EXPERIMENTS) -> float | None:
    """Read the run's mean effective rate multiplier from its exported telemetry.

    A mechanism that held the clone with a multiplier near zero stopped writing; one that
    held it near one consolidated while still learning. ``None`` when no export is found.
    """
    experiment = _experiment_json(log.read_text(), experiments)
    exports = experiment.get("exports_path") if experiment else None
    if not exports:
        return None
    series = REPO / exports / "session" / "data" / "tracking_plasticity_rate_multiplier.csv"
    if not series.is_file():
        return None
    with series.open(newline="") as handle:
        values = [
            float(row["plasticity_rate_multiplier"])
            for row in csv.DictReader(handle)
            if row.get("plasticity_rate_multiplier")
        ]
    return float(np.mean(values)) if values else None


def assess(values: dict[int, float]) -> dict[str, Any]:
    """Apply the assay's pass rule to one arm's per-seed values."""
    seeds = sorted(values)
    missing = [s for s in SEEDS if s not in values]
    deltas = {s: values[s] - FROZEN_CLONE[s] for s in seeds}
    mean = float(np.mean([values[s] for s in seeds])) if seeds else float("nan")
    at_or_above = sum(1 for s in seeds if deltas[s] >= 0.0)
    within_seed = sum(1 for s in seeds if deltas[s] >= -HOLD_SEED)
    complete = not missing
    holds = complete and (mean >= FROZEN_MEAN - HOLD_MEAN) and within_seed >= HOLD_SEEDS
    improves = complete and mean > FROZEN_MEAN and at_or_above >= HOLD_SEEDS
    return {
        "per_seed": {str(s): values[s] for s in seeds},
        "delta": {str(s): deltas[s] for s in seeds},
        "missing_seeds": missing,
        "n": len(seeds),
        "mean": mean,
        "mean_delta": mean - FROZEN_MEAN if seeds else float("nan"),
        "seeds_at_or_above": at_or_above,
        "seeds_within_hold": within_seed,
        "holds": holds,
        "improves": improves,
        "pass": holds or improves,
    }


def analyse(
    panel: dict[str, dict[int, SeedRecord]],
    logs: dict[str, dict[int, Path]],
    experiments: Path = EXPERIMENTS,
) -> dict:
    """Score every screened arm and describe how it got there."""
    out: dict[str, Any] = {
        "comparator": {
            "arm": "wt_clone_frozen",
            "per_seed": {str(s): v for s, v in FROZEN_CLONE.items()},
            "mean": FROZEN_MEAN,
            "source": "the warm-start panel's committed per-seed table",
        },
        "rule": {
            "hold_mean_points": HOLD_MEAN,
            "hold_seed_points": HOLD_SEED,
            "hold_seeds_of_eight": HOLD_SEEDS,
        },
        "screen_not_test": (
            "Reuses seeds already reported; declares no multiple-comparisons family and no "
            "verdict. A pass licenses running the registered panel and nothing more."
        ),
        "arms": {},
    }
    for arm in ARM_KEYS:
        seeds = panel.get(arm, {})
        result = assess({s: r.success for s, r in seeds.items()})
        cosines = {str(s): endpoint_cosine(logs[arm][s], s, experiments) for s in sorted(seeds)}
        multipliers = {str(s): rate_multiplier(logs[arm][s], experiments) for s in sorted(seeds)}
        read = [v for v in cosines.values() if v is not None]
        used = [v for v in multipliers.values() if v is not None]
        result["cosine_to_clone"] = cosines
        result["cosine_mean"] = float(np.mean(read)) if read else float("nan")
        result["rate_multiplier"] = multipliers
        result["rate_multiplier_mean"] = float(np.mean(used)) if used else float("nan")
        out["arms"][arm] = result
    return out


def _print_arm(arm: str, result: dict) -> None:
    """Print one arm's line of the screen."""
    if not result["n"]:
        print(f"  {arm:10} no runs read")
        return
    outcome = "improves" if result["improves"] else ("holds" if result["holds"] else "FAILS")
    print(
        f"  {arm:10} mean {result['mean']:5.1f} ({result['mean_delta']:+.1f} vs frozen clone)  "
        f"{result['seeds_within_hold']}/8 within hold, {result['seeds_at_or_above']}/8 at or above"
        f"  cos {result['cosine_mean']:.2f}  rate x{result['rate_multiplier_mean']:.2f}"
        f"  -> {outcome}",
    )
    if result["missing_seeds"]:
        print(f"    incomplete: seeds {result['missing_seeds']} not read - no pass is possible")


def _print_screen(out: dict) -> None:
    """Print the screen, comparator first."""
    print(
        f"\nClone assay against the committed frozen clone (mean {FROZEN_MEAN:.1f}, seeds 1-8, "
        f"{BUDGET} episodes)",
    )
    print(f"  {out['screen_not_test']}\n")
    for arm in ARM_KEYS:
        _print_arm(arm, out["arms"][arm])
    passed = [a for a in ARM_KEYS if out["arms"][a]["pass"]]
    print(f"\n  Passed: {', '.join(passed) if passed else 'none'}")


def write_per_seed_csv(out: dict, path: Path) -> None:
    """One row per arm and seed: the metric, its delta, the cosine and the multiplier."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "seed", "success", "frozen_clone", "delta", "cosine", "rate_mult"])
        for arm in ARM_KEYS:
            result = out["arms"][arm]
            for seed in sorted(result["per_seed"], key=int):
                writer.writerow(
                    [
                        arm,
                        seed,
                        f"{result['per_seed'][seed]:.4f}",
                        f"{FROZEN_CLONE[int(seed)]:.4f}",
                        f"{result['delta'][seed]:.4f}",
                        result["cosine_to_clone"].get(seed),
                        result["rate_multiplier"].get(seed),
                    ],
                )


def main(argv: list[str] | None = None) -> int:
    """Read the screen's runs, apply the pass rule and write the records."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--campaign-dir", type=Path, help="campaign directory holding logs/")
    source.add_argument("--manifest", type=Path, help="<arm> <seed> <log> per line")
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS)
    parser.add_argument("--out", type=Path, help="write the screen as JSON")
    parser.add_argument("--csv", type=Path, help="write the per-seed table")
    args = parser.parse_args(argv)

    scanned = (
        read_manifest(args.manifest, args.experiments)
        if args.manifest
        else scan_campaign(args.campaign_dir, args.experiments)
    )
    if not scanned:
        print("no runs read", file=sys.stderr)
        return 1
    panel, logs = group(scanned)
    out = analyse(panel, logs, args.experiments)
    _print_screen(out)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    if args.csv:
        write_per_seed_csv(out, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
