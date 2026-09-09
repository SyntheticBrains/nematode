#!/usr/bin/env python
"""The structured-instruction test: does routing the third factor through the wiring matter.

Every panel so far broadcast one reward-prediction error to every plastic synapse. This compares
that against a third factor routed through the aminergic wiring -- applied only where a synapse's
post-synaptic neuron receives chemical input from an aminergic neuron, with the unmodulated
Hebbian term everywhere else -- on both wirings.

What is confirmatory is fixed here, in code, before any data exist:

- four one-sided paired tests corrected together as one BH-FDR family -- **S1** the wild-type
  routed arm over the wild-type global arm (does routing help at all), **S2** the same on the
  rewired null (does it help without the real wiring), **S3** wild-type over rewired null under
  routing, **S4** the same under the global scalar;
- an ordered verdict map: ``insufficient_seeds``, ``no_routing_effect`` (neither S1 nor S2
  confirms -- the outcome in which the global scalar was not the limitation),
  ``routing_helps_both``, ``routing_helps_wild_type_only``, ``routing_helps_rewired_only``.
  S3 and S4 annotate and never decide.

``routing_helps_both`` is named here so it cannot be reported as a win: a routed third factor
helping the scramble as much as the animal is a fact about running two learning regimes in one
network, not about this animal's connectivity.

The global arms are re-run rather than read from the first panel's committed table, whose plastic
arm carries eight seeds at this budget; those values are reported as a consistency check only.

The pathway is a model of aminergic reach by SYNAPTIC connectivity, and a lower bound: these
amines are released by volume onto receptors expressed by cells that need not be synaptic partners
(Bentley et al., PLoS Comput Biol, 2016). A negative here refutes that proxy, not structured
instruction, and each arm's instructed fraction is reported so the proxy's coverage is visible.

Usage::

    uv run python scripts/analysis/l4_structured_instruction.py --campaign-dir campaigns/l4-routing
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

_STEM = "connectomeppo_small_continuous2d_combined_klinotaxis_plastic"

ARMS: dict[str, str] = {
    f"{_STEM}_pathway": "wt_pathway",
    f"{_STEM}_rewired_null_pathway": "rn_pathway",
    _STEM: "wt_global",
    f"{_STEM}_rewired_null": "rn_global",
}
ARM_KEYS = ("wt_pathway", "rn_pathway", "wt_global", "rn_global")

SEEDS = tuple(range(1, 17))
BUDGET = 3000
EXTENSION = 1.5
ALPHA = 0.05

# The first panel's committed plastic values, read as a consistency check on the re-run global
# arm and NEVER as a comparator: it carries eight seeds where this test runs sixteen.
PANEL1_CSV = (
    REPO / "docs" / "experiments" / "logbooks" / "supporting" / "040-l4-panel" / "per-seed.csv"
)
PANEL1_ARMS = {"wt_global": "wt_plastic", "rn_global": "rn_plastic"}

Scanned = list[tuple[str, int, SeedRecord, Path]]
LogOf = dict[str, dict[int, Path]]


def read_panel1(path: Path = PANEL1_CSV) -> dict[str, dict[int, float]]:
    """Read the first panel's committed plastic values, for the consistency check."""
    out: dict[str, dict[int, float]] = {arm: {} for arm in PANEL1_ARMS}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            for ours, theirs in PANEL1_ARMS.items():
                if row["arm"] == theirs:
                    out[ours][int(row["seed"])] = float(row["success"])
    return out


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
        record = read_log(REPO / parts[2], experiments)
        if record is None:
            print(f"  WARN {parts[0]} seed {parts[1]}: no parseable plateau - dropped")
            continue
        found.append((parts[0], int(parts[1]), record, REPO / parts[2]))
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


def family_tests(values: dict[str, dict[int, float]]) -> dict[str, dict]:
    """Compute the four registered tests, corrected together."""
    definitions = {
        "S1": ("wt_pathway", "wt_global", "wt routed - wt global"),
        "S2": ("rn_pathway", "rn_global", "rn routed - rn global"),
        "S3": ("wt_pathway", "rn_pathway", "wt - rn under routing"),
        "S4": ("wt_global", "rn_global", "wt - rn under the global scalar"),
    }
    tests: dict[str, dict] = {}
    for name, (left, right, label) in definitions.items():
        result = paired(values.get(left, {}), values.get(right, {}))
        result["contrast"] = label
        result["complete"] = len(result.get("seeds", [])) == len(SEEDS)
        tests[name] = result
    qs = bh_fdr([tests[n]["wilcoxon_p"] for n in ("S1", "S2", "S3", "S4")])
    for name, q in zip(("S1", "S2", "S3", "S4"), qs, strict=True):
        tests[name]["q"] = q
        tests[name]["confirms"] = bool(tests[name]["complete"] and q < ALPHA)
    return tests


def verdict(tests: dict[str, dict]) -> str:
    """Assign the ordered verdict, whose map is fixed before any data exist."""
    if not (tests["S1"]["complete"] and tests["S2"]["complete"]):
        return "insufficient_seeds"
    s1, s2 = tests["S1"]["confirms"], tests["S2"]["confirms"]
    if not s1 and not s2:
        return "no_routing_effect"
    if s1 and s2:
        return "routing_helps_both"
    return "routing_helps_wild_type_only" if s1 else "routing_helps_rewired_only"


def instructed(logs: LogOf, experiments: Path = EXPERIMENTS) -> dict[str, dict]:
    """Each arm's instructed fraction and mean instructed share, from its runs' telemetry.

    Reported per arm because the rewired null derives its own pathway from its own edges, and
    reported at all because a proxy covering nearly none or nearly all of the substrate would
    make the arm uninterpretable in opposite directions.
    """
    out: dict[str, dict] = {}
    for arm in ARM_KEYS:
        fractions: list[float] = []
        shares: list[float] = []
        for log in logs.get(arm, {}).values():
            experiment = _experiment_json(log.read_text(), experiments)
            exports = experiment.get("exports_path") if experiment else None
            if not exports:
                continue
            data = REPO / exports / "session" / "data"
            for name, sink in (
                ("plasticity_instructed_fraction", fractions),
                ("plasticity_instructed_share", shares),
            ):
                series = data / f"tracking_{name}.csv"
                if not series.is_file():
                    continue
                with series.open(newline="") as handle:
                    run = [
                        float(row[name])
                        for row in csv.DictReader(handle)
                        if row.get(name) not in (None, "", "nan")
                    ]
                if run:
                    sink.append(float(np.mean(run)))
        out[arm] = {
            "fraction": float(np.mean(fractions)) if fractions else float("nan"),
            "share": float(np.mean(shares)) if shares else float("nan"),
            "n_read": len(shares),
        }
    return out


def panel1_check(values: dict[str, dict[int, float]]) -> dict[str, dict]:
    """Compare the re-run global arms against the first panel's committed values on its seeds."""
    committed = read_panel1()
    out: dict[str, dict] = {}
    for arm, theirs in committed.items():
        shared = sorted(set(theirs) & set(values.get(arm, {})))
        out[arm] = {
            "seeds": shared,
            "committed_mean": float(np.mean([theirs[s] for s in shared]))
            if shared
            else float("nan"),
            "rerun_mean": float(np.mean([values[arm][s] for s in shared]))
            if shared
            else float("nan"),
            "note": "consistency check only; never a comparator",
        }
    return out


def extensions_needed(panel: dict[str, dict[int, SeedRecord]]) -> list[dict]:
    """List the runs the plateau detector marks non-converged: the one registered extension."""
    return [
        {"arm": arm, "seed": seed, "extend_to": int(BUDGET * EXTENSION)}
        for arm in ARM_KEYS
        for seed, record in sorted(panel.get(arm, {}).items())
        if record.episodes == BUDGET and record.converged is False
    ]


def analyse(
    panel: dict[str, dict[int, SeedRecord]],
    logs: LogOf,
    experiments: Path = EXPERIMENTS,
) -> dict:
    """Score the four arms, apply the registered family and assign the verdict."""
    values = successes(panel)
    tests = family_tests(values)
    return {
        "seeds": list(SEEDS),
        "budget": BUDGET,
        "alpha": ALPHA,
        "pathway_model": (
            "aminergic reach by synaptic connectivity: a lower bound, since these amines are "
            "released by volume onto receptors expressed by cells that need not be synaptic "
            "partners. A negative refutes this proxy, not structured instruction."
        ),
        "arms": {
            arm: {
                "per_seed": {str(s): v for s, v in sorted(values.get(arm, {}).items())},
                "n": len(values.get(arm, {})),
                "mean": (
                    float(np.mean(list(values[arm].values()))) if values.get(arm) else float("nan")
                ),
            }
            for arm in ARM_KEYS
        },
        "tests": tests,
        "verdict": verdict(tests),
        "annotations": {
            "instructed": instructed(logs, experiments),
            "wiring_contrast_routed": tests["S3"]["confirms"],
            "wiring_contrast_global": tests["S4"]["confirms"],
            "panel1_consistency": panel1_check(values),
            "routing_helps_both_reading": (
                "a routed third factor helping the rewired null as much as the wild type is a "
                "fact about running two learning regimes in one network, not about this "
                "animal's connectivity"
            ),
        },
        "extensions_needed": extensions_needed(panel),
    }


def _print_panel(out: dict) -> None:
    """Print the arms, the family and the annotations."""
    print(f"\nStructured-instruction test (seeds 1-{max(out['seeds'])}, {out['budget']} episodes)")
    for arm in ARM_KEYS:
        row = out["arms"][arm]
        ann = out["annotations"]["instructed"][arm]
        print(
            f"  {arm:12} mean {row['mean']:5.1f} (n={row['n']})  "
            f"instructed {ann['fraction']:.3f}  share {ann['share']:.3f}",
        )
    print("\n  Registered family (BH-FDR, alpha = 0.05):")
    for name in ("S1", "S2", "S3", "S4"):
        test = out["tests"][name]
        mark = (
            "CONFIRMS" if test["confirms"] else ("incomplete" if not test["complete"] else "fail")
        )
        print(
            f"    {name} {test['contrast']:32} {test['mean_delta']:+6.2f}  "
            f"[{test['ci_lo']:+6.2f}, {test['ci_hi']:+6.2f}]  q={test['q']:.3f}  "
            f"{test.get('positive_seeds', 0)}/{len(test.get('seeds', []))}  {mark}",
        )
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
