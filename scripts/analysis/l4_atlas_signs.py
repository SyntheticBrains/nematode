#!/usr/bin/env python
"""Sign grounding: does making the synapse signs real change the earlier answers.

The first rung of the substrate fidelity ladder. Two committed protocols are re-run with
atlas-grounded synapse signs and compared against the committed random-sign values rather
than re-running those arms: the second panel's 64-seed frozen prior sweep, and its Hebbian
wiring contrast on seeds 1-16, the latter once with Dale's law off and once on.

What is confirmatory is fixed here, in code, before any data exist:

- four one-sided paired tests corrected together as one BH-FDR family -- G1 the grounded
  wild-type frozen arm over the committed random-sign one (does the sign structure alone
  change the prior over untrained policies?); G2 the grounded wild-type over the grounded
  rewired-null Hebbian arm without enforcement (the primary: is the wiring contrast
  confirmable once signs are real?); G3 the same with enforcement; G4 enforced over
  unenforced on the wild-type;
- an ordered verdict map: ``insufficient_seeds``; ``substrate_fail`` when the grounded frozen
  arms' competent fraction falls below half the committed random-sign value, because an
  overwhelmingly excitatory network of saturating units may express nothing and that outcome
  is named in advance rather than rationalised afterwards; then from G2 alone
  ``specific_wiring``, ``rewired_beats_wild_type``, ``degree_statistics``, ``inconclusive``.
  G1, G3 and G4 annotate the verdict and never change it.

Everything else -- the remaining pairs, the distributions, the sign-flip telemetry -- is
descriptive and labelled so.

Usage::

    uv run python scripts/analysis/l4_atlas_signs.py --campaign-dir campaigns/l4-atlas-signs
        --out panel.json --csv per-seed.csv --curves curves.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

from l4_panel import _LABEL as LABEL
from l4_panel import EXPERIMENTS, REPO, SeedRecord, _experiment_json, paired, read_log
from l4_panel2 import COMPETENT_THRESHOLD, distribution, restrict
from weight_search_architecture_ranking import bh_fdr

_STEM = "connectomeppo_small_continuous2d_combined_klinotaxis_plastic"
PANEL2_CSV = (
    REPO / "docs" / "experiments" / "logbooks" / "supporting" / "041-l4-panel2" / "per-seed.csv"
)

# Config stem -> arm key. Grounded arms only: the random-sign comparators are read from
# panel 2's committed table, never re-run.
ARMS: dict[str, str] = {
    f"{_STEM}_frozen_atlassigns": "wt_frozen_atlas",
    f"{_STEM}_frozen_rewired_null_atlassigns": "rn_frozen_atlas",
    f"{_STEM}_hebbian_atlassigns": "wt_hebbian_atlas",
    f"{_STEM}_hebbian_rewired_null_atlassigns": "rn_hebbian_atlas",
    f"{_STEM}_hebbian_atlassigns_dale": "wt_hebbian_dale",
    f"{_STEM}_hebbian_rewired_null_atlassigns_dale": "rn_hebbian_dale",
}
ARM_KEYS: tuple[str, ...] = tuple(ARMS.values())
FROZEN_ARMS: tuple[str, ...] = ("wt_frozen_atlas", "rn_frozen_atlas")
HEBBIAN_ARMS: tuple[str, ...] = (
    "wt_hebbian_atlas",
    "rn_hebbian_atlas",
    "wt_hebbian_dale",
    "rn_hebbian_dale",
)
SWEEP_SEEDS: tuple[int, ...] = tuple(range(1, 65))
HEBBIAN_SEEDS: tuple[int, ...] = tuple(range(1, 17))
SEEDS_OF: dict[str, tuple[int, ...]] = {
    **dict.fromkeys(FROZEN_ARMS, SWEEP_SEEDS),
    **dict.fromkeys(HEBBIAN_ARMS, HEBBIAN_SEEDS),
}
BUDGETS: dict[str, int] = {
    **dict.fromkeys(FROZEN_ARMS, 600),
    **dict.fromkeys(HEBBIAN_ARMS, 1000),
}
EXTENSION = 1.5
SIG_Q = 0.05
CURVE_WINDOW = 250
MIN_PAIRED_SEEDS = 2
# Panel 2's random-sign counterparts, by arm key in its committed table.
RANDOM_OF: dict[str, str] = {
    "wt_frozen_atlas": "wt_frozen",
    "rn_frozen_atlas": "rn_frozen",
    "wt_hebbian_atlas": "wt_hebbian",
    "rn_hebbian_atlas": "rn_hebbian",
    "wt_hebbian_dale": "wt_hebbian",
    "rn_hebbian_dale": "rn_hebbian",
}
# A grounded prior this far below the committed one means grounding broke the substrate.
SUBSTRATE_FAIL_RATIO = 0.5
FAMILY: tuple[str, ...] = ("G1", "G2", "G3", "G4")
READS: dict[str, str] = {
    "G1": "grounded wild-type frozen > random-sign wild-type frozen (does the prior change?)",
    "G2": "grounded wild-type > grounded rewired-null Hebbian, no enforcement (primary)",
    "G3": "the same under Dale's law",
    "G4": "enforced > unenforced wild-type Hebbian (is sign enforcement a useful constraint?)",
}
GATE_TESTS: tuple[str, ...] = ("G1", "G2")

Scanned = list[tuple[str, int, SeedRecord]]
# Per run: the log the record came from. The log names the experiment, whose export carries the
# auto-saved endpoint weights the sign-flip telemetry reads.
LogOf = dict[tuple[str, int], Path]


# --- reading ----------------------------------------------------------------------------


def scan_campaign(
    campaign_dir: Path,
    experiments: Path = EXPERIMENTS,
    logs: LogOf | None = None,
) -> Scanned:
    """Read every registered run log under ``<campaign_dir>/logs`` (or the dir itself)."""
    logs = {} if logs is None else logs
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
        logs[(ARMS[stem], int(match.group("seed")))] = log
    return found


def read_manifest(
    manifest: Path,
    experiments: Path = EXPERIMENTS,
    logs: LogOf | None = None,
) -> Scanned:
    """``<arm> <seed> <log>`` per line; blank and ``#`` lines skipped."""
    logs = {} if logs is None else logs
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
        logs[(parts[0], int(parts[1]))] = REPO / parts[2]
    return found


def group_panel(scanned: Scanned) -> dict[str, dict[int, SeedRecord]]:
    """Group by arm and seed; each arm's registered seeds and run lengths only."""
    panel: dict[str, dict[int, SeedRecord]] = {}
    for arm, seed, record in scanned:
        allowed_seeds = SEEDS_OF[arm]
        if seed not in allowed_seeds:
            msg = (
                f"seed {seed} ({arm}) is outside the arm's registered seeds "
                f"{allowed_seeds[0]}-{allowed_seeds[-1]}"
            )
            raise ValueError(msg)
        allowed_lengths = {BUDGETS[arm], int(BUDGETS[arm] * EXTENSION)}
        if record.episodes not in allowed_lengths:
            msg = (
                f"{arm} seed {seed}: a run of {record.episodes} episodes is neither the budget "
                f"nor its registered extension ({sorted(allowed_lengths)})"
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


def read_panel2(path: Path = PANEL2_CSV) -> dict[str, dict[int, float]]:
    """Panel 2's committed per-seed table: the random-sign comparator for every grounded arm."""
    values: dict[str, dict[int, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["arm"], {})[int(row["seed"])] = float(row["success"])
    missing = [arm for arm in set(RANDOM_OF.values()) if arm not in values]
    if missing:
        msg = f"panel 2's table lacks the random-sign comparators {sorted(missing)}"
        raise ValueError(msg)
    return values


def successes(panel: dict[str, dict[int, SeedRecord]]) -> dict[str, dict[int, float]]:
    """Reduce records to the ranked metric."""
    return {arm: {s: r.success for s, r in seeds.items()} for arm, seeds in panel.items()}


# --- confirmatory -----------------------------------------------------------------------


def family_tests(
    values: dict[str, dict[int, float]],
    random_signs: dict[str, dict[int, float]],
) -> dict[str, dict]:
    """Compute G1-G4 on their registered seeds, corrected together."""
    pairs: dict[str, tuple[dict[int, float], dict[int, float], tuple[int, ...]]] = {
        "G1": (
            values.get("wt_frozen_atlas", {}),
            random_signs.get("wt_frozen", {}),
            SWEEP_SEEDS,
        ),
        "G2": (
            values.get("wt_hebbian_atlas", {}),
            values.get("rn_hebbian_atlas", {}),
            HEBBIAN_SEEDS,
        ),
        "G3": (
            values.get("wt_hebbian_dale", {}),
            values.get("rn_hebbian_dale", {}),
            HEBBIAN_SEEDS,
        ),
        "G4": (
            values.get("wt_hebbian_dale", {}),
            values.get("wt_hebbian_atlas", {}),
            HEBBIAN_SEEDS,
        ),
    }
    raw = {
        test: paired(restrict(a, seeds), restrict(b, seeds))
        for test, (a, b, seeds) in pairs.items()
    }
    qs = bh_fdr([raw[t]["wilcoxon_p"] for t in FAMILY])
    for test, q in zip(FAMILY, qs, strict=True):
        stats = raw[test]
        stats["bh_q"] = q
        stats["sufficient"] = stats["n"] >= MIN_PAIRED_SEEDS
        stats["complete"] = tuple(stats["seeds"]) == pairs[test][2]
        stats["pass"] = bool(stats["sufficient"] and q < SIG_Q and stats["mean_delta"] > 0)
        stats["reverse"] = bool(stats["sufficient"] and stats["ci_hi"] < 0.0)
        stats["reads"] = READS[test]
    return raw


def substrate_broken(
    values: dict[str, dict[int, float]],
    random_signs: dict[str, dict[int, float]],
) -> dict:
    """Compare the grounded and committed frozen priors' competent fractions."""
    out: dict = {"threshold_ratio": SUBSTRATE_FAIL_RATIO, "arms": {}, "broken": False}
    for arm in FROZEN_ARMS:
        grounded = distribution(restrict(values.get(arm, {}), SWEEP_SEEDS))
        committed = distribution(restrict(random_signs.get(RANDOM_OF[arm], {}), SWEEP_SEEDS))
        broken = bool(
            grounded["n"]
            and committed["n"]
            and grounded["competent_fraction"]
            < SUBSTRATE_FAIL_RATIO * committed["competent_fraction"],
        )
        out["arms"][arm] = {
            "grounded_competent_fraction": grounded.get("competent_fraction"),
            "random_competent_fraction": committed.get("competent_fraction"),
            "grounded_mean": grounded.get("mean"),
            "random_mean": committed.get("mean"),
            "broken": broken,
        }
        out["broken"] = out["broken"] or broken
    return out


def verdict(tests: dict[str, dict], substrate: dict) -> str:
    """Assign the verdict by the ordered map: the first condition that holds names it."""
    g2 = tests["G2"]
    ordered: tuple[tuple[str, bool], ...] = (
        ("insufficient_seeds", not all(tests[t]["sufficient"] for t in GATE_TESTS)),
        ("substrate_fail", bool(substrate["broken"])),
        ("rewired_beats_wild_type", g2["ci_hi"] < 0.0),
        ("specific_wiring", bool(g2["pass"])),
        ("degree_statistics", g2["ci_lo"] <= 0.0 <= g2["ci_hi"]),
    )
    return next((name for name, holds in ordered if holds), "inconclusive")


def annotate(tests: dict[str, dict]) -> dict[str, bool]:
    """Describe what G1, G3 and G4 say; none changes the verdict."""
    return {
        "prior_changed": bool(tests["G1"]["pass"]),
        "prior_worsened": bool(tests["G1"]["reverse"]),
        "contrast_holds_under_dale": bool(tests["G3"]["pass"]),
        "enforcement_helps": bool(tests["G4"]["pass"]),
    }


# --- descriptive ------------------------------------------------------------------------


def against_random(
    values: dict[str, dict[int, float]],
    random_signs: dict[str, dict[int, float]],
) -> list[dict]:
    """Every grounded arm against its committed random-sign counterpart, descriptive."""
    rows = []
    for arm in ARM_KEYS:
        comparator = RANDOM_OF[arm]
        seeds = SEEDS_OF[arm]
        stats = paired(
            restrict(values.get(arm, {}), seeds),
            restrict(random_signs.get(comparator, {}), seeds),
        )
        stats.update({"a": arm, "b": f"{comparator} (panel 2, random signs)", "descriptive": True})
        rows.append(stats)
    return rows


def descriptive_pairs(values: dict[str, dict[int, float]]) -> list[dict]:
    """Every pair of grounded arms outside the family, uncorrected."""
    family = {
        ("wt_hebbian_atlas", "rn_hebbian_atlas"),
        ("wt_hebbian_dale", "rn_hebbian_dale"),
        ("wt_hebbian_dale", "wt_hebbian_atlas"),
    }
    rows = []
    for a, b in combinations(ARM_KEYS, 2):
        if (a, b) in family or (b, a) in family or a not in values or b not in values:
            continue
        seeds = SWEEP_SEEDS if a in FROZEN_ARMS and b in FROZEN_ARMS else HEBBIAN_SEEDS
        stats = paired(restrict(values[a], seeds), restrict(values[b], seeds))
        stats.update({"a": a, "b": b, "descriptive": True})
        rows.append(stats)
    return rows


def run_sign_flips(log: Path, experiments: Path = EXPERIMENTS) -> dict[str, float] | None:
    """Return what a run did to its grounded sign structure, read from its own endpoint.

    Two distinct quantities, because they are what separates the enforced arms from the rest:

    ``violated``
        the share of grounded synapses whose final weight carries the *opposite* sign. Dale's
        law exists to hold this at zero; the unenforced arms say how much it removes.
    ``silenced``
        the share driven to exactly zero. Enforcement produces these — it projects a synapse
        that tried to change sign onto zero rather than preserving it — so a synapse it "held"
        may be one it switched off. Counting a zero as a flip would confuse the two.

    ``None`` when the endpoint is not on disk or the arm grounds nothing.
    """
    import torch

    experiment = _experiment_json(log.read_text(), experiments)
    exports = experiment.get("exports_path") if experiment else None
    weights = (REPO / exports / "weights" / "final.pt") if exports else None
    if weights is None or not weights.is_file():
        return None
    topology = torch.load(weights, weights_only=True)["topology"]
    signs = topology["chem_sign"]
    grounded = signs != 0
    if not bool(grounded.any()):
        return None
    final = topology["w_chem"][grounded]
    expected = signs[grounded].to(final.dtype)
    return {
        "violated": float(((final * expected) < 0).to(torch.float64).mean()),
        "silenced": float((final == 0).to(torch.float64).mean()),
    }


def sign_flips(
    panel: dict[str, dict[int, SeedRecord]],
    logs: LogOf,
    experiments: Path = EXPERIMENTS,
) -> dict:
    """Report per arm how much of the grounded sign structure each rule walked away from.

    Enforcement exists to hold this at zero; the unenforced arms say how much it removes.
    """
    out: dict = {}
    for arm in ARM_KEYS:
        per_seed: dict[int, dict[str, float]] = {}
        for seed in sorted(panel.get(arm, {})):
            log = logs.get((arm, seed))
            measured = run_sign_flips(log, experiments) if log else None
            if measured is not None:
                per_seed[seed] = measured
        rows = list(per_seed.values())
        out[arm] = {
            "per_seed": per_seed,
            "n_read": len(rows),
            "violated_mean": (sum(r["violated"] for r in rows) / len(rows)) if rows else None,
            "violated_max": max((r["violated"] for r in rows), default=None),
            "silenced_mean": (sum(r["silenced"] for r in rows) / len(rows)) if rows else None,
            "silenced_max": max((r["silenced"] for r in rows), default=None),
        }
    return out


def extensions_needed(panel: dict[str, dict[int, SeedRecord]]) -> list[dict]:
    """List the runs marked non-converged at their budget; each gets one fresh run at 1.5x."""
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
    random_signs: dict[str, dict[int, float]],
    out: dict,
    *,
    logs: LogOf | None = None,
    experiments: Path = EXPERIMENTS,
) -> dict:
    """Family, substrate check, verdict, annotations and the descriptive layers."""
    values = successes(panel)
    tests = family_tests(values, random_signs)
    substrate = substrate_broken(values, random_signs)
    result = verdict(tests, substrate)
    out["family"] = {t: tests[t] for t in FAMILY}
    out["substrate"] = substrate
    out["verdict"] = {
        "verdict": result,
        "annotations": annotate(tests),
        "ensemble_invariance": {
            t: {"positive_seeds": tests[t]["positive_seeds"], "n": tests[t]["n"]}
            for t in ("G1", "G2")
        },
    }
    out["per_arm"] = {
        arm: {
            **distribution(restrict(values.get(arm, {}), SEEDS_OF[arm])),
            "per_seed": dict(sorted(values.get(arm, {}).items())),
        }
        for arm in ARM_KEYS
    }
    out["competent_threshold"] = COMPETENT_THRESHOLD
    out["against_random_signs"] = against_random(values, random_signs)
    out["descriptive_pairs"] = descriptive_pairs(values)
    out["sign_flips"] = sign_flips(panel, logs or {}, experiments)
    out["extensions_needed"] = extensions_needed(panel)
    return out


# --- output -----------------------------------------------------------------------------


def _print_sign_flips(flips: dict) -> None:
    """Print what each arm did to its grounded sign structure, when any endpoint was readable."""
    if not any(row["n_read"] for row in flips.values()):
        return
    print("\n  Grounded synapses at each run's end (share violating their sign / silenced):")
    for arm in ARM_KEYS:
        row = flips.get(arm, {})
        if row.get("n_read"):
            print(
                f"    {arm:18} violated={row['violated_mean']:.3f} (max {row['violated_max']:.3f})"
                f"  silenced={row['silenced_mean']:.3f} (max {row['silenced_max']:.3f})"
                f"  ({row['n_read']} runs read)",
            )


def _print_panel(out: dict) -> None:
    print("\n" + "=" * 78)
    print("L4 SIGN GROUNDING - plateau-tail full-clear success, paired seeds")
    print("=" * 78)
    for arm in ARM_KEYS:
        row = out["per_arm"][arm]
        if row["n"]:
            print(
                f"  {arm:18} n={row['n']:2d}  mean={row['mean']:5.1f}  median={row['median']:5.1f}  "
                f"competent={row['competent_fraction']:.2f}",
            )
    print("\n  Substrate check (competent fraction, grounded vs committed random signs):")
    for arm, row in out["substrate"]["arms"].items():
        grounded, random_value = (
            row["grounded_competent_fraction"],
            row["random_competent_fraction"],
        )
        if grounded is None or random_value is None:
            continue
        print(
            f"    {arm:18} {grounded:.2f} vs {random_value:.2f}"
            f"{'   BROKEN' if row['broken'] else ''}",
        )
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
    _print_sign_flips(out.get("sign_flips") or {})
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
    """Write one row per arm and seed with the ranked metric and the sub-metrics."""
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
    """Run the sign-grounding analysis from the command line; return the exit code."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--campaign-dir",
        type=Path,
        action="append",
        help="campaign dir; repeatable",
    )
    source.add_argument("--manifest", type=Path, help="<arm> <seed> <log> per line")
    ap.add_argument("--panel2-csv", type=Path, default=PANEL2_CSV)
    ap.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--curves", type=Path, default=None)
    args = ap.parse_args(argv)

    scanned: Scanned = []
    logs: LogOf = {}
    if args.campaign_dir:
        for directory in args.campaign_dir:
            scanned += scan_campaign(directory, args.experiments_dir, logs)
    else:
        scanned = read_manifest(args.manifest, args.experiments_dir, logs)
    out: dict = {}
    try:
        panel = group_panel(scanned)
        analyse(
            panel,
            read_panel2(args.panel2_csv),
            out,
            logs=logs,
            experiments=args.experiments_dir,
        )
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
