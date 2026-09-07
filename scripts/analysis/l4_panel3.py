#!/usr/bin/env python
"""L4 panel 3: replicate the Hebbian wiring contrast on fresh seeds.

The two degree-scaled Hebbian arms of panel 2 -- wild-type and its degree-preserving
rewired-null under the unmodulated Hebbian rule -- on paired seeds 17-64, which no earlier
panel used. The metric, reader and statistics layer are panel 2's; the frozen floors for
these seeds and the descriptive pooling of seeds 1-16 are read from panel 2's committed
per-seed table, so the analysis reproduces from the repository alone.

What is confirmatory is fixed here, in code, before any data exist:

- two one-sided paired tests corrected together as one BH-FDR family -- R1 wild-type
  Hebbian > rewired-null Hebbian by the committed paired Wilcoxon (the primary); R2 the
  same direction in competent-fraction discordance: with ``b`` seeds where only the
  wild-type is competent (plateau tail at or above ``COMPETENT_THRESHOLD``) and ``c`` where
  only the rewired-null is, the exact binomial ``P(X >= b)`` for ``X ~ Bin(b + c, 1/2)``;
- a verdict assigned from R1 alone in panel 2's vocabulary: ``insufficient_seeds``,
  ``specific_wiring``, ``rewired_beats_wild_type``, ``degree_statistics``,
  ``inconclusive``. R2 annotates the verdict and never changes it.

Seeds 1-16 enter only the pooled descriptive summary. Every other number -- the learning
gains against panel 2's frozen floors, the distributions, the pooled 64-seed contrast -- is
descriptive and labelled so.

Usage::

    uv run python scripts/analysis/l4_panel3.py --campaign-dir campaigns/l4-panel3
        --out panel3.json --csv per-seed.csv --curves curves.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

# Panel 2's registry, reader, statistics helpers and writers, reused verbatim.
from l4_panel2 import (
    COMPETENT_THRESHOLD,
    EXPERIMENTS,
    EXTENSION_BUDGET,
    MIN_PAIRED_SEEDS,
    REPO,
    SIG_Q,
    Scanned,
    SeedRecord,
    distribution,
    paired,
    read_manifest,
    restrict,
    scan_campaign,
    write_curves_csv,
    write_per_seed_csv,
)
from scipy.stats import binomtest
from weight_search_architecture_ranking import bh_fdr

PANEL2_CSV = (
    REPO / "docs" / "experiments" / "logbooks" / "supporting" / "041-l4-panel2" / "per-seed.csv"
)
PANEL2_SWEEP = "campaigns/l4-panel2-sweep"

ARMS3: tuple[str, ...] = ("wt_hebbian", "rn_hebbian")
FLOOR_OF: dict[str, str] = {"wt_hebbian": "wt_frozen", "rn_hebbian": "rn_frozen"}
REPLICATION_SEEDS: tuple[int, ...] = tuple(range(17, 65))
PANEL2_SEEDS: tuple[int, ...] = tuple(range(1, 17))
POOLED_SEEDS: tuple[int, ...] = PANEL2_SEEDS + REPLICATION_SEEDS
BUDGET = 1000
FAMILY: tuple[str, ...] = ("R1", "R2")
READS: dict[str, str] = {
    "R1": "wild-type Hebbian > rewired-null Hebbian, seeds 17-64 (primary, paired Wilcoxon)",
    "R2": "wild-type > rewired-null in competent-fraction discordance, seeds 17-64 (exact binomial)",
}


# --- reading ----------------------------------------------------------------------------


def group_panel(scanned: Scanned) -> dict[str, dict[int, SeedRecord]]:
    """``{arm: {seed: record}}``; only the two Hebbian arms on the replication seeds may enter."""
    panel: dict[str, dict[int, SeedRecord]] = {}
    for arm, seed, record in scanned:
        if arm not in ARMS3:
            msg = f"{arm} is not a panel-3 arm; only {ARMS3} enter the replication"
            raise ValueError(msg)
        if seed not in REPLICATION_SEEDS:
            msg = (
                f"seed {seed} ({arm}) is outside the replication seeds "
                f"{REPLICATION_SEEDS[0]}-{REPLICATION_SEEDS[-1]}"
            )
            raise ValueError(msg)
        seeds = panel.setdefault(arm, {})
        held = seeds.get(seed)
        if held is not None:
            # An extension is a fresh, longer run replacing its predecessor, and the two logs may
            # arrive in either order, so the longer run wins by episode count rather than by
            # position. Two runs of equal length are a genuine conflict, not an extension.
            if held.episodes == record.episodes:
                msg = (
                    f"{arm} seed {seed}: two runs of {record.episodes} episodes - a duplicate that "
                    "is not an extension; resolve the campaign before analysing"
                )
                raise ValueError(msg)
            keep = max(held, record, key=lambda r: r.episodes)
            drop = min(held, record, key=lambda r: r.episodes)
            print(
                f"  WARN {arm} seed {seed}: keeping the {keep.episodes}-episode run over the "
                f"{drop.episodes}-episode one (the registered extension)",
            )
            seeds[seed] = keep
            continue
        seeds[seed] = record
    return panel


def read_panel2_csv(path: Path = PANEL2_CSV) -> dict[str, dict[int, float]]:
    """Panel 2's committed per-seed table: ``{arm: {seed: success}}`` for every arm it holds."""
    values: dict[str, dict[int, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["arm"], {})[int(row["seed"])] = float(row["success"])
    return values


def successes(panel: dict[str, dict[int, SeedRecord]]) -> dict[str, dict[int, float]]:
    """Reduce records to the ranked metric."""
    return {arm: {s: r.success for s, r in seeds.items()} for arm, seeds in panel.items()}


# --- confirmatory -----------------------------------------------------------------------


def discordance(wt: dict[int, float], rn: dict[int, float], seeds: tuple[int, ...]) -> dict:
    """Competent-fraction discordance on the common seeds, with the exact one-sided p-value."""
    common = sorted(s for s in seeds if s in wt and s in rn)
    wt_c = {s for s in common if wt[s] >= COMPETENT_THRESHOLD}
    rn_c = {s for s in common if rn[s] >= COMPETENT_THRESHOLD}
    b = len(wt_c - rn_c)
    c = len(rn_c - wt_c)
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5, alternative="greater").pvalue)
    return {
        "n": len(common),
        "seeds": common,
        "b_wild_type_only": b,
        "c_rewired_only": c,
        "both": len(wt_c & rn_c),
        "wild_type_competent": len(wt_c),
        "rewired_competent": len(rn_c),
        "exact_p": p,  # an exact binomial, never a Wilcoxon; the family reads ``p_value``
        "p_value": p,
        "mean_delta": float(b - c),
        "positive_seeds": b,
    }


def family_tests(values: dict[str, dict[int, float]]) -> dict[str, dict]:
    """Compute R1 and R2 on the replication seeds, corrected together."""
    wt = restrict(values.get("wt_hebbian", {}), REPLICATION_SEEDS)
    rn = restrict(values.get("rn_hebbian", {}), REPLICATION_SEEDS)
    raw = {"R1": paired(wt, rn), "R2": discordance(wt, rn, REPLICATION_SEEDS)}
    raw["R1"]["p_value"] = raw["R1"]["wilcoxon_p"]  # R1 is the Wilcoxon; R2 carries ``exact_p``
    qs = bh_fdr([raw[t]["p_value"] for t in FAMILY])
    for test, q in zip(FAMILY, qs, strict=True):
        stats = raw[test]
        stats["bh_q"] = q
        stats["sufficient"] = stats["n"] >= MIN_PAIRED_SEEDS
        stats["complete"] = tuple(stats["seeds"]) == REPLICATION_SEEDS
        stats["reads"] = READS[test]
    r1, r2 = raw["R1"], raw["R2"]
    r1["pass"] = bool(r1["sufficient"] and r1["bh_q"] < SIG_Q and r1["mean_delta"] > 0)
    r1["reverse"] = bool(r1["sufficient"] and r1["ci_hi"] < 0.0)
    r2["pass"] = bool(
        r2["sufficient"] and r2["bh_q"] < SIG_Q and r2["b_wild_type_only"] > r2["c_rewired_only"],
    )
    return raw


def verdict(tests: dict[str, dict]) -> str:
    """Assign the verdict from R1 alone, by panel 2's ordered map."""
    r1 = tests["R1"]
    if not r1["sufficient"]:
        return "insufficient_seeds"
    if r1["pass"]:
        return "specific_wiring"
    if r1["ci_hi"] < 0.0:
        return "rewired_beats_wild_type"
    if r1["ci_lo"] <= 0.0 <= r1["ci_hi"]:
        return "degree_statistics"
    return "inconclusive"


def annotate(result: str, tests: dict[str, dict]) -> dict[str, bool | str]:
    """Describe what R2 says about the verdict; it never changes it."""
    r2_pass = bool(tests["R2"]["pass"])
    out: dict[str, bool | str] = {"competent_fraction_confirms": r2_pass}
    if result == "specific_wiring":
        out["reading"] = "the wild-type advantage is confirmed on the ranked metric" + (
            " and on the competent fraction"
            if r2_pass
            else "; the competent fraction does not reach alpha"
        )
    elif r2_pass:
        out["reading"] = (
            "the competent fraction reaches alpha while the ranked metric does not: the bimodal case "
            "the rank test cannot carry; reported, not promoted -- the registered claim requires R1"
        )
    else:
        out["reading"] = "no confirmed wild-type Hebbian advantage on either test"
    return out


# --- descriptive ------------------------------------------------------------------------


def check_panel2_inputs(panel2: dict[str, dict[int, float]]) -> None:
    """Refuse to analyse on an incomplete panel-2 table: the floors and the pooling are registered."""
    missing: list[str] = []
    for arm in FLOOR_OF.values():
        absent = [s for s in REPLICATION_SEEDS if s not in panel2.get(arm, {})]
        if absent:
            missing.append(f"{arm} floors for seeds {absent}")
    for arm in ARMS3:
        absent = [s for s in PANEL2_SEEDS if s not in panel2.get(arm, {})]
        if absent:
            missing.append(f"{arm} values for the pooled seeds {absent}")
    if missing:
        msg = f"panel 2's table is incomplete, so the gains and the pooling would silently drop seeds: {'; '.join(missing)}"
        raise ValueError(msg)


def learning_gains(
    values: dict[str, dict[int, float]],
    floors: dict[str, dict[int, float]],
) -> dict:
    """Each Hebbian arm minus its own frozen floor, on the replication seeds and pooled."""
    out: dict = {"floors_from": str(PANEL2_CSV.relative_to(REPO)), "floors_campaign": PANEL2_SWEEP}
    for hebbian, frozen in FLOOR_OF.items():
        h, f = values.get(hebbian, {}), floors.get(frozen, {})
        for label, seeds in (("replication", REPLICATION_SEEDS), ("pooled", POOLED_SEEDS)):
            per_seed = {s: h[s] - f[s] for s in seeds if s in h and s in f}
            out[f"{hebbian}_{label}"] = {
                "floor": frozen,
                "n": len(per_seed),
                "mean": float(np.mean(list(per_seed.values()))) if per_seed else math.nan,
                "positive_seeds": int(sum(1 for d in per_seed.values() if d > 0)),
            }
    return out


def pooled(values: dict[str, dict[int, float]]) -> dict:
    """Seeds 1-64 together: the contrast, the discordance and the competent fractions, descriptive."""
    wt = restrict(values.get("wt_hebbian", {}), POOLED_SEEDS)
    rn = restrict(values.get("rn_hebbian", {}), POOLED_SEEDS)
    contrast = paired(wt, rn)
    contrast.update({"descriptive": True, "seeds_panel2": PANEL2_SEEDS})
    return {
        "contrast": contrast,
        "discordance": {**discordance(wt, rn, POOLED_SEEDS), "descriptive": True},
        "wt_hebbian": distribution(wt),
        "rn_hebbian": distribution(rn),
    }


def extensions_needed(panel: dict[str, dict[int, SeedRecord]]) -> list[dict]:
    """List the runs the plateau detector marks non-converged; each gets one fresh run at 1500."""
    return [
        {"arm": arm, "seed": seed, "episodes": record.episodes}
        for arm in ARMS3
        for seed, record in sorted(panel.get(arm, {}).items())
        if record.converged is False and record.episodes < EXTENSION_BUDGET
    ]


def analyse(
    panel: dict[str, dict[int, SeedRecord]],
    panel2: dict[str, dict[int, float]],
    out: dict,
) -> dict:
    """Family, verdict, annotation, gains, distributions and the pooled descriptive."""
    check_panel2_inputs(panel2)
    values = successes(panel)
    tests = family_tests(values)
    result = verdict(tests)
    out["family"] = {t: tests[t] for t in FAMILY}
    out["verdict"] = {
        "verdict": result,
        "annotations": annotate(result, tests),
        "ensemble_invariance": {
            "R1": {"positive_seeds": tests["R1"]["positive_seeds"], "n": tests["R1"]["n"]},
        },
    }
    out["per_arm"] = {
        arm: {
            **distribution(restrict(values.get(arm, {}), REPLICATION_SEEDS)),
            "per_seed": dict(sorted(values.get(arm, {}).items())),
        }
        for arm in ARMS3
    }
    merged = {
        arm: {**restrict(panel2.get(arm, {}), PANEL2_SEEDS), **values.get(arm, {})} for arm in ARMS3
    }
    out["learning_gains"] = learning_gains(merged, panel2)
    out["pooled_descriptive"] = pooled(merged)
    out["extensions_needed"] = extensions_needed(panel)
    return out


# --- output -----------------------------------------------------------------------------


def _print_panel(out: dict) -> None:
    print("\n" + "=" * 78)
    print("L4 PANEL 3 - Hebbian wiring contrast, replication seeds 17-64")
    print("=" * 78)
    for arm in ARMS3:
        d = out["per_arm"][arm]
        if d["n"]:
            print(
                f"  {arm:12} n={d['n']:2d}  mean={d['mean']:5.1f}  median={d['median']:5.1f}  "
                f"competent={d['competent_fraction']:.2f}",
            )
    print("\n  Confirmatory family (BH-FDR, alpha 0.05):")
    r1, r2 = out["family"]["R1"], out["family"]["R2"]
    flag = "PASS" if r1["pass"] else ("REVERSE" if r1["reverse"] else "fail")
    print(
        f"    R1  d={r1['mean_delta']:+6.2f}  CI[{r1['ci_lo']:+6.2f},{r1['ci_hi']:+6.2f}]  "
        f"p={r1['wilcoxon_p']:.3f}  q={r1['bh_q']:.3f}  +seeds={r1['positive_seeds']}/{r1['n']}  {flag}",
    )
    print(
        f"    R2  b={r2['b_wild_type_only']} c={r2['c_rewired_only']} both={r2['both']}  "
        f"p={r2['exact_p']:.3f}  q={r2['bh_q']:.3f}  {'PASS' if r2['pass'] else 'fail'}",
    )
    for test in FAMILY:
        if not out["family"][test]["complete"]:
            print(
                f"        {test} INCOMPLETE: {out['family'][test]['n']} of the registered seeds present",
            )
    g = out["learning_gains"]
    print("\n  Learning gains (Hebbian minus own frozen; floors from panel 2's table):")
    for arm in ARMS3:
        r, p = g[f"{arm}_replication"], g[f"{arm}_pooled"]
        print(
            f"    {arm:12} 17-64: {r['mean']:+6.2f} ({r['positive_seeds']}/{r['n']})   "
            f"1-64: {p['mean']:+6.2f} ({p['positive_seeds']}/{p['n']})",
        )
    pc, pd = out["pooled_descriptive"]["contrast"], out["pooled_descriptive"]["discordance"]
    print(
        f"\n  Pooled 1-64 (descriptive): d={pc['mean_delta']:+6.2f}  CI[{pc['ci_lo']:+6.2f},{pc['ci_hi']:+6.2f}]  "
        f"+seeds={pc['positive_seeds']}/{pc['n']}   discordance b={pd['b_wild_type_only']} c={pd['c_rewired_only']}",
    )
    v = out["verdict"]
    print("-" * 78)
    print(f"  VERDICT: {v['verdict']}   -- {v['annotations']['reading']}")
    if out["extensions_needed"]:
        print(f"  EXTENSION NEEDED: {out['extensions_needed']}")


def main(argv: list[str] | None = None) -> int:
    """Run the panel-3 analysis from the command line; return the exit code."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--campaign-dir",
        type=Path,
        help="replication campaign dir (reads logs/*.log)",
    )
    source.add_argument("--manifest", type=Path, help="<arm> <seed> <log> per line")
    ap.add_argument(
        "--panel2-csv",
        type=Path,
        default=PANEL2_CSV,
        help="panel 2's committed per-seed table",
    )
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
    out: dict = {}
    try:
        panel = group_panel(scanned)
        analyse(panel, read_panel2_csv(args.panel2_csv), out)
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
