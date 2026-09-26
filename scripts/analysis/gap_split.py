#!/usr/bin/env python
"""A.6's gap-only split: how much of A.6's move the gap junctions reproduce, on an exact pairing.

A.6 held a rewired null's gap junctions *and* autapses at the wild type's and moved the wiring gap
toward the null on both learners. The gap-held null runs the degree-preserving null's chemical swap
unchanged -- the same chemical graph at the same seed, the same autapses lost -- and skips only the
gap-junction swap. Against the current null it therefore differs in its gap junctions alone,
placement and strength **jointly**; nothing here separates the two.

It reuses A.6's seeds and runs. Only the gap-held arms are new; the wild type, the current null and
the chemical-only null come from ``campaigns/a6-*``, licensed by a parsed-field identity check (one
seed per reused arm, re-run and compared on every ``Run:`` line and on the final chemical matrix).

**The primary**, per learner: ``gap(gap_held) - gap(full)`` on ``auc_success``, the current null
minus the gap-held null on the same chemical graph. It is read against 2/3 of **A.6's committed
move**, since the question is what share of that move the gap junctions reproduce.

**A breakdown, as description**: A.6's move equals ``[gap(gap_held) - gap(full)] + [gap(chemical) -
gap(gap_held)]`` seed by seed. The second term mixes the autapses with the chemical-graph difference
in A.6's chemical-only null, and is never attributed to the autapses alone.

Every statistic is A.2's; the stems and seeds extend A.6's; the states and the drift rule are B.1c's.
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import measured_prior_contrast as mc  # noqa: E402  # pyright: ignore[reportMissingImports]
import measured_prior_pilot as mp  # noqa: E402  # pyright: ignore[reportMissingImports]
import null_strength_control as nsc  # noqa: E402  # pyright: ignore[reportMissingImports]
import operating_point_surface as ops  # noqa: E402  # pyright: ignore[reportMissingImports]
import wiring_premise as wp  # noqa: E402  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    import torch

SEEDS_BY_HALF = nsc.SEEDS_BY_HALF
HALVES = nsc.HALVES
FULL, CHEMICAL, GAP_HELD = nsc.FULL, nsc.CHEMICAL, "gap_held"
LEVELS: tuple[str, ...] = (FULL, CHEMICAL, GAP_HELD)
ARMS = mp.ARMS
REUSED_LEVELS: tuple[str, ...] = (FULL, CHEMICAL)

# The gap-held arms, in each family's own spelling, from an explicit table as A.6's were.
_GAP_HELD_NULL: dict[str, dict[str, str]] = {
    "ppo": {
        "rn_learn": f"{nsc._PPO}_rewired_gap_held_null",
        "rn_frozen": f"{nsc._PPO}_rewired_gap_held_null_frozen",
    },
    "reading": {
        "rn_learn": f"{nsc._READ}_rewired_gap_held_null",
        "rn_frozen": f"{nsc._READ_FROZEN}_rewired_gap_held_null",
    },
}
STEMS: dict[str, dict[str, dict[str, str]]] = {
    half: {
        **nsc.STEMS[half],
        GAP_HELD: {
            **{a: nsc.STEMS[half][FULL][a] for a in ("wt_learn", "wt_frozen")},
            **_GAP_HELD_NULL[half],
        },
    }
    for half in HALVES
}
# The new configs: each gap-held arm from the current-null arm of the same half.
NEW_ARMS: dict[str, tuple[str, str]] = {
    STEMS[half][GAP_HELD][arm]: (half, STEMS[half][FULL][arm])
    for half in HALVES
    for arm in ("rn_learn", "rn_frozen")
}
# The arms reused from A.6, each re-run at one seed for the identity check.
REUSED_STEMS: dict[str, tuple[str, ...]] = {
    half: tuple(dict.fromkeys(STEMS[half][lv][a] for lv in REUSED_LEVELS for a in ARMS))
    for half in HALVES
}
IDENTITY_SEED: dict[str, int] = {half: SEEDS_BY_HALF[half][0] for half in HALVES}


def _levels_by_stem() -> dict[str, tuple[str, str, tuple[str, ...]]]:
    """Config stem -> ``(half, arm, levels it serves)``: a wild-type run serves every level."""
    out: dict[str, tuple[str, str, tuple[str, ...]]] = {}
    for half in HALVES:
        for level in LEVELS:
            for arm, stem in STEMS[half][level].items():
                prior = out.get(stem)
                if prior and (prior[0], prior[1]) != (half, arm):
                    msg = f"stem {stem!r} is claimed by {prior[:2]} and by {(half, arm)}"
                    raise ValueError(msg)
                out[stem] = (half, arm, (*prior[2], level) if prior else (level,))
    return out


LEVELS_BY_STEM = _levels_by_stem()

# ── Reference and minimum ────────────────────────────────────────────────────────────────────
PRIMARY_METRIC = ops.UNCENSORED_METRIC
BESIDE_METRIC = ops.CENSORED_METRIC
# A.6's committed interaction on the primary (Logbook 074's control.json, `interaction_mean`).
A6_MOVE: dict[str, float] = {"ppo": -0.027791666666666662, "reading": -0.09859027777777779}


def minimum(half: str) -> float:
    """Return the registered minimum: 2/3 of A.6's committed move."""
    return mc.MINIMUM_FRACTION * abs(A6_MOVE[half])


class SplitError(ValueError):
    """The panel on disk is not the panel this module scores."""


_VERDICT_BY_STATE = {
    "move_null": "gap_junctions",
    "below": "partial",
    "no_move": "not_gap_junctions",
    "move_wt": "opposite",
    "unresolved": "unresolved",
}


def read_learner(half: str, gates: dict[str, Any], interaction: dict[str, Any]) -> dict[str, Any]:
    """Gates first, then the state and the verdict, for one learner."""
    readable = {level: mp.level_passes(g) for level, g in gates.items()}
    out: dict[str, Any] = {
        "minimum": minimum(half),
        "a6_move": A6_MOVE[half],
        "level_readable": readable,
    }
    if not all(readable.values()):
        out["verdict"] = "unreadable"
        out["why"] = "a level fails its floor on a wiring, or both its arms saturate"
        return out
    test = interaction["test"]
    out["state"] = mc.classify(
        interaction["interaction_mean"],
        test["ci_lo"],
        test["ci_hi"],
        test["bh_q"],
        minimum(half),
    )
    out["verdict"] = _VERDICT_BY_STATE[out["state"]]
    return out


# ── Manifest over two campaigns ──────────────────────────────────────────────────────────────
def build_manifest(
    campaign_dirs: tuple[Path, ...],
    path: Path,
    half: str,
    seeds: tuple[int, ...],
) -> Path:
    """Write ``<arm> <level> <seed> <log>`` from every campaign, a wild-type run under each level.

    Every stem A.6's campaign holds is one of this panel's, so a stray log from either campaign is
    an error rather than a silently dropped row.
    """
    seen: set[tuple[str, str, int]] = set()
    lines: list[str] = []
    for campaign_dir in campaign_dirs:
        logs = campaign_dir / "logs" if (campaign_dir / "logs").is_dir() else campaign_dir
        for log in sorted(logs.glob("*.log")):
            stem, sep, seed_part = log.stem.rpartition("-seed")
            if not sep:
                msg = f"{log.name} has no -seedN suffix, so it cannot be placed in the panel"
                raise SplitError(msg)
            entry = LEVELS_BY_STEM.get(stem)
            if entry is None:
                msg = f"{log.name} names config {stem!r}, which this panel does not have"
                raise SplitError(msg)
            log_half, arm, levels = entry
            seed = int(seed_part)
            if log_half != half or seed not in seeds:
                continue
            resolved = log.resolve()
            try:
                out = str(resolved.relative_to(wp.REPO))
            except ValueError:
                out = str(resolved)
            for level in levels:
                key = (arm, level, seed)
                if key in seen:
                    msg = f"{key} appears twice across {[str(c) for c in campaign_dirs]}"
                    raise SplitError(msg)
                seen.add(key)
                lines.append(f"{arm} {level} {seed} {out}")
    path.write_text("\n".join(lines) + "\n")
    return path


# ── The identity check and the drift evidence ────────────────────────────────────────────────
_RUN_LINE = re.compile(r"^.*Run:\s+\d+\s+Status:.*$", re.MULTILINE)


def run_lines(log: Path) -> list[str]:
    """Every per-episode ``Run:`` line, the series every instrument here parses."""
    return [m.group(0).strip() for m in _RUN_LINE.finditer(log.read_text(errors="ignore"))]


def _final_w_chem(log: Path) -> torch.Tensor | None:
    import l4_reduced_perturbation as rp  # pyright: ignore[reportMissingImports]

    return rp._chemical_weights(log)


def compare_runs(committed: Path, rerun: Path) -> dict[str, Any]:
    """Compare a re-run with its committed run on every field the analysis reads."""
    import torch

    a, b = run_lines(committed), run_lines(rerun)
    first = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), None)
    wa, wb = _final_w_chem(committed), _final_w_chem(rerun)
    weights_equal = wa is not None and wb is not None and torch.equal(wa, wb)
    return {
        "committed": str(committed),
        "rerun": str(rerun),
        "run_lines": len(a),
        "run_lines_equal": a == b,
        "first_difference": first if a != b else None,
        "w_chem_found": wa is not None and wb is not None,
        "w_chem_equal": weights_equal,
        "identical": a == b and bool(a) and weights_equal,
    }


def identity_check(a6_campaign: Path, identity_campaign: Path, half: str) -> dict[str, Any]:
    """Compare each reused arm's re-run at the identity seed with its A.6 run."""
    seed = IDENTITY_SEED[half]
    results = {}
    for stem in REUSED_STEMS[half]:
        committed = a6_campaign / "logs" / f"{stem}-seed{seed}.log"
        rerun = identity_campaign / "logs" / f"{stem}-seed{seed}.log"
        if not committed.is_file() or not rerun.is_file():
            results[stem] = {
                "identical": False,
                "missing": [str(p) for p in (committed, rerun) if not p.is_file()],
            }
            continue
        results[stem] = compare_runs(committed, rerun)
    return {
        "half": half,
        "seed": seed,
        "arms": results,
        "all_identical": all(r["identical"] for r in results.values()),
    }


def reused_evidence(a6_campaign: Path, half: str) -> dict[str, Any]:
    """Confirm every reused A.6 run still resolves to its final chemical matrix."""
    missing: list[str] = []
    checked = 0
    for stem in REUSED_STEMS[half]:
        for seed in SEEDS_BY_HALF[half]:
            log = a6_campaign / "logs" / f"{stem}-seed{seed}.log"
            checked += 1
            if not log.is_file() or _final_w_chem(log) is None:
                missing.append(f"{stem}-seed{seed}")
    return {"half": half, "checked": checked, "missing": missing, "complete": not missing}


A6_PER_SEED = (
    wp.REPO / "docs/experiments/logbooks/supporting/074-null-strength-control/per-seed.csv"
)


def reproduces_a6(
    half: str,
    a6_move_per_seed: dict[int, float],
    committed: Path = A6_PER_SEED,
) -> dict[str, Any]:
    """Check A.6's move, recomputed from the reused runs, against A.6's committed per-seed CSV.

    The reuse is end to end only if re-scoring the current and chemical-only nulls gives back the
    interaction A.6 committed, seed by seed, to the CSV's six decimals.
    """
    with committed.open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["half"] == half]
    want = {int(r["seed"]): float(r[f"interaction_{PRIMARY_METRIC}"]) for r in rows}
    diffs = {
        seed: a6_move_per_seed.get(seed)
        for seed in want
        if a6_move_per_seed.get(seed) is None or abs(a6_move_per_seed[seed] - want[seed]) > 5e-7
    }
    return {"seeds": len(want), "mismatched": sorted(diffs), "reproduced": not diffs}


# ── Scoring ──────────────────────────────────────────────────────────────────────────────────
def score_half(
    campaign_dirs: tuple[Path, ...],
    half: str,
    out_dir: Path,
    seeds: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Score one learner: gates, gaps, the primary, and A.6's two terms for the breakdown."""
    seeds = seeds if seeds is not None else SEEDS_BY_HALF[half]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(campaign_dirs, out_dir / f"manifest-{half}.txt", half, seeds)
    mp.require_complete(manifest, half, seeds, LEVELS)
    reports = {
        level: ops.score_level(manifest, half, level, out_dir / f"tmp-{half}-{level}")
        for level in LEVELS
    }
    rates = {level: ops.censoring_rates(report) for level, report in reports.items()}
    metrics = (PRIMARY_METRIC, BESIDE_METRIC)
    drift = ops.substrate_drift(manifest, half, seeds, LEVELS, floor_levels={s: s for s in LEVELS})
    drift["obligation_applies"] = half == "reading"
    return {
        "half": half,
        "cell": ops.CELL,
        "seeds": list(seeds),
        "gates": {
            level: ops.learning_gates(manifest, half, seeds, level, floor_level=level)
            for level in LEVELS
        },
        "gaps": {m: {lv: ops.wiring_gap(reports[lv], m) for lv in LEVELS} for m in metrics},
        # The primary: the gap junctions' joint effect, on the same chemical graph.
        "interactions": {m: ops.interaction(reports[FULL], reports[GAP_HELD], m) for m in metrics},
        # The breakdown, as description: A.6's move and its remainder term.
        "breakdown": {
            m: {
                "a6_move": ops.interaction(reports[FULL], reports[CHEMICAL], m),
                "remainder": ops.interaction(reports[GAP_HELD], reports[CHEMICAL], m),
            }
            for m in metrics
        },
        "censoring_rule_choice": ops.choose_metric(rates),
        "substrate_drift": drift,
        "verdicts": {level: report.get("verdict") for level, report in reports.items()},
    }


def score(campaigns: dict[str, tuple[Path, ...]], out_dir: Path) -> dict[str, Any]:
    """Score both learners, correct the two primaries together, then read the verdicts."""
    missing = set(HALVES) - set(campaigns)
    if missing:
        msg = f"the family spans both learners; missing {sorted(missing)}"
        raise SplitError(msg)
    halves = {half: score_half(dirs, half, out_dir) for half, dirs in campaigns.items()}
    for metric in (PRIMARY_METRIC, BESIDE_METRIC):
        nsc.correct_family(halves, metric)
    for half, result in halves.items():
        result["a6_reproduced"] = reproduces_a6(
            half,
            {
                int(k): v
                for k, v in result["breakdown"][PRIMARY_METRIC]["a6_move"]["per_seed"].items()
            },
        )
        result["reading"] = mc.honour_drift(
            half,
            read_learner(half, result["gates"], result["interactions"][PRIMARY_METRIC]),
            result["substrate_drift"],
        )
    return {"primary_metric": PRIMARY_METRIC, "beside_metric": BESIDE_METRIC, "halves": halves}


def write_csv(result: dict[str, Any], path: Path) -> Path:
    """One row per (learner, seed): each arm's plateau and floor, the gaps, and the breakdown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = (PRIMARY_METRIC, BESIDE_METRIC)
    header = ["half", "seed"]
    for level in LEVELS:
        header += [f"{level}_{c}" for c in ("wt_plateau", "wt_floor", "rn_plateau", "rn_floor")]
        header += [f"{level}_gap_{m}" for m in metrics]
    for m in metrics:
        header += [f"gap_junctions_{m}", f"a6_move_{m}", f"remainder_{m}"]
    with path.open("w", newline="") as handle:
        # csv defaults to CRLF, which would make every regeneration read as a whole-file diff.
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for half, res in result["halves"].items():
            for seed in res["seeds"]:
                row: list[Any] = [half, seed]
                for level in LEVELS:
                    gates = res["gates"][level]
                    for wiring in ("wt", "rn"):
                        per_seed = gates[wiring]["per_seed"].get(seed, {})
                        row += [_fmt(per_seed.get("learn")), _fmt(per_seed.get("floor"))]
                    row += [_fmt(res["gaps"][m][level]["per_seed"].get(seed)) for m in metrics]
                for m in metrics:
                    row += [
                        _fmt(res["interactions"][m]["per_seed"].get(seed)),
                        _fmt(res["breakdown"][m]["a6_move"]["per_seed"].get(seed)),
                        _fmt(res["breakdown"][m]["remainder"]["per_seed"].get(seed)),
                    ]
                writer.writerow(row)
    return path


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def main(argv: list[str] | None = None) -> int:
    """CLI: the identity check, the reused-evidence check, or the scoring."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)
    ident = sub.add_parser("identity", help="compare the identity re-runs with A.6's runs")
    ident.add_argument("--half", choices=HALVES, required=True)
    ident.add_argument("--a6-campaign", type=Path, required=True)
    ident.add_argument("--identity-campaign", type=Path, required=True)
    ident.add_argument("--out", type=Path)
    evid = sub.add_parser("evidence", help="confirm every reused A.6 run keeps its drift evidence")
    evid.add_argument("--half", choices=HALVES, required=True)
    evid.add_argument("--a6-campaign", type=Path, required=True)
    sc = sub.add_parser("score", help="score both learners together")
    sc.add_argument("--a6-ppo", type=Path, required=True)
    sc.add_argument("--a6-reading", type=Path, required=True)
    sc.add_argument("--split-ppo", type=Path, required=True)
    sc.add_argument("--split-reading", type=Path, required=True)
    sc.add_argument("--out-dir", type=Path, required=True)
    sc.add_argument("--out", type=Path)
    sc.add_argument("--csv", type=Path)
    args = ap.parse_args(argv)

    if args.command == "identity":
        result = identity_check(args.a6_campaign, args.identity_campaign, args.half)
    elif args.command == "evidence":
        result = reused_evidence(args.a6_campaign, args.half)
    else:
        result = score(
            {
                "ppo": (args.a6_ppo, args.split_ppo),
                "reading": (args.a6_reading, args.split_reading),
            },
            args.out_dir,
        )
        if args.csv:
            write_csv(result, args.csv)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if getattr(args, "out", None):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    else:
        print(payload)
    ok = result.get("all_identical", result.get("complete", True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
