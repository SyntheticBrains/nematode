#!/usr/bin/env python
r"""C.1e: the wiring contrast through the kinematic body. The pilot that sets the panel's minimum.

The wild type against the chemical-only null through the body, at the 500-step cell, under two
learners: PPO, which writes the chemical weights, and frozen-wiring PPO, which reads them without
writing. Each wiring has one frozen floor, shared by both learners. Seeds 1701-1716.

**What the pilot fixes, per learner, before the panel registers:**

* the reference effect: the paired wild-type-minus-null ``auc_success`` mean;
* the minimum: 2/3 of |reference|, floored at 0.0367, a judgement carried from the point worm so a
  near-zero pilot cannot make a trivial difference count as a move;
* the panel's seeds: the smallest n at which ``2.487 * sd / sqrt(n)`` is at most the minimum, capped
  at 64;
* the gates: both learning arms beat their floors, and the level is not saturated. A learner that
  fails leaves the panel.

Beside them, the frequency of competent seeds (plateau >= 30%) under each wiring, compared by an
exact McNemar test on the seeds where the wirings disagree, reported as description.

Every statistic is A.2's (``operating_point_surface``); the gates' completeness check is B.1b's.

Usage::

    uv run python scripts/analysis/body_wiring.py pilot --logs campaigns/c1e-pilot/logs \\
        --out-dir build/c1e --out pilot.json
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parent / "campaigns"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import gate_preflight as gp  # noqa: E402  # pyright: ignore[reportMissingImports]
import generate_body_wiring_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]
import measured_prior_pilot as mp  # noqa: E402  # pyright: ignore[reportMissingImports]
import operating_point_surface as ops  # noqa: E402  # pyright: ignore[reportMissingImports]
from scipy.stats import binomtest  # noqa: E402

HALF = "ppo"
# The efficiency harness's name for hard350's food cell; the body's 500-step cell is scored there.
CELL = "hard_food"
PILOT_SEEDS: tuple[int, ...] = tuple(range(1701, 1717))
LEARNERS: tuple[str, ...] = ("ppo", "fw")
MINIMUM_FLOOR = 0.0367
MINIMUM_FRACTION = 2.0 / 3.0
MDE_Z = 2.487
MAX_PANEL_SEEDS = 64
COMPETENCE = 30.0
PRIMARY_METRIC = ops.UNCENSORED_METRIC
BESIDE_METRIC = ops.CENSORED_METRIC

# Learner -> arm -> config stem: the gate preflight's table, one level per learner.
STEMS: dict[str, dict[str, str]] = {
    learner: {
        "wt_learn": gen.stem("wt", learner),
        "wt_frozen": gen.stem("wt", "frozen"),
        "rn_learn": gen.stem("chemnull", learner),
        "rn_frozen": gen.stem("chemnull", "frozen"),
    }
    for learner in LEARNERS
}


class BodyWiringError(ValueError):
    """The runs on disk are not the panel this module scores."""


def build_manifest(log_dirs: list[Path], path: Path, seeds: tuple[int, ...]) -> Path:
    """Write ``<arm> <learner> <seed> <log>`` for every run on ``seeds``."""
    seen: dict[tuple[str, str, int], Path] = {}
    lines: list[str] = []
    for arm, level, seed, log in gp.evidence(STEMS, log_dirs):
        if seed not in seeds:
            continue
        prior = seen.setdefault((arm, level, seed), log)
        if prior != log:
            msg = f"{arm} {level} seed {seed} has two runs: {prior} and {log}"
            raise BodyWiringError(msg)
        lines.append(f"{arm} {level} {seed} {log.resolve()}")
    path.write_text("\n".join(lines) + "\n")
    return path


def minimum(reference: float) -> float:
    """Return the registered minimum: 2/3 of |reference|, never below the judged floor."""
    return max(MINIMUM_FRACTION * abs(reference), MINIMUM_FLOOR)


def panel_seeds(sd: float, minimum_effect: float) -> dict[str, Any]:
    """Return the smallest n whose MDE is at most the minimum, capped at ``MAX_PANEL_SEEDS``."""
    for n in range(4, MAX_PANEL_SEEDS + 1):
        if MDE_Z * sd / math.sqrt(n) <= minimum_effect:
            return {"n": n, "capped": False, "mde": MDE_Z * sd / math.sqrt(n)}
    return {"n": MAX_PANEL_SEEDS, "capped": True, "mde": MDE_Z * sd / math.sqrt(MAX_PANEL_SEEDS)}


def competence_frequency(gates: dict[str, Any]) -> dict[str, Any]:
    """Compare how often each wiring's seeds reach competence, by exact McNemar on discordant pairs."""
    wt = gates["wt"]["per_seed"]
    rn = gates["rn"]["per_seed"]
    seeds = sorted(set(wt) & set(rn))
    wt_ok = {s: wt[s]["learn"] >= COMPETENCE for s in seeds}
    rn_ok = {s: rn[s]["learn"] >= COMPETENCE for s in seeds}
    only_wt = sum(1 for s in seeds if wt_ok[s] and not rn_ok[s])
    only_rn = sum(1 for s in seeds if rn_ok[s] and not wt_ok[s])
    discordant = only_wt + only_rn
    p = float(binomtest(only_wt, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "n_seeds": len(seeds),
        "wt_competent": sum(wt_ok.values()),
        "rn_competent": sum(rn_ok.values()),
        "only_wt": only_wt,
        "only_rn": only_rn,
        "mcnemar_exact_p": p,
    }


def read_learner(gates: dict[str, Any], gap: dict[str, Any]) -> dict[str, Any]:
    """Fix one learner's reference, minimum and panel size, or record why it leaves the panel."""
    readable = mp.level_passes(gates)
    per_seed = list(gap["per_seed"].values())
    reference = float(gap["gap_mean"])
    sd = statistics.stdev(per_seed) if len(per_seed) > 1 else float("nan")
    minimum_effect = minimum(reference)
    out: dict[str, Any] = {
        "readable": readable,
        "reference": reference,
        "reference_test": gap["test"],
        "sd": sd,
        "minimum": minimum_effect,
        "minimum_floored": minimum_effect == MINIMUM_FLOOR,
        "panel": panel_seeds(sd, minimum_effect),
    }
    if not readable:
        out["leaves_panel"] = (
            "a learning arm does not beat its floor" if not gates["gate_passes"] else "saturated"
        )
    return out


def pilot(
    log_dirs: list[Path],
    out_dir: Path,
    seeds: tuple[int, ...] = PILOT_SEEDS,
) -> dict[str, Any]:
    """Score the pilot: each learner's gates, wiring gap, minimum and the panel's seed count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(log_dirs, out_dir / "manifest-body-wiring-pilot.txt", seeds)
    mp.require_complete(manifest, HALF, seeds, LEARNERS)
    result: dict[str, Any] = {"seeds": list(seeds), "cell": CELL, "learners": {}}
    for learner in LEARNERS:
        report = ops.score_level(manifest, HALF, learner, out_dir / f"tmp-{learner}", cell=CELL)
        gates = ops.learning_gates(manifest, HALF, seeds, learner, floor_level=learner)
        gaps = {m: ops.wiring_gap(report, m) for m in (PRIMARY_METRIC, BESIDE_METRIC)}
        result["learners"][learner] = {
            "gates": gates,
            "gaps": gaps,
            "competence": competence_frequency(gates),
            **read_learner(gates, gaps[PRIMARY_METRIC]),
        }
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)
    pl = sub.add_parser("pilot", help="score the pilot")
    pl.add_argument("--logs", type=Path, action="append", required=True)
    pl.add_argument("--out-dir", type=Path, required=True)
    pl.add_argument("--out", type=Path, help="write the JSON here")
    args = ap.parse_args(argv)
    result = pilot(args.logs, args.out_dir)
    payload = json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
