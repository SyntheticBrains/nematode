#!/usr/bin/env python
"""C.0's validation: whether each learner still learns hard350 with reversal on.

Two learners, each learning and frozen, on hard350 with ``allow_reversal`` and ``signed_speed`` on,
seeds 1301-1308: MLP-PPO (width 64) and the connectome's settling wild type on Emmons 2024, 8b's
substrate. **The gate**: a learner passes if its plateau success beats its frozen floor, paired by
seed, with the 80% bootstrap interval of the difference above zero. Both pass: signed speed
validates and the substrate freezes. Plateaus are reported beside, never read.

Usage::

    uv run python scripts/analysis/body_prerequisites.py --logs campaigns/c0-pilot/logs \
        --out validation.json
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

SEEDS: tuple[int, ...] = tuple(range(1301, 1309))
_MLP = "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_ppo_w64"
_CONNECTOME = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350"
# Learner -> (learning stem, frozen stem).
LEARNERS: dict[str, tuple[str, str]] = {
    "mlp": (f"{_MLP}_reversal", f"{_MLP}_reversal_frozen"),
    "connectome": (f"{_CONNECTOME}_emmons_reversal", f"{_CONNECTOME}_frozen_emmons_reversal"),
}


def plateaus(log_dir: Path, stem: str, seeds: tuple[int, ...] = SEEDS) -> dict[int, float]:
    """Return each seed's plateau success, in percent, for one config's runs."""
    out: dict[int, float] = {}
    for seed in seeds:
        log = log_dir / f"{stem}-seed{seed}.log"
        tail = wp.plateau_tail(log) if log.is_file() else None
        if tail is not None:
            out[seed] = float(tail[0])
    return out


def gate(learn: dict[int, float], frozen: dict[int, float]) -> dict[str, Any]:
    """Read one learner's floor gate: paired plateau minus floor, 80% interval above zero."""
    seeds = sorted(set(learn) & set(frozen))
    test = wp.paired_seed_wilcoxon_bootstrap([learn[s] - frozen[s] for s in seeds])
    complete = len(seeds) == len(SEEDS)
    passes = complete and float(test["ci_lo"]) > 0.0
    return {
        "n_seeds": len(seeds),
        "plateau_mean": sum(learn[s] for s in seeds) / len(seeds) if seeds else None,
        "floor_mean": sum(frozen[s] for s in seeds) / len(seeds) if seeds else None,
        "test": test,
        "verdict": "passes" if passes else ("incomplete" if not complete else "fails"),
    }


def score(log_dirs: list[Path]) -> dict[str, Any]:
    """Every learner's plateaus, floors and gate."""
    result: dict[str, Any] = {"seeds": list(SEEDS), "learners": {}}
    for learner, (learn_stem, frozen_stem) in LEARNERS.items():
        learn: dict[int, float] = {}
        frozen: dict[int, float] = {}
        for log_dir in log_dirs:
            learn.update(plateaus(log_dir, learn_stem))
            frozen.update(plateaus(log_dir, frozen_stem))
        result["learners"][learner] = {
            "plateau": learn,
            "floor": frozen,
            **gate(learn, frozen),
        }
    verdicts = [entry["verdict"] for entry in result["learners"].values()]
    result["validated"] = all(v == "passes" for v in verdicts)
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--logs", type=Path, action="append", required=True, help="a run-log dir")
    ap.add_argument("--out", type=Path, help="write the validation JSON here")
    args = ap.parse_args(argv)
    result = score(args.logs)
    payload = json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
