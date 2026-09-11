"""Examine the three settings pinned since A.3, on platforms that can hold them.

Two of them act on a one-step trial and are examined on the committed control: the homeostatic
norm sphere, which returns every unit to its construction norm after each update, and the
exploration noise, which sets the action distribution the reward scores. The third, the
eligibility horizon, cannot act there at all -- the control resets the trace every trial, which is
what removes the horizon confound from the undelayed question -- so it is examined on the same
control with the reward delayed.

What a delay imposes is dilution rather than decay. The credited step's contribution to the trace
is one term among ``delay + 1`` by the time the modulator arrives, and a rule that normalises its
trace by a running RMS rescales the whole sum while leaving that share alone. A delay whose
intervening steps added nothing would be a pure scalar decay and would be divided straight out, so
the control's filler drives the plastic layer and carries nothing about the cue.

Each cell is scored by the control's own registered pass rule, against bounds computed at that
cell's exploration noise -- the floor and the optimum both move with it, while the gap between
them is the target variance and does not, so the fraction of the gap closed stays comparable
across the grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import l4_rule_positive_control as pc  # noqa: E402  # pyright: ignore[reportMissingImports]

# The variant this examines: the eligibility that passed the control, at the scale that passed it.
ARM = "node_perturbation"
NODE_NOISE = 0.2


def _cell(  # noqa: PLR0913 - one parameter per knob under examination
    task: pc.ContextualAssociation,
    trials: int,
    *,
    noise: float = pc.NOISE,
    delay: int = 0,
    trace_decay: float = pc.TRACE_DECAY,
    homeostasis: bool = True,
) -> dict[str, Any]:
    """Run one cell over the registered seeds and score it by the control's own rule."""
    runs = [
        pc.run_arm(
            ARM,
            seed,
            task,
            trials=trials,
            noise=noise,
            node_noise=NODE_NOISE,
            delay=delay,
            trace_decay=trace_decay,
            homeostasis=homeostasis,
        )
        for seed in pc.SEEDS
    ]
    floor, optimum = task.cue_blind_floor(noise), task.optimum(noise)
    scored = pc.assess([r["score"] for r in runs], floor, optimum)
    scored["gap_closed"] = (
        (scored["mean"] - floor) / (optimum - floor) if optimum != floor else float("nan")
    )
    scored["alignment"] = float(
        np.mean([r["alignment"] for r in runs if not math.isnan(r["alignment"])] or [math.nan]),
    )
    scored["nominal_credit_ratio"] = runs[0]["nominal_credit_ratio"]
    scored["knobs"] = {
        "delay": runs[0]["delay"],
        "trace_decay": runs[0]["trace_decay"],
        "homeostasis": runs[0]["homeostasis"],
        "action_noise": runs[0]["action_noise"],
    }
    scored["per_seed"] = {str(r["seed"]): r["score"] for r in runs}
    return scored


def homeostasis_grid(task: pc.ContextualAssociation, trials: int) -> dict[str, Any]:
    """Run the homeostatic norm sphere on and off, at the pinned values otherwise."""
    return {str(on): _cell(task, trials, homeostasis=on) for on in (True, False)}


def action_noise_grid(task: pc.ContextualAssociation, trials: int) -> dict[str, Any]:
    """Run the exploration noise over the grid the panels' own history brackets."""
    return {f"{noise:.4f}": _cell(task, trials, noise=noise) for noise in pc.ACTION_NOISE_GRID}


def horizon_grid(task: pc.ContextualAssociation, trials: int) -> dict[str, Any]:
    """Run the eligibility horizon: trace decay against the delay it has to bridge."""
    return {
        f"{decay}|{delay}": _cell(task, trials, trace_decay=decay, delay=delay)
        for decay in pc.TRACE_DECAY_GRID
        for delay in pc.DELAY_GRID
    }


def _jsonable(value: object) -> object:
    """Replace not-a-number with null, recursively, so the record is strict JSON."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _print(out: dict[str, Any]) -> None:
    """Print the grids, baselines marked."""
    print(f"\nThe unexamined knobs, on the positive control ({len(pc.SEEDS)} seeds)")
    print(f"  arm: {ARM} at sigma {NODE_NOISE}; each cell scored by the control's own rule\n")

    print("  homeostasis | gap closed | above floor | passes")
    for key, cell in out["homeostasis"].items():
        mark = "  (pinned)" if key == "True" else ""
        print(
            f"  {key:>11} | {cell['gap_closed']:9.1%} | {cell['seeds_above_floor']}/8"
            f"         | {'PASS' if cell['passes'] else '-'}{mark}",
        )

    print("\n  action noise | gap closed | above floor | passes")
    for key, cell in out["action_noise"].items():
        mark = "  (pinned)" if abs(float(key) - pc.NOISE) < 1e-4 else ""
        print(
            f"  {float(key):12.4f} | {cell['gap_closed']:9.1%} | {cell['seeds_above_floor']}/8"
            f"         | {'PASS' if cell['passes'] else '-'}{mark}",
        )

    print("\n  horizon: gap closed by delay (nominal credit ratio in brackets)")
    header = "".join(f"{d:>16}" for d in pc.DELAY_GRID)
    print(f"  {'trace_decay':>12}{header}")
    for decay in pc.TRACE_DECAY_GRID:
        row = ""
        for delay in pc.DELAY_GRID:
            cell = out["horizon"][f"{decay}|{delay}"]
            row += f"{cell['gap_closed']:>9.1%} [{cell['nominal_credit_ratio']:.2f}]"
        mark = "  (pinned)" if decay == pc.TRACE_DECAY else ""
        print(f"  {decay:>12}{row}{mark}")


def main(argv: list[str] | None = None) -> int:
    """Run the knob grids and write their records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=pc.TRIALS)
    parser.add_argument("--out", type=Path, help="write the grids as JSON")
    parser.add_argument("--csv", type=Path, help="write the per-cell table")
    args = parser.parse_args(argv)

    task = pc.ContextualAssociation.default()
    out: dict[str, Any] = {
        "arm": ARM,
        "node_noise": NODE_NOISE,
        "trials": args.trials,
        "seeds": list(pc.SEEDS),
        "pinned": {
            "trace_decay": pc.TRACE_DECAY,
            "homeostasis": True,
            "action_noise": pc.NOISE,
            "delay": 0,
        },
        "homeostasis": homeostasis_grid(task, args.trials),
        "action_noise": action_noise_grid(task, args.trials),
        "horizon": horizon_grid(task, args.trials),
    }
    _print(out)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(_jsonable(out), indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "grid",
                    "cell",
                    "delay",
                    "trace_decay",
                    "homeostasis",
                    "action_noise",
                    "nominal_credit_ratio",
                    "mean",
                    "gap_closed",
                    "seeds_above_floor",
                    "alignment",
                    "passes",
                ],
            )
            for grid in ("homeostasis", "action_noise", "horizon"):
                for key, cell in out[grid].items():
                    knobs = cell["knobs"]
                    writer.writerow(
                        [
                            grid,
                            key,
                            knobs["delay"],
                            knobs["trace_decay"],
                            knobs["homeostasis"],
                            f"{knobs['action_noise']:.4f}",
                            f"{cell['nominal_credit_ratio']:.4f}",
                            f"{cell['mean']:.6f}",
                            f"{cell['gap_closed']:.4f}",
                            cell["seeds_above_floor"],
                            f"{cell['alignment']:.4f}",
                            cell["passes"],
                        ],
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
