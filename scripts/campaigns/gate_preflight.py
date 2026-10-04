#!/usr/bin/env python
r"""Evaluate a panel's registered gates on existing runs before its campaign launches.

A wiring contrast is read only at a level where both learning arms beat their frozen floors and do
not both sit at or above the saturation bar. This evaluates exactly those gates, through the same
function the analyses use, on runs that already exist at the registered point: committed campaigns
where they cover it, otherwise a short pilot on disjoint seeds configured the way the campaign will
be. It exits nonzero if any level would be unreadable, has no or incomplete evidence, or sits within a
margin of the bar, so a launch can be blocked on it.

The panel's arms come from an analysis module's ``STEMS`` table, ``level -> arm -> config stem``,
or ``half -> level -> arm -> stem`` with ``--half``. Every log under each ``--logs`` directory whose
stem matches an arm is used, whatever its seed.

Usage::

    uv run python scripts/campaigns/gate_preflight.py --panel thermal_null_strength \
        --logs campaigns/init-sharing-control/logs
    uv run python scripts/campaigns/gate_preflight.py --panel null_strength_control --half ppo \
        --logs campaigns/a6-ppo/logs --margin-points 5
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

_ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import operating_point_surface as ops  # noqa: E402  # pyright: ignore[reportMissingImports]
import wiring_premise as wp  # noqa: E402  # pyright: ignore[reportMissingImports]

ARMS = ("wt_learn", "wt_frozen", "rn_learn", "rn_frozen")
# A level whose higher learning plateau sits within this many percentage points of the bar is
# flagged: a new seed band can move a plateau by a few points, so a level just under the bar on old
# runs can cross it.
DEFAULT_MARGIN_POINTS = 5.0


def panel_stems(module: str, half: str | None = None) -> dict[str, dict[str, str]]:
    """Return ``level -> arm -> stem`` from an analysis module's ``STEMS`` table."""
    stems = importlib.import_module(module).STEMS
    if half is not None:
        stems = stems[half]
    for level, arms in stems.items():
        if set(arms) != set(ARMS):
            msg = f"{module} level {level!r} has arms {sorted(arms)}, expected {sorted(ARMS)}"
            raise ValueError(msg)
    return stems


def evidence(
    stems: dict[str, dict[str, str]],
    log_dirs: list[Path],
) -> list[tuple[str, str, int, Path]]:
    """Find every ``(arm, level, seed, log)`` the given directories hold for the panel's arms."""
    by_stem: dict[str, list[tuple[str, str]]] = {}
    for level, arms in stems.items():
        for arm, stem in arms.items():
            by_stem.setdefault(stem, []).append((arm, level))
    rows: list[tuple[str, str, int, Path]] = []
    for log_dir in log_dirs:
        for log in sorted(log_dir.glob("*.log")):
            stem, sep, seed = log.stem.rpartition("-seed")
            if not sep or stem not in by_stem or not seed.isdigit():
                continue
            rows.extend((arm, level, int(seed), log) for arm, level in by_stem[stem])
    return rows


def preflight(
    stems: dict[str, dict[str, str]],
    log_dirs: list[Path],
    margin_points: float = DEFAULT_MARGIN_POINTS,
) -> dict[str, Any]:
    """Evaluate each level's floor and saturation gates on the runs found, and say whether to launch."""
    rows = evidence(stems, log_dirs)
    # One run per arm, level and seed: two logs for the same key (the same stem and seed in two
    # directories) would leave the gate to read whichever it saw last.
    seen: dict[tuple[str, str, int], Path] = {}
    for arm, level, seed, log in rows:
        prior = seen.setdefault((arm, level, seed), log)
        if prior != log:
            msg = f"{arm} {level} seed {seed} has two runs: {prior} and {log}"
            raise ValueError(msg)
    bar = wp.SATURATION_SUCCESS
    levels: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "manifest.txt"
        manifest.write_text(
            "".join(f"{arm} {level} {seed} {log.resolve()}\n" for arm, level, seed, log in rows),
        )
        for level in stems:
            seeds = tuple(
                sorted(
                    set.intersection(
                        *({s for a, lv, s, _ in rows if a == arm and lv == level} for arm in ARMS),
                    ),
                ),
            )
            if not seeds:
                levels[level] = {"status": "no_evidence", "n_seeds": 0}
                continue
            gates = ops.learning_gates(manifest, "ppo", seeds, level, floor_level=level)
            top = max(gates["wt"]["plateau_success"], gates["rn"]["plateau_success"])
            # The gate skips a seed whose log yields no plateau (a run still going, or cut short),
            # so a level is complete only if every selected seed was scored on both wirings.
            scored = min(gates["wt"]["n_seeds"], gates["rn"]["n_seeds"])
            if scored < len(seeds):
                status = "incomplete_evidence"
            elif not gates["gate_passes"]:
                status = "fails_floor"
            elif gates["saturated"]:
                status = "saturated"
            elif top >= bar - margin_points:
                status = "near_bar"
            else:
                status = "readable"
            levels[level] = {
                "status": status,
                "n_seeds": len(seeds),
                "n_scored": scored,
                "wt_plateau": gates["wt"]["plateau_success"],
                "rn_plateau": gates["rn"]["plateau_success"],
                "wt_floor": gates["wt"]["floor_success"],
                "rn_floor": gates["rn"]["floor_success"],
            }
    return {
        "saturation_bar": bar,
        "margin_points": margin_points,
        "levels": levels,
        "launch": all(entry["status"] == "readable" for entry in levels.values()),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: print the preflight and exit 1 unless every level is readable with room to spare."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--panel", required=True, help="analysis module with a STEMS table")
    ap.add_argument("--half", help="the learner key, for a module whose STEMS is keyed by learner")
    ap.add_argument("--logs", type=Path, action="append", required=True, help="a run-log directory")
    ap.add_argument(
        "--margin-points",
        type=float,
        default=DEFAULT_MARGIN_POINTS,
        help="flag a level whose higher plateau is within this many points of the bar",
    )
    args = ap.parse_args(argv)
    result = preflight(panel_stems(args.panel, args.half), args.logs, args.margin_points)
    print(json.dumps(result, indent=2))
    return 0 if result["launch"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
