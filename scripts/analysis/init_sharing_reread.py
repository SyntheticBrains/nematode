#!/usr/bin/env python
"""A.1's control, re-read at a second settling depth.

A.1 asked whether block V's wiring advantage survives a shared initialisation, and answered yes at
the committed settling depth of 4. A.2's PPO surface then showed the advantage is **depth-critical**:
abolished at depth 3, reversed at depth 2, and surviving at depth 6. The operating-point rule makes
a positive result carrying an inherited learner setting re-readable at a calibrated point, and A.2's
registration drew the line before the panel read out: the re-read is owed where the sign moves at or
adjacent to the committed setting, which depth 3 is.

**Depth 6 is the only depth the re-read can use.** At 2 and 3 there is no wiring advantage left, so
a shared-initialisation control there would be testing whether an absent effect survives -- a
question with no content. Depth 6 is the one other setting where the effect exists to be dissolved.

**This is A.1's design, not a new one.** Same cell, same four arms, same two definitions of a shared
initialisation, same interaction as the primary, same censoring rule, same committed instruments. The
only thing that moves is `forward_pass_depth`, and the seeds are fresh so nothing is reused across
campaigns. The statistics come from A.1's own module rather than being rewritten here, because a
re-read that re-implements its own arithmetic cannot distinguish a changed reading from a changed
world.

**32 seeds, and the first 16 were not enough by A.1's own arithmetic.** The panel was launched at
193-208 and read out with a minimum detectable interaction of 2.00 and 1.72 times the baseline
effect on the primary metric -- it could not have detected a total dissolution. A.1's launch record
had already rejected exactly this, computing 1.25 and 1.14 at 16 seeds and moving to 32 for that
reason. 209-224 were run into the same campaign directory under an unchanged execution path, so this
is one 32-seed panel rather than two merged ones.
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import init_sharing_control as isc  # noqa: E402  # pyright: ignore[reportMissingImports]
import wiring_premise as wp  # noqa: E402  # pyright: ignore[reportMissingImports]


class RereadError(ValueError):
    """The panel on disk is not the panel this re-read scores."""


# Fresh: 1-96, 101-108, 129-160 were burnt before A.2, and A.2 took 161-176 and 177-192.
SEEDS = tuple(range(193, 225))
CELL = "hard_food"
DEPTH = 6

_BASE = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350"
_ARM_SUFFIX = {
    "wt_ppo": "",
    "rn_ppo": "_rewired_null",
    "wt_frozen": "_frozen",
    "rn_frozen": "_rewired_null_frozen",
}
_MODE_TAG = {"edge_order": "", "dense_mask": "_densemask", "per_neuron_fanin": "_fanin"}


def _arms_by_stem() -> dict[str, tuple[str, str]]:
    """Config stem -> ``(arm, draw mode)``, stated explicitly rather than derived by regex.

    The suffix order is not free in this tree: the arm part follows `hard350`, the draw mode follows
    the arm, and the depth tag is last. A mis-keyed arm silently drops one side of a paired test.
    """
    out: dict[str, tuple[str, str]] = {}
    for mode, tag in _MODE_TAG.items():
        for arm, arm_suffix in _ARM_SUFFIX.items():
            stem = f"{_BASE}{arm_suffix}{tag}_d{DEPTH}"
            if stem in out:
                msg = f"stem {stem!r} is claimed twice"
                raise ValueError(msg)
            out[stem] = (arm, mode)
    return out


ARM_BY_STEM: dict[str, tuple[str, str]] = _arms_by_stem()


def build_manifest(campaign_dir: Path, path: Path, mode: str, seeds: tuple[int, ...]) -> Path:
    """Write the instrument's own ``<cell> <arm> <seed> <out>`` lines for one draw mode."""
    logs = campaign_dir / "logs"
    if not logs.is_dir():
        logs = campaign_dir
    seen: set[tuple[str, int]] = set()
    lines: list[str] = []
    for log in sorted(logs.glob("*.log")):
        stem, sep, seed_part = log.stem.rpartition("-seed")
        if not sep:
            msg = f"{log.name} has no -seedN suffix"
            raise RereadError(msg)
        entry = ARM_BY_STEM.get(stem)
        if entry is None:
            msg = f"{log.name} names config {stem!r}, which this re-read does not have"
            raise RereadError(msg)
        arm, log_mode = entry
        seed = int(seed_part)
        if log_mode != mode or seed not in seeds:
            continue
        if (arm, seed) in seen:
            msg = f"{(arm, seed)} appears twice in {campaign_dir}"
            raise RereadError(msg)
        seen.add((arm, seed))
        resolved = log.resolve()
        try:
            out = str(resolved.relative_to(wp.REPO))
        except ValueError:
            out = str(resolved)
        lines.append(f"{CELL} {arm} {seed} {out}")
    path.write_text("\n".join(lines) + "\n")
    return path


def require_complete(manifest: Path, mode: str, seeds: tuple[int, ...]) -> None:
    """Refuse a partial panel: the instrument only warns, and a re-read must not read a gap."""
    have: dict[str, set[int]] = {}
    for raw in manifest.read_text().splitlines():
        _, arm, seed, _ = raw.split()
        have.setdefault(arm, set()).add(int(seed))
    missing = [
        f"{arm}: {sorted(set(seeds) - have.get(arm, set()))}"
        for arm in wp.TESTED_ARMS
        if set(seeds) - have.get(arm, set())
    ]
    if missing:
        msg = f"{mode} panel is incomplete — " + "; ".join(missing)
        raise RereadError(msg)


def score(campaign_dir: Path, out_dir: Path, seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """Read the interaction of each sharing mode with wiring, at depth 6."""
    out_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    for mode in isc.MODES:
        manifest = build_manifest(campaign_dir, out_dir / f"manifest-{mode}.txt", mode, seeds)
        require_complete(manifest, mode, seeds)
        tmp = out_dir / f"tmp-{mode}"
        tmp.mkdir(parents=True, exist_ok=True)
        report = wp.efficiency_contrast(manifest, CELL, tmp)
        if report is None:
            msg = f"{mode} produced no paired contrast"
            raise RereadError(msg)
        reports[mode] = report

    rates = {mode: isc.censoring_rates(report) for mode, report in reports.items()}
    metric_choice = isc.choose_metric(rates)

    interactions: dict[str, Any] = {}
    for mode in isc.SHARING_MODES:
        interactions[mode] = {
            metric: isc.interaction(reports[isc.BASELINE_MODE], reports[mode], metric)
            for metric in (metric_choice["primary_metric"], metric_choice["reported_beside"])
        }
    return {
        "cell": CELL,
        "forward_pass_depth": DEPTH,
        "seeds": list(seeds),
        "metric_choice": metric_choice,
        "interactions": interactions,
        "verdicts": {mode: report.get("verdict") for mode, report in reports.items()},
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: score the re-read from its campaign directory."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args(argv)

    result = score(args.campaign, args.out_dir)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    else:
        print(payload)
    if args.csv:
        isc.write_csv({"cells": {CELL: result}}, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
