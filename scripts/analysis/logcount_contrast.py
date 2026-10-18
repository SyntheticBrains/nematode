#!/usr/bin/env python
"""The wiring contrast on log-compressed gap counts: the wild type against its gap-only null.

With count-scaled additive gap coupling, a few high-count pairs between low-degree neurons couple far
above 1, and the wild type's lead over the gap-only null follows that scale. With each count replaced
by ``log1p(count)`` the coupling stays additive but those pairs no longer dominate. This module reads
the wiring contrast on that model.

**The panel**: the thermal cell at target 35 on Emmons 2024, 3,000 episodes, seeds 3901-3964; the wild
type and the gap-only null on log counts, each learning and frozen (256 runs).

**The reading**: ``gap(log)`` on ``auc_success``, wild type - null, paired by seed, with an 80% bootstrap
interval and the two-sided Wilcoxon p as its q (a family of one), classified by ``mc.classify`` at
m = 0.0577. Gates: each learning arm beats its wiring's frozen floor, and the two do not both plateau at
or above 90%.

==========================  ========================
state                       verdict
==========================  ========================
gates fail                  ``unreadable``
``move_wt``                 ``placement``
``below`` (either way)      ``below_minimum``
``no_move``                 ``no_wiring_effect``
``move_null``               ``reversed``
``unresolved``              ``unresolved``
==========================  ========================

Under ``placement`` the lead also fixes the minimum a mechanism panel on this model carries: 0.0577 if
the lead sits in [0.058, 0.13] or below 0.058, otherwise two-thirds of the lead.

Usage::

    uv run python scripts/analysis/logcount_contrast.py plan
    uv run python scripts/analysis/logcount_contrast.py preflight --logs <pilot logs> --out preflight.json
    uv run python scripts/analysis/logcount_contrast.py table --out table.json
    uv run python scripts/analysis/logcount_contrast.py score --logs <panel logs> --out-dir <scratch> \
        --out panel.json --csv per-seed.csv
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parent / "campaigns"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import gap_operating_point as gop  # noqa: E402  # pyright: ignore[reportMissingImports]
import gate_preflight as gp  # noqa: E402  # pyright: ignore[reportMissingImports]
import generate_gap_substrate_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]
import measured_prior_contrast as mc  # noqa: E402  # pyright: ignore[reportMissingImports]
import measured_prior_pilot as mp  # noqa: E402  # pyright: ignore[reportMissingImports]
import operating_point_surface as ops  # noqa: E402  # pyright: ignore[reportMissingImports]
import thermal_split as ts  # noqa: E402  # pyright: ignore[reportMissingImports]

HALF = "ppo"
CELL = ts.CELL
LEVEL = "log"
SEEDS: tuple[int, ...] = tuple(range(3901, 3965))
# The runs that chose this model, on the same configs: the gate evidence before launch.
PILOT_SEEDS: tuple[int, ...] = tuple(range(9871, 9887))
# Seed bands already read on this cell's gap contrasts: G.0's panel, anchor, smokes and gate pilot,
# the substrate pilot and its smokes, and Logbook 082's Cook draws.
COMMITTED_BANDS: tuple[tuple[int, int], ...] = (
    (2401, 2464),
    (9801, 9816),
    (9851, 9868),
    (9871, 9886),
    (9891, 9892),
    (513, 576),
)

_ARMS = gen.ARMS["thermal"]
STEMS: dict[str, dict[str, str]] = {
    LEVEL: {
        "wt_learn": _ARMS["log_wt_learn"],
        "wt_frozen": _ARMS["log_wt_frozen"],
        "rn_learn": _ARMS["log_gaponly_learn"],
        "rn_frozen": _ARMS["log_gaponly_frozen"],
    },
}

PRIMARY_METRIC = ops.UNCENSORED_METRIC
BESIDE_METRIC = ops.CENSORED_METRIC
METRICS: tuple[str, ...] = (PRIMARY_METRIC, BESIDE_METRIC)
MINIMUM = gop.CARRIED_MINIMUM
EPISODES = 3000

VERDICTS: dict[str, str] = {
    "move_wt": "placement",
    "below": "below_minimum",
    "no_move": "no_wiring_effect",
    "move_null": "reversed",
    "unresolved": "unresolved",
}


class ContrastError(ValueError):
    """The panel's runs cannot be read as registered."""


# ── The verdict map ──────────────────────────────────────────────────────────────────────────
def mechanism_minimum(lead: float, ci_lo: float, ci_hi: float) -> dict[str, Any]:
    """Return the minimum a mechanism panel on this model carries, from a ``placement`` lead.

    The anchor rule carries 0.0577 for a lead in [0.058, 0.13] and re-fixes it at two-thirds of the
    lead otherwise. A lead that clears 0.0577 but sits below 0.058 would re-fix the minimum under the
    bar the lead itself had to clear, so it carries 0.0577.
    """
    rule = gop.anchor_minimum(lead, ci_lo, ci_hi, True)  # noqa: FBT003
    if rule["case"] == "refixed" and lead < gop.CARRY_RANGE[0]:
        return {"case": "carried_below_range", "minimum": MINIMUM}
    return rule


def read(gates: dict[str, Any], gap: dict[str, Any], minimum: float = MINIMUM) -> dict[str, Any]:
    """Gates, then the state of ``gap(log)``, then the verdict."""
    out: dict[str, Any] = {"minimum": minimum, "readable": mp.level_passes(gates)}
    if not out["readable"]:
        out["verdict"] = "unreadable"
        out["why"] = "a learning arm fails its floor, or both saturate"
        return out
    test = gap["test"]
    mean = float(gap["gap_mean"])
    state = mc.classify(mean, test["ci_lo"], test["ci_hi"], test["bh_q"], minimum)
    out["state"] = state
    out["verdict"] = VERDICTS[state]
    if state == "below":
        out["direction"] = "wild_type" if mean > 0.0 else "null"
    if state == "move_wt":
        out["mechanism_minimum"] = mechanism_minimum(mean, test["ci_lo"], test["ci_hi"])
    return out


# ── Manifest ─────────────────────────────────────────────────────────────────────────────────
def _ran_config(log: Path) -> str | None:
    """Return the config file name a tracked run records on completion, or ``None``."""
    for raw in reversed(log.read_text(errors="replace").splitlines()):
        line = raw.strip()
        if line.startswith("Config:"):
            return Path(line.removeprefix("Config:").strip()).name
    return None


def build_manifest(log_dirs: list[Path], path: Path, seeds: tuple[int, ...] = SEEDS) -> Path:
    """Write ``<arm> <level> <seed> <log>`` for the panel's runs that exited 0.

    A log is named by the campaign runner from the config and seed it launched, and its ``.exit``
    marker is written by the runner after the run ends. A run that exited 0 must also record, in its
    own log, the config it ran under that name; one that does not is refused rather than scored.
    """
    seen: dict[tuple[str, str, int], Path] = {}
    lines: list[str] = []
    for arm, level, seed, log in gp.evidence(STEMS, log_dirs):
        status = log.with_suffix(".exit")
        if seed not in seeds or not status.is_file() or status.read_text().strip() != "0":
            continue
        expected = f"{log.stem.rpartition('-seed')[0]}.yml"
        ran = _ran_config(log)
        if ran != expected:
            msg = f"{log} exited 0 but records config {ran!r}, not {expected!r}"
            raise ContrastError(msg)
        prior = seen.setdefault((arm, level, seed), log)
        if prior != log:
            msg = f"{arm} {level} seed {seed} has two runs: {prior} and {log}"
            raise ContrastError(msg)
        lines.append(f"{arm} {level} {seed} {log.resolve()}")
    path.write_text("\n".join(lines) + "\n")
    return path


def require_complete(manifest: Path, seeds: tuple[int, ...] = SEEDS) -> None:
    """Refuse a partial panel: every arm on every seed, each run exited 0."""
    have: dict[str, set[int]] = {}
    for raw in manifest.read_text().splitlines():
        if raw.strip():
            arm, _, seed, _ = raw.split()
            have.setdefault(arm, set()).add(int(seed))
    missing = {
        arm: sorted(set(seeds) - have.get(arm, set()))
        for arm in gp.ARMS
        if set(seeds) - have.get(arm, set())
    }
    if missing:
        msg = "the log-count panel is incomplete: " + "; ".join(
            f"{arm}: {s}" for arm, s in missing.items()
        )
        raise ContrastError(msg)


def overlapping_bands(seeds: tuple[int, ...] = SEEDS) -> list[tuple[int, int]]:
    """Return the committed seed bands the panel's seeds touch; empty when they are disjoint."""
    return [(lo, hi) for lo, hi in COMMITTED_BANDS if any(lo <= s <= hi for s in seeds)]


# ── Scoring ──────────────────────────────────────────────────────────────────────────────────
def score(log_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    """Score the panel, gate it on its own floors, and read the verdict."""
    if overlapping_bands():
        msg = f"the panel's seeds overlap committed bands {overlapping_bands()}"
        raise ContrastError(msg)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(log_dirs, out_dir / "manifest-logcount.txt")
    require_complete(manifest)
    report = ops.score_level(manifest, HALF, LEVEL, out_dir / "tmp-logcount", cell=CELL)
    gaps = {m: ops.wiring_gap(report, m) for m in METRICS}
    for metric in METRICS:
        ops.apply_family_correction({LEVEL: {metric: {"interaction": gaps[metric]}}})
    gates = ops.learning_gates(manifest, HALF, SEEDS, LEVEL, floor_level=LEVEL)
    return {
        "cell": CELL,
        "seeds": list(SEEDS),
        "primary_metric": PRIMARY_METRIC,
        "beside_metric": BESIDE_METRIC,
        "gates": gates,
        "gaps": gaps,
        "censoring_rule_choice": {
            "note": "description: the reading is fixed on auc_success whatever this field says",
            **ops.choose_metric({LEVEL: ops.censoring_rates(report)}),
        },
        "collapse": (
            f"not read: no run here exceeds {EPISODES} episodes, the horizon below which the "
            "collapse statistic does not apply"
        ),
        "reading": read(gates, gaps[PRIMARY_METRIC]),
    }


def write_csv(result: dict[str, Any], path: Path) -> Path:
    """One row per panel seed: both wirings' plateaus and floors, and the gap on both metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["seed", "wt_plateau", "wt_floor", "rn_plateau", "rn_floor"]
    header += [f"gap_{m}" for m in METRICS]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for seed in result["seeds"]:
            row: list[Any] = [seed]
            for wiring in ("wt", "rn"):
                per_seed = result["gates"][wiring]["per_seed"].get(seed, {})
                row += [gop._fmt(per_seed.get("learn")), gop._fmt(per_seed.get("floor"))]
            row += [gop._fmt(result["gaps"][m]["per_seed"].get(seed)) for m in METRICS]
            writer.writerow(row)
    return path


# ── Before launch ────────────────────────────────────────────────────────────────────────────
def preflight(log_dirs: list[Path]) -> dict[str, Any]:
    """Evaluate the panel level's floor and saturation gates on the pilot's runs."""
    return gp.preflight(STEMS, log_dirs)


def table(seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """D24's matched statistics for both log-count wirings; the wild type's matrix takes no seed."""
    wt, rn = STEMS[LEVEL]["wt_learn"], STEMS[LEVEL]["rn_learn"]
    per_seed = {seed: gop.matched_statistics(gop._brain(rn, seed)) for seed in seeds}
    return {
        wt: {"seed_independent": True, "stats": gop.matched_statistics(gop._brain(wt, seeds[0]))},
        rn: {"seed_independent": False, "per_seed": per_seed, "summary": gop._summary(per_seed)},
    }


def plan() -> dict[str, Any]:
    """Return the panel's configs and seeds."""
    return {
        "configs": [f"configs/scenarios/thermal_foraging/{s}.yml" for s in STEMS[LEVEL].values()],
        "seeds": [SEEDS[0], SEEDS[-1]],
        "runs": len(STEMS[LEVEL]) * len(SEEDS),
        "episodes": EPISODES,
        "overlapping_bands": overlapping_bands(),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)
    pl = sub.add_parser("plan", help="the panel's configs and seeds")
    pf = sub.add_parser("preflight", help="the gates on the pilot's runs of the same configs")
    pf.add_argument("--logs", type=Path, action="append", required=True)
    tb = sub.add_parser("table", help="D24's matched statistics on the panel's draws")
    sc = sub.add_parser("score", help="read the panel")
    sc.add_argument("--logs", type=Path, action="append", required=True)
    sc.add_argument("--out-dir", type=Path, required=True)
    sc.add_argument("--csv", type=Path, help="write the per-seed CSV here")
    for parser in (pl, pf, tb, sc):
        parser.add_argument("--out", type=Path, help="write the JSON here")
    args = ap.parse_args(argv)
    ok = True
    if args.command == "plan":
        result = plan()
        ok = not result["overlapping_bands"]
    elif args.command == "preflight":
        result = preflight(args.logs)
        ok = bool(result["launch"])
    elif args.command == "table":
        result = table()
    else:
        result = score(args.logs, args.out_dir)
        if args.csv:
            write_csv(result, args.csv)
    payload = json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
