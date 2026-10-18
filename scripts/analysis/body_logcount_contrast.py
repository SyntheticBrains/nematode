#!/usr/bin/env python
"""The wiring contrast through the kinematic body on log-compressed gap counts, read before collapse.

The thermal cell through the kinematic body at target 20, on Emmons 2024 with ``log1p(count)`` gap
weights: the wild type against its gap-only null, 6,000 episodes, learning on seeds 4001-4128 and frozen
on 4001-4016 (288 runs).

**The window.** Small PPO learners on this cell can lose a policy late in training, at a stable rate set
by the training stream rather than the seed. The verdict and its gates are therefore read on episodes
1-3,000, which end before every collapse onset observed on this cell under the unmodified learner. The
rest of the horizon is reported beside, with the collapse statistic for each learning arm.

**The reading**: ``gap(log, body)``, wild type - null, each run's mean success over episodes 1-3,000,
paired by seed, with an 80% bootstrap interval and the two-sided Wilcoxon p as its q (a family of one),
classified by ``mc.classify`` at m = 0.0577.

**The gates**, on the window: each learning arm beats its wiring's frozen floor (the paired interval of
learning - frozen plateau over the 16 floor seeds lies above zero), and the two learning arms' mean
plateaus over all 128 seeds are not both at or above 90%. The plateau on the window is the final quarter
of episodes 1-3,000.

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

Usage::

    uv run python scripts/analysis/body_logcount_contrast.py plan
    uv run python scripts/analysis/body_logcount_contrast.py preflight --logs <pilot logs> \
        --out preflight.json
    uv run python scripts/analysis/body_logcount_contrast.py table --out table.json
    uv run python scripts/analysis/body_logcount_contrast.py score --logs <panel logs> \
        --out-dir <scratch> --out panel.json --csv per-seed.csv
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parent / "campaigns"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import collapse as col  # noqa: E402  # pyright: ignore[reportMissingImports]
import connectome_structure_efficiency as cse  # noqa: E402  # pyright: ignore[reportMissingImports]
import gap_operating_point as gop  # noqa: E402  # pyright: ignore[reportMissingImports]
import gate_preflight as gp  # noqa: E402  # pyright: ignore[reportMissingImports]
import generate_body_logcount_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]
import logcount_contrast as lc  # noqa: E402  # pyright: ignore[reportMissingImports]
import measured_prior_contrast as mc  # noqa: E402  # pyright: ignore[reportMissingImports]
import operating_point_surface as ops  # noqa: E402  # pyright: ignore[reportMissingImports]
import t7_continuous_ranking as t7  # noqa: E402  # pyright: ignore[reportMissingImports]
import wiring_premise as wp  # noqa: E402  # pyright: ignore[reportMissingImports]

LEVEL = "logcount_body"
SEEDS: tuple[int, ...] = tuple(range(4001, 4129))
FLOOR_SEEDS: tuple[int, ...] = SEEDS[:16]
PILOT_SEEDS: tuple[int, ...] = tuple(range(9895, 9899))
SMOKE_SEEDS: tuple[int, ...] = (9899, 9900)
WINDOW = 3000
HORIZON = 6000
MINIMUM = gop.CARRIED_MINIMUM
PLATEAU_FRACTION = t7._FINAL_WINDOW_FRAC
NEAR_BAR_POINTS = gp.DEFAULT_MARGIN_POINTS
# Seed bands already read on this cell or registered elsewhere in the phase that sit near this one.
COMMITTED_BANDS: tuple[tuple[int, int], ...] = (
    (2301, 2332),
    (3901, 3964),
    (9701, 9728),
    (9871, 9894),
)

STEMS: dict[str, dict[str, str]] = {LEVEL: {role: child for role, (child, _) in gen.ARMS.items()}}
LEARN: dict[str, str] = {"wt": STEMS[LEVEL]["wt_learn"], "rn": STEMS[LEVEL]["rn_learn"]}
FROZEN: dict[str, str] = {"wt": STEMS[LEVEL]["wt_frozen"], "rn": STEMS[LEVEL]["rn_frozen"]}


class ContrastError(ValueError):
    """The panel's runs cannot be read as registered."""


# ── Per-run quantities ───────────────────────────────────────────────────────────────────────
def auc(series: np.ndarray, start: int = 0, stop: int = WINDOW) -> float:
    """Return mean success over episodes ``start + 1`` to ``stop``."""
    return float(series[start:stop].mean())


def plateau(series: np.ndarray, stop: int = WINDOW) -> float:
    """Return the instrument's final-quarter plateau, in percent, of episodes 1 to ``stop``."""
    window = series[:stop]
    tail = max(1, int(len(window) * PLATEAU_FRACTION))
    return 100.0 * float(window[-tail:].mean())


# ── Manifest ─────────────────────────────────────────────────────────────────────────────────
def _seeds_for(role: str, seeds: tuple[int, ...], floor_seeds: tuple[int, ...]) -> tuple[int, ...]:
    return floor_seeds if role.endswith("_frozen") else seeds


def load_runs(
    log_dirs: list[Path],
    seeds: tuple[int, ...] = SEEDS,
    floor_seeds: tuple[int, ...] = FLOOR_SEEDS,
) -> dict[str, dict[int, np.ndarray]]:
    """Every registered run's success series, by role and seed; refuse a panel that is not whole.

    A run is read only if its ``.exit`` is 0, its log records the config its name claims, and it has
    every one of the horizon's episodes. A missing run is refused, never read as a failure.
    """
    found: dict[str, dict[int, Path]] = {role: {} for role in STEMS[LEVEL]}
    for role, level, seed, log in gp.evidence(STEMS, log_dirs):
        del level
        if seed not in _seeds_for(role, seeds, floor_seeds):
            continue
        status = log.with_suffix(".exit")
        if not status.is_file() or status.read_text().strip() != "0":
            continue
        expected = f"{log.stem.rpartition('-seed')[0]}.yml"
        ran = lc._ran_config(log)
        if ran != expected:
            msg = f"{log} exited 0 but records config {ran!r}, not {expected!r}"
            raise ContrastError(msg)
        if seed in found[role] and found[role][seed] != log:
            msg = f"{role} seed {seed} has two runs: {found[role][seed]} and {log}"
            raise ContrastError(msg)
        found[role][seed] = log
    missing = {
        role: sorted(set(_seeds_for(role, seeds, floor_seeds)) - set(found[role]))
        for role in found
        if set(_seeds_for(role, seeds, floor_seeds)) - set(found[role])
    }
    if missing:
        msg = "the body panel is incomplete: " + "; ".join(f"{r}: {s}" for r, s in missing.items())
        raise ContrastError(msg)
    runs: dict[str, dict[int, np.ndarray]] = {}
    for role, by_seed in found.items():
        runs[role] = {}
        for seed, log in sorted(by_seed.items()):
            series = col.success_series(log)
            if len(series) < HORIZON:
                msg = f"{log} has {len(series)} of {HORIZON} episodes"
                raise ContrastError(msg)
            runs[role][seed] = series[:HORIZON]
    return runs


def overlapping_bands(seeds: tuple[int, ...] = SEEDS) -> list[tuple[int, int]]:
    """Return the committed seed bands the panel's seeds touch; empty when they are disjoint."""
    return [(lo, hi) for lo, hi in COMMITTED_BANDS if any(lo <= s <= hi for s in seeds)]


# ── The reading ──────────────────────────────────────────────────────────────────────────────
def paired(values: dict[int, float]) -> dict[str, Any]:
    """Return the paired test on per-seed values, with the two-sided q the classifier reads."""
    seeds = sorted(values)
    test = wp.paired_seed_wilcoxon_bootstrap([values[s] for s in seeds])
    q = ops.two_sided(test["wilcoxon_p"])
    return {
        "n": len(seeds),
        "mean": float(np.mean([values[s] for s in seeds])),
        "ci_lo": test["ci_lo"],
        "ci_hi": test["ci_hi"],
        "q": q,
        "per_seed": values,
    }


def gap(
    runs: dict[str, dict[int, np.ndarray]],
    seeds: tuple[int, ...],
    start: int = 0,
    stop: int = WINDOW,
) -> dict[str, Any]:
    """Return wild type - null in ``auc_success`` over one window, paired by seed."""
    return paired(
        {
            s: auc(runs["wt_learn"][s], start, stop) - auc(runs["rn_learn"][s], start, stop)
            for s in seeds
        },
    )


def episodes_gap(runs: dict[str, dict[int, np.ndarray]], seeds: tuple[int, ...]) -> dict[str, Any]:
    """Return null - wild type in episodes to a 30% rolling success rate, censored at the horizon.

    Oriented so that a positive value means the wild type got there sooner.
    """

    def first(series: np.ndarray) -> float:
        return cse._eps_to_threshold(list(series), cse._SUCCESS_THRESHOLD, HORIZON)

    return paired({s: first(runs["rn_learn"][s]) - first(runs["wt_learn"][s]) for s in seeds})


def gates(
    runs: dict[str, dict[int, np.ndarray]],
    seeds: tuple[int, ...],
    floor_seeds: tuple[int, ...],
    stop: int = WINDOW,
) -> dict[str, Any]:
    """Return the floor and saturation gates on one window."""
    out: dict[str, Any] = {"window": [1, stop]}
    beats: list[bool] = []
    means: list[float] = []
    for wiring in ("wt", "rn"):
        learn = {s: plateau(runs[f"{wiring}_learn"][s], stop) for s in seeds}
        floor = {s: plateau(runs[f"{wiring}_frozen"][s], stop) for s in floor_seeds}
        vs_floor = paired({s: learn[s] - floor[s] for s in floor_seeds})
        means.append(float(np.mean(list(learn.values()))))
        beats.append(float(vs_floor["ci_lo"]) > 0.0)
        out[wiring] = {
            "plateau_mean": means[-1],
            "floor_mean": float(np.mean(list(floor.values()))),
            "vs_floor": vs_floor,
            "never_learned": sorted(s for s, v in learn.items() if v == 0.0),
            "per_seed": {s: {"learn": learn[s], "floor": floor.get(s)} for s in seeds},
        }
    out["beats_floor"] = all(beats)
    out["saturated"] = all(m >= wp.SATURATION_SUCCESS for m in means)
    out["top_plateau"] = max(means)
    out["readable"] = out["beats_floor"] and not out["saturated"]
    return out


def read(gate: dict[str, Any], primary: dict[str, Any], minimum: float = MINIMUM) -> dict[str, Any]:
    """Gates, then the state of the windowed gap, then the verdict and its joint outcome."""
    out: dict[str, Any] = {"minimum": minimum, "readable": bool(gate["readable"])}
    if not out["readable"]:
        out["verdict"] = "unreadable"
        out["why"] = "a learning arm fails its floor, or both saturate, on the window"
    else:
        mean = float(primary["mean"])
        state = mc.classify(mean, primary["ci_lo"], primary["ci_hi"], primary["q"], minimum)
        out["state"] = state
        out["verdict"] = lc.VERDICTS[state]
        if state == "below":
            out["direction"] = "wild_type" if mean > 0.0 else "null"
    out["with_the_point_worm"] = (
        "the STOP branch is taken" if out["verdict"] == "no_wiring_effect" else "a dated decision"
    )
    return out


def score(log_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    """Read the panel: the windowed verdict, then everything beside it."""
    if overlapping_bands():
        msg = f"the panel's seeds overlap committed bands {overlapping_bands()}"
        raise ContrastError(msg)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(log_dirs)
    gate = gates(runs, SEEDS, FLOOR_SEEDS)
    primary = gap(runs, SEEDS)
    return {
        "cell": "thermal_body_t20",
        "seeds": list(SEEDS),
        "floor_seeds": [FLOOR_SEEDS[0], FLOOR_SEEDS[-1]],
        "window": [1, WINDOW],
        "horizon": HORIZON,
        "primary": primary,
        "gates": gate,
        "reading": read(gate, primary),
        "beside": {
            "gap_to_horizon": gap(runs, SEEDS, 0, HORIZON),
            "gap_late": gap(runs, SEEDS, WINDOW, HORIZON),
            "episodes_to_30pct_success": episodes_gap(runs, SEEDS),
            "gates_at_horizon": gates(runs, SEEDS, FLOOR_SEEDS, HORIZON),
            "collapse": {
                wiring: col.panel(
                    {str(s): runs[f"{wiring}_learn"][s] for s in SEEDS},
                    early_horizon=WINDOW,
                )
                for wiring in ("wt", "rn")
            },
        },
    }


def write_csv(result: dict[str, Any], path: Path) -> Path:
    """One row per seed: both wirings' plateaus and floors on the window, and the gap per window."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["seed", "wt_plateau", "wt_floor", "rn_plateau", "rn_floor", "gap_window"]
    header += ["gap_horizon", "gap_late", "wt_collapsed", "rn_collapsed"]
    beside = result["beside"]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for seed in result["seeds"]:
            row: list[Any] = [seed]
            for wiring in ("wt", "rn"):
                per_seed = result["gates"][wiring]["per_seed"][seed]
                row += [gop._fmt(per_seed["learn"]), gop._fmt(per_seed["floor"])]
            row += [gop._fmt(result["primary"]["per_seed"][seed])]
            row += [gop._fmt(beside[k]["per_seed"][seed]) for k in ("gap_to_horizon", "gap_late")]
            row += [beside["collapse"][w]["runs"][str(seed)]["collapsed"] for w in ("wt", "rn")]
            writer.writerow(row)
    return path


# ── Before launch ────────────────────────────────────────────────────────────────────────────
def preflight(log_dirs: list[Path], seeds: tuple[int, ...] = PILOT_SEEDS) -> dict[str, Any]:
    """Evaluate the gates on the pilot's runs, on the verdict's window; its gap is never read."""
    try:
        runs = load_runs(log_dirs, seeds, seeds)
    except ContrastError as exc:
        return {"status": "no_evidence", "why": str(exc), "launch": False}
    gate = gates(runs, seeds, seeds)
    if not gate["beats_floor"]:
        status = "fails_floor"
    elif gate["saturated"]:
        status = "saturated"
    elif gate["top_plateau"] >= wp.SATURATION_SUCCESS - NEAR_BAR_POINTS:
        status = "near_bar"
    else:
        status = "readable"
    return {
        "seeds": list(seeds),
        "window": [1, WINDOW],
        "status": status,
        "launch": status == "readable",
        "saturation_bar": wp.SATURATION_SUCCESS,
        "margin_points": NEAR_BAR_POINTS,
        "gates": {k: v for k, v in gate.items() if k not in ("wt", "rn")},
        "plateaus": {
            w: {
                "window": gate[w]["plateau_mean"],
                "floor": gate[w]["floor_mean"],
                "horizon": float(np.mean([plateau(runs[f"{w}_learn"][s], HORIZON) for s in seeds])),
            }
            for w in ("wt", "rn")
        },
        "never_read": "the gap between the wild type and the null: on four seeds it licenses nothing",
    }


def table(seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """D24's matched statistics for both wirings; the wild type's matrix takes no seed."""
    wt, rn = LEARN["wt"], LEARN["rn"]
    per_seed = {seed: gop.matched_statistics(gop._brain(rn, seed)) for seed in seeds}
    return {
        wt: {"seed_independent": True, "stats": gop.matched_statistics(gop._brain(wt, seeds[0]))},
        rn: {"seed_independent": False, "per_seed": per_seed, "summary": gop._summary(per_seed)},
    }


def plan() -> dict[str, Any]:
    """Return the panel's configs and seeds."""
    return {
        "configs": [f"configs/scenarios/thermal_foraging/{s}.yml" for s in STEMS[LEVEL].values()],
        "learning_seeds": [SEEDS[0], SEEDS[-1]],
        "floor_seeds": [FLOOR_SEEDS[0], FLOOR_SEEDS[-1]],
        "runs": 2 * len(SEEDS) + 2 * len(FLOOR_SEEDS),
        "episodes": HORIZON,
        "window": [1, WINDOW],
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
    pf = sub.add_parser("preflight", help="the gates on the pilot, on the verdict's window")
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
