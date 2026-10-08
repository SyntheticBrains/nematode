#!/usr/bin/env python
"""C.1d: the MLP positive control through the kinematic body, its calibration and its kinematics.

Four readings, each fixed before its runs:

* ``pilot``: the steering calibration. MLP-PPO at four steering gains, learning and frozen, seeds
  1501-1504. **The rule**: the gain with the highest mean plateau success; gains within 5 points of
  the best are ties, broken toward the smaller gain. If no gain's learning arm beats its frozen floor
  at any seed, the pilot is the diagnosis and no control runs.
* ``control``: the positive control at the chosen gain, seeds 1505-1512. **Passes** if the plateau
  beats the frozen floor paired by seed with the 80% bootstrap interval above zero, and every seed's
  plateau is at least 30%. A passing floor with failing competence sends it to the fallback.
* ``fallback``: the learning arm at 500, 700 and 1,000 steps on seeds 1513-1516. The shortest length
  at which every seed reaches competence is chosen; the control re-runs there on seeds 1517-1524.
* ``kinematics``: the instruments on each run's final weights, frozen, over 10 held-out episodes.
  For the control, the same weights are re-read at 40 sub-steps, and each instrument's mean over the
  runs must agree within 10% (reversal fraction: 10% or 0.01 absolute, whichever is larger), and
  each band is read at both: ``in``, ``out``, or ``edge`` where the two readings differ. For
  the pilot, each seed's untrained and trained reversal fractions are reported beside the rule.

Usage::

    uv run python scripts/analysis/body_control.py pilot --logs campaigns/c1d-pilot/logs \
        --out pilot.json
    uv run python scripts/analysis/body_control.py control --logs campaigns/c1d-control/logs \
        --out control.json
    uv run python scripts/analysis/body_control.py fallback --logs campaigns/c1d-fallback/logs
    uv run python scripts/analysis/body_control.py kinematics --stage control \
        --logs campaigns/c1d-control/logs --out kinematics.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_CAMPAIGNS = _HERE.parent / "campaigns"
if str(_CAMPAIGNS) not in sys.path:
    sys.path.insert(0, str(_CAMPAIGNS))

import body_kinematics_eval as harness  # noqa: E402  # pyright: ignore[reportMissingImports]
import generate_body_control_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]
import wiring_premise as wp  # noqa: E402  # pyright: ignore[reportMissingImports]
from quantumnematode.validation import body_kinematics as bk  # noqa: E402

if TYPE_CHECKING:
    from quantumnematode.validation.body_kinematics import Kinematics

REPO = _HERE.parents[1]
EXPERIMENTS = REPO / "experiments"
PILOT_SEEDS: tuple[int, ...] = tuple(range(1501, 1505))
CONTROL_SEEDS: tuple[int, ...] = tuple(range(1505, 1513))
FALLBACK_PILOT_SEEDS: tuple[int, ...] = tuple(range(1513, 1517))
FALLBACK_CONTROL_SEEDS: tuple[int, ...] = tuple(range(1517, 1525))
TIE_POINTS = 5.0
COMPETENCE = 30.0
HALF_STEP_TOLERANCE = 0.10
REVERSAL_ABSOLUTE_TOLERANCE = 0.01
EPISODES = 10
HALF_STEP_SUBSTEPS = 40
_EXPERIMENT_LINE = re.compile(r"Experiment ID:\s+(\S+)")
_INSTRUMENTS = ("frequency_hz", "wavelength_bl", "speed_bl_per_s", "reversal_fraction")


def plateaus(log_dirs: list[Path], stem: str, seeds: tuple[int, ...]) -> dict[int, float]:
    """Return each seed's plateau success, in percent, for one config's runs."""
    out: dict[int, float] = {}
    for log_dir in log_dirs:
        for seed in seeds:
            log = log_dir / f"{stem}-seed{seed}.log"
            tail = wp.plateau_tail(log) if log.is_file() else None
            if tail is not None:
                out[seed] = float(tail[0])
    return out


def _mean(values: dict[int, float]) -> float | None:
    return statistics.fmean(values.values()) if values else None


# ── The calibration ─────────────────────────────────────────────────────────────────────────────


def select_gain(mean_plateau: dict[float, float]) -> float:
    """Apply the rule: the best mean plateau, ties within ``TIE_POINTS`` to the smaller gain."""
    best = max(mean_plateau.values())
    return min(g for g, m in mean_plateau.items() if m >= best - TIE_POINTS)


def calibration(
    learn: dict[float, dict[int, float]],
    frozen: dict[float, dict[int, float]],
    seeds: tuple[int, ...] = PILOT_SEEDS,
) -> dict[str, Any]:
    """Read the pilot from each gain's per-seed learning plateaus and frozen floors."""
    gains: dict[str, Any] = {}
    complete = True
    learned_anywhere = False
    for gain in gen.GAINS:
        lp, fp = learn.get(gain, {}), frozen.get(gain, {})
        paired = sorted(set(lp) & set(fp))
        complete &= len(paired) == len(seeds)
        beats = [s for s in paired if lp[s] > fp[s]]
        learned_anywhere |= bool(beats)
        gains[f"{gain:g}"] = {
            "plateau": lp,
            "floor": fp,
            "plateau_mean": _mean(lp),
            "floor_mean": _mean(fp),
            "seeds_beating_floor": beats,
        }
    result: dict[str, Any] = {"seeds": list(seeds), "gains": gains, "complete": complete}
    if not complete:
        result["verdict"] = "incomplete"
    elif not learned_anywhere:
        result["verdict"] = "diagnosis"
    else:
        means = {g: float(gains[f"{g:g}"]["plateau_mean"]) for g in gen.GAINS}
        chosen = select_gain(means)
        index = gen.GAINS.index(chosen)
        neighbours = [g for g in gen.GAINS[max(0, index - 1) : index + 2] if g != chosen]
        result |= {
            "verdict": "chosen",
            "gain": chosen,
            "sensitivity": {f"{g:g}": means[g] - means[chosen] for g in neighbours},
        }
    return result


def pilot(log_dirs: list[Path]) -> dict[str, Any]:
    """Read the calibration pilot from campaign logs."""
    learn = {g: plateaus(log_dirs, gen.stem("learn", gain=g), PILOT_SEEDS) for g in gen.GAINS}
    frozen = {g: plateaus(log_dirs, gen.stem("frozen", gain=g), PILOT_SEEDS) for g in gen.GAINS}
    return calibration(learn, frozen)


# ── The control and its fallback ────────────────────────────────────────────────────────────────


def control_gate(
    learn: dict[int, float],
    frozen: dict[int, float],
    seeds: tuple[int, ...] = CONTROL_SEEDS,
) -> dict[str, Any]:
    """Read the floor gate and competence: ``passes``, ``fallback``, ``fails`` or ``incomplete``."""
    paired = sorted(set(learn) & set(frozen))
    test = wp.paired_seed_wilcoxon_bootstrap([learn[s] - frozen[s] for s in paired])
    competent = [s for s in paired if learn[s] >= COMPETENCE]
    floor_passes = float(test["ci_lo"]) > 0.0
    if len(paired) != len(seeds):
        verdict = "incomplete"
    elif not floor_passes:
        verdict = "fails"
    elif len(competent) < len(paired):
        verdict = "fallback"
    else:
        verdict = "passes"
    return {
        "n_seeds": len(paired),
        "plateau": learn,
        "floor": frozen,
        "plateau_mean": _mean(learn),
        "floor_mean": _mean(frozen),
        "test": test,
        "floor_gate": floor_passes,
        "competent_seeds": competent,
        "verdict": verdict,
    }


def control(log_dirs: list[Path], *, steps: int | None = None) -> dict[str, Any]:
    """Read the positive control, at the registered episode or at a fallback length."""
    seeds = CONTROL_SEEDS if steps is None else FALLBACK_CONTROL_SEEDS
    learn = plateaus(log_dirs, gen.stem("learn", steps=steps), seeds)
    frozen = plateaus(log_dirs, gen.stem("frozen", steps=steps), seeds)
    return {"seeds": list(seeds), "max_steps": steps, **control_gate(learn, frozen, seeds)}


def choose_length(by_length: dict[int, dict[int, float]]) -> int | None:
    """Return the shortest length at which every fallback-pilot seed reaches competence."""
    for steps in gen.FALLBACK_STEPS:
        reached = by_length.get(steps, {})
        complete = set(reached) == set(FALLBACK_PILOT_SEEDS)
        if complete and all(v >= COMPETENCE for v in reached.values()):
            return steps
    return None


def fallback(log_dirs: list[Path]) -> dict[str, Any]:
    """Read the competence fallback's gate-only pilot."""
    by_length = {
        steps: plateaus(log_dirs, gen.stem("learn", steps=steps), FALLBACK_PILOT_SEEDS)
        for steps in gen.FALLBACK_STEPS
    }
    return {
        "seeds": list(FALLBACK_PILOT_SEEDS),
        "plateau": {str(k): v for k, v in by_length.items()},
        "chosen_steps": choose_length(by_length),
    }


# ── The kinematics ──────────────────────────────────────────────────────────────────────────────


def final_weights(log: Path, experiments: Path = EXPERIMENTS) -> Path | None:
    """Return a run's final weights, through its tracked-experiment record, if both exist."""
    match = _EXPERIMENT_LINE.search(log.read_text())
    if match is None:
        return None
    record = experiments / match.group(1) / f"{match.group(1)}.json"
    if not record.is_file():
        return None
    exports = json.loads(record.read_text()).get("exports_path")
    weights = REPO / exports / "weights" / "final.pt" if exports else None
    return weights if weights is not None and weights.is_file() else None


def pooled(readings: list[Kinematics]) -> dict[str, float | None]:
    """Return each instrument's mean over runs, over the runs where it could be read."""
    out: dict[str, float | None] = {}
    for name in _INSTRUMENTS:
        values = [v for k in readings if (v := getattr(k, name)) is not None]
        out[name] = statistics.fmean(values) if values else None
    return out


def half_step_agreement(
    base: dict[str, float | None],
    doubled: dict[str, float | None],
) -> dict[str, Any]:
    """Compare pooled readings at 20 and 40 sub-steps against the registered tolerance."""
    per: dict[str, Any] = {}
    for name in _INSTRUMENTS:
        a, b = base[name], doubled[name]
        if a is None or b is None:
            per[name] = {"base": a, "doubled": b, "agrees": None}
            continue
        allowed = HALF_STEP_TOLERANCE * abs(a)
        if name == "reversal_fraction":
            allowed = max(allowed, REVERSAL_ABSOLUTE_TOLERANCE)
        per[name] = {"base": a, "doubled": b, "allowed": allowed, "agrees": abs(b - a) <= allowed}
    readable = [v["agrees"] for v in per.values() if v["agrees"] is not None]
    return {"instruments": per, "agrees": bool(readable) and all(readable)}


_BANDS = {
    "frequency_hz": bk.FREQUENCY_BAND_HZ,
    "wavelength_bl": bk.WAVELENGTH_BAND_BL,
    "speed_bl_per_s": bk.SPEED_BAND_BL_PER_S,
}


def band_readings(
    base: dict[str, float | None],
    doubled: dict[str, float | None],
) -> dict[str, str | None]:
    """Read each band at 20 and 40 sub-steps: ``in``, ``out``, or ``edge`` where the two differ."""
    out: dict[str, str | None] = {}
    for name, (lo, hi) in _BANDS.items():
        a, b = base[name], doubled[name]
        if a is None or b is None:
            out[name] = None
            continue
        inside = (lo <= a <= hi, lo <= b <= hi)
        out[name] = "edge" if inside[0] != inside[1] else ("in" if inside[0] else "out")
    return out


def _evaluate(config: Path, seed: int, weights: Path | None, substeps: int | None) -> Kinematics:
    return harness.evaluate(config, seed, weights, episodes=EPISODES, substeps=substeps)


def kinematics(log_dirs: list[Path], stage: str, *, steps: int | None = None) -> dict[str, Any]:
    """Read the instruments on a stage's learning runs."""
    if stage == "pilot":
        runs = [(g, gen.stem("learn", gain=g)) for g in gen.GAINS]
        seeds = PILOT_SEEDS
    else:
        runs = [(None, gen.stem("learn", steps=steps))]
        seeds = CONTROL_SEEDS if steps is None else FALLBACK_CONTROL_SEEDS
    result: dict[str, Any] = {"stage": stage, "seeds": list(seeds), "runs": {}}
    for gain, run_stem in runs:
        config = gen.FORAGING / f"{run_stem}.yml"
        base: list[Kinematics] = []
        doubled: list[Kinematics] = []
        per_seed: dict[int, Any] = {}
        for seed in seeds:
            logs = [d / f"{run_stem}-seed{seed}.log" for d in log_dirs]
            log = next((p for p in logs if p.is_file()), None)
            weights = final_weights(log) if log is not None else None
            if weights is None:
                per_seed[seed] = None
                continue
            trained = _evaluate(config, seed, weights, None)
            base.append(trained)
            entry: dict[str, Any] = {"trained": asdict(trained)}
            if stage == "pilot":
                entry["untrained_reversal_fraction"] = _evaluate(
                    config,
                    seed,
                    None,
                    None,
                ).reversal_fraction
            else:
                doubled.append(_evaluate(config, seed, weights, HALF_STEP_SUBSTEPS))
                entry["half_step"] = asdict(doubled[-1])
            per_seed[seed] = entry
        summary: dict[str, Any] = {"per_seed": per_seed, "pooled": pooled(base)}
        if doubled:
            summary["half_step"] = half_step_agreement(pooled(base), pooled(doubled))
            summary["bands"] = band_readings(pooled(base), pooled(doubled))
        result["runs"][run_stem if gain is None else f"{gain:g}"] = summary
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("reading", choices=("pilot", "control", "fallback", "kinematics"))
    ap.add_argument("--logs", type=Path, action="append", required=True, help="a run-log dir")
    ap.add_argument("--stage", choices=("pilot", "control"), default="control")
    ap.add_argument("--steps", type=int, default=None, help="a fallback episode length")
    ap.add_argument("--out", type=Path, help="write the JSON here")
    args = ap.parse_args(argv)
    if args.reading == "pilot":
        result = pilot(args.logs)
    elif args.reading == "control":
        result = control(args.logs, steps=args.steps)
    elif args.reading == "fallback":
        result = fallback(args.logs)
    else:
        result = kinematics(args.logs, args.stage, steps=args.steps)
    payload = json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
