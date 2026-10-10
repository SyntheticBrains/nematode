#!/usr/bin/env python
r"""C.3: body-level validation of C.1e's trained runs, graded against bands fixed in advance.

Every run's final weights are evaluated frozen for 30 held-out episodes with posture capture (every
sub-step, 4 Hz) and behaviour capture (every step), then at 40 sub-steps for the half-step check.

**Body checks**, which the body's generator largely sets: frequency, wavelength, the variance the
first four eigenworms capture, and the half-step agreement. **Behaviour readings**, which are
emergent: speed, the share of episodes with a 20-second forward bout, and Logbook 035's klinokinesis
and weathervane curves. Omega turns, with the share whose posture reaches an omega's, and amplitude
are reported, not graded.

The arms are C.1e's wild type, chemical-only null and MLP (seeds 1801-1864), their floors (the
connectome's frozen runs; the MLP's untrained policy, since C.1e trained no frozen MLP), and a
derivative-sensing MLP control with its floor (seeds 1801-1816), which says whether the weathervane
needs the synthetic head-sweep. Each arm's bias statistics are paired with its floor's by seed.

Usage::

    uv run python scripts/analysis/body_validation.py \\
        --logs campaigns/c1e-panel/logs --logs campaigns/c3-control/logs \\
        --out-dir build/c3 --out validation.json --csv per-run.csv [--workers 16]

A cost pilot evaluates runs outside the panel's seeds: ``--logs campaigns/c1e-pilot/logs --seeds
1701 1702 --arms wild_type chemical_only_null wild_type_frozen chemical_only_null_frozen``.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parent / "campaigns"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import behavioural_chemotaxis_validation as bcv  # noqa: E402  # pyright: ignore[reportMissingImports]
import body_control as bc  # noqa: E402  # pyright: ignore[reportMissingImports]
import body_kinematics_eval as harness  # noqa: E402  # pyright: ignore[reportMissingImports]
import generate_body_validation_configs as control  # noqa: E402  # pyright: ignore[reportMissingImports]
import generate_body_wiring_configs as wiring  # noqa: E402  # pyright: ignore[reportMissingImports]
import wiring_premise as wp  # noqa: E402  # pyright: ignore[reportMissingImports]
from quantumnematode.validation import body_kinematics as bk  # noqa: E402
from quantumnematode.validation import posture  # noqa: E402
from quantumnematode.validation.datasets import load_bias_signatures  # noqa: E402

if TYPE_CHECKING:
    from quantumnematode.report.dtypes import BehaviourStep

EPISODES = 30
HALF_STEP_SUBSTEPS = 40
WALL_MARGIN_MM = 1.0
THETA_SHARP = 0.45
PANEL_SEEDS: tuple[int, ...] = tuple(range(1801, 1865))
CONTROL_SEEDS: tuple[int, ...] = tuple(range(1801, 1817))
AMPLITUDE_SAMPLE_EVERY = 20
OMEGA_POSTURE_PERCENTILE = 99.0

# Arm -> (config stem, seeds, graded). Graded arms get the bands and the half-step check.
ARMS: dict[str, tuple[str, tuple[int, ...], bool]] = {
    "wild_type": (wiring.stem("wt", "ppo"), PANEL_SEEDS, True),
    "chemical_only_null": (wiring.stem("chemnull", "ppo"), PANEL_SEEDS, True),
    "mlp": (wiring.MLP_STEM, PANEL_SEEDS, True),
    "wild_type_frozen": (wiring.stem("wt", "frozen"), PANEL_SEEDS, False),
    "chemical_only_null_frozen": (wiring.stem("chemnull", "frozen"), PANEL_SEEDS, False),
    "mlp_untrained": (wiring.MLP_STEM, PANEL_SEEDS, False),
    "mlp_derivative": (control.stem("learn"), CONTROL_SEEDS, False),
    "mlp_derivative_frozen": (control.stem("frozen"), CONTROL_SEEDS, False),
}

# Arms evaluated with each seed's untrained policy instead of a run's final weights: C.1e trained no
# frozen MLP, so the MLP's floor is its seeds' initial weights, as a frozen run's would be.
UNTRAINED: frozenset[str] = frozenset({"mlp_untrained"})

# Each arm whose bias curves are attributed to learning, and the floor it is paired with by seed.
FLOORS: dict[str, str] = {
    "wild_type": "wild_type_frozen",
    "chemical_only_null": "chemical_only_null_frozen",
    "mlp": "mlp_untrained",
    "mlp_derivative": "mlp_derivative_frozen",
}
BIAS_STATISTICS = ("klinokinesis", "klinokinesis_magnitude", "klinotaxis", "klinotaxis_all")

# The gate preflight's table. The control is the one trained arm; it has no contrast, so it fills
# both of the preflight's pairs and the preflight reads its floor and saturation gates.
STEMS: dict[str, dict[str, str]] = {
    "control": {
        "wt_learn": control.stem("learn"),
        "wt_frozen": control.stem("frozen"),
        "rn_learn": control.stem("learn"),
        "rn_frozen": control.stem("frozen"),
    },
}

# (pass band, partial band), each (low, high); None for an open end.
BANDS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "frequency_hz": ((0.20, 0.45), (0.10, 0.60)),
    "wavelength_bl": ((0.50, 0.80), (0.40, 1.00)),
    "speed_bl_per_s": ((0.12, 0.30), (0.06, 0.50)),
    "eigenworm_variance": ((0.85, math.inf), (0.70, math.inf)),
    "forward_bout_share": ((0.80, math.inf), (0.50, math.inf)),
}


@functools.cache
def omega_posture_threshold() -> float:
    """Return the real postures' 99th percentile of the third eigenworm's magnitude.

    A real omega turn bends the body deeply, which loads the third eigenworm (Stephens et al. 2008);
    a turn whose posture never passes the real postures' tail turned by steering, not by an omega
    posture.
    """
    real = posture.load_real_postures()
    third = np.abs(real @ posture.load_eigenworms()[:, 2])
    return float(np.percentile(third, OMEGA_POSTURE_PERCENTILE))


def grade(name: str, value: float | None) -> str | None:
    """Grade a reading against its registered bands: ``pass``, ``partial`` or ``fail``."""
    if value is None:
        return None
    (pass_lo, pass_hi), (part_lo, part_hi) = BANDS[name]
    if pass_lo <= value <= pass_hi:
        return "pass"
    if part_lo <= value <= part_hi:
        return "partial"
    return "fail"


def _log_for(log_dirs: list[Path], stem: str, seed: int) -> Path | None:
    logs = (d / f"{stem}-seed{seed}.log" for d in log_dirs)
    return next((log for log in logs if log.is_file()), None)


def evaluate_run(job: tuple[str, int, list[str], str]) -> dict[str, Any]:
    """Evaluate one run: kinematics, posture readings, omega turns, bouts; write its capture."""
    arm, seed, log_dirs, out_dir = job
    stem, _seeds, graded = ARMS[arm]
    weights: Path | None = None
    if arm not in UNTRAINED:
        log = _log_for([Path(d) for d in log_dirs], stem, seed)
        weights = bc.final_weights(log) if log is not None else None
        if weights is None:
            return {"arm": arm, "seed": seed, "missing": True}
    config = wiring.FORAGING / f"{stem}.yml"
    capture = harness.run_capture(config, seed, weights, episodes=EPISODES, capture_behaviour=True)
    step_seconds = capture.body.step_seconds
    basis = posture.load_eigenworms()
    per_episode = [
        posture.body_tangent_angles(
            np.array([s[1] for step in ep for s in step["substeps"]]),
        )
        for ep in capture.episodes
    ]
    postures = np.vstack(per_episode)
    projections = postures @ basis[:, :4]
    amplitude = posture.mode_amplitude(postures, basis)
    turns: list[float] = []
    turn_a3: list[float] = []
    for ep, angles in zip(capture.episodes, per_episode, strict=True):
        third = np.abs(angles @ basis[:, 2])
        for change, start, end in bk.omega_turn_swings(
            ep,
            world_size_mm=capture.world_size_mm,
            wall_margin_mm=WALL_MARGIN_MM,
        ):
            turns.append(change)
            turn_a3.append(float(third[start : end + 1].max()))
    capture_path = write_capture(
        Path(out_dir) / "captures" / f"{arm}-seed{seed}.json",
        seed,
        capture.behaviour,
    )
    result: dict[str, Any] = {
        "arm": arm,
        "seed": seed,
        "world_size_mm": capture.world_size_mm,
        "kinematics": asdict(_measure(capture)),
        "eigenworm_captured_ss": float((projections**2).sum()),
        "eigenworm_total_ss": float((postures**2).sum()),
        "amplitude_sample": amplitude[::AMPLITUDE_SAMPLE_EVERY].tolist(),
        "omega_turns": turns,
        "omega_turn_a3": turn_a3,
        "worm_minutes": sum(len(ep) for ep in capture.episodes) * step_seconds / 60.0,
        "forward_bout_share": bk.forward_bout_share(capture.episodes, step_seconds=step_seconds),
        "capture": str(capture_path),
    }
    if graded:
        doubled = harness.run_capture(
            config,
            seed,
            weights,
            episodes=EPISODES,
            substeps=HALF_STEP_SUBSTEPS,
        )
        result["half_step"] = asdict(_measure(doubled))
    return result


def write_capture(path: Path, seed: int, behaviour: list[list[BehaviourStep]]) -> Path:
    """Write a run's behaviour as the simulation's capture file, one series per episode."""
    runs = [
        {"run": i, "seed": seed, "steps": [asdict(step) for step in episode]}
        for i, episode in enumerate(behaviour)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"runs": runs}))
    return path


def _measure(capture: harness.Capture) -> bk.Kinematics:
    return bk.measure(
        capture.episodes,
        world_size_mm=capture.world_size_mm,
        body_length_mm=capture.body.body_length_mm,
        step_seconds=capture.body.step_seconds,
        reversal_threshold=capture.body.reversal_threshold,
        wall_margin_mm=WALL_MARGIN_MM,
        min_wave_amplitude=capture.body.min_wave_amplitude,
    )


def bias_curves(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Read Logbook 035's four bias statistics over an arm's captures, with the wall margin."""
    seeds: dict[int, list[list[Any]]] = {}
    for run in runs:
        data = json.loads(Path(run["capture"]).read_text())
        series = [bcv._steps_from_dicts(r["steps"]) for r in data["runs"] if r["steps"]]
        if series:
            seeds[run["seed"]] = series
    if not seeds:
        return None
    seeds, wall = bcv.exclude_walls(seeds, runs[0]["world_size_mm"], WALL_MARGIN_MM)
    summary = bcv.analyse(seeds, THETA_SHARP)
    summary["wall_exclusion"] = wall
    return summary


def summarise_arm(runs: list[dict[str, Any]], *, graded: bool) -> dict[str, Any]:
    """Pool an arm's runs into its readings and, for a graded arm, its grades."""
    present = [r for r in runs if not r.get("missing")]
    out: dict[str, Any] = {
        "n_runs": len(present),
        "missing_seeds": [r["seed"] for r in runs if r.get("missing")],
    }
    if not present:
        return out
    pooled = bc.pooled([bk.Kinematics(**r["kinematics"]) for r in present])
    captured = sum(r["eigenworm_captured_ss"] for r in present)
    total = sum(r["eigenworm_total_ss"] for r in present)
    bouts = [r["forward_bout_share"] for r in present if r["forward_bout_share"] is not None]
    turns = [t for r in present for t in r["omega_turns"]]
    turn_a3 = [a for r in present for a in r.get("omega_turn_a3", [])]
    minutes = sum(r["worm_minutes"] for r in present)
    amplitude = np.array([a for r in present for a in r["amplitude_sample"]])
    readings = {
        **pooled,
        "eigenworm_variance": captured / total if total else None,
        "forward_bout_share": statistics.fmean(bouts) if bouts else None,
    }
    out |= {
        "readings": readings,
        "omega_turns": {
            "count": len(turns),
            "per_worm_minute": len(turns) / minutes if minutes else None,
            "median_heading_change_deg": (
                float(np.degrees(np.median(np.abs(turns)))) if turns else None
            ),
            "omega_posture_share": (
                float(np.mean(np.array(turn_a3) > omega_posture_threshold())) if turn_a3 else None
            ),
            "omega_posture_threshold_a3": omega_posture_threshold(),
        },
        "amplitude": _percentiles(amplitude),
        "bias_curves": bias_curves(present),
    }
    if graded:
        doubled = bc.pooled([bk.Kinematics(**r["half_step"]) for r in present if "half_step" in r])
        out["grades"] = {name: grade(name, readings.get(name)) for name in BANDS}
        out["half_step"] = bc.half_step_agreement(pooled, doubled)
        out["edge"] = edges(pooled, doubled)
    return out


def edges(base: dict[str, float | None], doubled: dict[str, float | None]) -> dict[str, bool]:
    """Flag a banded reading whose grade differs between 20 and 40 sub-steps: it sits on an edge."""
    return {
        name: grade(name, base.get(name)) != grade(name, doubled.get(name))
        for name in ("frequency_hz", "wavelength_bl", "speed_bl_per_s")
    }


def _percentiles(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0:
        return None
    p5, p50, p95 = np.percentile(values, [5, 50, 95])
    return {"p5": float(p5), "median": float(p50), "p95": float(p95)}


def floor_comparison(arms: dict[str, Any]) -> dict[str, Any]:
    """Pair each arm's bias statistics with its floor's by seed: what learning added.

    Each difference is oriented so that positive is the reference's direction; learning added a
    bias where its 80% interval lies above zero. The floor's own verdicts are reported beside, so a
    floor that leans is seen, without voiding the arm's reading.
    """
    signs = {key: ref.sign for key, ref in load_bias_signatures().items()}
    out: dict[str, Any] = {}
    for arm, floor in FLOORS.items():
        arm_curves = arms.get(arm, {}).get("bias_curves")
        floor_curves = arms.get(floor, {}).get("bias_curves")
        if arm_curves is None or floor_curves is None:
            continue
        readings: dict[str, Any] = {}
        for key in BIAS_STATISTICS:
            a = arm_curves["statistics"][key]["per_seed"]
            f = floor_curves["statistics"][key]["per_seed"]
            paired = [
                signs[key] * (float(a[s]) - float(f[s]))
                for s in sorted(set(a) & set(f), key=int)
                if _finite(a[s]) and _finite(f[s])
            ]
            test = wp.paired_seed_wilcoxon_bootstrap(paired) if paired else None
            readings[key] = {
                "n_seeds": len(paired),
                "arm_minus_floor": test,
                "learned": test is not None and float(test["ci_lo"]) > 0.0,
            }
        out[arm] = {
            "floor": floor,
            "statistics": readings,
            "floor_verdicts": floor_curves["strategy_verdicts"],
        }
    return out


def control_comparison(
    arms: dict[str, Any],
    learn: dict[int, float],
    frozen: dict[int, float],
) -> dict[str, Any]:
    """Compare the derivative control with the MLP: its floor gate first, then the weathervane.

    The control is readable only when every seed is present and its learning arm beats its frozen
    floor; otherwise the weathervane's specificity is untested, never inferred.
    """
    gate = bc.control_gate(learn, frozen, seeds=CONTROL_SEEDS) if learn and frozen else None
    readable = gate is not None and gate["verdict"] in {"passes", "fallback"}
    out: dict[str, Any] = {"gate": gate, "readable": readable}
    mlp_curves = arms.get("mlp", {}).get("bias_curves")
    control_curves = arms.get("mlp_derivative", {}).get("bias_curves")
    if not readable or mlp_curves is None or control_curves is None:
        out["weathervane_specificity"] = "untested"
        return out
    out["weathervane"] = {}
    for key in ("klinotaxis", "klinotaxis_all"):
        mlp = mlp_curves["statistics"][key]["per_seed"]
        ctl = control_curves["statistics"][key]["per_seed"]
        paired = [
            float(mlp[str(s)]) - float(ctl[str(s)])
            for s in CONTROL_SEEDS
            if _finite(mlp.get(str(s))) and _finite(ctl.get(str(s)))
        ]
        out["weathervane"][key] = {
            "n_seeds": len(paired),
            "mlp_minus_control": wp.paired_seed_wilcoxon_bootstrap(paired) if paired else None,
        }
    return out


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


CSV_FIELDS = (
    "arm",
    "seed",
    "frequency_hz",
    "wavelength_bl",
    "speed_bl_per_s",
    "reversal_fraction",
    "steps_used",
    "steps_near_wall",
    "eigenworm_variance",
    "forward_bout_share",
    "omega_turns",
    "omega_postures",
    "worm_minutes",
    "half_step_frequency_hz",
    "half_step_wavelength_bl",
    "half_step_speed_bl_per_s",
    "half_step_reversal_fraction",
    *BIAS_STATISTICS,
)


def write_csv(results: list[dict[str, Any]], arms: dict[str, Any], path: Path) -> Path:
    """Write one row per evaluated run: its readings, its half-step pass, its bias statistics."""
    threshold = omega_posture_threshold()
    rows = []
    for run in sorted(results, key=lambda r: (r["arm"], r["seed"])):
        row: dict[str, Any] = {"arm": run["arm"], "seed": run["seed"]}
        if not run.get("missing"):
            kin = run["kinematics"]
            row |= {name: kin[name] for name in CSV_FIELDS if name in kin}
            total = run["eigenworm_total_ss"]
            row |= {
                "eigenworm_variance": run["eigenworm_captured_ss"] / total if total else None,
                "forward_bout_share": run["forward_bout_share"],
                "omega_turns": len(run["omega_turns"]),
                "omega_postures": sum(a > threshold for a in run["omega_turn_a3"]),
                "worm_minutes": run["worm_minutes"],
            }
            for name, value in run.get("half_step", {}).items():
                if f"half_step_{name}" in CSV_FIELDS:
                    row[f"half_step_{name}"] = value
            curves = arms.get(run["arm"], {}).get("bias_curves") or {"statistics": {}}
            for key in BIAS_STATISTICS:
                per_seed = curves["statistics"].get(key, {}).get("per_seed", {})
                row[key] = per_seed.get(str(run["seed"]))
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI: evaluate every arm's runs, then grade and compare."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--logs", type=Path, action="append", required=True, help="a run-log dir")
    ap.add_argument("--out-dir", type=Path, required=True, help="where the captures go")
    ap.add_argument("--out", type=Path, help="write the JSON here")
    ap.add_argument("--csv", type=Path, help="write one row per evaluated run here")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seeds-per-arm", type=int, default=None, help="each arm's first N seeds")
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="evaluate these seeds in every arm instead: a pilot on seeds outside the panel",
    )
    ap.add_argument("--arms", nargs="*", choices=list(ARMS), default=list(ARMS))
    args = ap.parse_args(argv)
    jobs = [
        (arm, seed, [str(d) for d in args.logs], str(args.out_dir))
        for arm in args.arms
        for seed in (tuple(args.seeds) if args.seeds else ARMS[arm][1])[: args.seeds_per_arm]
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(evaluate_run, jobs))
    arms = {
        arm: summarise_arm([r for r in results if r["arm"] == arm], graded=ARMS[arm][2])
        for arm in args.arms
    }
    learn, frozen = (
        bc.plateaus(args.logs, ARMS[arm][0], CONTROL_SEEDS)
        for arm in ("mlp_derivative", "mlp_derivative_frozen")
    )
    result = {
        "episodes": EPISODES,
        "theta_sharp": THETA_SHARP,
        "wall_margin_mm": WALL_MARGIN_MM,
        "arms": arms,
        "floors": floor_comparison(arms),
        "control": control_comparison(arms, learn, frozen),
    }
    payload = json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    if args.csv:
        write_csv(results, arms, args.csv)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
