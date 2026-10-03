#!/usr/bin/env python
r"""Re-read the real-worm chemotaxis validation with wall-proximal transitions excluded.

The continuous environment clamps the worm to a square arena, so a worm heading into an edge slides
along it and its heading, displacement and bearing change for a reason that is not taxis. This
script re-reads a re-capture of the validation's three arms at three settings — the exclusion off, a
primary margin and a sensitivity margin — and compares them: first with the committed statistics the
original analysis produced (identity), then with each other (what moved).

Subcommands::

    # Per-arm manifests from a run_campaign.py directory: "<seed> <behaviour_capture.json>".
    uv run python scripts/analysis/wall_exclusion_check.py manifests \
        --campaign campaigns/h3-wall-recapture --out-dir <dir>

    # The three readings per arm, one summary JSON each.
    uv run python scripts/analysis/wall_exclusion_check.py read --out-dir <dir>

    # The two weathervane slopes with each seed's creep floor held at its unexcluded value.
    uv run python scripts/analysis/wall_exclusion_check.py floor --out-dir <dir>

    # Identity against the committed summaries, and the verdicts at each setting.
    uv run python scripts/analysis/wall_exclusion_check.py compare --out-dir <dir> \
        --committed docs/experiments/logbooks/supporting/035-realworm-chemotaxis-validation
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import behavioural_chemotaxis_validation as bcv  # noqa: E402  # pyright: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parents[2]

# Config stem -> (arm, the committed summary the original analysis wrote for it).
ARMS: dict[str, tuple[str, str]] = {
    "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_capture": ("mlp", "mlp-curves.json"),
    "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_capture": (
        "connectome",
        "connectome-curves.json",
    ),
    "mlpppo_small_continuous2d_fick_adaptive_derivative_capture": (
        "control",
        "control-derivative-curves.json",
    ),
}
SEEDS = tuple(range(42, 50))
TAIL_RUNS = 100
THETA_SHARP = 0.45
ARENA_MM = 20.0
# Primary: the cells' max_step_mm, so a transition starting within one step of an edge can have been
# clamped. Sensitivity: twice that. Registered in the launch record before any capture ran.
SETTINGS: dict[str, float | None] = {"off": None, "m1.0": 1.0, "m2.0": 2.0}
PRIMARY = "m1.0"
# Floats in the committed summaries are compared to this relative tolerance: JSON round-trips a
# float exactly, so anything looser than rounding noise is a real difference.
_REL_TOL = 1e-9


class CheckError(RuntimeError):
    """The re-capture cannot be read as the registered panel."""


def session_of(log: Path) -> str:
    """Return the session id a run log names."""
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("Session ID:"):
            return line.split(":", 1)[1].strip()
    msg = f"{log.name} names no session"
    raise CheckError(msg)


def build_manifests(campaign: Path, out_dir: Path, exports: Path = REPO / "exports") -> list[Path]:
    """Write one ``<seed> <capture>`` manifest per arm, refusing a missing or failed run."""
    logs = campaign / "logs"
    rows: dict[str, list[str]] = {arm: [] for arm, _ in ARMS.values()}
    for stem, (arm, _) in ARMS.items():
        for seed in SEEDS:
            log = logs / f"{stem}-seed{seed}.log"
            exit_file = log.with_suffix(".exit")
            if not log.is_file() or not exit_file.is_file():
                msg = f"{arm} seed {seed}: no finished run at {log}"
                raise CheckError(msg)
            if exit_file.read_text().strip() != "0":
                msg = f"{arm} seed {seed}: the run exited {exit_file.read_text().strip()}"
                raise CheckError(msg)
            capture = exports / session_of(log) / "session" / "data" / "behaviour_capture.json"
            if not capture.is_file():
                msg = f"{arm} seed {seed}: no capture at {capture}"
                raise CheckError(msg)
            rows[arm].append(f"{seed} {capture}")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for arm, lines in rows.items():
        path = out_dir / f"manifest-{arm}.txt"
        path.write_text("\n".join(lines) + "\n")
        written.append(path)
    return written


def load_registered(manifest: Path) -> dict[int, list[list[Any]]]:
    """Load an arm's post-convergence tail, refusing a panel missing any registered seed."""
    seeds = bcv.tail_runs(bcv.load_manifest(manifest), TAIL_RUNS)
    if sorted(seeds) != list(SEEDS):
        msg = f"{manifest.name}: seeds {sorted(seeds)}, expected {list(SEEDS)}"
        raise CheckError(msg)
    return seeds


def read_arm(manifest: Path, margin_mm: float | None) -> dict[str, Any]:
    """Grade one arm at one setting, exactly as the original analysis did except for the margin."""
    seeds = load_registered(manifest)
    report = None
    if margin_mm is not None:
        seeds, report = bcv.exclude_walls(seeds, ARENA_MM, margin_mm)
    summary = bcv.analyse(seeds, THETA_SHARP, modality="food")
    if report is not None:
        summary["wall_exclusion"] = report
    return summary


def floor_held(manifest: Path, margin_mm: float) -> dict[str, Any]:
    """Grade the two weathervane slopes with each seed's creep floor held at its unexcluded value.

    The harness recomputes its curving-rate floor, a fraction of the median stride, on whatever data
    it is given. Dropping wall steps, which are often short, raises that floor slightly, so part of
    a change under the exclusion can come from the floor rather than from the dropped transitions.
    Holding the floor at the value the full data gives separates the two.
    """
    from quantumnematode.validation import behavioural_curves as bc
    from quantumnematode.validation.behavioural_agreement import grade_statistic
    from quantumnematode.validation.datasets import load_bias_signatures

    seeds = load_registered(manifest)
    refs = load_bias_signatures(modality="food")
    kept, _ = bcv.exclude_walls(seeds, ARENA_MM, margin_mm)
    slopes: dict[str, list[float]] = {"klinotaxis": [], "klinotaxis_all": []}
    for seed in sorted(seeds):
        full = [k for run in seeds[seed] for k in bc.kinematics(run, THETA_SHARP)]
        kin = [k for run in kept[seed] for k in bc.kinematics(run, THETA_SHARP)]
        floor = bc.suggest_min_path_len(full, bcv._MIN_PATH_LEN_FRACTION)
        for key, fn in (
            ("klinotaxis", bc.weathervane_slope),
            ("klinotaxis_all", bc.weathervane_slope_all),
        ):
            value = fn(kin, min_path_len=floor)
            if value is not None and math.isfinite(value):
                slopes[key].append(value)
    out: dict[str, Any] = {}
    for key, values in slopes.items():
        graded = grade_statistic(values, refs[key])
        out[key] = {
            "verdict": graded.verdict.value,
            "mean": graded.mean,
            "ci_lo": graded.ci_lo,
            "ci_hi": graded.ci_hi,
            "n": graded.n,
        }
    return out


def _same(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=0.0)


def identity(recaptured: dict[str, Any], committed: dict[str, Any]) -> dict[str, Any]:
    """Compare every per-seed statistic and verdict of a re-read with the committed summary."""
    differences: list[str] = []
    for key, stat in committed["statistics"].items():
        new = recaptured["statistics"][key]
        for seed, value in stat["per_seed"].items():
            if not _same(value, new["per_seed"].get(seed)):
                differences.append(
                    f"{key} seed {seed}: {value} then, {new['per_seed'].get(seed)} now",
                )
        if stat["verdict"] != new["verdict"]:
            differences.append(f"{key} verdict: {stat['verdict']} then, {new['verdict']} now")
    return {"identical": not differences, "differences": differences}


def verdicts(summary: dict[str, Any]) -> dict[str, Any]:
    """Extract the verdicts and intervals a reading turns on."""
    return {
        "statistics": {
            key: {k: stat[k] for k in ("verdict", "mean", "ci_lo", "ci_hi", "n")}
            for key, stat in summary["statistics"].items()
        },
        "strategies": {k: v["combined"] for k, v in summary["strategy_verdicts"].items()},
        "fraction_kept": summary.get("wall_exclusion", {}).get("fraction_kept", 1.0),
    }


def moved(off: dict[str, Any], on: dict[str, Any]) -> list[str]:
    """List what changed verdict between two readings, by the launch record's definition."""
    changes = [
        f"{key}: {off['statistics'][key]['verdict']} -> {on['statistics'][key]['verdict']}"
        for key in off["statistics"]
        if off["statistics"][key]["verdict"] != on["statistics"][key]["verdict"]
    ]
    changes += [
        f"{strategy} (combined): {off['strategies'][strategy]} -> {on['strategies'][strategy]}"
        for strategy in off["strategies"]
        if off["strategies"][strategy] != on["strategies"][strategy]
    ]
    return changes


def compare(out_dir: Path, committed_dir: Path) -> dict[str, Any]:
    """Compare with the committed summaries, then report the readings and what moved."""
    result: dict[str, Any] = {"primary": PRIMARY, "arms": {}}
    for arm, committed_name in ARMS.values():
        readings = {
            label: json.loads((out_dir / f"{arm}-{label}.json").read_text()) for label in SETTINGS
        }
        committed = json.loads((committed_dir / committed_name).read_text())
        table = {label: verdicts(summary) for label, summary in readings.items()}
        result["arms"][arm] = {
            "identity_with_committed": identity(readings["off"], committed),
            "readings": table,
            "moved": {
                label: moved(table["off"], table[label]) for label in SETTINGS if label != "off"
            },
        }
    result["moved_at_primary"] = any(a["moved"][PRIMARY] for a in result["arms"].values())
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)
    m = sub.add_parser("manifests")
    m.add_argument("--campaign", type=Path, required=True)
    m.add_argument("--out-dir", type=Path, required=True)
    r = sub.add_parser("read")
    r.add_argument("--out-dir", type=Path, required=True)
    f = sub.add_parser("floor")
    f.add_argument("--out-dir", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--out-dir", type=Path, required=True)
    c.add_argument("--committed", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.command == "manifests":
        for path in build_manifests(args.campaign, args.out_dir):
            print(f"wrote {path}")
    elif args.command == "read":
        for arm, _ in ARMS.values():
            for label, margin in SETTINGS.items():
                summary = read_arm(args.out_dir / f"manifest-{arm}.txt", margin)
                out = args.out_dir / f"{arm}-{label}.json"
                out.write_text(json.dumps(summary, indent=2, default=str) + "\n")
                print(f"wrote {out}")
    elif args.command == "floor":
        result = {
            arm: {
                label: floor_held(args.out_dir / f"manifest-{arm}.txt", margin)
                for label, margin in SETTINGS.items()
                if margin is not None
            }
            for arm, _ in ARMS.values()
        }
        out = args.out_dir / "floor-held.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    else:
        result = compare(args.out_dir, args.committed)
        out = args.out_dir / "comparison.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
