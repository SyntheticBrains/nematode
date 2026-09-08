#!/usr/bin/env python
r"""Warm-start panel campaign steps: select and record the teacher, clone every student.

``teacher``: scan the MLP-PPO teacher campaign, pick the seed with the highest committed
plateau tail, copy its auto-saved weights beside the results, derive a frozen recording
config (the teacher config plus ``freeze_updates: true`` and ``weights_path``), run it for
the registered number of episodes at the recording seed with ``--record-rollouts``, and
record the teacher's frozen plateau tail at that seed as the ceiling every clone is read
against.

``clone``: for every panel seed and both wirings, clone the teacher's recorded policy into
the plastic-set student (the plastic frozen arm) and the full-set student (the low-noise PPO
arm) with the registered hyperparameters, and write ``clones.json`` with every fit and a
flag on any clone whose held-out loss did not fall below half its initial.

Usage::

    uv run python scripts/campaigns/l4_warm_start_campaign.py teacher \\
        --campaign-dir campaigns/l4-teacher --out-dir campaigns/l4-warm-start
    uv run python scripts/campaigns/l4_warm_start_campaign.py clone --out-dir campaigns/l4-warm-start
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# The analysis scripts hold the committed metric and the panel harness helpers; this
# script lives beside the campaign runners, so put them on the path explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import l4_panel  # pyright: ignore[reportMissingImports]
import t7_continuous_ranking as t7  # pyright: ignore[reportMissingImports]

EXPERIMENTS = l4_panel.EXPERIMENTS
REPO = l4_panel.REPO

CONFIG_DIR = REPO / "configs" / "scenarios" / "foraging_predator_thermal"
TEACHER_STEM = "mlpppo_small_continuous2d_combined_klinotaxis"
STUDENT_STEM = "connectomeppo_small_continuous2d_combined_klinotaxis"
STUDENT_CONFIGS: dict[tuple[str, str], str] = {
    ("plastic", "wt"): f"{STUDENT_STEM}_plastic_frozen",
    ("plastic", "rn"): f"{STUDENT_STEM}_plastic_frozen_rewired_null",
    ("full", "wt"): f"{STUDENT_STEM}_lowstd",
    ("full", "rn"): f"{STUDENT_STEM}_rewired_null_lowstd",
}
PANEL_SEEDS: tuple[int, ...] = tuple(range(1, 9))
TEACHER_SEEDS: tuple[int, ...] = tuple(range(1, 9))
RECORD_SEED = 101
RECORD_EPISODES = 300
CLONE_EPOCHS = 300
CLONE_LR = 1e-3
CLONE_BATCH = 256
CLONE_HOLDOUT = 0.2
WEAK_CLONE_RATIO = 0.5  # flagged when held-out loss / initial held-out loss is not below this

_LABEL = re.compile(r"^(?P<stem>.+?)-seed(?P<seed>\d+)\.log$")


# --- teacher ----------------------------------------------------------------------------


def scan_teacher_campaign(
    campaign_dir: Path,
    experiments: Path = EXPERIMENTS,
) -> list[dict[str, Any]]:
    """One record per teacher run: seed, plateau tail, and the auto-saved weights path."""
    log_dir = campaign_dir / "logs" if (campaign_dir / "logs").is_dir() else campaign_dir
    records: list[dict[str, Any]] = []
    for log in sorted(log_dir.glob("*.log")):
        match = _LABEL.match(log.name)
        if match is None or match.group("stem") != TEACHER_STEM:
            continue
        tail = t7.plateau_tail(log)
        if tail is None:
            print(f"  WARN: no parseable plateau in {log.name} - skipped")
            continue
        experiment = l4_panel._experiment_json(log.read_text(), experiments)
        exports = experiment.get("exports_path") if experiment else None
        weights = (REPO / exports / "weights" / "final.pt") if exports else None
        records.append(
            {
                "seed": int(match.group("seed")),
                "plateau_tail": float(tail[0]),
                "log": str(log),
                "weights": str(weights) if weights else None,
            },
        )
    return records


def select_teacher(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the seed with the highest plateau tail; ties go to the lower seed."""
    usable = [r for r in records if r.get("weights")]
    if not usable:
        msg = "no teacher run with auto-saved weights was found"
        raise ValueError(msg)
    return max(usable, key=lambda r: (r["plateau_tail"], -r["seed"]))


def derive_recording_config(parent_text: str, weights_path: str) -> str:
    """Derive the teacher config with updates frozen and the teacher's weights loaded; nothing else."""
    lines = parent_text.splitlines(keepends=True)
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        i += 1
    body = "".join(lines[i:])
    if body.count("  config:\n") != 1:
        msg = "the teacher config has no single `  config:` block to derive from"
        raise ValueError(msg)
    header = (
        "# Frozen teacher for rollout recording -- MLPPPO, no learning.\n"
        "#\n"
        "# Derived beside the campaign results from the teacher config: `freeze_updates: true`\n"
        "# and `weights_path` naming the selected teacher's weights, nothing else, so the\n"
        "# recorded policy is exactly the trained one and stationary across the recording.\n"
    )
    keys = f"    freeze_updates: true\n    weights_path: {weights_path}\n"
    return header + body.replace("  config:\n", "  config:\n" + keys, 1)


def teacher(args: argparse.Namespace) -> int:
    """Select, copy, derive, record."""
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    records = scan_teacher_campaign(args.campaign_dir, args.experiments_dir)
    chosen = select_teacher(records)
    teacher_pt = out_dir / "teacher.pt"
    shutil.copyfile(chosen["weights"], teacher_pt)
    recording_config = out_dir / f"{TEACHER_STEM}_teacher_frozen.yml"
    recording_config.write_text(
        derive_recording_config(
            (CONFIG_DIR / f"{TEACHER_STEM}.yml").read_text(),
            str(teacher_pt.relative_to(REPO))
            if teacher_pt.is_relative_to(REPO)
            else str(teacher_pt),
        ),
    )
    rollouts = out_dir / "rollouts.jsonl"
    log = out_dir / "teacher_recording.log"
    command = [
        sys.executable,
        str(REPO / "scripts" / "run_simulation.py"),
        "--config",
        str(recording_config),
        "--runs",
        str(args.record_episodes),
        "--seed",
        str(args.record_seed),
        "--theme",
        "headless",
        "--record-rollouts",
        str(rollouts),
    ]
    print("recording:", " ".join(command))
    with log.open("w") as handle:
        result = subprocess.run(  # noqa: S603 -- the entry point, with paths this script built
            command,
            cwd=REPO,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        print(f"error: the recording run exited {result.returncode}; see {log}", file=sys.stderr)
        return result.returncode
    ceiling = t7.plateau_tail(log)
    record = {
        "candidates": records,
        "selected_seed": chosen["seed"],
        "selected_plateau_tail": chosen["plateau_tail"],
        "weights": str(teacher_pt),
        "recording_config": str(recording_config),
        "record_seed": args.record_seed,
        "record_episodes": args.record_episodes,
        "rollouts": str(rollouts),
        "ceiling_plateau_tail": float(ceiling[0]) if ceiling else None,
    }
    (out_dir / "teacher.json").write_text(json.dumps(record, indent=2))
    print(
        f"teacher: seed {chosen['seed']} ({chosen['plateau_tail']:.1f}%), ceiling {record['ceiling_plateau_tail']}",
    )
    return 0


# --- clones -----------------------------------------------------------------------------


def clone_file(parameter_set: str, wiring: str, seed: int) -> str:
    """Return the clone file name the arm configs address through their ``{seed}`` placeholder."""
    return f"{parameter_set}_{wiring}_seed{seed}.pt"


def flag_clone(record: dict[str, Any]) -> bool:
    """Return True when the held-out loss did not fall below half its initial value."""
    initial = record.get("initial_held_out_loss")
    final = record.get("held_out_loss")
    if initial is None or final is None or math.isnan(initial) or math.isnan(final):
        return True
    return not final < WEAK_CLONE_RATIO * initial


def clone(args: argparse.Namespace) -> int:
    """Clone every student; write ``clones.json``."""
    import l4_behavioural_clone as bc

    out_dir: Path = args.out_dir
    rollouts = out_dir / "rollouts.jsonl"
    if not rollouts.is_file():
        print(f"error: {rollouts} not found; run the teacher step first", file=sys.stderr)
        return 2
    clones_dir = out_dir / "clones"
    clones_dir.mkdir(parents=True, exist_ok=True)
    fits: list[dict[str, Any]] = []
    for (parameter_set, wiring), stem in STUDENT_CONFIGS.items():
        for seed in args.seeds:
            out = clones_dir / clone_file(parameter_set, wiring, seed)
            record_file = out.with_name(f"{out.stem}.clone.json")
            if args.skip_existing and out.is_file() and record_file.is_file():
                record = {
                    "parameter_set": parameter_set,
                    "wiring": wiring,
                    "seed": seed,
                    "file": str(out),
                }
                record.update(json.loads(record_file.read_text()))
                record["weak"] = flag_clone(record)
                fits.append(record)
                print(
                    f"  {parameter_set} {wiring} seed {seed}: kept existing clone (weak={record['weak']})",
                )
                continue
            code = bc.main(
                [
                    "--config",
                    str(CONFIG_DIR / f"{stem}.yml"),
                    "--seed",
                    str(seed),
                    "--rollouts",
                    str(rollouts),
                    "--parameter-set",
                    parameter_set,
                    "--out",
                    str(out),
                    "--epochs",
                    str(CLONE_EPOCHS),
                    "--lr",
                    str(CLONE_LR),
                    "--batch-size",
                    str(CLONE_BATCH),
                    "--holdout",
                    str(CLONE_HOLDOUT),
                ],
            )
            record: dict[str, Any] = {
                "parameter_set": parameter_set,
                "wiring": wiring,
                "seed": seed,
                "file": str(out),
            }
            if code != 0:
                record["failed"] = code
            else:
                record.update(json.loads(out.with_name(f"{out.stem}.clone.json").read_text()))
            record["weak"] = flag_clone(record)
            fits.append(record)
            print(
                f"  {parameter_set} {wiring} seed {seed}: held-out {record.get('held_out_loss')} weak={record['weak']}",
            )
    name = f"clones.{args.part}.json" if args.part else "clones.json"
    (out_dir / name).write_text(json.dumps(fits, indent=2))
    failed = [f for f in fits if "failed" in f]
    if failed:
        print(f"error: {len(failed)} clone(s) failed", file=sys.stderr)
        return 1
    return 0


def merge(args: argparse.Namespace) -> int:
    """Combine the per-part clone records into ``clones.json``, ordered by set, wiring, seed."""
    parts = sorted(args.out_dir.glob("clones.*.json"))
    if not parts:
        print(f"error: no clones.<part>.json under {args.out_dir}", file=sys.stderr)
        return 2
    merged: dict[tuple[str, str, int], dict[str, Any]] = {}
    for part in parts:
        for record in json.loads(part.read_text()):
            merged[(record["parameter_set"], record["wiring"], int(record["seed"]))] = record
    order = {key: i for i, key in enumerate(STUDENT_CONFIGS)}
    fits = [merged[k] for k in sorted(merged, key=lambda k: (order[(k[0], k[1])], k[2]))]
    (args.out_dir / "clones.json").write_text(json.dumps(fits, indent=2))
    print(f"merged {len(fits)} clone records from {len(parts)} parts")
    return 0


def _seeds(text: str) -> tuple[int, ...]:
    if "-" in text:
        lo, hi = text.split("-", 1)
        return tuple(range(int(lo), int(hi) + 1))
    return tuple(int(s) for s in text.split(","))


def main(argv: list[str] | None = None) -> int:
    """Run one campaign step from the command line; return the exit code."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="step", required=True)
    t = sub.add_parser("teacher")
    t.add_argument("--campaign-dir", type=Path, required=True)
    t.add_argument("--out-dir", type=Path, required=True)
    t.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS)
    t.add_argument("--record-seed", type=int, default=RECORD_SEED)
    t.add_argument("--record-episodes", type=int, default=RECORD_EPISODES)
    c = sub.add_parser("clone")
    c.add_argument("--out-dir", type=Path, required=True)
    c.add_argument("--seeds", type=_seeds, default=PANEL_SEEDS)
    c.add_argument(
        "--part",
        type=str,
        default=None,
        help="write clones.<part>.json (for parallel workers)",
    )
    c.add_argument("--skip-existing", action="store_true", help="keep clones already on disk")
    m = sub.add_parser("merge")
    m.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    steps = {"teacher": teacher, "clone": clone, "merge": merge}
    return steps[args.step](args)


if __name__ == "__main__":
    sys.exit(main())
