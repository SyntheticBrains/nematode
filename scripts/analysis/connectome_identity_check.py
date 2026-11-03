#!/usr/bin/env python
r"""Check that a change to the connectome brain leaves every existing connectome output byte-identical.

Two stages. ``baseline``, run on the code before the change, runs a fixed set of jobs on discarded
seeds and stores a SHA-256 of each job's outputs. ``check``, run after it, re-runs the jobs and
compares every digest exactly.

The jobs cover the hand-placed and the data-layer connectome arms, untrained:

- the batched evaluator on each of benchmark v3's four assays (Scheer's small lawn and uniform
  well, López-Cruz's food removal, Chen's chemotaxis), 16 animals over the whole assay: every
  trajectory array;
- the per-step runner on Scheer's small lawn for 300 steps: its features, action means, actions,
  modulator concentrations and behaviour.

Usage::

    uv run python scripts/analysis/connectome_identity_check.py baseline --build build/c2-encoder/identity
    uv run python scripts/analysis/connectome_identity_check.py check --build build/c2-encoder/identity \
        --out <identity.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import body_kinematics_eval as harness

DESCRIPTION = "Check a connectome brain change leaves every existing connectome output identical."
_REPO = Path(__file__).resolve().parents[2]
FORAGING = _REPO / "configs" / "scenarios" / "foraging"
STEM = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_modulated_s1_primitives"
ARMS: dict[str, dict[str, object]] = {"hand_placed": {}, "data_layer": {"modulator_layer": "data"}}
ASSAYS = ("scheer_lawn", "uniform_lawn_scheer", "food_removal_lopezcruz", "chemotaxis_chen")
BRAIN_SEED = 99_810_000
ANIMAL_BASE = 99_820_000
ANIMALS = 16
PER_STEP_SEED = 99_830_000
PER_STEP_STEPS = 300
ARRAYS = (
    "x",
    "y",
    "heading",
    "concentration",
    "derivative",
    "satiety",
    "intake",
    "on_lawn",
    "primitive",
    "phase",
)


def _digest(parts: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    for part in parts:
        array = np.ascontiguousarray(part)
        h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def _path(assay: str) -> Path:
    return FORAGING / f"{STEM}_assay_{assay}.yml"


def batched_job(arm: str, assay: str) -> str:
    """Run the batched evaluator for one arm and assay; return the digest of its trajectories."""
    from quantumnematode.evaluation.batched import BatchedEvaluator
    from quantumnematode.evaluation.controllers import BatchedConnectome
    from quantumnematode.evaluation.odour_field import chen_field, chen_starts
    from quantumnematode.utils.config_loader import load_simulation_config

    path = _path(assay)
    config = load_simulation_config(str(path))
    brain, _, _ = harness.build_brain(path, BRAIN_SEED, ARMS[arm])
    seeds = [ANIMAL_BASE + i for i in range(ANIMALS)]
    extra: dict[str, Any] = {}
    if assay == "chemotaxis_chen":
        field = chen_field()
        extra = {"odour_field": field, "field_starts": chen_starts(seeds, field)}
    trajectories = BatchedEvaluator(config, BatchedConnectome(brain), seeds, **extra).run()  # type: ignore[arg-type]
    return _digest([np.asarray(getattr(trajectories, name)) for name in ARRAYS])


def per_step_job(arm: str) -> str:
    """Run the per-step runner for one arm on the small lawn; return the digest of its log."""
    from quantumnematode.evaluation.reference import run_reference
    from quantumnematode.utils.config_loader import load_simulation_config

    path = _path("scheer_lawn")
    config = load_simulation_config(str(path))
    brain, _, _ = harness.build_brain(path, BRAIN_SEED, ARMS[arm])
    episode = run_reference(config, brain, PER_STEP_SEED, PER_STEP_STEPS, record_density=True)
    behaviour = np.array(
        [[s.x, s.y, s.heading_rad, s.concentration, s.dc_dt] for s in episode.behaviour],
        dtype=np.float64,
    )
    parts = [
        np.asarray(episode.features, dtype=np.float64),
        np.asarray(episode.means, dtype=np.float64),
        np.asarray(episode.actions, dtype=np.float64),
        np.asarray(episode.modulators, dtype=np.float64),
        behaviour,
        np.array([len(episode.behaviour)]),
        np.array([str(s.primitive) for s in episode.behaviour]).astype("U16").view(np.uint32),
    ]
    return _digest(parts)


def jobs() -> list[tuple[str, str, str]]:
    """Return every job as (kind, arm, assay)."""
    out = [("batched", arm, assay) for arm in ARMS for assay in ASSAYS]
    out += [("per_step", arm, "scheer_lawn") for arm in ARMS]
    return out


def run_job(job: tuple[str, str, str]) -> str:
    """Return one job's digest."""
    kind, arm, assay = job
    return batched_job(arm, assay) if kind == "batched" else per_step_job(arm)


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607 - the repository's own git
        capture_output=True,
        text=True,
        check=True,
        cwd=_REPO,
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    """CLI: record the baseline, or check against it."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("stage", choices=("baseline", "check"))
    parser.add_argument("--build", type=Path, required=True, help="where the baseline is kept")
    parser.add_argument("--out", type=Path, help="the identity record (check)")
    args = parser.parse_args(argv)
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=len(jobs())) as pool:
        digests = dict(zip(map("/".join, jobs()), pool.map(run_job, jobs()), strict=True))
    baseline_path = args.build / "baseline.json"
    if args.stage == "baseline":
        args.build.mkdir(parents=True, exist_ok=True)
        record = {"code": _head(), "digests": digests}
        baseline_path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"{len(digests)} jobs recorded at {record['code'][:10]}")
        return 0
    baseline = json.loads(baseline_path.read_text())
    rows = [
        {"job": job, "identical": digests[job] == baseline["digests"][job]}
        for job in baseline["digests"]
    ]
    record = {
        "description": "Description: every existing connectome output re-run after the change, "
        "against a baseline from the code before it.",
        "baseline_code": baseline["code"],
        "code": _head(),
        "jobs": len(rows),
        "identical": sum(r["identical"] for r in rows),
        "all_identical": all(r["identical"] for r in rows),
        "rows": rows,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("jobs", "identical", "all_identical")}, indent=2))
    return 0 if record["all_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
