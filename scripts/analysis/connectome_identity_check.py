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
  modulator concentrations and behaviour;
- a built brain's topology state (every chemical and gap weight, sensory gain, readout and wiring
  buffer, the encoder's among them; not the PPO critic, which no sign reaches), for each
  per-connection config whose runs are recorded and for one encoded config, since the jobs above
  do not read the per-connection sign table.

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
# Configs whose built brains are compared whole: the per-connection configs behind recorded runs,
# and one encoded config.
_LAWN = "foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_"
_THERMAL = "thermal_foraging/connectomeppo_small_continuous2d_thermal_klinotaxis_"
BRAIN_CONFIGS = (
    *(f"{_LAWN}modulated_perconn{v}.yml" for v in ("", "_frozen", "_pdfr1", "_satiety", "_tph1")),
    *(
        f"{_THERMAL}{null}per_connection_signs{v}_t35.yml"
        for null in ("", "rewired_chemical_null_")
        for v in ("_frozen", "")
    ),
    f"{_LAWN}encoded_s1_primitives_assay_scheer_lawn.yml",
)
# Per-animal totals the trajectories return beside their per-step arrays.
TOTALS = ("final_satiety", "total_intake")
# What a record's source digest covers: the package that runs, the configs the jobs read, this
# script and the harness it builds the brains with.
SOURCE_PATHS = (
    "packages/quantum-nematode/quantumnematode",
    "configs/scenarios/foraging",
    "scripts/analysis/connectome_identity_check.py",
    "scripts/analysis/body_kinematics_eval.py",
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
    return _digest([np.asarray(getattr(trajectories, name)) for name in (*ARRAYS, *TOTALS)])


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


def brain_job(config: str) -> str:
    """Build one config's brain at the fixed seed; return the digest of its topology state."""
    brain, _, _ = harness.build_brain(_REPO / "configs" / "scenarios" / config, BRAIN_SEED)
    state = brain.topology.state_dict()  # type: ignore[attr-defined]
    return _digest([state[name].detach().cpu().numpy() for name in sorted(state)])


def jobs() -> list[tuple[str, str, str]]:
    """Return every job as (kind, arm, assay)."""
    out = [("batched", arm, assay) for arm in ARMS for assay in ASSAYS]
    out += [("per_step", arm, "scheer_lawn") for arm in ARMS]
    out += [("brain", config, "-") for config in BRAIN_CONFIGS]
    return out


def run_job(job: tuple[str, str, str]) -> str:
    """Return one job's digest."""
    kind, arm, assay = job
    if kind == "brain":
        return brain_job(arm)
    return batched_job(arm, assay) if kind == "batched" else per_step_job(arm)


def _git(*args: str) -> str:
    return subprocess.run(  # noqa: S603 - fixed git subcommands and this repository's paths
        ["git", *args],  # noqa: S607 - the repository's own git
        capture_output=True,
        text=True,
        check=True,
        cwd=_REPO,
    ).stdout


def source_state() -> dict[str, Any]:
    """Return the checked source: its commit, whether it carries uncommitted edits, and a digest.

    The digest covers every tracked or untracked, non-ignored file under :data:`SOURCE_PATHS` as
    it stands in the working tree, so it describes the code that ran, edits included.
    """
    files = sorted(
        set(
            _git("ls-files", "--cached", "--others", "--exclude-standard", "--", *SOURCE_PATHS)
            .strip()
            .splitlines(),
        ),
    )
    h = hashlib.sha256()
    for name in files:
        path = _REPO / name
        if path.is_file():
            h.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    status = _git("status", "--porcelain", "--", *SOURCE_PATHS).strip()
    return {
        "head": _git("rev-parse", "HEAD").strip(),
        "uncommitted_edits": bool(status),
        "source_sha256": h.hexdigest(),
        "files": len(files),
    }


def _describe(state: dict[str, Any]) -> str:
    edits = " with uncommitted edits" if state["uncommitted_edits"] else ""
    return f"{state['head'][:10]}{edits} (source {state['source_sha256'][:12]})"


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
        state = source_state()
        record = {"code": state["head"], "source": state, "digests": digests}
        baseline_path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"{len(digests)} jobs recorded at {_describe(state)}")
        return 0
    baseline = json.loads(baseline_path.read_text())
    rows = [
        {"job": job, "identical": digests[job] == baseline["digests"][job]}
        for job in baseline["digests"]
    ]
    state = source_state()
    baseline_state = baseline.get("source", {"head": baseline["code"]})
    baseline_described = (
        _describe(baseline_state) if "source_sha256" in baseline_state else baseline["code"][:10]
    )
    record = {
        "description": (
            f"Description: every connectome output in the baseline re-run at {_describe(state)}, "
            f"against the baseline recorded at {baseline_described}."
        ),
        "baseline_code": baseline["code"],
        "baseline_source": baseline_state,
        "code": state["head"],
        "source": state,
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
