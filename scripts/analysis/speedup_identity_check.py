#!/usr/bin/env python
r"""Check that a speed-up of C.1's simulation path leaves every output byte-identical.

Two stages. ``baseline``, run on the code before the change, simulates a fixed set of jobs on
discarded seeds and stores each result. ``check``, run after it, re-simulates those jobs and a set
of stored records whose inputs are on disk, and compares every graded statistic, distance and
presence reading exactly. A record's ``seconds`` is its wall time and is not compared.

The jobs cover each path the changed code serves:

- benchmark v2 without presence (a fit's objective), and with presence (registered gradings and the
  candidates' observations);
- benchmark v1 (``npe_recovery.simulate``), which C.1's v1 fit and the v1 recoveries used.

Usage::

    uv run python scripts/analysis/speedup_identity_check.py baseline
    uv run python scripts/analysis/speedup_identity_check.py check
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cma_recovery_v2 as v2
import npe_recovery as nr

DESCRIPTION = "Check that a speed-up of C.1's simulation path leaves every output byte-identical."
_REPO = Path(__file__).resolve().parents[2]
BUILD = _REPO / "build" / "v2-speedup"
IDENTITY_PATH = (
    _REPO / "docs" / "experiments" / "logbooks" / "supporting" / "118-v2-speedup" / "identity.json"
)
BASELINE_BASE = 99_700_000
F0A_VECTORS = (0, 1)
STORED_V1 = (0, 1, 2)
STORED_CANDIDATES = 3
COMPARED = ("statistics", "D", "D_fit", "presence")


def _vectors() -> list[tuple[str, list[float]]]:
    sims = nr.load_simulations()
    return [("centre", [0.5] * len(sims[0]["z"]))] + [
        (f"f0a_{k}", list(sims[k]["z"])) for k in F0A_VECTORS
    ]


def baseline_jobs() -> list[dict[str, Any]]:
    """Return the baseline's jobs: three paths at the box centre and two F.0a vectors."""
    return [
        {"name": f"{path}/{name}", "path": path, "k": j, "z": z}
        for j, (name, z) in enumerate(_vectors())
        for path in ("v2", "v2_presence", "v1")
    ]


def _encode(value: Any) -> Any:  # noqa: ANN401 - any JSON-shaped value
    """Return a value with each non-finite float as its own string: nan, inf or -inf."""
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    return nr._finite(value)


def _collapse(value: Any) -> Any:  # noqa: ANN401 - any JSON-shaped value
    """Return an encoded value with every non-finite reading as null, as stored records keep it."""
    if isinstance(value, dict):
        return {k: _collapse(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_collapse(v) for v in value]
    return None if value in ("nan", "inf", "-inf") else value


def run(job: dict[str, Any]) -> dict[str, Any]:
    """Run one job on its path and return its comparable fields, each non-finite value distinct."""
    base = job.get("seed_base", BASELINE_BASE)
    if job["path"] == "v1":
        record = nr.simulate(job["k"], job["z"], base)
    else:
        record = v2.simulate_v2(job["k"], job["z"], base, presence=job["path"] == "v2_presence")
    return _encode({key: record[key] for key in COMPARED if key in record})


def _run_all(jobs: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        return list(pool.map(run, jobs))


def stored_jobs() -> list[dict[str, Any]]:
    """Return jobs whose expected output is already stored, with that output."""
    jobs = []
    for k, r in sorted(v2.read_records(v2.BUILD_B / "probe" / "answers.jsonl").items()):
        jobs.append(
            {
                "name": f"probe_answer/{k}",
                "path": "v2",
                "k": k,
                "z": r["z"],
                "seed_base": r["seed_base"],
                "expected": {key: r[key] for key in COMPARED if key in r},
            },
        )
    candidates = v2.read_records(v2.CANDIDATES_PATH)
    for c in range(STORED_CANDIDATES):
        r = candidates[v2.OBSERVATION_K + c]
        jobs.append(
            {
                "name": f"candidate_observation/{c}",
                "path": "v2_presence",
                "k": r["k"],
                "z": r["z"],
                "seed_base": r.get("seed_base", v2.CANDIDATE_BASE),
                "expected": {key: r[key] for key in COMPARED if key in r},
            },
        )
    sims = nr.load_simulations()
    for k in STORED_V1:
        r = sims[k]
        jobs.append(
            {
                "name": f"f0a_v1/{k}",
                "path": "v1",
                "k": k,
                "z": r["z"],
                "seed_base": nr.SEED_BASE,
                "expected": {key: r[key] for key in COMPARED if key in r},
            },
        )
    return jobs


def _inputs() -> dict[str, Any]:
    """Return what a check needs on disk: the baseline's jobs and the stored records' counts."""
    return {
        "baseline_jobs": [j["name"] for j in baseline_jobs()],
        "probe_answers": len(v2.read_records(v2.BUILD_B / "probe" / "answers.jsonl")),
        "candidate_observations": STORED_CANDIDATES,
        "f0a_v1": len(STORED_V1),
    }


def _baseline(workers: int, build: Path) -> int:
    """Record the baseline once, on the code before the change; never overwrite one."""
    baseline_path, manifest_path = build / "baseline.jsonl", build / "baseline_manifest.json"
    if baseline_path.exists() or manifest_path.exists():
        print(
            f"{baseline_path} already exists: a baseline is recorded once, on the code before the "
            "change. Remove it and its manifest to record a new one.",
            file=sys.stderr,
        )
        return 1
    jobs = baseline_jobs()
    results = _run_all(jobs, workers)
    build.mkdir(parents=True, exist_ok=True)
    with baseline_path.open("w") as out:
        for job, result in zip(jobs, results, strict=True):
            out.write(json.dumps({**job, "expected": result}, allow_nan=False) + "\n")
    manifest_path.write_text(json.dumps(_inputs(), indent=2) + "\n")
    print(f"wrote {len(jobs)} baseline records to {baseline_path}")
    return 0


def _incomplete(
    baseline: list[dict[str, Any]],
    stored: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[str]:
    """Return why the check's inputs differ from those the baseline recorded, if they do."""
    names = [j["name"] for j in baseline]
    problems = []
    if names != manifest["baseline_jobs"] or names != [j["name"] for j in baseline_jobs()]:
        problems.append("the baseline's jobs differ from those recorded")
    counts = {
        "probe_answers": sum(j["name"].startswith("probe_answer/") for j in stored),
        "candidate_observations": sum(
            j["name"].startswith("candidate_observation/") for j in stored
        ),
        "f0a_v1": sum(j["name"].startswith("f0a_v1/") for j in stored),
    }
    problems += [
        f"{key}: {n} found, {manifest[key]} recorded"
        for key, n in counts.items()
        if n != manifest[key]
    ]
    return problems


def _same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _check(workers: int, build: Path, out: Path) -> int:
    baseline = [json.loads(line) for line in (build / "baseline.jsonl").read_text().splitlines()]
    manifest = json.loads((build / "baseline_manifest.json").read_text())
    stored = stored_jobs()
    problems = _incomplete(baseline, stored, manifest)
    if problems:
        print("the check's inputs are incomplete: " + "; ".join(problems), file=sys.stderr)
        return 1
    jobs = baseline + stored
    results = _run_all([{k: v for k, v in j.items() if k != "expected"} for j in jobs], workers)
    rows = []
    for i, (job, raw) in enumerate(zip(jobs, results, strict=True)):
        expected = job["expected"]
        # Stored records kept every non-finite reading as null; the baseline keeps each distinct.
        result = raw if i < len(baseline) else _collapse(raw)
        compared = {key: expected[key] for key in result if key in expected}
        same = _same(result, compared) and set(compared) == set(expected)
        rows.append({"name": job["name"], "path": job["path"], "identical": same})
    record = {
        "description": "Description: every statistic, D and presence reading re-simulated after "
        "the speed-up, against the baseline from the code before it and against stored records.",
        "inputs": manifest,
        "jobs": len(rows),
        "identical": sum(r["identical"] for r in rows),
        "all_identical": all(r["identical"] for r in rows),
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("jobs", "identical", "all_identical")}, indent=2))
    return 0 if record["all_identical"] else 1


def main(argv: list[str] | None = None) -> int:
    """CLI: record the baseline before the change, or check against it after."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("stage", choices=("baseline", "check"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--build",
        type=Path,
        default=BUILD,
        help="this change's baseline directory",
    )
    parser.add_argument("--out", type=Path, default=IDENTITY_PATH, help="the identity record")
    args = parser.parse_args(argv)
    build, out = args.build.resolve(), args.out.resolve()
    if args.stage == "baseline":
        return _baseline(args.workers, build)
    return _check(args.workers, build, out)


if __name__ == "__main__":
    sys.exit(main())
