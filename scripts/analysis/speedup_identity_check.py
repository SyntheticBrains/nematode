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
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cma_recovery_v2 as v2
import npe_recovery as nr

DESCRIPTION = "Check that a speed-up of C.1's simulation path leaves every output byte-identical."
_REPO = Path(__file__).resolve().parents[2]
BUILD = _REPO / "build" / "v2-speedup"
BASELINE_PATH = BUILD / "baseline.jsonl"
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


def run(job: dict[str, Any]) -> dict[str, Any]:
    """Run one job on its path and return its comparable fields, non-finite values as null."""
    base = job.get("seed_base", BASELINE_BASE)
    if job["path"] == "v1":
        record = nr.simulate(job["k"], job["z"], base)
    else:
        record = v2.simulate_v2(job["k"], job["z"], base, presence=job["path"] == "v2_presence")
    return nr._finite({key: record[key] for key in COMPARED if key in record})


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


def _baseline(workers: int) -> int:
    jobs = baseline_jobs()
    results = _run_all(jobs, workers)
    BUILD.mkdir(parents=True, exist_ok=True)
    with BASELINE_PATH.open("w") as out:
        for job, result in zip(jobs, results, strict=True):
            out.write(json.dumps({**job, "expected": result}, allow_nan=False) + "\n")
    print(f"wrote {len(jobs)} baseline records to {BASELINE_PATH}")
    return 0


def _same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _check(workers: int) -> int:
    baseline = [json.loads(line) for line in BASELINE_PATH.read_text().splitlines()]
    jobs = baseline + stored_jobs()
    results = _run_all([{k: v for k, v in j.items() if k != "expected"} for j in jobs], workers)
    rows = []
    for job, result in zip(jobs, results, strict=True):
        expected = job["expected"]
        compared = {key: expected[key] for key in result if key in expected}
        same = _same(result, compared) and set(compared) == set(expected)
        rows.append({"name": job["name"], "path": job["path"], "identical": same})
    record = {
        "description": "Description: every statistic, D and presence reading re-simulated after "
        "the speed-up, against the baseline from the code before it and against stored records.",
        "jobs": len(rows),
        "identical": sum(r["identical"] for r in rows),
        "all_identical": all(r["identical"] for r in rows),
        "rows": rows,
    }
    IDENTITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    IDENTITY_PATH.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("jobs", "identical", "all_identical")}, indent=2))
    return 0 if record["all_identical"] else 1


def main(argv: list[str] | None = None) -> int:
    """CLI: record the baseline before the change, or check against it after."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("stage", choices=("baseline", "check"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    return _baseline(args.workers) if args.stage == "baseline" else _check(args.workers)


if __name__ == "__main__":
    sys.exit(main())
