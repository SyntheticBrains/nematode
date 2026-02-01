# pragma: no cover

"""Nightly end-to-end regression tests.

These tests run full training sessions and assert that success rates
fall within established benchmark ranges. They are designed to detect
performance regressions from code changes.

Benchmark ranges are defined in e2e_benchmarks.json and derived from
experiment logbooks and benchmark submissions.

Usage:
    uv run pytest -m nightly -v
    uv run pytest -m nightly -k "foraging_small" -v
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
BENCHMARKS_FILE = Path(__file__).parent / "e2e_benchmarks.json"


def _load_benchmarks() -> list[dict]:
    """Load benchmark configurations from JSON file."""
    with BENCHMARKS_FILE.open() as f:
        data = json.load(f)
    return data["configs"]


def _parse_success_rate(work_dir: Path) -> float:
    """Parse success rate from simulation_results.csv export.

    Parameters
    ----------
    work_dir : Path
        Working directory where the simulation was run.

    Returns
    -------
    float
        Success rate as a fraction (0.0 to 1.0).
    """
    # Find the exports directory (exports/<timestamp>/session/data/)
    exports_dir = work_dir / "exports"
    if not exports_dir.exists():
        msg = f"No exports directory found in {work_dir}"
        raise FileNotFoundError(msg)

    # Find the timestamped session directory
    session_dirs = sorted(exports_dir.iterdir(), key=lambda p: p.name)
    if not session_dirs:
        msg = f"No session directories found in {exports_dir}"
        raise FileNotFoundError(msg)

    csv_path = session_dirs[0] / "session" / "data" / "simulation_results.csv"
    if not csv_path.exists():
        msg = f"simulation_results.csv not found at {csv_path}"
        raise FileNotFoundError(msg)

    total = 0
    successes = 0
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row.get("success", "").strip().lower() == "true":
                successes += 1

    if total == 0:
        msg = f"No rows found in {csv_path}"
        raise ValueError(msg)

    return successes / total


BENCHMARKS = _load_benchmarks()
BENCHMARK_NAMES = [b["name"] for b in BENCHMARKS]


@pytest.mark.nightly
@pytest.mark.parametrize(
    "benchmark",
    BENCHMARKS,
    ids=BENCHMARK_NAMES,
)
def test_nightly_regression(benchmark: dict, tmp_path: Path) -> None:
    """Run a full training session and verify success rate is within benchmark range."""
    config_path = PROJECT_ROOT / benchmark["config"]
    assert config_path.exists(), f"Config not found: {config_path}"

    runs = benchmark["runs"]
    expected_min = benchmark["success_rate_min"]
    expected_max = benchmark["success_rate_max"]

    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(SCRIPTS_DIR / "run_simulation.py"),
            "--config",
            str(config_path),
            "--runs",
            str(runs),
            "--seed",
            "12345",
            "--log-level",
            "NONE",
            "--theme",
            "ascii",
        ],
        check=False,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour max per config
    )

    assert result.returncode == 0, (
        f"Simulation failed for {benchmark['name']}.\nstderr:\n{result.stderr[-3000:]}"
    )

    success_rate = _parse_success_rate(tmp_path)

    assert success_rate >= expected_min, (
        f"REGRESSION: {benchmark['name']} success rate {success_rate:.1%} "
        f"is below floor {expected_min:.1%}. "
        f"Expected range: [{expected_min:.1%}, {expected_max:.1%}]. "
        f"Source: {benchmark['source']}"
    )

    if success_rate > expected_max:
        warnings.warn(
            f"{benchmark['name']} success rate {success_rate:.1%} "
            f"exceeds ceiling {expected_max:.1%} — consider updating benchmark.",
            UserWarning,
            stacklevel=1,
        )
