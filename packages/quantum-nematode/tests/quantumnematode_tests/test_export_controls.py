"""The two flags that keep a campaign from filling the disk, tested end to end.

A 3000-episode run writes ~380 MB of step-level CSVs under
``exports/<session>/session/data/detailed/`` and ~250 MB to ``logs/simulation_<session>.log``,
against a few MB for everything an analysis reads. A 768-run campaign filled the volume mid-run
on exactly that. ``--no-detailed-export`` and ``--no-file-log`` have to remove those two outputs
and nothing else -- and the campaign log, which is captured stdout, must be untouched.

The child runs with the pytest environment markers stripped, because ``configure_file_logging``
is a no-op under test and the control case has to actually write the file for the contrast to
mean anything.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "scenarios"
    / "foraging"
    / "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_eprop_frozen.yml"
)
_TEST_MARKERS = ("PYTEST_CURRENT_TEST", "TESTING")


def _run(cwd: Path, *flags: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in _TEST_MARKERS}
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_simulation.py"),
            "--config",
            str(CONFIG),
            "--runs",
            "1",
            "--seed",
            "42",
            "--log-level",
            "NONE",
            "--theme",
            "headless",
            *flags,
        ],
        check=False,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _one(cwd: Path, pattern: str) -> list[Path]:
    return sorted(cwd.glob(pattern))


@pytest.mark.smoke
def test_the_control_writes_both_heavy_outputs(tmp_path: Path) -> None:
    """Without the flags, both outputs appear -- so the flags test below is not vacuous."""
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr[-2000:]
    assert _one(tmp_path, "exports/*/session/data/detailed"), "control wrote no detailed/ dir"
    assert _one(tmp_path, "logs/simulation_*.log"), "control wrote no file log"
    assert _one(tmp_path, "exports/*/session/data/simulation_results.csv")


@pytest.mark.smoke
def test_the_flags_remove_exactly_those_two_outputs(tmp_path: Path) -> None:
    """With both flags, the two heavy outputs are gone and nothing an analysis reads is."""
    result = _run(tmp_path, "--no-detailed-export", "--no-file-log")
    assert result.returncode == 0, result.stderr[-2000:]
    assert not _one(tmp_path, "exports/*/session/data/detailed"), "detailed/ was still written"
    assert not _one(tmp_path, "logs/simulation_*.log"), "the file log was still written"
    # Everything an analysis reads is still there.
    assert _one(tmp_path, "exports/*/session/data/simulation_results.csv")
    assert _one(tmp_path, "exports/*/weights")
    # And the console stream -- what a campaign captures and `read_log` parses -- is intact.
    assert "Run:" in result.stdout or "Average foods collected" in result.stdout, (
        "stdout lost the per-run summary the campaign log depends on"
    )
