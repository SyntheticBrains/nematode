# pragma: no cover

"""Smoke tests that run entry-point scripts with minimal episodes.

These tests verify that the CLI scripts don't crash for key configurations.
They are excluded from pre-commit hooks but run on every PR.

Usage:
    uv run pytest -m smoke -v
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIGS_DIR = PROJECT_ROOT / "configs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


SIMULATION_CONFIGS = [
    "scenarios/foraging/mlpppo_small_oracle.yml",
    "scenarios/foraging/mlpreinforce_small_oracle.yml",
    "scenarios/foraging/spikingreinforce_small_oracle.yml",
    "special/qvarcircuit_foraging_small_validate_oracle.yml",
    "scenarios/thermal_foraging/mlpppo_small_oracle.yml",
    "scenarios/pursuit/mlpppo_small_oracle.yml",
    "scenarios/foraging/qrh_small_oracle.yml",
    "scenarios/foraging/qef_small_oracle.yml",
    "scenarios/foraging/crh_small_oracle.yml",
    "scenarios/foraging/mlpppo_small_temporal.yml",
    "scenarios/foraging/mlpppo_small_derivative.yml",
    "scenarios/foraging/lstmppo_small_derivative.yml",
    "scenarios/foraging/lstmppo_small_temporal.yml",
]


@pytest.mark.smoke
@pytest.mark.parametrize("config_name", SIMULATION_CONFIGS)
def test_run_simulation_smoke(config_name: str, tmp_path: Path) -> None:
    """Verify run_simulation.py exits cleanly with minimal runs."""
    config_path = CONFIGS_DIR / config_name
    assert config_path.exists(), f"Config not found: {config_path}"

    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(SCRIPTS_DIR / "run_simulation.py"),
            "--config",
            str(config_path),
            "--runs",
            "2",
            "--seed",
            "42",
            "--log-level",
            "NONE",
            "--theme",
            "headless",
        ],
        check=False,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"run_simulation.py failed for {config_name}.\n"
        f"stdout:\n{result.stdout[-2000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )
    assert "Traceback" not in result.stderr, (
        f"Traceback in stderr for {config_name}:\n{result.stderr[-2000:]}"
    )


@pytest.mark.smoke
def test_run_evolution_smoke_mlpppo(tmp_path: Path) -> None:
    """Verify run_evolution.py exits cleanly against the MLPPPO pilot config.

    The framework targets classical brains; quantum brains are not currently
    supported.
    """
    config_path = CONFIGS_DIR / "evolution" / "mlpppo_foraging_small.yml"
    assert config_path.exists(), f"Config not found: {config_path}"

    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(SCRIPTS_DIR / "run_evolution.py"),
            "--config",
            str(config_path),
            "--generations",
            "1",
            "--population",
            "4",
            "--episodes",
            "2",
            "--seed",
            "42",
            "--log-level",
            "WARNING",
            "--output-dir",
            str(tmp_path / "evolution_results"),
        ],
        check=False,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"run_evolution.py failed.\n"
        f"stdout:\n{result.stdout[-2000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )
    assert "Traceback" not in result.stderr, f"Traceback in stderr:\n{result.stderr[-2000:]}"


@pytest.mark.smoke
def test_run_evolution_smoke_mlpppo_resume(tmp_path: Path) -> None:
    """Verify the ``--resume`` CLI flag exercises the resume code path end-to-end.

    The Python-level resume contract is verified by
    ``test_loop_resume_from_checkpoint`` in the evolution unit tests; this
    smoke test confirms the CLI wiring.  Sequence: run 1 generation with
    checkpoint_every=1 (forces a checkpoint), then resume from that
    checkpoint and run more generations.
    """
    config_path = CONFIGS_DIR / "evolution" / "mlpppo_foraging_small.yml"
    output_root = tmp_path / "evolution_results"

    # First session: 1 generation, force checkpoint.
    first = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(SCRIPTS_DIR / "run_evolution.py"),
            "--config",
            str(config_path),
            "--generations",
            "1",
            "--population",
            "4",
            "--episodes",
            "1",
            "--seed",
            "42",
            "--log-level",
            "WARNING",
            "--output-dir",
            str(output_root),
        ],
        check=False,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert first.returncode == 0, f"First run failed:\n{first.stderr[-2000:]}"

    # Locate the session directory and its checkpoint.
    session_dirs = list(output_root.iterdir())
    assert len(session_dirs) == 1, f"Expected 1 session dir, got {session_dirs}"
    checkpoint = session_dirs[0] / "checkpoint.pkl"
    assert checkpoint.exists(), f"Checkpoint not found: {checkpoint}"

    # Second session: resume and run 1 more generation.
    resume = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(SCRIPTS_DIR / "run_evolution.py"),
            "--config",
            str(config_path),
            "--generations",
            "2",
            "--population",
            "4",
            "--episodes",
            "1",
            "--seed",
            "42",
            "--log-level",
            "WARNING",
            "--output-dir",
            str(output_root),
            "--resume",
            str(checkpoint),
        ],
        check=False,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert resume.returncode == 0, f"Resume run failed:\n{resume.stderr[-2000:]}"
    assert "Traceback" not in resume.stderr, f"Traceback in resume:\n{resume.stderr[-2000:]}"

    # Resume MUST write into the original session directory so lineage.csv
    # stays a single chronological history — per the evolution-framework
    # spec scenario "Append mode preserves history across resume".  A
    # second session directory would mean the resumed half fragmented away
    # from the original, breaking lineage continuity.
    session_dirs_after = sorted(output_root.iterdir())
    assert len(session_dirs_after) == 1, (
        f"Resume created a new session dir; lineage is fragmented. Got: {session_dirs_after}"
    )
    # Both generations end up in the same lineage.csv: 1 header row +
    # (population * generations) data rows = 1 + 4 * 2 = 9 lines.
    lineage = session_dirs_after[0] / "lineage.csv"
    assert lineage.exists(), f"lineage.csv missing: {lineage}"
    line_count = sum(1 for _ in lineage.open())
    assert line_count == 9, f"Expected 9 lineage lines (1 header + 8 data), got {line_count}"
