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
    """Verify the new run_evolution.py exits cleanly against the MLPPPO pilot config.

    Replaces the legacy QVarCircuit smoke test (removed alongside the legacy
    script in M0 Phase 7).  The new framework targets classical brains; quantum
    brain support is deferred to a future Phase 6 re-evaluation.
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
