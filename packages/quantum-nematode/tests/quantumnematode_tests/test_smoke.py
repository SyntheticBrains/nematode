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
CONFIGS_DIR = PROJECT_ROOT / "configs" / "examples"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


SIMULATION_CONFIGS = [
    "ppo_foraging_small.yml",
    "mlp_foraging_small.yml",
    "spiking_foraging_small.yml",
    "modular_foraging_small_validate.yml",
    "modular_predators_small_validate.yml",
    "ppo_thermotaxis_foraging_small.yml",
    "ppo_predators_small.yml",
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
def test_run_evolution_smoke(tmp_path: Path) -> None:
    """Verify run_evolution.py exits cleanly with minimal parameters."""
    config_path = CONFIGS_DIR / "evolution_modular_foraging_small.yml"
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
