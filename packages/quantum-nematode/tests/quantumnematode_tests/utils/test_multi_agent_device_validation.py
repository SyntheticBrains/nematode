"""Multi-agent runs validate every agent's brain against the shared device.

The multi-agent path builds one brain per agent and is dispatched before the
single-brain device check in ``main`` ever runs, so without its own check a
heterogeneous config could carry a brain the selected device cannot serve —
past the validation entirely, into brain construction.

Each agent may name a different architecture, so it is not enough to check one
of them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4].parent
SIMULATION = PROJECT_ROOT / "scripts" / "run_simulation.py"
MIXED_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "scenarios"
    / "multi_agent_foraging"
    / "lstmppo_large_5agents_mixed_1F_4L_klinotaxis.yml"
)

_REJECTED_EXIT = 2


@pytest.fixture
def quantum_agent_config(tmp_path: Path) -> Path:
    """Build a multi-agent config whose first agent uses a Qiskit-backed brain."""
    text = MIXED_CONFIG.read_text(encoding="utf-8").replace("name: lstmppo", "name: qvarcircuit", 1)
    path = tmp_path / "multi_agent_quantum.yml"
    path.write_text(text, encoding="utf-8")
    return path


def _run(config: Path, device: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        [
            sys.executable,
            str(SIMULATION),
            "--config",
            str(config),
            "--runs",
            "1",
            "--theme",
            "headless",
            "--log-level",
            "NONE",
            "--seed",
            "1",
            "--device",
            device,
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        check=False,
    )


@pytest.mark.slow
class TestMultiAgentDeviceValidation:
    """The per-agent check runs before any brain is constructed."""

    def test_quantum_agent_rejects_torch_only_device(self, quantum_agent_config: Path) -> None:
        result = _run(quantum_agent_config, "mps")

        assert result.returncode == _REJECTED_EXIT
        assert "PyTorch-only accelerator" in result.stderr
        # The message must say which agent is at fault, not just which brain.
        assert "agent" in result.stderr
        # Rejected before construction, so no traceback from inside a brain.
        assert "Traceback" not in result.stderr

    def test_all_classical_agents_are_accepted(self) -> None:
        """The guard must not refuse a configuration that works."""
        assert _run(MIXED_CONFIG, "mps").returncode == 0
