"""``--update-telemetry`` and ``--checkpoint-every``, end to end through ``run_simulation.py``.

Covers the training-telemetry requirements "Per-update PPO telemetry" (scenarios "One line per
update", "Off writes nothing", "Telemetry does not change the run") and "Periodic checkpoints"
(scenario "Checkpoints appear on schedule", and the refusal of a non-positive interval).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "scenarios"
    / "foraging"
    / "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350.yml"
)
_RUNS = 4


def _run(cwd: Path, *flags: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_simulation.py"),
            "--config",
            str(CONFIG),
            "--runs",
            str(_RUNS),
            "--seed",
            "42",
            "--log-level",
            "NONE",
            "--theme",
            "headless",
            "--no-detailed-export",
            "--no-file-log",
            *flags,
        ],
        check=False,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=600,
    )


def _session(cwd: Path) -> Path:
    (session,) = sorted((cwd / "exports").iterdir())
    return session


def _tensors(path: Path) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for component, state in torch.load(path, weights_only=False).items():
        if isinstance(state, dict):
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    out[f"{component}.{key}"] = value
    return out


def _same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Run the same seed with both flags and with neither."""
    out: dict[str, Path] = {}
    for name, flags in (("on", ("--update-telemetry", "--checkpoint-every", "2")), ("off", ())):
        cwd = tmp_path_factory.mktemp(name)
        result = _run(cwd, *flags)
        assert result.returncode == 0, result.stderr[-2000:]
        out[name] = _session(cwd)
    return out


@pytest.mark.smoke
def test_one_line_per_update_in_order(runs: dict[str, Path]) -> None:
    """Scenario "One line per update"."""
    path = runs["on"] / "telemetry" / "updates.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert lines
    assert [line["update"] for line in lines] == list(range(len(lines)))
    episodes = [line["episode"] for line in lines]
    assert episodes == sorted(episodes)
    assert set(episodes) <= set(range(1, _RUNS + 1))
    assert all("settling_residual" in line for line in lines)


@pytest.mark.smoke
def test_checkpoints_appear_on_schedule(runs: dict[str, Path]) -> None:
    """Scenario "Checkpoints appear on schedule": the last one is the final weights."""
    weights = runs["on"] / "weights"
    assert sorted(p.name for p in weights.glob("ep*.pt")) == ["ep00002.pt", "ep00004.pt"]
    assert _same(_tensors(weights / "ep00004.pt"), _tensors(weights / "final.pt"))


@pytest.mark.smoke
def test_off_writes_nothing_new(runs: dict[str, Path]) -> None:
    """Scenario "Off writes nothing"."""
    assert not (runs["off"] / "telemetry").exists()
    assert not list((runs["off"] / "weights").glob("ep*.pt"))


@pytest.mark.smoke
def test_the_flags_do_not_change_the_run(runs: dict[str, Path]) -> None:
    """Scenario "Telemetry does not change the run": the same weights and the same episodes."""
    on, off = runs["on"], runs["off"]
    assert _same(_tensors(on / "weights" / "final.pt"), _tensors(off / "weights" / "final.pt"))
    results = "session/data/simulation_results.csv"
    assert (on / results).read_text() == (off / results).read_text()


@pytest.mark.smoke
def test_a_non_positive_interval_is_refused(tmp_path: Path) -> None:
    """A checkpoint interval of zero episodes is refused before anything runs."""
    result = _run(tmp_path, "--checkpoint-every", "0")
    assert result.returncode == 2
    assert "--checkpoint-every" in result.stderr
    assert not (tmp_path / "exports").exists()
