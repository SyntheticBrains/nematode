"""Rollout recording: the recorder's lines, and an end-to-end run through the entry point."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from quantumnematode.brain.actions import ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.rollouts import RolloutRecorder, params_record, read_rollouts

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SCRIPT = _REPO_ROOT / "scripts" / "run_simulation.py"
_CONFIG = (
    _REPO_ROOT
    / "configs"
    / "scenarios"
    / "foraging"
    / "connectomeppo_small_continuous2d_klinotaxis.yml"
)


class TestRecorderUnit:
    def test_one_line_per_step_with_the_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "r.jsonl"
        recorder = RolloutRecorder(path)
        params = BrainParams(food_gradient_strength=0.3, food_gradient_direction=-0.5, satiety=0.7)
        action = ActionData(
            state="continuous",
            probability=0.42,
            continuous=(0.6, -0.1),
            continuous_mean=(0.55, 0.0),
        )
        recorder.record(params, action)
        recorder.record(
            params,
            ActionData(state="continuous", probability=0.5, continuous=(0.1, 0.2)),
        )
        recorder.end_episode()
        recorder.record(params, action)
        recorder.close()
        rows = read_rollouts(path)
        assert [(r["episode"], r["step"]) for r in rows] == [(0, 0), (0, 1), (1, 0)]
        assert rows[0]["params"] == params_record(params)
        assert rows[0]["params"]["food_gradient_strength"] == 0.3
        assert "action" not in rows[0]["params"]
        assert "predator_contact" not in rows[0]["params"]  # unset fields are dropped
        assert rows[0]["action"] == [0.6, -0.1]
        assert rows[0]["action_mean"] == [0.55, 0.0]
        assert rows[0]["probability"] == 0.42
        assert rows[1]["action_mean"] is None

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        recorder = RolloutRecorder(tmp_path / "r.jsonl")
        recorder.close()
        recorder.close()


@pytest.mark.slow
class TestRecorderEndToEnd:
    def _run(self, tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(_SCRIPT),
                "--config",
                str(_CONFIG),
                "--runs",
                "2",
                "--seed",
                "3",
                "--theme",
                "headless",
                *extra,
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )

    def test_recording_has_one_line_per_step_of_every_run(self, tmp_path: Path) -> None:
        out = tmp_path / "rollouts.jsonl"
        result = self._run(tmp_path, "--record-rollouts", str(out))
        assert result.returncode == 0, result.stderr[-2000:]
        rows = read_rollouts(out)
        assert rows
        assert {r["episode"] for r in rows} == {0, 1}
        for row in rows:
            assert set(row) == {"episode", "step", "params", "action", "action_mean", "probability"}
            assert len(row["action"]) == 2
            assert len(row["action_mean"]) == 2
            assert row["probability"] >= 0.0
            assert (
                "food_concentration" in row["params"] or "food_gradient_strength" in row["params"]
            )
        # The run log states each run's step count; the recording must carry exactly that many
        # lines per episode, numbered from zero.
        steps_by_run = {
            int(m.group(1)): int(m.group(2))
            for m in re.finditer(r"Run:\s+(\d+).*?Steps:\s+(\d+)", result.stdout)
        }
        assert set(steps_by_run) == {1, 2}
        for episode, run in ((0, 1), (1, 2)):
            steps = [r["step"] for r in rows if r["episode"] == episode]
            assert steps == list(range(steps_by_run[run]))

    def test_manyworlds_is_rejected(self, tmp_path: Path) -> None:
        result = self._run(tmp_path, "--record-rollouts", str(tmp_path / "r.jsonl"), "--manyworlds")
        assert result.returncode == 2
        assert "cannot be combined with --manyworlds" in result.stderr
        assert not (tmp_path / "r.jsonl").exists()

    def test_discrete_brain_is_rejected(self, tmp_path: Path) -> None:
        discrete = (
            _REPO_ROOT / "configs" / "scenarios" / "foraging" / "connectomeppo_small_klinotaxis.yml"
        )
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(_SCRIPT),
                "--config",
                str(discrete),
                "--runs",
                "1",
                "--seed",
                "3",
                "--theme",
                "headless",
                "--record-rollouts",
                str(tmp_path / "r.jsonl"),
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        assert result.returncode == 2
        assert "discrete actions" in result.stderr
        assert not (tmp_path / "r.jsonl").exists()

    def test_no_flag_writes_nothing(self, tmp_path: Path) -> None:
        result = self._run(tmp_path)
        assert result.returncode == 0, result.stderr[-2000:]
        assert not list(tmp_path.rglob("*.jsonl"))
        json.loads("{}")  # keep the json import honest for the module
