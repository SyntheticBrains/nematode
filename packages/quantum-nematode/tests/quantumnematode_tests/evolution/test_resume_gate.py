"""The resume gate: a pickle checkpoint is loaded only under an explicit opt-in.

Both evolution drivers resume from Python pickles, so loading one executes arbitrary code
from that file. The risk lives in the file's provenance, which the program cannot check,
so the gate is an explicit per-invocation flag rather than a warning the caller reads
past. These tests assert the refusal happens **before** anything is unpickled, which is
the only place a gate is worth having.

A companion smoke test covers the happy path end to end; here the point is that the
refusal is reached without a valid checkpoint existing at all.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    while here.parent != here and not (here / "scripts").is_dir():
        here = here.parent
    return here


_PROJECT_ROOT = _find_project_root()
_SCRIPTS = _PROJECT_ROOT / "scripts"
_CONFIG = _PROJECT_ROOT / "configs" / "evolution" / "mlpppo_foraging_small.yml"
_COEVO_CONFIG = _PROJECT_ROOT / "configs" / "evolution" / "coevolution_pilot_arm_a.yml"


def _boom() -> None:
    """Raise: `pickle.load` calls this if — and only if — the gate let the file through."""
    msg = "the gate let an untrusted checkpoint reach pickle.load"
    raise AssertionError(msg)


class _Exploding:
    """A payload that pickles cleanly and raises when **loaded**.

    A real hostile checkpoint runs code at load time via exactly this mechanism, so this
    stands in for one without doing anything: `__reduce__` records a reference to
    `_boom`, and `pickle.load` calls it.
    """

    def __reduce__(self) -> tuple:
        return (_boom, ())


def _run(script: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Invoke a driver the way a user does, with pytest's fingerprints removed.

    The package decides at import whether it is under test, and `PYTEST_CURRENT_TEST` is
    inherited by a subprocess — so a child launched from a test installs a stream log
    handler that production never installs. Stripping those markers is what makes these
    tests exercise the real path: an earlier draft of the gate logged its refusal, which
    looked correct here and was **silent in production**.
    """
    env = {k: v for k, v in os.environ.items() if k not in {"PYTEST_CURRENT_TEST", "TESTING"}}
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        [sys.executable, str(_SCRIPTS / script), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )


def _hostile_checkpoint(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(_Exploding(), handle)
    return path


class TestSinglePopulationDriver:
    def test_resume_without_the_flag_is_refused(self, tmp_path: Path) -> None:
        checkpoint = _hostile_checkpoint(tmp_path / "session" / "checkpoint.pkl")
        result = _run(
            "run_evolution.py",
            ["--config", str(_CONFIG), "--resume", str(checkpoint), "--log-level", "ERROR"],
        )
        assert result.returncode == 1
        assert "--allow-unsafe-resume" in result.stderr + result.stdout
        # The payload raises on unpickling, so a traceback here means the gate was bypassed.
        assert "the gate let an untrusted checkpoint" not in result.stderr

    def test_the_flag_is_off_by_default(self, tmp_path: Path) -> None:
        # No checkpoint at the path at all: the refusal must still come from the gate,
        # not from a missing-file check, so the gate is ordered before any file access.
        result = _run(
            "run_evolution.py",
            [
                "--config",
                str(_CONFIG),
                "--resume",
                str(tmp_path / "absent.pkl"),
                "--log-level",
                "ERROR",
            ],
        )
        assert result.returncode == 1
        combined = result.stderr + result.stdout
        assert "--allow-unsafe-resume" in combined
        assert "Checkpoint not found" not in combined

    def test_a_run_without_resume_does_not_need_the_flag(self) -> None:
        result = _run("run_evolution.py", ["--help"])
        assert result.returncode == 0
        assert "--allow-unsafe-resume" in result.stdout

    def test_the_refusal_is_visible_without_a_log_handler(self, tmp_path: Path) -> None:
        # The gate runs before `configure_file_logging()`, so a logged refusal reaches a
        # NullHandler and the user gets a bare exit 1. `_run` strips pytest's env markers,
        # so this asserts the production behaviour rather than the test-mode one.
        result = _run(
            "run_evolution.py",
            [
                "--config",
                str(_CONFIG),
                "--resume",
                str(tmp_path / "absent.pkl"),
                "--log-level",
                "ERROR",
            ],
        )
        assert result.returncode == 1
        assert "Refusing to resume" in result.stderr, (
            "the refusal must reach stderr without a configured log handler; "
            f"stderr was {result.stderr!r}"
        )


class TestCoevolutionDriver:
    def test_resume_without_the_flag_is_refused(self, tmp_path: Path) -> None:
        session = tmp_path / "session"
        _hostile_checkpoint(session / "coevolution_rng.pkl")
        result = _run(
            "run_coevolution.py",
            ["--config", str(_COEVO_CONFIG), "--resume", str(session), "--log-level", "ERROR"],
        )
        assert result.returncode == 1
        assert "--allow-unsafe-resume" in result.stderr + result.stdout
        assert "the gate let an untrusted checkpoint" not in result.stderr

    def test_the_flag_is_documented(self) -> None:
        result = _run("run_coevolution.py", ["--help"])
        assert result.returncode == 0
        assert "--allow-unsafe-resume" in result.stdout


class TestTheGateIsNotMerelyAdvisory:
    @pytest.mark.parametrize(
        ("script", "args"),
        [
            ("run_evolution.py", ["--config", "{cfg}", "--resume", "{ckpt}"]),
            ("run_coevolution.py", ["--config", "{coevo}", "--resume", "{session}"]),
        ],
    )
    def test_the_refusal_names_the_risk_not_just_the_flag(
        self,
        tmp_path: Path,
        script: str,
        args: list[str],
    ) -> None:
        session = tmp_path / "session"
        checkpoint = _hostile_checkpoint(session / "checkpoint.pkl")
        filled = [
            a.format(
                cfg=str(_CONFIG),
                coevo=str(_COEVO_CONFIG),
                ckpt=str(checkpoint),
                session=str(session),
            )
            for a in args
        ]
        result = _run(script, [*filled, "--log-level", "ERROR"])
        combined = (result.stderr + result.stdout).lower()
        assert result.returncode == 1
        assert "pickle" in combined
        assert "arbitrary code" in combined
