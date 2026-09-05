"""Tests for the parallel campaign runner.

``scripts/`` is not an importable package, so the runner is loaded by path —
the pattern the other campaign-script tests use.

Every test here drives a **stub child script** rather than the real
simulation, so the suite stays fast and hermetic. The one test that proves a
campaign-launched run matches a hand-launched run needs real simulations and
lives in ``test_campaign_run_equivalence.py``, marked ``slow``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

RUNNER_PATH = Path(__file__).resolve().parents[4].parent / "scripts" / "run_campaign.py"


@pytest.fixture(scope="module")
def campaign() -> ModuleType:
    """Load ``scripts/run_campaign.py`` by path."""
    spec = importlib.util.spec_from_file_location("run_campaign", RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before executing: the runner's dataclasses resolve their
    # `from __future__` annotations through sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def stub_runner(tmp_path: Path) -> Path:
    """Build a child script that records when it ran, then exits.

    Exits non-zero when handed ``--fail``, so failure handling can be driven
    deterministically.
    """
    script = tmp_path / "stub_runner.py"
    script.write_text(
        "import argparse, json, os, sys, time\n"
        "start = time.perf_counter()\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--config')\n"
        "parser.add_argument('--seed')\n"
        "parser.add_argument('--runs', default=None)\n"
        "parser.add_argument('--fail', action='store_true')\n"
        "parser.add_argument('--marker', default=None)\n"
        "args, extra = parser.parse_known_args()\n"
        "time.sleep(0.25)\n"
        "record = {\n"
        "    'config': args.config, 'seed': args.seed, 'runs': args.runs,\n"
        "    'marker': args.marker, 'extra': extra,\n"
        "    'start': start, 'end': time.perf_counter(),\n"
        "    'omp': os.environ.get('OMP_NUM_THREADS'),\n"
        "}\n"
        "out = os.environ['STUB_RECORD_DIR']\n"
        "with open(os.path.join(out, f'run-{args.seed}.json'), 'w') as handle:\n"
        "    json.dump(record, handle)\n"
        "sys.exit(3 if args.fail else 0)\n",
        encoding="utf-8",
    )
    return script


@pytest.fixture
def configs(tmp_path: Path) -> list[Path]:
    """Two config files that exist on disk (contents are never read here)."""
    paths = []
    for name in ("arm_a", "arm_b"):
        path = tmp_path / f"{name}.yml"
        path.write_text("max_steps: 1\n", encoding="utf-8")
        paths.append(path)
    return paths


def _records(record_dir: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(record_dir.glob("run-*.json"))]


class TestSeedParsing:
    """Seed specifications expand to ordered, de-duplicated integers."""

    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            ("1-4", [1, 2, 3, 4]),
            ("1,3,5", [1, 3, 5]),
            ("1 3 5", [1, 3, 5]),
            ("1-3,9", [1, 2, 3, 9]),
            ("7", [7]),
            ("42-42", [42]),
        ],
    )
    def test_valid_specs(self, campaign: ModuleType, spec: str, expected: list[int]) -> None:
        assert campaign.parse_seeds(spec) == expected

    def test_duplicates_collapse_preserving_order(self, campaign: ModuleType) -> None:
        """A repeated seed would collide in log filenames and add nothing."""
        assert campaign.parse_seeds("5,1,5,2,1") == [5, 1, 2]

    @pytest.mark.parametrize("spec", ["", "abc", "1-x", "4-2", "1,,x"])
    def test_malformed_specs_raise(self, campaign: ModuleType, spec: str) -> None:
        with pytest.raises(ValueError, match=r"seed|Invalid|No seeds"):
            campaign.parse_seeds(spec)


class TestPlanning:
    """Configs x seeds, config-major."""

    def test_cross_product(self, campaign: ModuleType, configs: list[Path]) -> None:
        plan = campaign.plan_runs(configs, [1, 2, 3, 4])
        assert len(plan) == 8
        assert {(run.config, run.seed) for run in plan} == {
            (config, seed) for config in configs for seed in (1, 2, 3, 4)
        }

    def test_config_major_ordering(self, campaign: ModuleType, configs: list[Path]) -> None:
        """An interrupted campaign leaves whole arms done, not a ragged slice."""
        plan = campaign.plan_runs(configs, [1, 2])
        assert [run.config for run in plan] == [configs[0], configs[0], configs[1], configs[1]]

    def test_labels_are_unique(self, campaign: ModuleType, configs: list[Path]) -> None:
        plan = campaign.plan_runs(configs, [1, 2])
        assert len({run.label for run in plan}) == len(plan)


class TestLabelDisambiguation:
    """Config basenames are not unique across scenario directories."""

    def test_same_basename_different_directories_get_distinct_labels(
        self,
        campaign: ModuleType,
        tmp_path: Path,
    ) -> None:
        """Otherwise both runs of a seed share a log file and one is lost."""
        first_dir = tmp_path / "scenario_a"
        second_dir = tmp_path / "scenario_b"
        first_dir.mkdir()
        second_dir.mkdir()
        first = first_dir / "mlpppo_small_oracle.yml"
        second = second_dir / "mlpppo_small_oracle.yml"
        for path in (first, second):
            path.write_text("max_steps: 1\n", encoding="utf-8")

        plan = campaign.plan_runs([first, second], [1, 2])

        labels = [run.label for run in plan]
        assert len(set(labels)) == len(plan)

    def test_unique_basenames_keep_plain_labels(
        self,
        campaign: ModuleType,
        configs: list[Path],
    ) -> None:
        """The common case stays readable — no tag where none is needed."""
        plan = campaign.plan_runs(configs, [1])
        assert [run.label for run in plan] == ["arm_a-seed1", "arm_b-seed1"]

    def test_labels_are_stable_across_invocations(
        self,
        campaign: ModuleType,
        tmp_path: Path,
    ) -> None:
        """Log filenames must not move between runs of the same campaign."""
        first_dir = tmp_path / "a"
        second_dir = tmp_path / "b"
        first_dir.mkdir()
        second_dir.mkdir()
        paths = [first_dir / "same.yml", second_dir / "same.yml"]
        for path in paths:
            path.write_text("max_steps: 1\n", encoding="utf-8")

        first_plan = [run.label for run in campaign.plan_runs(paths, [1])]
        second_plan = [run.label for run in campaign.plan_runs(paths, [1])]
        assert first_plan == second_plan

    def test_colliding_runs_write_separate_logs(
        self,
        campaign: ModuleType,
        stub_runner: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        record_dir = tmp_path / "records"
        record_dir.mkdir()
        monkeypatch.setenv("STUB_RECORD_DIR", str(record_dir))
        first_dir = tmp_path / "one"
        second_dir = tmp_path / "two"
        first_dir.mkdir()
        second_dir.mkdir()
        for directory in (first_dir, second_dir):
            (directory / "dup.yml").write_text("max_steps: 1\n", encoding="utf-8")
        output_dir = tmp_path / "out"

        campaign.main(
            [
                "--config",
                str(first_dir / "dup.yml"),
                "--config",
                str(second_dir / "dup.yml"),
                "--seeds",
                "1",
                "--runner",
                str(stub_runner),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert len(list((output_dir / "logs").glob("*.log"))) == 2


class TestCampaignOwnedOptionsAreAuthoritative:
    """Passthrough must not be able to rewrite the plan."""

    @pytest.mark.parametrize(
        "argument",
        ["--seed", "--config", "--runs"],
    )
    def test_conflicting_option_rejected(
        self,
        campaign: ModuleType,
        configs: list[Path],
        argument: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Argparse is last-wins, so a passthrough copy would silently win."""
        exit_code = campaign.main(
            ["--config", str(configs[0]), "--seeds", "1-8", "--", argument, "9"],
        )
        assert exit_code == 2
        assert argument in capsys.readouterr().err

    def test_equals_form_also_rejected(
        self,
        campaign: ModuleType,
        configs: list[Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = campaign.main(
            ["--config", str(configs[0]), "--seeds", "1-8", "--", "--seed=9"],
        )
        assert exit_code == 2
        assert "--seed" in capsys.readouterr().err

    def test_unrelated_passthrough_still_allowed(
        self,
        campaign: ModuleType,
        configs: list[Path],
    ) -> None:
        exit_code = campaign.main(
            [
                "--config",
                str(configs[0]),
                "--seeds",
                "1",
                "--dry-run",
                "--",
                "--theme",
                "headless",
                "--track-experiment",
            ],
        )
        assert exit_code == 0

    def test_detector_reports_each_conflict_once(self, campaign: ModuleType) -> None:
        assert campaign.conflicting_passthrough(["--seed", "1", "--seed=2", "--theme", "x"]) == [
            "--seed",
        ]


class TestCommandConstruction:
    """Each child gets the command a person would type."""

    def test_command_names_config_and_seed(
        self,
        campaign: ModuleType,
        configs: list[Path],
    ) -> None:
        run = campaign.Run(config=configs[0], seed=7)
        command = campaign.build_command(run, Path("runner.py"), 25, [])
        assert command[0] == sys.executable
        assert "--config" in command
        assert str(configs[0]) in command
        assert command[command.index("--seed") + 1] == "7"
        assert command[command.index("--runs") + 1] == "25"

    def test_runs_omitted_when_unset(self, campaign: ModuleType, configs: list[Path]) -> None:
        """Without --runs the child keeps its own default."""
        run = campaign.Run(config=configs[0], seed=1)
        assert "--runs" not in campaign.build_command(run, Path("runner.py"), None, [])

    def test_passthrough_appended(self, campaign: ModuleType, configs: list[Path]) -> None:
        run = campaign.Run(config=configs[0], seed=1)
        command = campaign.build_command(run, Path("runner.py"), None, ["--track-experiment"])
        assert command[-1] == "--track-experiment"


class TestPassthroughSplitting:
    """A bare '--' separates our arguments from the children's."""

    def test_splits_at_first_double_dash(self, campaign: ModuleType) -> None:
        own, passthrough = campaign.split_passthrough(
            ["--seeds", "1-2", "--", "--theme", "headless"],
        )
        assert own == ["--seeds", "1-2"]
        assert passthrough == ["--theme", "headless"]

    def test_absent_double_dash_yields_no_passthrough(self, campaign: ModuleType) -> None:
        own, passthrough = campaign.split_passthrough(["--seeds", "1-2"])
        assert own == ["--seeds", "1-2"]
        assert passthrough == []


class TestDryRun:
    """Dry run plans without executing."""

    def test_starts_no_process(  # noqa: PLR0913
        self,
        campaign: ModuleType,
        configs: list[Path],
        stub_runner: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        record_dir = tmp_path / "records"
        record_dir.mkdir()
        monkeypatch.setenv("STUB_RECORD_DIR", str(record_dir))

        exit_code = campaign.main(
            [
                "--config",
                str(configs[0]),
                "--seeds",
                "1-3",
                "--runner",
                str(stub_runner),
                "--dry-run",
            ],
        )

        assert exit_code == 0
        assert list(record_dir.iterdir()) == []
        assert len(capsys.readouterr().out.strip().splitlines()) == 3


class TestExecution:
    """Real subprocesses, stub child."""

    def test_every_run_executes_with_its_own_seed(
        self,
        campaign: ModuleType,
        configs: list[Path],
        stub_runner: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        record_dir = tmp_path / "records"
        record_dir.mkdir()
        monkeypatch.setenv("STUB_RECORD_DIR", str(record_dir))

        exit_code = campaign.main(
            [
                "--config",
                str(configs[0]),
                "--seeds",
                "1-4",
                "--runner",
                str(stub_runner),
                "--workers",
                "4",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

        assert exit_code == 0
        records = _records(record_dir)
        assert sorted(int(record["seed"]) for record in records) == [1, 2, 3, 4]

    def test_passthrough_reaches_the_child(
        self,
        campaign: ModuleType,
        configs: list[Path],
        stub_runner: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        record_dir = tmp_path / "records"
        record_dir.mkdir()
        monkeypatch.setenv("STUB_RECORD_DIR", str(record_dir))

        campaign.main(
            [
                "--config",
                str(configs[0]),
                "--seeds",
                "1",
                "--runner",
                str(stub_runner),
                "--output-dir",
                str(tmp_path / "out"),
                "--",
                "--marker",
                "sentinel",
                "--extra-flag",
            ],
        )

        record = _records(record_dir)[0]
        assert record["marker"] == "sentinel"
        assert "--extra-flag" in record["extra"]

    def test_children_get_single_threaded_env(
        self,
        campaign: ModuleType,
        configs: list[Path],
        stub_runner: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Workers must not each open a full BLAS pool."""
        record_dir = tmp_path / "records"
        record_dir.mkdir()
        monkeypatch.setenv("STUB_RECORD_DIR", str(record_dir))

        campaign.main(
            [
                "--config",
                str(configs[0]),
                "--seeds",
                "1",
                "--runner",
                str(stub_runner),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

        assert _records(record_dir)[0]["omp"] == "1"

    def test_writes_one_log_per_run(
        self,
        campaign: ModuleType,
        configs: list[Path],
        stub_runner: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        record_dir = tmp_path / "records"
        record_dir.mkdir()
        monkeypatch.setenv("STUB_RECORD_DIR", str(record_dir))
        output_dir = tmp_path / "out"

        campaign.main(
            [
                "--config",
                str(configs[0]),
                "--seeds",
                "1-3",
                "--runner",
                str(stub_runner),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert len(list((output_dir / "logs").glob("*.log"))) == 3


class TestConcurrencyBound:
    """At most `workers` children run at once."""

    def test_worker_limit_respected(
        self,
        campaign: ModuleType,
        configs: list[Path],
        stub_runner: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        record_dir = tmp_path / "records"
        record_dir.mkdir()
        monkeypatch.setenv("STUB_RECORD_DIR", str(record_dir))

        campaign.main(
            [
                "--config",
                str(configs[0]),
                "--seeds",
                "1-6",
                "--runner",
                str(stub_runner),
                "--workers",
                "2",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

        # Children record their own start/end; count the maximum overlap. Each
        # sleeps 0.25s, far longer than process startup jitter.
        records = _records(record_dir)
        assert len(records) == 6
        events = [(record["start"], 1) for record in records]
        events += [(record["end"], -1) for record in records]
        events.sort()
        concurrent = maximum = 0
        for _, delta in events:
            concurrent += delta
            maximum = max(maximum, concurrent)
        assert maximum <= 2


class TestFailureHandling:
    """A failing run is reported without aborting the campaign."""

    def test_failure_sets_exit_code_and_others_still_run(  # noqa: PLR0913
        self,
        campaign: ModuleType,
        configs: list[Path],
        stub_runner: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        record_dir = tmp_path / "records"
        record_dir.mkdir()
        monkeypatch.setenv("STUB_RECORD_DIR", str(record_dir))

        exit_code = campaign.main(
            [
                "--config",
                str(configs[0]),
                "--seeds",
                "1-3",
                "--runner",
                str(stub_runner),
                "--output-dir",
                str(tmp_path / "out"),
                "--",
                "--fail",
            ],
        )

        assert exit_code == 1
        # Every run was still attempted despite all of them failing.
        assert len(_records(record_dir)) == 3
        assert "failed runs:" in capsys.readouterr().out


class TestArgumentValidation:
    """Bad input is rejected before anything is launched."""

    def test_missing_config_reported(
        self,
        campaign: ModuleType,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = campaign.main(
            ["--config", str(tmp_path / "absent.yml"), "--seeds", "1"],
        )
        assert exit_code == 2
        assert "config not found" in capsys.readouterr().err

    def test_bad_seed_spec_reported(
        self,
        campaign: ModuleType,
        configs: list[Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = campaign.main(["--config", str(configs[0]), "--seeds", "nonsense"])
        assert exit_code == 2
        assert "error:" in capsys.readouterr().err

    def test_zero_workers_rejected(
        self,
        campaign: ModuleType,
        configs: list[Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = campaign.main(
            ["--config", str(configs[0]), "--seeds", "1", "--workers", "0"],
        )
        assert exit_code == 2
        assert "workers" in capsys.readouterr().err


class TestDefaultWorkers:
    """The default leaves the machine usable while a campaign runs."""

    def test_reserves_headroom(self, campaign: ModuleType) -> None:
        assert campaign.default_workers() >= 1

    def test_never_exceeds_cpu_count(self, campaign: ModuleType) -> None:
        import os

        assert campaign.default_workers() <= (os.cpu_count() or 1)


class TestInterruptTerminatesChildren:
    """Cancelling stops in-flight children rather than orphaning them."""

    def test_cancel_terminates_running_processes(
        self,
        campaign: ModuleType,
        tmp_path: Path,
    ) -> None:
        import subprocess

        executor = campaign.CampaignExecutor(log_dir=tmp_path, workers=1)
        # A child that would outlive the test if it were never terminated.
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
        )
        executor._processes.add(process)
        try:
            assert process.poll() is None

            executor.cancel()

            # terminate() is asynchronous; wait rather than racing it.
            assert process.wait(timeout=10) is not None
        finally:
            if process.poll() is None:  # pragma: no cover — cleanup only
                process.kill()
                process.wait(timeout=10)

    def test_cancel_stops_further_launches(
        self,
        campaign: ModuleType,
        configs: list[Path],
        stub_runner: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Once cancelled, queued runs are not started."""
        record_dir = tmp_path / "records"
        record_dir.mkdir()
        monkeypatch.setenv("STUB_RECORD_DIR", str(record_dir))

        executor = campaign.CampaignExecutor(log_dir=tmp_path, workers=1)
        executor.cancel()

        plan = campaign.plan_runs(configs[:1], [1, 2, 3])
        commands = [campaign.build_command(run, stub_runner, None, []) for run in plan]
        results = executor.run_all(plan, commands)

        assert all(not result.ok for result in results)
        assert list(record_dir.iterdir()) == []

    def test_cancel_racing_a_launch_still_terminates(
        self,
        campaign: ModuleType,
        configs: list[Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A cancel landing mid-launch must not leave the child running.

        The dangerous interleaving is narrow: a worker passes the pre-launch
        cancellation check, and only then does `cancel` run and snapshot the
        process set — without finding this child, because it has not been
        registered yet. Nothing else would ever terminate it, so the campaign
        would wait on a run the user already interrupted.

        Sleeping until the race happens by chance would be flaky and would
        usually miss, so the window is reproduced exactly: `Popen` is wrapped
        to fire the cancellation *after* the pre-check has passed and *before*
        the process is registered. Only the re-check under the registration
        lock can catch it, so this fails if that re-check is removed.
        """
        import subprocess

        sleeper = tmp_path / "sleeper.py"
        sleeper.write_text("import time; time.sleep(60)\n", encoding="utf-8")

        executor = campaign.CampaignExecutor(log_dir=tmp_path, workers=1)
        real_popen = subprocess.Popen

        def cancelling_popen(*args: object, **kwargs: object) -> object:
            # Stand in for `cancel` having run and snapshotted an empty set.
            executor._cancelled.set()
            return real_popen(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(campaign.subprocess, "Popen", cancelling_popen)

        plan = campaign.plan_runs(configs[:1], [1])
        results = executor.run_all(plan, [[sys.executable, str(sleeper)]])

        # The child sleeps 60s; anything near that means it was never killed.
        assert len(results) == 1
        assert results[0].seconds < 30
        assert not results[0].ok
