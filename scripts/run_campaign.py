#!/usr/bin/env python
r"""Run a campaign of simulations — configs x seeds — concurrently.

Paired-seed protocols are embarrassingly parallel: the runs are independent
processes with independent RNG streams writing to independent session
directories. Running them one at a time leaves most of a multi-core machine
idle for the whole campaign.

Each run is executed as a **subprocess of the single-run entry point**, given
byte-for-byte the command line a person would type by hand. That is the point
of the design rather than an implementation detail: it means a campaign
changes only *when* runs happen, never *what* they compute, so results stay
comparable with everything measured before. Nothing here imports or
reimplements any simulation, brain, environment, or learning-rule code.

Examples
--------
Four architectures across eight seeds, tracked::

    uv run ./scripts/run_campaign.py \
        --config configs/scenarios/foraging_predator_thermal/mlpppo_small_continuous2d_combined_klinotaxis.yml \
        --config configs/scenarios/foraging_predator_thermal/cfcppo_small_continuous2d_combined_klinotaxis.yml \
        --seeds 1-8 --runs 3000 -- --track-experiment

Preview without executing::

    uv run ./scripts/run_campaign.py --config <cfg> --seeds 1-4 --dry-run
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNNER = PROJECT_ROOT / "scripts" / "run_simulation.py"

# Reserve two cores so the machine stays usable while a campaign occupies it,
# and so workers are not competing with the OS for the last core.
_CORES_RESERVED = 2

# One thread per numerical library per worker, instead of each worker opening a
# full pool and oversubscribing the machine. Safe because results are
# bit-identical across thread counts at the tensor sizes used here — pinned by
# the thread-invariance test in the suite, which will fail if that ever stops
# holding.
_SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def default_workers() -> int:
    """Worker count that uses the machine without monopolising it."""
    return max(1, (os.cpu_count() or 1) - _CORES_RESERVED)


@dataclass(frozen=True)
class Run:
    """One planned (config, seed) execution."""

    config: Path
    seed: int

    @property
    def label(self) -> str:
        """Stable identifier used for log filenames and the summary table."""
        return f"{self.config.stem}-seed{self.seed}"


@dataclass
class RunResult:
    """Outcome of one executed run."""

    run: Run
    returncode: int
    seconds: float
    log_path: Path

    @property
    def ok(self) -> bool:
        """Whether the child exited cleanly."""
        return self.returncode == 0


def parse_seeds(spec: str) -> list[int]:
    """Expand a seed specification into an ordered, de-duplicated list.

    Accepts inclusive ranges (``1-8``), comma or whitespace separated values
    (``1,3,5`` / ``1 3 5``), and any mixture (``1-4,9``).

    Raises
    ------
    ValueError
        If a token is not an integer or an ascending integer range.
    """
    seeds: list[int] = []
    for raw in spec.replace(",", " ").split():
        token = raw.strip()
        if not token:
            continue
        if "-" in token.lstrip("-"):
            start_text, _, end_text = token.partition("-")
            try:
                start, end = int(start_text), int(end_text)
            except ValueError as exc:
                msg = f"Invalid seed range {token!r}; expected 'START-END' with integers."
                raise ValueError(msg) from exc
            if end < start:
                msg = f"Invalid seed range {token!r}; end {end} is below start {start}."
                raise ValueError(msg)
            seeds.extend(range(start, end + 1))
        else:
            try:
                seeds.append(int(token))
            except ValueError as exc:
                msg = f"Invalid seed {token!r}; expected an integer."
                raise ValueError(msg) from exc
    if not seeds:
        msg = f"No seeds parsed from {spec!r}."
        raise ValueError(msg)
    # De-duplicate while preserving order: a repeated seed would collide in the
    # log filenames and add nothing to a paired-seed design.
    return list(dict.fromkeys(seeds))


def plan_runs(configs: list[Path], seeds: list[int]) -> list[Run]:
    """Cross product of configs and seeds, config-major.

    Config-major ordering means an interrupted campaign has completed whole
    arms rather than a ragged slice of every arm.
    """
    return [Run(config=config, seed=seed) for config in configs for seed in seeds]


def build_command(run: Run, runner: Path, runs: int | None, passthrough: list[str]) -> list[str]:
    """Build the child command line for one run."""
    command = [sys.executable, str(runner), "--config", str(run.config), "--seed", str(run.seed)]
    if runs is not None:
        command += ["--runs", str(runs)]
    return command + passthrough


class CampaignExecutor:
    """Runs a plan with bounded concurrency, terminating children on interrupt."""

    def __init__(self, log_dir: Path, workers: int) -> None:
        self.log_dir = log_dir
        self.workers = workers
        self._processes: set[subprocess.Popen[bytes]] = set()
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self._completed = 0

    def cancel(self) -> None:
        """Stop launching new runs and terminate everything in flight."""
        self._cancelled.set()
        with self._lock:
            processes = list(self._processes)
        for process in processes:
            if process.poll() is None:
                process.terminate()

    def _execute(self, run: Run, command: list[str], total: int) -> RunResult:
        log_path = self.log_dir / f"{run.label}.log"
        if self._cancelled.is_set():
            return RunResult(run=run, returncode=130, seconds=0.0, log_path=log_path)

        env = {**os.environ, **_SINGLE_THREAD_ENV}
        started = time.perf_counter()
        with log_path.open("wb") as log_file:
            process = subprocess.Popen(  # noqa: S603 — command is built from our own argv
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT,
                env=env,
            )
            with self._lock:
                self._processes.add(process)
            try:
                returncode = process.wait()
            finally:
                with self._lock:
                    self._processes.discard(process)
        elapsed = time.perf_counter() - started

        with self._lock:
            self._completed += 1
            done = self._completed
        status = "ok" if returncode == 0 else f"FAILED (exit {returncode})"
        print(f"[{done}/{total}] {run.label}: {status} in {elapsed:.1f}s", flush=True)

        return RunResult(run=run, returncode=returncode, seconds=elapsed, log_path=log_path)

    def run_all(self, plan: list[Run], commands: list[list[str]]) -> list[RunResult]:
        """Execute every planned run, returning results in plan order."""
        total = len(plan)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [
                pool.submit(self._execute, run, command, total)
                for run, command in zip(plan, commands, strict=True)
            ]
            return [future.result() for future in futures]


def _print_summary(results: list[RunResult], wall_seconds: float) -> None:
    """Print the per-run status table and campaign totals."""
    failed = [result for result in results if not result.ok]
    serial_seconds = sum(result.seconds for result in results)

    # Size the label column to the longest run so nothing wraps or misaligns.
    label_width = max((len(result.run.label) for result in results), default=3) + 2
    table_width = label_width + 22

    print("\n" + "=" * table_width, flush=True)
    print(f"{'run':<{label_width}}{'status':>10}{'seconds':>12}", flush=True)
    print("-" * table_width, flush=True)
    for result in results:
        status = "ok" if result.ok else f"exit {result.returncode}"
        print(
            f"{result.run.label:<{label_width}}{status:>10}{result.seconds:>12.1f}",
            flush=True,
        )
    print("-" * table_width, flush=True)

    summary = f"{len(results) - len(failed)}/{len(results)} succeeded in {wall_seconds:.1f}s"
    if wall_seconds > 0 and serial_seconds > 0:
        summary += (
            f" (sum of run times {serial_seconds:.1f}s — {serial_seconds / wall_seconds:.1f}x)"
        )
    print(summary, flush=True)
    if failed:
        print(f"failed runs: {', '.join(result.run.label for result in failed)}", flush=True)
        print(f"logs: {failed[0].log_path.parent}", flush=True)


def split_passthrough(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split argv at the first bare ``--``; everything after goes to children."""
    if "--" in argv:
        index = argv.index("--")
        return argv[:index], argv[index + 1 :]
    return argv, []


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    """Parse the campaign runner's own arguments."""
    parser = argparse.ArgumentParser(
        description="Run configs x seeds concurrently as isolated simulation subprocesses.",
        epilog="Arguments after a bare '--' are passed to every run unchanged.",
    )
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        type=Path,
        help="Config to run; repeat for multiple arms (crossed with every seed).",
    )
    parser.add_argument(
        "--seeds",
        required=True,
        help="Seeds as a range, list, or mixture: '1-8', '1,3,5', '1-4,9'.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Episodes per run, forwarded as --runs (default: the run script's own default).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help=(
            "Maximum concurrent runs "
            f"(default: {default_workers()} — CPU count minus {_CORES_RESERVED})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for per-run logs (default: campaigns/<timestamp>/).",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=DEFAULT_RUNNER,
        help="Entry point to invoke per run (default: scripts/run_simulation.py).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands and exit without starting anything.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""
    own_argv, passthrough = split_passthrough(
        list(sys.argv[1:]) if argv is None else list(argv),
    )
    args = parse_arguments(own_argv)

    try:
        seeds = parse_seeds(args.seeds)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    missing = [config for config in args.config if not config.is_file()]
    if missing:
        print(
            f"error: config not found: {', '.join(str(path) for path in missing)}",
            file=sys.stderr,
        )
        return 2

    if args.workers < 1:
        print(f"error: --workers must be at least 1, got {args.workers}.", file=sys.stderr)
        return 2

    plan = plan_runs(args.config, seeds)
    commands = [build_command(run, args.runner, args.runs, passthrough) for run in plan]

    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return 0

    output_dir = args.output_dir or PROJECT_ROOT / "campaigns" / time.strftime("%Y%m%d_%H%M%S")
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    workers = min(args.workers, len(plan))
    print(
        f"Campaign: {len(args.config)} config(s) x {len(seeds)} seed(s) = {len(plan)} runs, "
        f"{workers} workers\nLogs: {log_dir}",
        flush=True,
    )

    executor = CampaignExecutor(log_dir=log_dir, workers=workers)

    def _handle_interrupt(_signum: int, _frame: object) -> None:
        print("\nInterrupted — terminating running simulations...", file=sys.stderr)
        executor.cancel()

    previous_handler = signal.signal(signal.SIGINT, _handle_interrupt)
    started = time.perf_counter()
    try:
        results = executor.run_all(plan, commands)
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    wall_seconds = time.perf_counter() - started

    _print_summary(results, wall_seconds)
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
