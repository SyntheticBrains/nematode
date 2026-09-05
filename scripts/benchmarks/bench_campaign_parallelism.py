#!/usr/bin/env python
"""Measure how campaign wall-clock scales with worker count.

Answers the practical question behind ``--workers``: how many concurrent runs
is this machine actually worth, and where does adding more stop helping?

The sweep reports three numbers per level, because they disagree and the
disagreement is the point:

- **wall** — what you wait for. Usually the number to minimise.
- **speedup** — wall against the sequential baseline.
- **efficiency** — speedup per worker. This falls well before wall-clock
  stops improving, because cores are rarely uniform and memory bandwidth is
  shared. A level with poor efficiency can still be the right choice if the
  machine has nothing else to do; it is the wrong choice if it does.

Wall-clock can get *worse* at high levels, where workers contend with each
other and the OS. The sweep runs high levels last so that shows up plainly.

Usage::

    uv run ./scripts/benchmarks/bench_campaign_parallelism.py --seeds 1-16 --runs 20
    uv run ./scripts/benchmarks/bench_campaign_parallelism.py --workers 1,4,8,16
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_SUMMARY = (__doc__ or "").splitlines()[0]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CAMPAIGN_RUNNER = PROJECT_ROOT / "scripts" / "run_campaign.py"
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "scenarios"
    / "foraging_predator_thermal"
    / "mlpppo_small_continuous2d_combined_klinotaxis.yml"
)


def _default_levels() -> list[int]:
    """Powers of two up to the core count, plus the core count itself."""
    cores = os.cpu_count() or 1
    levels = [level for level in (2, 4, 6, 8, 12, 16, 24, 32) if level <= cores]
    if cores not in levels:
        levels.append(cores)
    return sorted(levels)


def run_campaign(config: Path, seeds: str, runs: int, workers: int, output_dir: Path) -> float:
    """Run one campaign at a given worker count; return wall-clock seconds."""
    command = [
        sys.executable,
        str(CAMPAIGN_RUNNER),
        "--config",
        str(config),
        "--seeds",
        seeds,
        "--runs",
        str(runs),
        "--workers",
        str(workers),
        "--output-dir",
        str(output_dir),
        "--",
        "--theme",
        "headless",
        "--log-level",
        "NONE",
    ]
    started = time.perf_counter()
    result = subprocess.run(  # noqa: S603 — fixed argv, no shell
        command,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-2000:] + result.stderr[-2000:])
        msg = f"campaign at workers={workers} failed with exit {result.returncode}"
        raise RuntimeError(msg)
    return elapsed


def parse_arguments() -> argparse.Namespace:
    """Parse benchmark options."""
    parser = argparse.ArgumentParser(description=_SUMMARY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Config to run.")
    parser.add_argument("--seeds", default="1-16", help="Seed spec for every level (default 1-16).")
    parser.add_argument("--runs", type=int, default=20, help="Episodes per run (default 20).")
    parser.add_argument(
        "--workers",
        default=None,
        help="Comma-separated levels to sweep (default: powers of two up to the core count).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where per-level logs go (default: a temporary directory).",
    )
    return parser.parse_args()


def main() -> int:
    """Run the sweep and print the table."""
    args = parse_arguments()
    if not args.config.is_file():
        sys.stderr.write(f"error: config not found: {args.config}\n")
        return 2

    levels = (
        [int(level) for level in args.workers.replace(",", " ").split()]
        if args.workers
        else _default_levels()
    )
    output_root = args.output_dir or PROJECT_ROOT / "campaigns" / "bench_parallelism"
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"config: {args.config.name}")
    print(f"seeds: {args.seeds} | episodes per run: {args.runs} | cores: {os.cpu_count()}\n")

    print("measuring sequential baseline (workers=1)...", flush=True)
    baseline = run_campaign(args.config, args.seeds, args.runs, 1, output_root / "workers-1")
    print(f"baseline: {baseline:.1f}s\n", flush=True)

    print(f"{'workers':>8}{'wall (s)':>12}{'speedup':>10}{'efficiency':>13}")
    print("-" * 43)
    print(f"{1:>8}{baseline:>12.1f}{1.0:>9.2f}x{'100%':>13}")

    for workers in levels:
        if workers == 1:
            continue
        elapsed = run_campaign(
            args.config,
            args.seeds,
            args.runs,
            workers,
            output_root / f"workers-{workers}",
        )
        speedup = baseline / elapsed
        print(
            f"{workers:>8}{elapsed:>12.1f}{speedup:>9.2f}x{speedup / workers * 100:>12.0f}%",
            flush=True,
        )

    print(
        "\nPick the level by what the machine is for: lowest wall-clock if it is dedicated "
        "to the campaign, a higher-efficiency level if you need it for anything else.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
