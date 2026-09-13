"""Fail when `.test_durations` has drifted too far behind the test suite.

CI splits the suite into five shards with ``pytest-split``, balanced by the committed
``.test_durations`` file. A test missing from that file is **not** placed without an estimate: it is
assigned the *mean* of the recorded durations. That default is what makes drift invisible and
expensive at the same time.

The failure mode this guard exists for, measured on 2026-09-13 with a file three weeks old:

* 1,376 of 5,520 collected tests had no recorded duration — 25% of the suite;
* those tests accounted for **679s of 1,127s**, and each was being budgeted at the file-wide mean of
  0.096s against a true mean of 0.494s;
* one of them ran for **100.7s**, 8.9% of the whole suite, budgeted at 0.096s;
* the shards' true loads were 135s, 172s, 184s, 319s and 317s — a **2.36x** spread.

**Predicted balance is not a usable signal for this.** ``pytest-split`` reported those same shards as
balanced to within 1.03x, because it scored its own split with the stale numbers that produced it. The
only signal available without re-timing the suite is how much of the suite the file still describes,
which is what this checks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DURATIONS = REPO / ".test_durations"
# The suite this balances is the one CI shards.
MARKER = "not nightly"
# Generous: a PR adding a whole new test module moves this by ~2 points, so the gate fires on
# accumulated drift rather than on any one contribution. The 2026-09-13 drift reached 25%.
MAX_UNKNOWN_FRACTION = 0.10
REFRESH = (
    'uv run pytest -m "not nightly" --store-durations --clean-durations '
    "--durations-path .test_durations"
)


def collect() -> list[str]:
    """Every test id CI's shards would collect, in collection order."""
    result = subprocess.run(  # noqa: S603 — fixed argv, no shell
        [
            # The running interpreter, so nothing is assumed about what is on PATH.
            sys.executable,
            "-m",
            "pytest",
            "-m",
            MARKER,
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "--no-cov",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stdout[-4000:], file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        msg = f"collection failed with exit code {result.returncode}"
        raise RuntimeError(msg)
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if "::" in line and not line.startswith(" ")
    ]


def main(argv: list[str] | None = None) -> int:
    """Report how much of the suite `.test_durations` still describes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-unknown-fraction",
        type=float,
        default=MAX_UNKNOWN_FRACTION,
        help=f"fail above this share of collected tests having no duration (default {MAX_UNKNOWN_FRACTION})",
    )
    args = parser.parse_args(argv)

    if not DURATIONS.is_file():
        print(f"::error::{DURATIONS.name} is missing; regenerate it with:\n  {REFRESH}")
        return 1

    durations: dict[str, float] = json.loads(DURATIONS.read_text())
    collected = collect()
    known = [t for t in collected if t in durations]
    unknown = len(collected) - len(known)
    fraction = unknown / len(collected) if collected else 0.0
    # Entries for tests that no longer exist. Harmless -- pytest-split filters them before taking the
    # mean -- so this is reported and never fails.
    orphaned = len(durations) - len(known)

    print(
        f"{len(collected)} tests collected under -m '{MARKER}'; "
        f"{len(known)} have a recorded duration, {unknown} do not ({fraction:.1%})",
    )
    if orphaned > 0:
        print(f"{orphaned} entries in {DURATIONS.name} no longer match a collected test")

    if fraction > args.max_unknown_fraction:
        print(
            f"::error::{fraction:.1%} of the suite has no recorded duration, above the "
            f"{args.max_unknown_fraction:.0%} limit. pytest-split budgets each of those tests at the "
            f"mean of the recorded ones, so the shards will be balanced on numbers that no longer "
            f"describe the suite. Regenerate with:\n  {REFRESH}",
        )
        return 1
    print(f"Within the {args.max_unknown_fraction:.0%} limit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
