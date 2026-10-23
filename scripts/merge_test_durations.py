#!/usr/bin/env python
"""Rebuild `.test_durations` from the five CI shards' measured durations.

Each CI shard times its own tests on the runner and uploads them as the artifact
``test-durations-shard-<n>`` (see `.github/workflows/tests.yml`). Every shard writes only the tests it
ran, so their union is the whole suite timed on CI hardware, which is what `pytest-split` has to
balance. A file timed on a laptop mis-weights tests whose speed differs between the two machines,
and the shards drift apart even though the split looks balanced by its own numbers.

Usage::

    dir=$(mktemp -d)
    gh run download <run-id> --pattern 'test-durations-shard-*' --dir "$dir"
    uv run python scripts/merge_test_durations.py "$dir"

Download into a new, empty directory every time: a reused one can still hold an earlier run's
``test-durations-shard-<n>/`` folders. The run must be one whose five shards all passed: a failed
shard uploads nothing, and the script refuses fewer than five files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DURATIONS = REPO / ".test_durations"
SHARDS = 5


def shard_files(directory: Path) -> list[Path]:
    """Return each shard's durations file under ``directory``, as `gh run download` lays them out."""
    return sorted(directory.glob("test-durations-shard-*/.test_durations"))


def merge(files: list[Path]) -> dict[str, float]:
    """Return the union of the shards' durations.

    Raises
    ------
    ValueError
        If there are not exactly ``SHARDS`` files, or a test appears in two shards: the shards
        partition the suite, so either means the files are not from one run.
    """
    if len(files) != SHARDS:
        msg = f"expected {SHARDS} shard files, found {len(files)}: {[str(f) for f in files]}"
        raise ValueError(msg)
    merged: dict[str, float] = {}
    for path in files:
        for test, seconds in json.loads(path.read_text()).items():
            if test in merged:
                msg = f"{test} appears in more than one shard; the files are not from one run"
                raise ValueError(msg)
            merged[test] = float(seconds)
    return merged


def main() -> int:
    """CLI: merge the shard files under a directory into `.test_durations`."""
    parser = argparse.ArgumentParser(description="Rebuild .test_durations from CI shard timings.")
    parser.add_argument("directory", type=Path, help="where `gh run download` put the artifacts")
    parser.add_argument("--out", type=Path, default=DURATIONS)
    args = parser.parse_args()
    try:
        merged = merge(shard_files(args.directory))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    # pytest-split's own layout, plus the trailing newline the end-of-file hook wants.
    args.out.write_text(json.dumps(merged, sort_keys=True, indent=4) + "\n")
    print(f"wrote {len(merged)} durations ({sum(merged.values()):.0f}s in all) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
