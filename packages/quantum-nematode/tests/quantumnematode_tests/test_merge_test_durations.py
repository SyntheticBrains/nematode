"""``scripts/merge_test_durations.py``: the union of the five CI shards' measured durations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
import merge_test_durations as mtd  # pyright: ignore[reportMissingImports]


def _shards(tmp_path: Path, contents: list[dict[str, float]]) -> Path:
    for n, durations in enumerate(contents, start=1):
        shard = tmp_path / f"test-durations-shard-{n}"
        shard.mkdir()
        (shard / ".test_durations").write_text(json.dumps(durations))
    return tmp_path


def test_the_five_shards_are_merged(tmp_path: Path) -> None:
    """Each shard's tests appear once, with the shard's timing."""
    directory = _shards(tmp_path, [{f"t{n}": float(n)} for n in range(1, 6)])
    merged = mtd.merge(mtd.shard_files(directory))
    assert merged == {"t1": 1.0, "t2": 2.0, "t3": 3.0, "t4": 4.0, "t5": 5.0}


def test_fewer_than_five_shards_are_refused(tmp_path: Path) -> None:
    """A failed shard uploads nothing, so a run missing one is not used."""
    directory = _shards(tmp_path, [{f"t{n}": 1.0} for n in range(1, 5)])
    with pytest.raises(ValueError, match="expected 5 shard files, found 4"):
        mtd.merge(mtd.shard_files(directory))


def test_a_test_in_two_shards_is_refused(tmp_path: Path) -> None:
    """The shards partition the suite; a repeated test means mixed runs."""
    directory = _shards(tmp_path, [{"t": 1.0}] + [{f"t{n}": 1.0} for n in range(2, 6)])
    (directory / "test-durations-shard-2" / ".test_durations").write_text(json.dumps({"t": 2.0}))
    with pytest.raises(ValueError, match="more than one shard"):
        mtd.merge(mtd.shard_files(directory))
