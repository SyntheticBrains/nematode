"""Fixtures for the benchmark tests: overriding a reference file, and the validity record."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from wormbench import data as wdata
from wormbench import grade as wgrade

if TYPE_CHECKING:
    from collections.abc import Callable

_REPOSITORY_DATA = Path(__file__).resolve().parents[5] / "data"


@pytest.fixture
def benchmark_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return a directory whose files override the repository's ``data/benchmark/`` ones.

    The override directory is the host data root, and the repository's ``data/`` is the next
    source, so every file not written here still resolves to the vendored one. Both are restored
    after the test.
    """
    root = tmp_path / "data-overrides"
    (root / "benchmark").mkdir(parents=True)
    (root / "roaming_dwelling").mkdir(parents=True)
    monkeypatch.setattr(wdata, "_root", root)
    monkeypatch.setenv(wdata.ENVIRONMENT_VARIABLE, str(_REPOSITORY_DATA))
    return root / "benchmark"


@pytest.fixture
def validity_record(monkeypatch: pytest.MonkeyPatch) -> Callable[[dict[str, Any]], None]:
    """Return a setter making the grading read a given validity record, not the shipped one."""
    shipped = wdata.package_record

    def use(record: dict[str, Any]) -> None:
        monkeypatch.setattr(
            wdata,
            "package_record",
            lambda name: record if name == wgrade.VALIDITY_RECORD else shipped(name),
        )

    return use
