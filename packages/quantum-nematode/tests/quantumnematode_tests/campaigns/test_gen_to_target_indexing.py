"""Regression tests for #277 — gen-to-target was off by one in m3/m4.

``history.csv`` is written **1-indexed**: its first data row is ``generation,1``.
Both aggregators added 1 to that column on the belief it was 0-indexed, inflating
every reported gen-to-target and every derived mean.

The error did not cancel between arms, because the never-reached fallback
(``max_gens + 1``) is not shifted while the reached values are — so the published
speed-gate margins were wrong too, not just the absolute figures.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
CAMPAIGNS = PROJECT_ROOT / "scripts" / "campaigns"


def _load(name: str) -> Any:
    """Load an aggregator by path; they are scripts, not importable modules."""
    spec = importlib.util.spec_from_file_location(name, CAMPAIGNS / f"{name}.py")
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        msg = f"could not load {name}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _history(rows: list[tuple[int, float]]) -> list[dict[str, float]]:
    """Build a history in the on-disk shape: ``generation`` starting at 1."""
    return [{"generation": float(g), "best_fitness": f} for g, f in rows]


@pytest.mark.parametrize("aggregator", ["aggregate_m3_pilot", "aggregate_m4_pilot"])
class TestGenToTargetIsOneIndexed:
    """The returned generation must be the CSV's own label, unmodified."""

    def test_target_reached_on_the_first_row_reports_one(self, aggregator: str) -> None:
        """The tightest case: hitting the target immediately is generation 1, not 2."""
        module = _load(aggregator)
        history = _history([(1, 0.95), (2, 0.99)])

        assert module._gen_first_reaches_target(history, 0.92) == 1

    def test_returns_the_csv_generation_label_verbatim(self, aggregator: str) -> None:
        module = _load(aggregator)
        history = _history([(1, 0.10), (2, 0.50), (3, 0.93), (4, 0.99)])

        assert module._gen_first_reaches_target(history, 0.92) == 3

    def test_never_reached_returns_none(self, aggregator: str) -> None:
        module = _load(aggregator)
        history = _history([(1, 0.1), (2, 0.2), (3, 0.3)])

        assert module._gen_first_reaches_target(history, 0.92) is None

    def test_last_generation_is_distinguishable_from_never_reached(
        self,
        aggregator: str,
    ) -> None:
        """The knock-on effect of the ``+ 1``, and why it mattered.

        With the off-by-one, a seed converging on the final generation ``G``
        reported ``G + 1`` — exactly the ``max_gens + 1`` sentinel used for
        "never reached", making the two indistinguishable in the CSV and in the
        mean.
        """
        module = _load(aggregator)
        n_gens = 5
        reached_last = _history([(g, 0.99 if g == n_gens else 0.1) for g in range(1, n_gens + 1)])
        never = _history([(g, 0.1) for g in range(1, n_gens + 1)])

        got = module._gen_first_reaches_target(reached_last, 0.92)
        sentinel = n_gens + 1

        assert got == n_gens
        assert got != sentinel, (
            "a seed reaching the target on the final generation is reported as the "
            "never-reached sentinel — the two are indistinguishable (see #277)"
        )
        assert module._gen_first_reaches_target(never, 0.92) is None


class TestAgainstCommittedData:
    """Cross-check against the real logbook data the published figures came from."""

    def test_matches_ground_truth_on_the_m3_lamarckian_arm(self) -> None:
        module = _load("aggregate_m3_pilot")
        root = PROJECT_ROOT / "artifacts/logbooks/013/m3_lamarckian_pilot/lamarckian"
        if not root.is_dir():  # pragma: no cover - data not vendored
            pytest.skip("logbook 013 pilot data not present")

        got = []
        for seed in (42, 43, 44, 45):
            with (root / f"seed-{seed}" / "history.csv").open() as handle:
                rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(handle)]
            got.append(module._gen_first_reaches_target(rows, 0.92))

        # Verified directly against the CSVs; the published figures were [3, 4, 4, 7].
        assert got == [2, 3, 3, 6]
