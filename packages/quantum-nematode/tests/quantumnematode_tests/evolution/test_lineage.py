"""Tests for :mod:`quantumnematode.evolution.lineage`."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import numpy as np
from quantumnematode.evolution.genome import Genome, genome_id_for
from quantumnematode.evolution.lineage import CSV_HEADER, LineageTracker

if TYPE_CHECKING:
    from pathlib import Path


def _make_genome(generation: int, index: int, parent_ids: list[str]) -> Genome:
    """Build a stub Genome with deterministic id (params content irrelevant)."""
    return Genome(
        params=np.zeros(0, dtype=np.float32),
        genome_id=genome_id_for(generation, index, parent_ids),
        parent_ids=parent_ids,
        generation=generation,
    )


def _read_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


# ---------------------------------------------------------------------------
# CSV append correctness across generations
# ---------------------------------------------------------------------------


def test_lineage_csv_appends_across_generations(tmp_path: Path) -> None:
    """Recording 5 generations x population 4 SHALL yield 20 rows + header.

    Every gen-N child shares the same ``parent_ids`` list (all of gen N-1's
    child_ids) per the framework's parent_ids convention.  Verify that data
    is correctly written.
    """
    csv_path = tmp_path / "lineage.csv"
    tracker = LineageTracker(csv_path)

    population = 4
    prev_ids: list[str] = []
    for gen in range(5):
        gen_ids: list[str] = []
        for idx in range(population):
            genome = _make_genome(gen, idx, prev_ids)
            tracker.record(genome, fitness=0.1 * gen + 0.01 * idx, brain_type="mlpppo")
            gen_ids.append(genome.genome_id)
        prev_ids = gen_ids

    rows = _read_rows(csv_path)
    # Header + 5 x 4 = 21 rows
    assert len(rows) == 21
    assert tuple(rows[0]) == CSV_HEADER

    # Every gen-N row's parent_ids SHALL be the joined set of all gen-(N-1) IDs.
    rows_by_gen: dict[int, list[list[str]]] = {}
    for row in rows[1:]:
        gen = int(row[0])
        rows_by_gen.setdefault(gen, []).append(row)

    for gen in range(1, 5):
        # Pick the parent_ids cell from every row at this generation
        # (cell index 2 = parent_ids), split, sort, deduplicate.
        all_parent_id_cells = {row[2] for row in rows_by_gen[gen]}
        # All children in this generation share the same parent_ids string
        assert len(all_parent_id_cells) == 1
        parent_ids = sorted(next(iter(all_parent_id_cells)).split(";"))
        prev_gen_child_ids = sorted(row[1] for row in rows_by_gen[gen - 1])
        assert parent_ids == prev_gen_child_ids


def test_lineage_csv_gen_zero_has_empty_parent_ids(tmp_path: Path) -> None:
    """Generation 0 rows SHALL have empty ``parent_ids`` field."""
    csv_path = tmp_path / "lineage.csv"
    tracker = LineageTracker(csv_path)

    for idx in range(3):
        genome = _make_genome(0, idx, parent_ids=[])
        tracker.record(genome, fitness=0.0, brain_type="mlpppo")

    rows = _read_rows(csv_path)
    # Skip header; check parent_ids column (index 2).
    for row in rows[1:]:
        assert row[2] == ""


def test_lineage_csv_header_only_written_once(tmp_path: Path) -> None:
    """Reinstantiating the tracker on an existing file SHALL NOT rewrite the header."""
    csv_path = tmp_path / "lineage.csv"

    # First session
    tracker1 = LineageTracker(csv_path)
    tracker1.record(_make_genome(0, 0, []), fitness=0.5, brain_type="mlpppo")

    # Second session — simulating resume
    tracker2 = LineageTracker(csv_path)
    tracker2.record(
        _make_genome(1, 0, [_make_genome(0, 0, []).genome_id]),
        fitness=0.6,
        brain_type="mlpppo",
    )

    rows = _read_rows(csv_path)
    # 1 header + 2 data rows
    assert len(rows) == 3
    assert tuple(rows[0]) == CSV_HEADER
    # Only the first row is the header.
    assert rows[1][0] == "0"
    assert rows[2][0] == "1"


def test_lineage_generation_indexing_is_zero_based(tmp_path: Path) -> None:
    """A run with G generations SHALL produce rows whose generation column is ``[0, G-1]``.

    Each generation value SHALL appear exactly P times (where P is the population size).
    """
    csv_path = tmp_path / "lineage.csv"
    tracker = LineageTracker(csv_path)

    population = 5
    generations = 7
    prev_ids: list[str] = []
    for gen in range(generations):
        gen_ids: list[str] = []
        for idx in range(population):
            genome = _make_genome(gen, idx, prev_ids)
            tracker.record(genome, fitness=0.0, brain_type="mlpppo")
            gen_ids.append(genome.genome_id)
        prev_ids = gen_ids

    rows = _read_rows(csv_path)
    gen_values = [int(row[0]) for row in rows[1:]]
    # Every generation in [0, G-1] appears exactly P times.
    expected: dict[int, int] = dict.fromkeys(range(generations), population)
    counts: dict[int, int] = {}
    for g in gen_values:
        counts[g] = counts.get(g, 0) + 1
    assert counts == expected


# ---------------------------------------------------------------------------
# Header column shape
# ---------------------------------------------------------------------------


def test_csv_header_columns(tmp_path: Path) -> None:
    """Header SHALL be the canonical 6-column schema."""
    csv_path = tmp_path / "lineage.csv"
    LineageTracker(csv_path)
    rows = _read_rows(csv_path)
    assert tuple(rows[0]) == (
        "generation",
        "child_id",
        "parent_ids",
        "fitness",
        "brain_type",
        "inherited_from",
    )


def test_record_writes_brain_type_column(tmp_path: Path) -> None:
    """The ``brain_type`` column SHALL hold the value passed to ``record``.

    This is what enables future co-evolution runs (where two brain populations
    share an output directory) to slice a single CSV by species.
    """
    csv_path = tmp_path / "lineage.csv"
    tracker = LineageTracker(csv_path)

    tracker.record(_make_genome(0, 0, []), fitness=0.5, brain_type="mlpppo")
    tracker.record(_make_genome(0, 1, []), fitness=0.4, brain_type="lstmppo")

    rows = _read_rows(csv_path)
    assert rows[1][4] == "mlpppo"
    assert rows[2][4] == "lstmppo"
