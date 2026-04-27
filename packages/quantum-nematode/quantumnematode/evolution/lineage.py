"""Lineage CSV writer for the evolution loop.

The :class:`LineageTracker` records every fitness evaluation as one row in a
CSV file at ``evolution_results/<session_id>/lineage.csv``.  The file is
append-mode so resume operations continue writing to the same file without
truncating prior history.

Generation indexing is 0-based: a run with ``generations: G`` populates rows
for generations ``0, 1, ..., G-1``.  Every member of generation ``N-1`` is
recorded as a parent of every member of generation ``N`` (uniform across
CMA-ES and GA, since neither optimiser exposes per-child parent provenance
to the loop).  Generation 0 rows have empty ``parent_ids``.

The tracker is owned by the **parent process only** — workers report
fitnesses back to the parent and the parent calls :meth:`record`.  Workers
MUST NOT instantiate or write to the tracker directly (no concurrent-write
hazard).
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from quantumnematode.evolution.genome import Genome


CSV_HEADER: tuple[str, ...] = (
    "generation",
    "child_id",
    "parent_ids",
    "fitness",
    "brain_type",
)


class LineageTracker:
    """Append-only CSV writer for parent → child lineage records.

    Parameters
    ----------
    output_path
        Path to the CSV file.  Parent directory must exist.  If the file
        does not exist it is created with a header row; if it does, no
        header is written (so resume continues seamlessly).
    """

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self._write_header_if_needed()

    def _write_header_if_needed(self) -> None:
        """Write the CSV header if the file does not already exist.

        Idempotent: if the file exists with content (header or data), this
        is a no-op.  This is the mechanism by which resume preserves a
        single-header invariant.
        """
        if self.output_path.exists() and self.output_path.stat().st_size > 0:
            return
        with self.output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_HEADER)

    def record(self, genome: Genome, fitness: float, brain_type: str) -> None:
        """Append one row for a fitness evaluation.

        Parameters
        ----------
        genome
            The evaluated genome.  ``genome.genome_id``, ``genome.parent_ids``,
            and ``genome.generation`` are written to the row.
        fitness
            The fitness value (typically in ``[0.0, 1.0]`` for
            :class:`EpisodicSuccessRate`).
        brain_type
            Brain identifier (e.g. ``"mlpppo"``, ``"lstmppo"``).  Included
            so future co-evolution runs (where two brain populations share
            an output directory) can slice the CSV by species.
        """
        with self.output_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                (
                    genome.generation,
                    genome.genome_id,
                    ";".join(genome.parent_ids),
                    f"{fitness:.6f}",
                    brain_type,
                ),
            )
