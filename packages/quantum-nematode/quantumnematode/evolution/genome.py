"""Genome dataclass and deterministic ID helper for the evolution framework.

The :class:`Genome` is the unit of variation in the evolution loop: a flat
parameter vector plus the metadata needed for lineage tracking and round-trip
encoding.  IDs are derived deterministically from ``(generation, index,
parent_ids)`` via :func:`genome_id_for` so two runs with identical population
structure produce identical IDs — useful for checkpoint resume and for
cross-referencing logs.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


@dataclass
class Genome:
    """A flat parameter vector plus lineage metadata.

    Parameters
    ----------
    params
        Flat parameter array.  Shape and dtype are encoder-specific; concrete
        encoders document their layout in ``birth_metadata``.
    genome_id
        Deterministic UUID derived from ``generation``, ``index``, and
        ``parent_ids`` via :func:`genome_id_for`.
    parent_ids
        IDs of the genomes from generation ``N-1`` that contributed to this
        candidate.  See ``Decision 5a`` in the change's ``design.md``: the
        framework records every member of generation ``N-1`` as a parent of
        every member of generation ``N`` (uniform across CMA-ES and GA).
        Empty for ``generation == 0``.
    generation
        Zero-based generation index.  A run with ``generations: G`` populates
        rows for generations ``0, 1, ..., G-1``.
    birth_metadata
        Encoder-defined metadata such as the shape map needed to unflatten
        ``params`` back into per-component weight tensors.
    """

    params: np.ndarray
    genome_id: str
    parent_ids: list[str]
    generation: int
    birth_metadata: dict[str, Any] = field(default_factory=dict)


def genome_id_for(
    generation: int,
    index: int,
    parent_ids: list[str],
) -> str:
    """Derive a deterministic UUID for a genome.

    Identical inputs produce identical IDs; any input change produces a
    different ID.  Uses ``uuid.uuid5`` with the OID namespace so the result
    is a valid RFC 4122 UUID.

    Parameters
    ----------
    generation
        Zero-based generation index.
    index
        Position of this genome within its generation's population.
    parent_ids
        IDs of prior-generation genomes.  Sorted before hashing so that
        order-of-iteration on the parent list does not affect the result.

    Returns
    -------
    str
        A UUID string.

    Examples
    --------
    >>> a = genome_id_for(0, 0, [])
    >>> b = genome_id_for(0, 0, [])
    >>> a == b
    True
    >>> c = genome_id_for(0, 1, [])
    >>> a == c
    False
    """
    name = f"gen{generation}-idx{index}-parents{','.join(sorted(parent_ids))}"
    return str(uuid.uuid5(uuid.NAMESPACE_OID, name))
