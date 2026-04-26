"""Tests for :mod:`quantumnematode.evolution.genome`."""

from __future__ import annotations

import uuid

from quantumnematode.evolution.genome import genome_id_for


def test_genome_id_deterministic() -> None:
    """Same inputs SHALL produce the same UUID."""
    a = genome_id_for(generation=3, index=2, parent_ids=["p1", "p2"])
    b = genome_id_for(generation=3, index=2, parent_ids=["p1", "p2"])
    assert a == b


def test_genome_id_distinct_for_distinct_inputs() -> None:
    """Any input change SHALL produce a different UUID.

    Note: under Decision 5a, all children in a generation share the same
    ``parent_ids`` list, so within-generation ``parent_ids`` variation is
    rare in practice — but the helper itself remains general for M3
    Lamarckian inheritance which may pass per-child parent IDs.
    """
    base = genome_id_for(generation=0, index=0, parent_ids=[])

    # Generation differs
    assert base != genome_id_for(generation=1, index=0, parent_ids=[])

    # Index differs
    assert base != genome_id_for(generation=0, index=1, parent_ids=[])

    # parent_ids differs
    assert base != genome_id_for(generation=0, index=0, parent_ids=["p1"])


def test_genome_id_format_is_uuid() -> None:
    """The returned string SHALL parse as a valid UUID."""
    result = genome_id_for(generation=0, index=0, parent_ids=[])
    parsed = uuid.UUID(result)
    assert str(parsed) == result


def test_genome_id_parent_ids_order_independent() -> None:
    """Reordering ``parent_ids`` SHALL NOT change the result.

    The helper sorts ``parent_ids`` internally so that order-of-iteration
    quirks (e.g. set iteration order, dict-key ordering) don't produce
    different IDs for what is conceptually the same parent set.
    """
    a = genome_id_for(generation=1, index=0, parent_ids=["p1", "p2", "p3"])
    b = genome_id_for(generation=1, index=0, parent_ids=["p3", "p1", "p2"])
    assert a == b
