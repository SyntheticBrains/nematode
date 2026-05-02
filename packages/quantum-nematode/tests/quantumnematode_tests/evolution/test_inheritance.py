"""Unit tests for :mod:`quantumnematode.evolution.inheritance`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from quantumnematode.evolution.inheritance import (
    LamarckianInheritance,
    NoInheritance,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# NoInheritance
# ---------------------------------------------------------------------------


def test_no_inheritance_returns_empty_parents() -> None:
    """``NoInheritance.select_parents`` SHALL always return ``[]``."""
    n = NoInheritance()
    assert n.select_parents([], [], 0) == []
    assert n.select_parents(["a", "b", "c"], [0.1, 0.5, 0.9], 7) == []


def test_no_inheritance_assign_parent_returns_none() -> None:
    """``NoInheritance.assign_parent`` SHALL always return ``None``."""
    n = NoInheritance()
    assert n.assign_parent(0, []) is None
    assert n.assign_parent(0, ["a", "b"]) is None
    assert n.assign_parent(99, ["a"]) is None


def test_no_inheritance_checkpoint_path_returns_none(tmp_path: Path) -> None:
    """``NoInheritance.checkpoint_path`` SHALL always return ``None``."""
    n = NoInheritance()
    assert n.checkpoint_path(tmp_path, 0, "gid") is None


# ---------------------------------------------------------------------------
# LamarckianInheritance.select_parents
# ---------------------------------------------------------------------------


def test_lamarckian_select_parents_returns_top_k_by_fitness() -> None:
    """``select_parents`` SHALL return the ``elite_count`` highest-fitness IDs."""
    l1 = LamarckianInheritance(elite_count=1)
    assert l1.select_parents(["a", "b", "c"], [0.1, 0.9, 0.5], 0) == ["b"]
    assert l1.select_parents(["x", "y", "z"], [0.7, 0.7, 0.3], 5) == ["x"]  # tie-break

    l3 = LamarckianInheritance(elite_count=3)
    assert l3.select_parents(["a", "b", "c", "d"], [0.1, 0.5, 0.9, 0.3], 0) == ["c", "b", "d"]


def test_lamarckian_select_parents_breaks_ties_lexicographically() -> None:
    """When fitnesses tie, ``select_parents`` SHALL break on ``genome_id`` order."""
    l3 = LamarckianInheritance(elite_count=3)
    # All three have the same fitness; lexicographic order on ID wins.
    assert l3.select_parents(["c", "a", "b"], [0.5, 0.5, 0.5], 0) == ["a", "b", "c"]
    # Mixed: top-fitness wins outright; tie among lower fitnesses breaks lexicographically.
    l2 = LamarckianInheritance(elite_count=2)
    assert l2.select_parents(["c", "a", "b"], [0.7, 0.5, 0.5], 0) == ["c", "a"]


def test_lamarckian_select_parents_empty_input_returns_empty() -> None:
    """Gen-0 case: empty prior generation → empty selection."""
    l1 = LamarckianInheritance()
    assert l1.select_parents([], [], 0) == []


def test_lamarckian_select_parents_length_mismatch_raises() -> None:
    """``select_parents`` SHALL raise on mismatched ``gen_ids``/``fitnesses``."""
    l1 = LamarckianInheritance()
    with pytest.raises(ValueError, match="len\\(gen_ids\\) == len\\(fitnesses\\)"):
        l1.select_parents(["a", "b"], [0.5], 0)


# ---------------------------------------------------------------------------
# LamarckianInheritance.assign_parent
# ---------------------------------------------------------------------------


def test_lamarckian_assign_parent_round_robin() -> None:
    """With ``elite_count > 1``, children rotate round-robin through parents."""
    l_any = LamarckianInheritance()  # elite_count is irrelevant to assign_parent
    parents = ["p0", "p1", "p2"]
    assert l_any.assign_parent(0, parents) == "p0"
    assert l_any.assign_parent(1, parents) == "p1"
    assert l_any.assign_parent(2, parents) == "p2"
    assert l_any.assign_parent(3, parents) == "p0"  # wraps
    assert l_any.assign_parent(7, parents) == "p1"  # wraps further


def test_lamarckian_assign_parent_returns_none_for_empty_parent_list() -> None:
    """When no parents (gen 0), every child is from-scratch."""
    l_any = LamarckianInheritance()
    for child_idx in range(10):
        assert l_any.assign_parent(child_idx, []) is None


def test_lamarckian_assign_parent_single_elite_broadcasts() -> None:
    """With ``elite_count=1``, all children inherit from the same single parent."""
    l1 = LamarckianInheritance(elite_count=1)
    parents = ["only_one"]
    for child_idx in range(20):
        assert l1.assign_parent(child_idx, parents) == "only_one"


# ---------------------------------------------------------------------------
# LamarckianInheritance.checkpoint_path
# ---------------------------------------------------------------------------


def test_lamarckian_checkpoint_path_format_round_trips(tmp_path: Path) -> None:
    """Path follows ``output_dir/inheritance/gen-NNN/genome-<gid>.pt``."""
    l1 = LamarckianInheritance()
    p = l1.checkpoint_path(tmp_path, 7, "abc-123")
    assert p is not None  # narrow Path | None for the type-checker
    assert p == tmp_path / "inheritance" / "gen-007" / "genome-abc-123.pt"
    # Generation index is 3-digit zero-padded
    p2 = l1.checkpoint_path(tmp_path, 0, "x")
    assert p2 == tmp_path / "inheritance" / "gen-000" / "genome-x.pt"
    # And ID extraction round-trips
    assert p.stem.removeprefix("genome-") == "abc-123"


# ---------------------------------------------------------------------------
# Constructor guards
# ---------------------------------------------------------------------------


def test_lamarckian_rejects_zero_or_negative_elite_count() -> None:
    """``elite_count < 1`` is meaningless and SHALL raise at construction."""
    with pytest.raises(ValueError, match="elite_count must be >= 1"):
        LamarckianInheritance(elite_count=0)
    with pytest.raises(ValueError, match="elite_count must be >= 1"):
        LamarckianInheritance(elite_count=-1)
