"""Unit tests for :mod:`quantumnematode.evolution.transgenerational_inheritance`.

Mirrors the structure of ``test_inheritance.py`` for the Baldwin
strategy. Covers the four ``InheritanceStrategy`` Protocol methods
plus the known-kind set guard. Commit 1 ships the strategy skeleton
only; F0 substrate extraction + worker forwarding land in commits 4
and 5 with their own integration tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from quantumnematode.evolution.inheritance import (
    BaldwinInheritance,
    LamarckianInheritance,
    NoInheritance,
)
from quantumnematode.evolution.transgenerational_inheritance import (
    TransgenerationalInheritance,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# TransgenerationalInheritance.kind
# ---------------------------------------------------------------------------


def test_transgenerational_kind() -> None:
    """``TransgenerationalInheritance.kind`` SHALL return the literal ``"transgenerational"``."""
    assert TransgenerationalInheritance().kind() == "transgenerational"


def test_kind_values_are_in_extended_known_set() -> None:
    """All four shipped strategies SHALL return a value from the known kind set.

    Extends the existing three-strategy guard in ``test_inheritance.py``
    to include the new ``"transgenerational"`` literal. Guards against
    future strategies leaking new literals without updating the loop's
    branching logic.
    """
    known = {"none", "weights", "trait", "transgenerational"}
    assert NoInheritance().kind() in known
    assert LamarckianInheritance().kind() in known
    assert BaldwinInheritance().kind() in known
    assert TransgenerationalInheritance().kind() in known


# ---------------------------------------------------------------------------
# TransgenerationalInheritance.checkpoint_path
# ---------------------------------------------------------------------------


def test_transgenerational_checkpoint_path_uses_tei_pt_extension(tmp_path: Path) -> None:
    """``checkpoint_path`` SHALL return ``inheritance/gen-NNN/genome-<gid>.tei.pt``.

    The ``.tei.pt`` extension (vs Lamarckian's ``.pt``) prevents
    on-disk collision if a future config ever mixes the two
    substrate types.
    """
    t = TransgenerationalInheritance()
    path = t.checkpoint_path(tmp_path, 0, "gid-a")
    assert path is not None
    assert path == tmp_path / "inheritance" / "gen-000" / "genome-gid-a.tei.pt"


def test_transgenerational_checkpoint_path_zero_pads_generation(tmp_path: Path) -> None:
    """Generation index SHALL be 3-digit zero-padded (``gen-007``, not ``gen-7``).

    Matches the Lamarckian / Baldwin / NoInheritance path-builder
    convention so lineage directory ordering stays stable across
    string sorts.
    """
    t = TransgenerationalInheritance()
    path = t.checkpoint_path(tmp_path, 7, "abc-123")
    assert path is not None
    assert "gen-007" in str(path)
    assert path.name == "genome-abc-123.tei.pt"


# ---------------------------------------------------------------------------
# TransgenerationalInheritance.assign_parent
# ---------------------------------------------------------------------------


def test_transgenerational_assign_parent_returns_none() -> None:
    """``assign_parent`` SHALL return ``None`` (no per-child warm-start under TEI).

    TEI's substrate flows through ``fitness.evaluate``'s
    ``tei_prior_source`` kwarg path (commit 5), not through the per-
    child weight-checkpoint warm-start mechanism. Matches Baldwin's
    return semantics.
    """
    t = TransgenerationalInheritance()
    assert t.assign_parent(0, []) is None
    assert t.assign_parent(0, ["a", "b"]) is None
    assert t.assign_parent(99, ["a"]) is None


# ---------------------------------------------------------------------------
# TransgenerationalInheritance.select_parents
# ---------------------------------------------------------------------------


def test_transgenerational_select_parents_returns_single_elite() -> None:
    """``select_parents`` SHALL return ``[best_genome_id]`` (single-element list)."""
    t = TransgenerationalInheritance()
    assert t.select_parents(["a", "b", "c"], [0.1, 0.9, 0.5], 0) == ["b"]


def test_transgenerational_select_parents_breaks_ties_lexicographically() -> None:
    """``select_parents`` SHALL break fitness ties on ``genome_id`` lex order.

    Matches the Lamarckian / Baldwin tie-break rule for deterministic
    F0 elite selection across runs.
    """
    t = TransgenerationalInheritance()
    # All three tied at 0.7 — lex-smallest "x" wins.
    assert t.select_parents(["x", "y", "z"], [0.7, 0.7, 0.7], 5) == ["x"]
    # Higher-fitness tie at 0.9, lower-fitness "z" at 0.3 — "a" wins over "b".
    assert t.select_parents(["b", "a", "z"], [0.9, 0.9, 0.3], 5) == ["a"]


def test_transgenerational_select_parents_matches_baldwin_and_lamarckian_single_elite() -> None:
    """Transgenerational selection SHALL match Baldwin and Lamarckian(1) byte-for-byte.

    All three single-elite strategies share the same selection rule
    (top fitness, lex-tie-broken on ``genome_id``). Asserting
    equivalence here pins the shared rule and guards against drift
    if any one implementation changes.
    """
    t = TransgenerationalInheritance()
    b = BaldwinInheritance()
    lam1 = LamarckianInheritance(elite_count=1)
    cases = [
        (["a", "b", "c"], [0.1, 0.9, 0.5]),
        (["x", "y", "z"], [0.7, 0.7, 0.3]),
        (["c", "a", "b"], [0.5, 0.5, 0.5]),
        (["only"], [0.42]),
    ]
    for ids, fits in cases:
        t_result = t.select_parents(ids, fits, 0)
        b_result = b.select_parents(ids, fits, 0)
        lam_result = lam1.select_parents(ids, fits, 0)
        assert t_result == b_result == lam_result


def test_transgenerational_select_parents_empty_input_returns_empty() -> None:
    """Gen-0 case: empty prior generation → empty selection."""
    assert TransgenerationalInheritance().select_parents([], [], 0) == []


def test_transgenerational_select_parents_length_mismatch_raises() -> None:
    """``select_parents`` SHALL raise on mismatched ``gen_ids`` / ``fitnesses``."""
    t = TransgenerationalInheritance()
    with pytest.raises(ValueError, match="len\\(gen_ids\\) == len\\(fitnesses\\)"):
        t.select_parents(["a", "b"], [0.5], 0)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_transgenerational_satisfies_inheritance_strategy_protocol() -> None:
    """``TransgenerationalInheritance`` SHALL satisfy the runtime-checkable Protocol.

    The Protocol is declared ``@runtime_checkable``, so ``isinstance``
    works as a structural check that all four required methods
    exist with the right shape. The loop relies on this Protocol
    contract (not on ``isinstance(strategy, NoInheritance)`` etc.)
    to dispatch on ``kind()``.
    """
    from quantumnematode.evolution.inheritance import InheritanceStrategy

    assert isinstance(TransgenerationalInheritance(), InheritanceStrategy)
