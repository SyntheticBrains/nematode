"""Unit tests for :mod:`quantumnematode.evolution.lamarckian_transgenerational_inheritance`.

The composed strategy mirrors :class:`LamarckianInheritance` exactly on
``select_parents`` / ``assign_parent`` / ``checkpoint_path`` (the M3
single-elite-broadcast contract) and adds the new
``"weights+transgenerational"`` kind literal to drive the M6.13 loop
dispatch. The tests below assert per-method parity with Lamarckian
where appropriate, distinctness from both parent strategies on
``kind()``, and protocol-conformance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from quantumnematode.evolution.inheritance import (
    BaldwinInheritance,
    InheritanceStrategy,
    LamarckianInheritance,
    NoInheritance,
)
from quantumnematode.evolution.lamarckian_transgenerational_inheritance import (
    LamarckianTransgenerationalInheritance,
)
from quantumnematode.evolution.transgenerational_inheritance import (
    TransgenerationalInheritance,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Construction + validation
# ---------------------------------------------------------------------------


def test_default_elite_count_is_one() -> None:
    """``LamarckianTransgenerationalInheritance()`` SHALL default ``elite_count=1``."""
    strategy = LamarckianTransgenerationalInheritance()
    assert strategy.elite_count == 1


def test_explicit_elite_count_is_preserved() -> None:
    """Explicit ``elite_count`` SHALL be preserved on the instance."""
    strategy = LamarckianTransgenerationalInheritance(elite_count=3)
    assert strategy.elite_count == 3


def test_rejects_zero_elite_count() -> None:
    """``elite_count=0`` SHALL raise ``ValueError`` at construction.

    Mirrors :class:`LamarckianInheritance`'s ``elite_count < 1`` rejection.
    """
    with pytest.raises(ValueError, match="elite_count must be >= 1"):
        LamarckianTransgenerationalInheritance(elite_count=0)


def test_rejects_negative_elite_count() -> None:
    """Negative ``elite_count`` SHALL raise ``ValueError`` at construction."""
    with pytest.raises(ValueError, match="elite_count must be >= 1"):
        LamarckianTransgenerationalInheritance(elite_count=-2)


# ---------------------------------------------------------------------------
# kind() return value
# ---------------------------------------------------------------------------


def test_kind_returns_composed_literal() -> None:
    """``kind()`` SHALL return the new ``"weights+transgenerational"`` literal."""
    assert LamarckianTransgenerationalInheritance().kind() == "weights+transgenerational"


def test_kind_distinct_from_either_parent_strategy() -> None:
    """The composed ``kind()`` SHALL differ from both Lamarckian and Transgenerational.

    The whole point of a fifth literal is to drive distinct loop dispatch;
    accidentally returning ``"weights"`` (Lamarckian) or ``"transgenerational"``
    (M6.9+ pure-TEI) would silently route the composed strategy through one
    of the existing branches.
    """
    composed = LamarckianTransgenerationalInheritance().kind()
    assert composed != LamarckianInheritance().kind()
    assert composed != TransgenerationalInheritance().kind()
    assert composed != BaldwinInheritance().kind()
    assert composed != NoInheritance().kind()


# ---------------------------------------------------------------------------
# select_parents — parity with Lamarckian
# ---------------------------------------------------------------------------


def test_select_parents_returns_top_fitness_lex_tie_broken() -> None:
    """Top-fitness selection with lex tie-break — identical to Lamarckian."""
    composed = LamarckianTransgenerationalInheritance(elite_count=1)
    result = composed.select_parents(["g0", "g1", "g2"], [0.5, 0.8, 0.6], 0)
    assert result == ["g1"]
    # Tie on fitness — lex order on genome_id decides.
    tied = composed.select_parents(["g0", "g1", "g2"], [0.8, 0.8, 0.6], 0)
    assert tied == ["g0"]


def test_select_parents_matches_lamarckian_byte_for_byte() -> None:
    """Composed.select_parents output SHALL equal Lamarckian.select_parents for same inputs.

    Cross-strategy parity test: the composed strategy reuses the M3
    selection rule by construction. This test pins that contract.
    """
    composed = LamarckianTransgenerationalInheritance(elite_count=1)
    lamarckian = LamarckianInheritance(elite_count=1)
    gen_ids = ["alpha", "bravo", "charlie", "delta"]
    fitnesses = [0.3, 0.9, 0.5, 0.7]
    assert composed.select_parents(gen_ids, fitnesses, 0) == lamarckian.select_parents(
        gen_ids,
        fitnesses,
        0,
    )


def test_select_parents_empty_returns_empty() -> None:
    """Empty input SHALL return empty list (matches Lamarckian's gen-0 contract)."""
    assert LamarckianTransgenerationalInheritance().select_parents([], [], 0) == []


def test_select_parents_length_mismatch_raises() -> None:
    """Length mismatch between gen_ids and fitnesses SHALL raise ``ValueError``."""
    composed = LamarckianTransgenerationalInheritance()
    with pytest.raises(ValueError, match="len\\(gen_ids\\) == len\\(fitnesses\\)"):
        composed.select_parents(["a", "b"], [0.5], 0)


def test_select_parents_multi_elite_returns_top_n() -> None:
    """``elite_count=3`` SHALL return the top 3 IDs ranked by fitness.

    Multi-elite is reserved for future work (the validator currently
    rejects ``elite_count != 1`` from YAML), but the strategy SHALL
    support the slice structurally so a future config-validator
    relaxation can enable it without a code change to the strategy.
    Tests parity with Lamarckian's existing multi-elite slice
    behaviour across an even-fitness 4-element population.
    """
    composed = LamarckianTransgenerationalInheritance(elite_count=3)
    lamarckian = LamarckianInheritance(elite_count=3)
    gen_ids = ["alpha", "bravo", "charlie", "delta"]
    fitnesses = [0.3, 0.9, 0.5, 0.7]
    expected = ["bravo", "delta", "charlie"]  # top-3 descending
    assert composed.select_parents(gen_ids, fitnesses, 0) == expected
    assert composed.select_parents(gen_ids, fitnesses, 0) == lamarckian.select_parents(
        gen_ids,
        fitnesses,
        0,
    )


# ---------------------------------------------------------------------------
# assign_parent — round-robin parity with Lamarckian
# ---------------------------------------------------------------------------


def test_assign_parent_round_robin_matches_lamarckian() -> None:
    """``assign_parent`` SHALL round-robin identically to Lamarckian."""
    composed = LamarckianTransgenerationalInheritance(elite_count=2)
    parent_ids = ["pA", "pB"]
    assert [composed.assign_parent(idx, parent_ids) for idx in range(4)] == [
        "pA",
        "pB",
        "pA",
        "pB",
    ]


def test_assign_parent_single_elite_broadcasts() -> None:
    """With ``elite_count=1`` (the only YAML-supported config), every child broadcasts."""
    composed = LamarckianTransgenerationalInheritance(elite_count=1)
    parent_ids = ["elite_x"]
    assert all(composed.assign_parent(idx, parent_ids) == "elite_x" for idx in range(8))


def test_assign_parent_empty_returns_none() -> None:
    """Empty parent list (gen-0 case) SHALL return ``None``."""
    assert LamarckianTransgenerationalInheritance().assign_parent(0, []) is None


# ---------------------------------------------------------------------------
# checkpoint_path — canonical .pt (NOT .tei.pt)
# ---------------------------------------------------------------------------


def test_checkpoint_path_is_canonical_pt(tmp_path: Path) -> None:
    """``checkpoint_path`` SHALL return ``inheritance/gen-NNN/genome-<gid>.pt``."""
    composed = LamarckianTransgenerationalInheritance()
    result = composed.checkpoint_path(tmp_path, 2, "abc")
    assert result == tmp_path / "inheritance" / "gen-002" / "genome-abc.pt"


def test_checkpoint_path_matches_lamarckian(tmp_path: Path) -> None:
    """Composed.checkpoint_path SHALL be identical to Lamarckian.checkpoint_path.

    The composed strategy owns the WEIGHTS checkpoint path; the substrate
    ``.tei.pt`` path is owned by the F0 substrate-extraction pipeline as a
    separate concern. This test pins that the strategy's
    ``checkpoint_path`` returns the ``.pt`` (M3) path, NOT the ``.tei.pt``
    (TEI substrate) path.
    """
    composed = LamarckianTransgenerationalInheritance()
    lamarckian = LamarckianInheritance()
    for gen, gid in [(0, "elite"), (3, "g-xyz"), (12, "abc-123")]:
        assert composed.checkpoint_path(tmp_path, gen, gid) == lamarckian.checkpoint_path(
            tmp_path,
            gen,
            gid,
        )


def test_checkpoint_path_extension_is_pt_not_tei_pt(tmp_path: Path) -> None:
    """The path SHALL end in ``.pt`` (NOT ``.tei.pt``)."""
    result = LamarckianTransgenerationalInheritance().checkpoint_path(tmp_path, 0, "g0")
    assert result is not None
    assert result.suffix == ".pt"
    assert not result.name.endswith(".tei.pt")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_implements_inheritance_strategy_protocol() -> None:
    """Composed strategy SHALL satisfy ``isinstance(_, InheritanceStrategy)``.

    The ``InheritanceStrategy`` Protocol is ``@runtime_checkable``; passing
    isinstance proves the class implements every Protocol method.
    Kind-set membership is covered cross-file in
    ``test_inheritance.test_kind_values_are_in_known_set`` to avoid
    duplicating the set-membership assertion across two files.
    """
    assert isinstance(LamarckianTransgenerationalInheritance(), InheritanceStrategy)
