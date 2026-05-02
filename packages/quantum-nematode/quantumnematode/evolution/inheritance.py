"""Inheritance strategies for the evolution loop.

Provides the :class:`InheritanceStrategy` Protocol and two concrete
implementations:

- :class:`NoInheritance` (the default): every child is from-scratch; no
  per-genome weight checkpoints written, no GC, no inheritance directory.
  The loop, fitness, and lineage code paths are byte-equivalent to a
  pre-inheritance run when this strategy is active.
- :class:`LamarckianInheritance`: each child of generation N+1 inherits
  trained weights from a *selected parent* of generation N (the highest
  fitness genome by default; ties broken on genome_id lexicographic
  order).  The hyperparameter genome continues to evolve via TPE — weights
  flow as a side-channel substrate keyed by parent ``genome_id``.

The Protocol intentionally exposes three methods so future strategies
(tournament, fitness-proportionate "roulette", soft-elite top-k sampling)
can plug in without touching the loop:

- ``select_parents`` — pick which prior-gen genomes survive into the
  next generation as inheritance sources.
- ``assign_parent`` — for a given child slot in the next generation,
  pick which surviving parent it inherits from (round-robin in
  Lamarckian; sampling in tournament/roulette/soft-elite variants).
- ``checkpoint_path`` — single canonical path-builder used by both the
  capture (writer) and warm-start (reader) sides so the two cannot drift.

Future-work strategies (Baldwin Effect, tournament selection, roulette
sampling, soft-elite top-k) are NOT implemented here — they each become
a new ``InheritanceStrategy`` subclass and a new ``Literal`` extension on
``EvolutionConfig.inheritance``.  Currently only single-elite-broadcast
is shipped; the validator on ``EvolutionConfig`` rejects
``inheritance_elite_count != 1`` when ``inheritance: lamarckian``, so
multi-elite paths are unreachable from YAML even though the round-robin
code path exists structurally below.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class InheritanceStrategy(Protocol):
    """Pluggable strategy controlling per-genome weight inheritance.

    Implementations decide (a) which prior-generation genomes propagate
    weights into the next generation (``select_parents``), (b) which of
    those parents each child inherits from (``assign_parent``), and (c)
    where per-genome weight checkpoints live on disk (``checkpoint_path``).
    """

    def select_parents(
        self,
        gen_ids: list[str],
        fitnesses: list[float],
        generation: int,
    ) -> list[str]:
        """Return the IDs whose checkpoints survive into the next generation.

        Called by the loop AFTER ``optimizer.tell`` returns for the just-
        completed generation, so ``gen_ids`` and ``fitnesses`` describe
        the freshly-evaluated population.  The returned list is the
        inheritance source set for the next generation's children.
        Empty list means "no inheritance for next gen" — used implicitly
        in the gen-0 case where the prior-gen lists are empty.
        """
        ...

    def assign_parent(
        self,
        child_index: int,
        parent_ids: list[str],
    ) -> str | None:
        """Return the parent ID this child inherits from, or ``None``.

        ``parent_ids`` is the list returned by ``select_parents`` for the
        prior generation.  ``child_index`` is the child's index within
        the about-to-evaluate generation's population.  Returning ``None``
        means "this child is from-scratch" — used for gen 0 (empty
        parent list) and for any from-scratch fallback path.
        """
        ...

    def checkpoint_path(
        self,
        output_dir: Path,
        generation: int,
        genome_id: str,
    ) -> Path | None:
        """Return the canonical on-disk path for one genome's checkpoint.

        Returning ``None`` signals that this strategy does not use
        per-genome checkpoints (the no-op case).  Non-no-op strategies
        return a deterministic path used by both the capture (writer)
        and warm-start (reader) sides; the loop uses ``Path.exists()``
        to guard reads.
        """
        ...


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


class NoInheritance:
    """No-op strategy — keeps the loop in frozen-weight evolution mode.

    ``select_parents`` always returns ``[]``, ``assign_parent`` always
    returns ``None``, and ``checkpoint_path`` returns ``None`` (the loop
    guards with ``isinstance(strategy, NoInheritance)`` so this method
    is never invoked under no-op).  Returning ``None`` rather than
    raising keeps the Protocol method shape uniform and avoids tripping
    type-checkers.
    """

    def select_parents(
        self,
        gen_ids: list[str],  # noqa: ARG002
        fitnesses: list[float],  # noqa: ARG002
        generation: int,  # noqa: ARG002
    ) -> list[str]:
        """Return ``[]`` — no-op strategy never selects parents."""
        return []

    def assign_parent(
        self,
        child_index: int,  # noqa: ARG002
        parent_ids: list[str],  # noqa: ARG002
    ) -> str | None:
        """Return ``None`` — no-op strategy never assigns a parent."""
        return None

    def checkpoint_path(
        self,
        output_dir: Path,  # noqa: ARG002
        generation: int,  # noqa: ARG002
        genome_id: str,  # noqa: ARG002
    ) -> Path | None:
        """Return ``None`` — no-op strategy uses no checkpoint paths."""
        return None


class LamarckianInheritance:
    """Single-elite (or top-K) parent broadcast.

    ``select_parents`` sorts the prior generation by fitness descending
    (ties broken on ``genome_id`` lexicographic order for determinism)
    and returns the top ``elite_count`` IDs.  ``assign_parent`` rotates
    children round-robin through the parent list, so with the default
    ``elite_count=1`` every child of the next generation inherits from
    the same single best parent.

    The validator on ``EvolutionConfig`` currently rejects
    ``elite_count != 1`` when ``inheritance: lamarckian``, so the
    round-robin path is documented future-work behaviour rather than
    shipped behaviour from YAML.
    """

    def __init__(self, elite_count: int = 1) -> None:
        if elite_count < 1:
            msg = f"elite_count must be >= 1, got {elite_count}."
            raise ValueError(msg)
        self.elite_count = elite_count

    def select_parents(
        self,
        gen_ids: list[str],
        fitnesses: list[float],
        generation: int,  # noqa: ARG002
    ) -> list[str]:
        """Return the top ``elite_count`` IDs by fitness, lex-tie-broken."""
        if len(gen_ids) != len(fitnesses):
            msg = (
                f"select_parents requires len(gen_ids) == len(fitnesses); "
                f"got {len(gen_ids)} vs {len(fitnesses)}."
            )
            raise ValueError(msg)
        if not gen_ids:
            return []
        # Sort by (-fitness, genome_id): higher fitness wins, lexicographic
        # tie-break ensures deterministic selection across runs.
        ranked = sorted(zip(gen_ids, fitnesses, strict=True), key=lambda x: (-x[1], x[0]))
        return [gid for gid, _ in ranked[: self.elite_count]]

    def assign_parent(
        self,
        child_index: int,
        parent_ids: list[str],
    ) -> str | None:
        """Round-robin: return ``parent_ids[child_index % len(parent_ids)]`` or ``None``."""
        if not parent_ids:
            return None
        return parent_ids[child_index % len(parent_ids)]

    def checkpoint_path(
        self,
        output_dir: Path,
        generation: int,
        genome_id: str,
    ) -> Path | None:
        """Return the canonical ``inheritance/gen-NNN/genome-<gid>.pt`` path."""
        return output_dir / "inheritance" / f"gen-{generation:03d}" / f"genome-{genome_id}.pt"
