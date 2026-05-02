"""Inheritance strategies for the evolution loop.

Provides the :class:`InheritanceStrategy` Protocol and three concrete
implementations:

- :class:`NoInheritance` (the default): every child is from-scratch; no
  per-genome weight checkpoints written, no GC, no inheritance directory.
  The loop, fitness, and lineage code paths are byte-equivalent to a
  frozen-weight evolution run when this strategy is active.
- :class:`LamarckianInheritance`: each child of generation N+1 inherits
  trained weights from a *selected parent* of generation N (the highest
  fitness genome by default; ties broken on genome_id lexicographic
  order).  The hyperparameter genome continues to evolve via TPE — weights
  flow as a side-channel substrate keyed by parent ``genome_id``.
- :class:`BaldwinInheritance`: trait-only inheritance — the prior
  generation's elite genome ID is recorded as ``inherited_from`` for
  every child of the next generation (so the lineage CSV captures the
  evolutionary trace), but no per-genome weight checkpoints are written.
  Mechanically equivalent to :class:`NoInheritance` on the weight-IO
  path; differs only by populating lineage rows from gen 1 onwards.

The Protocol exposes four methods so future strategies (tournament,
fitness-proportionate "roulette", soft-elite top-k sampling) can plug
in without touching the loop:

- ``select_parents`` — pick which prior-gen genomes survive into the
  next generation as inheritance sources.
- ``assign_parent`` — for a given child slot in the next generation,
  pick which surviving parent it inherits from (round-robin in
  Lamarckian; sampling in tournament/roulette/soft-elite variants).
- ``checkpoint_path`` — single canonical path-builder used by both the
  capture (writer) and warm-start (reader) sides so the two cannot drift.
- ``kind`` — returns one of ``"none"``, ``"weights"``, or ``"trait"``
  so the loop branches on intent rather than ``isinstance`` checks.
  ``"none"`` skips both lineage-tracking and weight-IO; ``"weights"``
  enables both; ``"trait"`` enables lineage-tracking only.

Future-work strategies (tournament selection, roulette sampling,
soft-elite top-k) are NOT implemented here — they each become a new
``InheritanceStrategy`` subclass and a new ``Literal`` extension on
``EvolutionConfig.inheritance``.  Single-elite-broadcast is currently
the only Lamarckian configuration shipped; the validator on
``EvolutionConfig`` rejects ``inheritance_elite_count != 1`` when
``inheritance: lamarckian``, so multi-elite paths are unreachable from
YAML even though the round-robin code path exists structurally below.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class InheritanceStrategy(Protocol):
    """Pluggable strategy controlling per-genome inheritance.

    Implementations decide (a) which prior-generation genomes survive
    into the next generation as inheritance sources (``select_parents``),
    (b) which of those parents each child inherits from
    (``assign_parent``), (c) where per-genome weight checkpoints live
    on disk (``checkpoint_path`` — may return ``None`` for strategies
    with no on-disk substrate), and (d) what kind of inheritance the
    strategy implements (``kind`` — gates loop behaviour).
    """

    def select_parents(
        self,
        gen_ids: list[str],
        fitnesses: list[float],
        generation: int,
    ) -> list[str]:
        """Return the IDs that survive into the next generation as inheritance sources.

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
        per-genome checkpoints (the no-op case OR the trait-only case).
        Non-``None`` strategies return a deterministic path used by both
        the capture (writer) and warm-start (reader) sides; the loop uses
        ``Path.exists()`` to guard reads.
        """
        ...

    def kind(self) -> Literal["none", "weights", "trait"]:
        """Return the inheritance kind so the loop can branch on intent.

        Three values, each gating different code paths:

        - ``"none"`` (e.g. :class:`NoInheritance`) — loop skips ALL
          inheritance code paths.  No `select_parents` call, no
          `_selected_parent_ids` update, no GC, no warm-start, no
          per-child `inherited_from` write.
        - ``"weights"`` (e.g. :class:`LamarckianInheritance`) — loop
          captures per-genome trained-weight checkpoints, calls
          `select_parents`, GCs non-selected files, and warm-starts
          each next-gen child from its assigned parent's checkpoint.
        - ``"trait"`` (e.g. :class:`BaldwinInheritance`) — loop calls
          `select_parents` and writes `inherited_from` to lineage rows
          (so the elite-parent ID flows in lineage), but does NOT
          capture or GC any weight checkpoints.

        The loop's ``_inheritance_active()`` helper SHALL evaluate
        ``kind() == "weights"`` (gates weight-IO code paths).  The loop's
        ``_inheritance_records_lineage()`` helper SHALL evaluate
        ``kind() != "none"`` (gates lineage-tracking + `select_parents`).
        """
        ...


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


class NoInheritance:
    """No-op strategy — keeps the loop in frozen-weight evolution mode.

    ``select_parents`` always returns ``[]``, ``assign_parent`` always
    returns ``None``, ``checkpoint_path`` returns ``None``, and
    ``kind`` returns ``"none"`` (the loop guards with
    ``strategy.kind() == "none"`` so the weight-IO and lineage-tracking
    code paths are never invoked under no-op).  Returning ``None``
    rather than raising keeps the Protocol method shape uniform and
    avoids tripping type-checkers.
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

    def kind(self) -> Literal["none"]:
        """Return ``"none"`` — gates the loop into the no-inheritance code path."""
        return "none"


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

    def kind(self) -> Literal["weights"]:
        """Return ``"weights"`` — gates the loop into the weight-IO code path."""
        return "weights"


class BaldwinInheritance:
    """Trait-only inheritance — the genome flows, the weights do NOT.

    The hyperparameter genome continues to evolve via TPE; the loop
    records the prior generation's elite genome ID as ``inherited_from``
    for every child of the next generation, so the lineage CSV captures
    the evolutionary trace.  Mechanically equivalent to
    :class:`NoInheritance` on the weight-IO path: no per-genome ``.pt``
    files written, no GC, no warm-start.

    The Baldwin Effect (in evolutionary biology): lifetime learning
    guides genetic evolution toward genomes that learn fast, *without*
    the genome ever encoding the learned policy directly.  In this
    implementation, the K-train phase still runs from-scratch for every
    child; the elite-ID lineage trace exists so post-pilot analysis can
    identify which prior-gen elite each child shares hyperparameters
    with via TPE's posterior.

    The ``inheritance_elite_count`` config field is unused under
    Baldwin (Baldwin is conceptually single-elite by construction; the
    field exists for forward-compatibility with future multi-elite
    Baldwin variants).  ``select_parents`` returns ``[best_genome_id]``
    using the same selection rule as :class:`LamarckianInheritance`
    with ``elite_count=1`` (top fitness, lex-tie-broken on
    ``genome_id``) so the loop's existing ``_selected_parent_ids``
    array carries the elite ID forward.
    """

    def select_parents(
        self,
        gen_ids: list[str],
        fitnesses: list[float],
        generation: int,  # noqa: ARG002
    ) -> list[str]:
        """Return ``[best_genome_id]`` (single-element list, lex-tie-broken).

        Same selection rule as :class:`LamarckianInheritance` with
        ``elite_count=1`` — they're factored as separate methods rather
        than shared because the strategies are conceptually distinct
        (Baldwin doesn't broadcast weights to descendants; it only
        records the lineage trace).
        """
        if len(gen_ids) != len(fitnesses):
            msg = (
                f"select_parents requires len(gen_ids) == len(fitnesses); "
                f"got {len(gen_ids)} vs {len(fitnesses)}."
            )
            raise ValueError(msg)
        if not gen_ids:
            return []
        ranked = sorted(zip(gen_ids, fitnesses, strict=True), key=lambda x: (-x[1], x[0]))
        return [ranked[0][0]]

    def assign_parent(
        self,
        child_index: int,  # noqa: ARG002
        parent_ids: list[str],  # noqa: ARG002
    ) -> str | None:
        """Return ``None`` — Baldwin doesn't warm-start any child."""
        return None

    def checkpoint_path(
        self,
        output_dir: Path,  # noqa: ARG002
        generation: int,  # noqa: ARG002
        genome_id: str,  # noqa: ARG002
    ) -> Path | None:
        """Return ``None`` — Baldwin uses no on-disk checkpoint substrate."""
        return None

    def kind(self) -> Literal["trait"]:
        """Return ``"trait"`` — gates the loop into the lineage-only code path."""
        return "trait"
