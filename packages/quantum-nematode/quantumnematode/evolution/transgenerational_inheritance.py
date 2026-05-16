"""Transgenerational inheritance strategy for the evolution loop.

Provides :class:`TransgenerationalInheritance`, the fourth concrete
implementation of the :class:`InheritanceStrategy` Protocol (alongside
``NoInheritance``, ``LamarckianInheritance``, and ``BaldwinInheritance``
in :mod:`quantumnematode.evolution.inheritance`).

The strategy carries an inheritable behavioural-bias substrate
(``TransgenerationalMemory``, a follow-up addition) extracted from the
F0 elite's policy and multiplicatively decayed across F1/F2/F3
generations. The substrate biases the actor's logits before softmax,
independently of trained weights.

Structurally mirrors ``BaldwinInheritance`` for the four Protocol
methods:

- ``select_parents`` — single-elite (top fitness, lex-tie-broken on
  ``genome_id``), matching the Lamarckian/Baldwin selection rule so
  the loop's existing ``_selected_parent_ids`` array carries the F0
  elite ID forward.
- ``assign_parent`` — returns ``None`` (transgenerational inheritance
  does not warm-start any child from a per-genome weight checkpoint;
  the substrate flows separately through ``fitness.evaluate``'s
  ``tei_prior_source`` kwarg, a follow-up addition).
- ``checkpoint_path`` — returns the ``.tei.pt`` path under
  ``inheritance/gen-NNN/genome-<gid>.tei.pt``. Distinct extension
  from Lamarckian's ``.pt`` so the two substrate types cannot
  collide on disk if a future config ever mixes them.
- ``kind()`` — returns the new literal ``"transgenerational"``.

This module ships ONLY the strategy skeleton + Protocol conformance.
The F0 substrate extraction pipeline (writing the ``.tei.pt`` after
F0 weight capture + telemetry pass) and the worker tuple extension +
``fitness.evaluate`` kwarg forwarding are follow-up additions tracked
in the OpenSpec change under ``openspec/changes/add-transgenerational-
memory/``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path


class TransgenerationalInheritance:
    """Single-elite substrate-flow inheritance — F0 substrate cascade.

    Selects the F0 elite via the same top-1 lex-tie-broken rule used
    by :class:`BaldwinInheritance` and :class:`LamarckianInheritance`
    (``elite_count=1``). The selected elite's substrate is extracted
    AFTER F0 fitness evaluation via a deterministic telemetry pass
    (``TransgenerationalMemory.extract_from_brain``, a follow-up
    addition) and saved to disk at ``checkpoint_path(...)``. F1+
    children load this single F0 substrate and apply
    ``inherit_from(...)`` N times where N = current generation,
    producing depth-N substrates mechanically (no per-gen storage).

    The ``EvolutionConfig.inheritance_elite_count`` field is unused
    under transgenerational (single-elite by construction, matching
    Baldwin). The structural ``inheritance_elite_count <=
    population_size`` validator on ``EvolutionConfig`` still applies
    uniformly across all inheritance modes.
    """

    def select_parents(
        self,
        gen_ids: list[str],
        fitnesses: list[float],
        generation: int,  # noqa: ARG002
    ) -> list[str]:
        """Return ``[best_genome_id]`` (single-element list, lex-tie-broken).

        Same selection rule as :class:`BaldwinInheritance.select_parents`
        and :class:`LamarckianInheritance.select_parents` with
        ``elite_count=1``. The three strategies factor the selection
        rule separately rather than sharing because they are
        conceptually distinct (Lamarckian broadcasts weights to
        descendants; Baldwin records only lineage; transgenerational
        flows a behavioural-bias substrate).
        """
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
        return [ranked[0][0]]

    def assign_parent(
        self,
        child_index: int,  # noqa: ARG002
        parent_ids: list[str],  # noqa: ARG002
    ) -> str | None:
        """Return ``None`` — transgenerational does not warm-start any child.

        The transgenerational substrate flows through the separate
        ``tei_prior_source`` kwarg path into ``fitness.evaluate`` (a
        follow-up addition), not through the per-child weight-
        checkpoint warm-start mechanism that Lamarckian uses. The
        F1+ branch of ``_resolve_per_child_inheritance`` returns
        ``(None, None, parent_id)`` — same shape as Baldwin's trait-
        only flow — so the lineage CSV records the F0 elite ID but
        no warm-start path.
        """
        return None

    def checkpoint_path(
        self,
        output_dir: Path,
        generation: int,
        genome_id: str,
    ) -> Path | None:
        """Return the canonical ``inheritance/gen-NNN/genome-<gid>.tei.pt`` path.

        Distinct ``.tei.pt`` extension (vs Lamarckian's ``.pt``) so
        the two substrate types cannot collide on disk if a future
        config ever mixes them. Both capture (writer) and load
        (reader) sides MUST use this path-builder to avoid drift.

        Used by the F0 Substrate Extraction Pipeline (a follow-up
        addition) to write the F0 elite's substrate. F1/F2/F3
        substrates are mechanically derived in-memory at load time
        (no per-gen storage), so this method is only invoked at
        gen 0 in practice — but it is structurally valid at any
        generation for forensic round-trip tests.
        """
        return output_dir / "inheritance" / f"gen-{generation:03d}" / f"genome-{genome_id}.tei.pt"

    def kind(self) -> Literal["transgenerational"]:
        """Return ``"transgenerational"`` — gates the loop into the substrate-flow code path."""
        return "transgenerational"
