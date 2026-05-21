"""Composed weights + substrate inheritance strategy.

Provides :class:`LamarckianTransgenerationalInheritance`, the fifth
``InheritanceStrategy`` implementation. Composes the Lamarckian
weight-inheritance path (per-genome ``.pt`` warm-start + capture + GC)
with the substrate-flow path (F0 substrate extraction +
decayed cascade applied via ``brain.tei_prior``).

The class lives in its own module — not as a subclass of
:class:`~quantumnematode.evolution.inheritance.LamarckianInheritance`
nor :class:`~quantumnematode.evolution.transgenerational_inheritance.TransgenerationalInheritance`
— so the two parent strategies remain auditable in isolation. Their
regression tests gate byte-equivalence on the Lamarckian and pure-TEI paths
respectively; this composed class is orthogonal and has its own tests.

Per-method behaviour mirrors :class:`LamarckianInheritance` exactly:

- ``select_parents``: top ``elite_count`` IDs by fitness, lex-tie-broken
  on ``genome_id`` (identical sort key + slice).
- ``assign_parent``: round-robin over the parent list (identical
  ``parent_ids[child_index % len(parent_ids)]``).
- ``checkpoint_path``: canonical ``inheritance/gen-NNN/genome-<gid>.pt``
  (the weight checkpoint path — NOT the substrate ``.tei.pt`` path,
  which is owned by the F0 substrate-extraction pipeline as a separate
  concern).

The distinguishing behaviour lives entirely in ``kind()``, which returns
the new Literal value ``"weights+transgenerational"``. The loop dispatches
on this value so both ``_inheritance_active()`` (weight-IO) and
``_substrate_inheritance_active()`` (substrate-flow) widen to fire under
composed mode, threading both ``warm_start_path_override`` and
``tei_prior_source`` into every F1+ child's ``fitness.evaluate`` call.

See the OpenSpec change ``openspec/changes/archive/2026-05-21-add-tei-prior-on-m3/``
for the full design rationale + biological framing (TEI as a *prior on
trained weights*, aligning with the wet-lab Kaletsky/mammalian
mechanism — not a transmitted policy).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path


class LamarckianTransgenerationalInheritance:
    """Composed weights + substrate inheritance strategy.

    Single-elite (or top-K) parent broadcast for the weight-IO path
    (Lamarckian pattern), composed with the substrate-flow path (F0
    substrate extraction → F1+ ``tei_prior`` application). The composition is
    orthogonal: this class only owns the four ``InheritanceStrategy``
    Protocol methods + the new ``"weights+transgenerational"`` kind
    literal; the loop's existing ``_substrate_inheritance_active()``
    predicate handles the substrate-flow side once it's widened to
    accept this kind.

    The validator on ``EvolutionConfig`` currently rejects
    ``elite_count != 1`` when ``inheritance: weights+transgenerational``
    (mirroring the Lamarckian single-elite-broadcast contract), so the
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
        """Return the top ``elite_count`` IDs by fitness, lex-tie-broken.

        Selection rule is identical to :class:`LamarckianInheritance`
        with the same ``elite_count`` — composed mode reuses the
        Lamarckian elite-selection contract by construction.
        """
        if len(gen_ids) != len(fitnesses):
            msg = (
                f"select_parents requires len(gen_ids) == len(fitnesses); "
                f"got {len(gen_ids)} vs {len(fitnesses)}."
            )
            raise ValueError(msg)
        if not gen_ids:
            return []
        # Sort by (-fitness, genome_id): higher fitness wins, lex tie-break
        # ensures deterministic selection across runs.
        ranked = sorted(zip(gen_ids, fitnesses, strict=True), key=lambda x: (-x[1], x[0]))
        return [gid for gid, _ in ranked[: self.elite_count]]

    def assign_parent(
        self,
        child_index: int,
        parent_ids: list[str],
    ) -> str | None:
        """Round-robin: return ``parent_ids[child_index % len(parent_ids)]`` or ``None``.

        Identical to :class:`LamarckianInheritance.assign_parent`. With
        ``elite_count=1`` (the only YAML-supported configuration today),
        every child broadcasts from the same single elite — the
        single-elite-broadcast pattern.
        """
        if not parent_ids:
            return None
        return parent_ids[child_index % len(parent_ids)]

    def checkpoint_path(
        self,
        output_dir: Path,
        generation: int,
        genome_id: str,
    ) -> Path | None:
        """Return the canonical ``inheritance/gen-NNN/genome-<gid>.pt`` path.

        Returns the WEIGHTS checkpoint path (the Lamarckian ``.pt`` artefact)
        — NOT the substrate ``.tei.pt`` path. The substrate path is owned
        by the F0 substrate-extraction pipeline (``_run_f0_substrate_extraction``
        in the loop) as a separate concern; the loop's per-child
        inheritance plumbing reads only this single path. Identical to
        :class:`LamarckianInheritance.checkpoint_path`.
        """
        return output_dir / "inheritance" / f"gen-{generation:03d}" / f"genome-{genome_id}.pt"

    def kind(self) -> Literal["weights+transgenerational"]:
        """Return ``"weights+transgenerational"`` — the new composed dispatch value.

        The loop's ``_inheritance_active()`` predicate evaluates
        ``kind() in {"weights", "weights+transgenerational"}`` and
        the ``_substrate_inheritance_active()`` predicate evaluates
        ``kind() in {"transgenerational", "weights+transgenerational"}``,
        so BOTH the weight-IO code path AND the substrate-flow code
        path fire under composed mode without any new plumbing.
        """
        return "weights+transgenerational"
