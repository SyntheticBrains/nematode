"""Connectome data model — Pydantic types for the *C. elegans* connectome.

Per phase6-tracking Decision 7, chemical synapses and gap junctions are
**separately-typed connection categories**. The dual-edge case (e.g.
AVAL <-> AVBL, connected by both a chemical synapse AND a gap junction) is
represented as two distinct entries — one ``ChemicalSynapse`` and one
``GapJunction`` — never as a single edge with two weight attributes.

Extra-synaptic / peptidergic signalling is NOT represented in this data
model; that work is reserved for Phase 7 L4 plasticity per phase6-tracking
Decision 7.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

CellClass = Literal["sensory", "interneuron", "motor", "muscle", "pharyngeal"]
"""Allowed cell-class labels per phase6-tracking Decision 7's taxonomy.

Source: derived from cect's ``Cells.py`` MIT-licensed Python constants,
which themselves attribute Cook et al. 2019 paper + WormAtlas.
"""


class Neuron(BaseModel):
    """A single named *C. elegans* neuron.

    Attributes
    ----------
    name
        Canonical *C. elegans* neuron name (e.g. ``"ASEL"``, ``"AVAL"``,
        ``"VB02"``). Bilateral pairs use suffixes ``L`` / ``R``; ventral-cord
        motor neurons use a numbered suffix (``VB01``..``VB11``,
        ``DB01``..``DB07``, etc.).
    cell_class
        One of ``"sensory"``, ``"interneuron"``, ``"motor"``, ``"muscle"``,
        ``"pharyngeal"``. Boundaries are convention-dependent for polymodal
        cells; this codebase defers to cect's ``Cells.py`` classification.
    neurotransmitter
        Primary neurotransmitter when known (e.g. ``"Glutamate"``,
        ``"GABA"``, ``"Acetylcholine"``), otherwise ``None``.
    """

    name: str = Field(..., min_length=1, description="Canonical neuron name.")
    cell_class: CellClass = Field(..., description="Anatomical cell class.")
    neurotransmitter: str | None = Field(
        default=None,
        description="Primary neurotransmitter when known.",
    )


class ChemicalSynapse(BaseModel):
    """A directed chemical synapse from ``pre`` to ``post``.

    Attributes
    ----------
    pre
        Name of the presynaptic neuron.
    post
        Name of the postsynaptic neuron.
    weight
        Synapse count from Cook et al. 2019 EM-derived data. Must be a
        positive integer; zero-weight rows are dropped at parse time (they
        indicate "no chemical synapse exists between this pair").
    """

    pre: str = Field(..., min_length=1)
    post: str = Field(..., min_length=1)
    weight: int = Field(..., gt=0, description="Cook 2019 synapse count.")


class GapJunction(BaseModel):
    """An undirected gap junction (electrical coupling).

    Stored in canonical form with ``neuron_a < neuron_b`` (lexicographic) so
    each unordered pair appears at most once. ``weight`` is the Cook 2019
    junction count; this is treated as a fixed physiological signal per
    phase6-tracking Decision 7 (NOT a learnable scalar).

    Attributes
    ----------
    neuron_a
        Name of one neuron in the gap-junction pair (lexicographically smaller).
    neuron_b
        Name of the other neuron in the pair (lexicographically larger).
    weight
        Junction count from Cook et al. 2019 EM-derived data.
    """

    neuron_a: str = Field(..., min_length=1)
    neuron_b: str = Field(..., min_length=1)
    weight: int = Field(..., ge=0, description="Cook 2019 junction count.")

    @model_validator(mode="after")
    def _enforce_canonical_pair_order(self) -> GapJunction:
        if self.neuron_a >= self.neuron_b:
            msg = (
                f"GapJunction requires neuron_a < neuron_b (got "
                f"{self.neuron_a!r}, {self.neuron_b!r}); "
                "gap junctions are undirected and stored in canonical order."
            )
            raise ValueError(msg)
        return self


class Connectome(BaseModel):
    """A complete *C. elegans* connectome — neurons + connections + provenance.

    Attributes
    ----------
    neurons
        Dict mapping neuron name to ``Neuron`` instance. For Cook 2019
        hermaphrodite, this contains 302 entries.
    chemical_synapses
        List of directed chemical synapses. Sorted by ``(pre, post)`` for
        determinism.
    gap_junctions
        List of undirected gap junctions in canonical (neuron_a < neuron_b)
        form. Sorted by ``(neuron_a, neuron_b)`` for determinism.
    source
        Short identifier for the upstream dataset
        (e.g. ``"cook_2019_hermaphrodite"``).
    version
        Free-form version / provenance string (vendored snapshot identifier;
        the actual value resolves from ``data/connectome/PROVENANCE.md``).
    """

    neurons: dict[str, Neuron]
    chemical_synapses: list[ChemicalSynapse]
    gap_junctions: list[GapJunction]
    source: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _enforce_neuron_set_integrity(self) -> Connectome:
        names = self.neurons.keys()
        for syn in self.chemical_synapses:
            if syn.pre not in names:
                msg = f"ChemicalSynapse references unknown pre={syn.pre!r}"
                raise ValueError(msg)
            if syn.post not in names:
                msg = f"ChemicalSynapse references unknown post={syn.post!r}"
                raise ValueError(msg)
        for gj in self.gap_junctions:
            if gj.neuron_a not in names:
                msg = f"GapJunction references unknown neuron_a={gj.neuron_a!r}"
                raise ValueError(msg)
            if gj.neuron_b not in names:
                msg = f"GapJunction references unknown neuron_b={gj.neuron_b!r}"
                raise ValueError(msg)
        return self
