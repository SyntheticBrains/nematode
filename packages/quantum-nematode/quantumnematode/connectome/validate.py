"""Connectome validation — structural + cross-dataset divergence checks.

Three public entrypoints:

- ``validate_neuron_count(c)`` — confirms the neuron count matches the
  expected size (302 for Cook 2019 hermaphrodite).
- ``validate_known_pathways(c)`` — confirms at least one of the canonical
  sensory → interneuron → motor pathways (klinotaxis, thermotaxis,
  nociception) traces successfully through the connectome's chemical
  synapses. Pathways are drawn from the Bargmann lab literature
  (Gray et al. 2005; Iino & Yoshida 2009).
- ``cross_validate(primary, secondary)`` — compares two connectomes
  (typically Cook 2019 hermaphrodite against Witvliet 2021 dataset 8)
  on the shared neuron subset, producing a ``DivergenceReport`` for
  forensic inspection.

The validators report ``ValidationResult`` (or ``DivergenceReport`` for
``cross_validate``) rather than raising — callers are expected to inspect
the result and decide whether the divergence is acceptable.
"""

from __future__ import annotations

from statistics import mean, median
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from quantumnematode.connectome.model import Connectome


EXPECTED_HERMAPHRODITE_NEURON_COUNT = 302
"""Expected neuron count for the *C. elegans* hermaphrodite connectome."""


KNOWN_PATHWAYS: dict[str, tuple[str, ...]] = {
    # Klinotaxis (chemotaxis via head-sweep gradient detection).
    # Gray, Hill & Bargmann 2005; Iino & Yoshida 2009.
    "klinotaxis": ("ASEL", "AIYL", "RIAL", "SMDDL"),
    # Thermotaxis (AFD thermosensory pathway).
    # Mori & Ohshima 1995; Hedgecock & Russell 1975.
    "thermotaxis": ("AFDL", "AIYL", "RIAL", "SMDDL"),
    # Nociception (ASH-mediated avoidance via command interneurons).
    "nociception": ("ASHL", "AVAL", "VA1"),
}
"""Canonical sensory → interneuron → motor pathways from the literature.

Each entry is an ordered tuple of neuron names; a pathway is "traced
successfully" if every consecutive pair has at least one chemical
synapse in the connectome.
"""


class ValidationResult(BaseModel):
    """Result of a single structural validator."""

    passed: bool
    summary: str
    details: dict[str, str | int | bool | list[str]] = {}


def validate_neuron_count(
    connectome: Connectome,
    *,
    expected: int = EXPECTED_HERMAPHRODITE_NEURON_COUNT,
) -> ValidationResult:
    """Check that the connectome's neuron count matches ``expected``."""
    actual = len(connectome.neurons)
    return ValidationResult(
        passed=actual == expected,
        summary=(
            f"neuron count: actual={actual}, expected={expected}, "
            f"{'pass' if actual == expected else 'FAIL'}"
        ),
        details={"actual": actual, "expected": expected},
    )


def _trace_pathway(
    connectome: Connectome,
    pathway: tuple[str, ...],
) -> bool:
    """Return True if every consecutive (a, b) in pathway has an a -> b synapse."""
    edges = {(s.pre, s.post) for s in connectome.chemical_synapses}
    return all((pathway[i], pathway[i + 1]) in edges for i in range(len(pathway) - 1))


def validate_known_pathways(connectome: Connectome) -> ValidationResult:
    """Check that at least one canonical sensory→interneuron→motor pathway traces.

    Passes if any of the documented pathways (klinotaxis, thermotaxis,
    nociception) can be traced through the connectome's chemical-synapse
    adjacency. The result's ``details`` lists which pathways traced.
    """
    traced: list[str] = []
    failed: list[str] = []
    for name, pathway in KNOWN_PATHWAYS.items():
        if _trace_pathway(connectome, pathway):
            traced.append(name)
        else:
            failed.append(name)

    passed = len(traced) > 0
    return ValidationResult(
        passed=passed,
        summary=(
            f"known pathways: {len(traced)}/{len(KNOWN_PATHWAYS)} traced"
            f" ({'pass' if passed else 'FAIL'})"
        ),
        details={"traced": traced, "failed": failed},
    )


class DivergenceReport(BaseModel):
    """Cross-dataset comparison between two connectomes on shared neurons.

    Attributes
    ----------
    primary_source
        ``source`` field of the primary (reference) connectome.
    secondary_source
        ``source`` field of the secondary (comparison) connectome.
    shared_neurons
        Count of neurons present in both connectomes.
    shared_pairs_agreement
        Count of (pre, post) pairs (within the shared neuron subset) where
        both connectomes either both report a chemical synapse OR both
        report no chemical synapse.
    shared_pairs_disagreement
        Count of (pre, post) pairs where the two connectomes disagree on
        chemical-synapse presence.
    primary_only_pairs
        Chemical-synapse (pre, post) pairs that appear in the primary
        connectome but not in the secondary, restricted to the shared
        neuron subset. Capped at 50 entries (full list available via
        ``cross_validate(..., truncate_lists=False)``).
    secondary_only_pairs
        Chemical-synapse pairs in the secondary but not the primary.
    weight_divergence_summary
        For pairs both connectomes report, the ratio
        ``secondary_weight / primary_weight``. Reports
        ``{"mean": …, "median": …, "n_pairs": …}``.
    """

    primary_source: str
    secondary_source: str
    shared_neurons: int
    shared_pairs_agreement: int
    shared_pairs_disagreement: int
    primary_only_pairs: list[tuple[str, str]]
    secondary_only_pairs: list[tuple[str, str]]
    weight_divergence_summary: dict[str, float]


def cross_validate(
    primary: Connectome,
    secondary: Connectome,
    *,
    list_cap: int = 50,
) -> DivergenceReport:
    """Diff two connectomes on the shared neuron subset.

    Parameters
    ----------
    primary
        Reference connectome (typically Cook 2019 hermaphrodite).
    secondary
        Comparison connectome (typically Witvliet 2021 dataset 8).
    list_cap
        Maximum number of entries to include in ``primary_only_pairs`` and
        ``secondary_only_pairs``. Caps prevent the report from ballooning;
        callers wanting the full list can post-process the connectomes
        directly.
    """
    shared_neurons = set(primary.neurons) & set(secondary.neurons)

    primary_pairs = {
        (s.pre, s.post): s.weight
        for s in primary.chemical_synapses
        if s.pre in shared_neurons and s.post in shared_neurons
    }
    secondary_pairs = {
        (s.pre, s.post): s.weight
        for s in secondary.chemical_synapses
        if s.pre in shared_neurons and s.post in shared_neurons
    }

    all_shared_pairs = set(primary_pairs) | set(secondary_pairs)
    primary_set = set(primary_pairs)
    secondary_set = set(secondary_pairs)

    agreement_count = sum(1 for p in all_shared_pairs if (p in primary_set) == (p in secondary_set))
    disagreement_count = len(all_shared_pairs) - agreement_count

    primary_only = sorted(primary_set - secondary_set)
    secondary_only = sorted(secondary_set - primary_set)

    # Weight-divergence ratios for pairs both connectomes report
    both = primary_set & secondary_set
    ratios = [secondary_pairs[p] / primary_pairs[p] for p in both if primary_pairs[p] > 0]
    weight_summary: dict[str, float] = {"n_pairs": float(len(ratios))}
    if ratios:
        weight_summary["mean"] = float(mean(ratios))
        weight_summary["median"] = float(median(ratios))

    return DivergenceReport(
        primary_source=primary.source,
        secondary_source=secondary.source,
        shared_neurons=len(shared_neurons),
        shared_pairs_agreement=agreement_count,
        shared_pairs_disagreement=disagreement_count,
        primary_only_pairs=primary_only[:list_cap],
        secondary_only_pairs=secondary_only[:list_cap],
        weight_divergence_summary=weight_summary,
    )
