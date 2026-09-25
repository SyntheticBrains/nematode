"""Degree-preserving rewired-null connectome generation.

A control for the connectome architecture ranking: rewire the wild-type wiring so that every
neuron keeps its exact in/out degree (chemical) and degree (gap junction), but *which* neurons
connect is scrambled. Comparing the wild-type connectome against these degree-matched nulls, under
matched initialisation and training budget, separates "the specific *C. elegans* wiring matters"
from "only the degree/sparsity statistics matter".

The rewiring is a seeded **double-edge-swap** - directed (configuration-model) for chemical
synapses, undirected for gap junctions - which preserves the degree sequence exactly by
construction. A naive random-rewiring null (which destroys the degree sequence) is a weaker,
uninteresting control; the degree-preserving swap is the standard.

Degree is all it preserves. Existing autapses can be swapped away and none are created (see
``rewire_degree_preserving``), and synapse and gap-junction counts travel with their edges, so a
neuron's total incoming count - and, since gap-junction counts are coupling weights, its total
gap-junction strength - generally changes.

No graph library is required: the swap is a few lines on the seeded ``numpy`` RNG.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quantumnematode.connectome.model import ChemicalSynapse, Connectome, GapJunction
from quantumnematode.logging_config import logger

if TYPE_CHECKING:
    import numpy as np

_DEFAULT_SWAPS_PER_EDGE = 10
_MAX_ATTEMPTS_PER_TARGET = 100  # safety cap: give up mixing rather than loop forever
_MIN_UNDIRECTED_SWAP_NODES = 4  # an undirected swap needs four distinct nodes (no self-loop)
_ALT_PAIRING_PROB = 0.5  # coin flip between the two undirected rewirings, for mixing


def _directed_double_edge_swap(
    edges: list[tuple[str, str]],
    weights: dict[tuple[str, str], int],
    rng: np.random.Generator,
    target_swaps: int,
) -> None:
    """Rewire a directed edge list in place, preserving every node's in/out degree.

    Each accepted swap takes two distinct edges ``(a→b), (c→d)`` to ``(a→d), (c→b)``, rejecting any
    swap that would create a self-loop or a duplicate edge. Out-degree (``a``, ``c``) and in-degree
    (``b``, ``d``) are conserved by construction. Synapse counts travel with the edge, so the
    graph's global multiset of counts is preserved; an individual neuron's incoming counts are not
    whenever the two swapped edges carried different counts, because ``d`` receives the count of
    ``(a→b)`` in place of the count of ``(c→d)``. The strict mask reads only presence, but
    count-scaled weight initialisation reads the counts, so on a rewired graph each edge inherits
    the count of the edge it was swapped from.
    """
    edge_set = set(edges)
    n = len(edges)
    accepted = 0
    attempts = 0
    max_attempts = target_swaps * _MAX_ATTEMPTS_PER_TARGET
    while accepted < target_swaps and attempts < max_attempts:
        attempts += 1
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i == j:
            continue
        a, b = edges[i]
        c, d = edges[j]
        new_i, new_j = (a, d), (c, b)
        if a == d or c == b:  # self-loop
            continue
        if new_i in edge_set or new_j in edge_set:  # duplicate edge
            continue
        old_i, old_j = edges[i], edges[j]
        edge_set.discard(old_i)
        edge_set.discard(old_j)
        edges[i], edges[j] = new_i, new_j
        edge_set.add(new_i)
        edge_set.add(new_j)
        weights[new_i] = weights.pop(old_i)  # (a→d) carries (a→b)'s weight
        weights[new_j] = weights.pop(old_j)  # (c→b) carries (c→d)'s weight
        accepted += 1
    if accepted < target_swaps:
        logger.warning(
            "Directed rewiring reached only %d/%d swaps in %d attempts "
            "(sparse graph, low acceptance) - reported, not reseeded.",
            accepted,
            target_swaps,
            attempts,
        )


def _undirected_double_edge_swap(
    edges: list[tuple[str, str]],
    weights: dict[tuple[str, str], int],
    rng: np.random.Generator,
    target_swaps: int,
) -> None:
    """Rewire an undirected (canonical ``a<b``) edge list in place, preserving every node's degree.

    Each accepted swap takes two edges on four distinct nodes ``{a,b}, {c,d}`` to one of the two
    alternative pairings (``{a,c},{b,d}`` or ``{a,d},{b,c}``), rejecting duplicates. Requiring four
    distinct nodes avoids self-loops; degree is conserved by construction.
    """
    edge_set = set(edges)
    n = len(edges)
    accepted = 0
    attempts = 0
    max_attempts = target_swaps * _MAX_ATTEMPTS_PER_TARGET
    while accepted < target_swaps and attempts < max_attempts:
        attempts += 1
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i == j:
            continue
        a, b = edges[i]
        c, d = edges[j]
        if len({a, b, c, d}) < _MIN_UNDIRECTED_SWAP_NODES:  # four distinct nodes -> no self-loop
            continue
        if rng.random() < _ALT_PAIRING_PROB:
            p, q = (a, c), (b, d)
        else:
            p, q = (a, d), (b, c)
        new_i = (p[0], p[1]) if p[0] < p[1] else (p[1], p[0])
        new_j = (q[0], q[1]) if q[0] < q[1] else (q[1], q[0])
        if new_i in edge_set or new_j in edge_set:
            continue
        old_i, old_j = edges[i], edges[j]
        edge_set.discard(old_i)
        edge_set.discard(old_j)
        edges[i], edges[j] = new_i, new_j
        edge_set.add(new_i)
        edge_set.add(new_j)
        weights[new_i] = weights.pop(old_i)
        weights[new_j] = weights.pop(old_j)
        accepted += 1
    if accepted < target_swaps:
        logger.warning(
            "Undirected (gap-junction) rewiring reached only %d/%d swaps in %d attempts.",
            accepted,
            target_swaps,
            attempts,
        )


def rewire_degree_preserving(
    connectome: Connectome,
    rng: np.random.Generator,
    swaps_per_edge: int = _DEFAULT_SWAPS_PER_EDGE,
    *,
    rewire_gap_junctions: bool = True,
    preserve_autapses: bool = False,
) -> Connectome:
    """Return a degree-preserving rewired copy of ``connectome``.

    The neuron set and ordering are untouched (so downstream index stability, per-post fan-in, and
    hence the strict-mask / weight-init scale / gap-junction normalisation are preserved); only the
    chemical and gap-junction edge sets are rewired by independent seeded double-edge-swaps, each
    running ``swaps_per_edge * |E|`` accepted swaps for mixing. Deterministic given ``rng``'s seed.

    Two options narrow what the null changes, so it differs from the input in chemical placement
    alone:

    * ``rewire_gap_junctions=False`` skips the undirected swap, so gap junctions keep their pairs
      and counts, and every neuron keeps its total gap-junction strength.
    * ``preserve_autapses=True`` takes the self-loops out of the directed swap and puts them back
      unchanged, so every autapse keeps its count. In- and out-degree stay exact, since each
      autapse contributes one of each to its own neuron.

    At their defaults both options leave the draws, and so the rewired graph, exactly as they were.
    With ``preserve_autapses`` the directed swap runs on a shorter list and draws differently, so at
    one seed its chemical graph is a different sample from the default null's; with
    ``rewire_gap_junctions=False`` alone the chemical graph is identical to the default null's.
    """
    chem_edges = [(s.pre, s.post) for s in connectome.chemical_synapses]
    gap_edges = [(g.neuron_a, g.neuron_b) for g in connectome.gap_junctions]
    # Guard against PARALLEL edges only (the weight dicts below key on the edge tuple and would
    # silently collapse duplicates). Self-loops (autapses) in the input are accepted: the real Cook
    # connectome contains 38 chemical autapses. They are NOT preserved, though. The swap rejects any
    # move that would create a self-loop but not one that removes one, so rewiring can remove
    # existing autapses and never creates new ones. In- and out-degree are still preserved exactly.
    # Gap-junction counts travel with their edges, so each neuron keeps its gap degree but not its
    # total gap-junction strength. The Cook loader dedups both edge types, so the parallel-edge
    # guard protects a hand-built/alternate-loader fixture, not a live path.
    if len(set(chem_edges)) != len(chem_edges) or len(set(gap_edges)) != len(gap_edges):
        msg = "rewire_degree_preserving requires a simple connectome (no duplicate/parallel edges)"
        raise ValueError(msg)
    chem_weights = {(s.pre, s.post): s.weight for s in connectome.chemical_synapses}
    autapses: list[tuple[str, str]] = []
    if preserve_autapses:
        autapses = [edge for edge in chem_edges if edge[0] == edge[1]]
        chem_edges = [edge for edge in chem_edges if edge[0] != edge[1]]
    _directed_double_edge_swap(chem_edges, chem_weights, rng, swaps_per_edge * len(chem_edges))
    chem_edges += autapses

    gap_weights = {(g.neuron_a, g.neuron_b): g.weight for g in connectome.gap_junctions}
    if rewire_gap_junctions:
        _undirected_double_edge_swap(gap_edges, gap_weights, rng, swaps_per_edge * len(gap_edges))

    chemical_synapses = sorted(
        (ChemicalSynapse(pre=p, post=q, weight=chem_weights[(p, q)]) for p, q in chem_edges),
        key=lambda s: (s.pre, s.post),
    )
    gap_junctions = sorted(
        (GapJunction(neuron_a=a, neuron_b=b, weight=gap_weights[(a, b)]) for a, b in gap_edges),
        key=lambda g: (g.neuron_a, g.neuron_b),
    )
    return Connectome(
        neurons=connectome.neurons,
        chemical_synapses=chemical_synapses,
        gap_junctions=gap_junctions,
        source=f"{connectome.source}+rewired_degree_preserving"
        + ("" if rewire_gap_junctions else "[gap_junctions_held]")
        + ("[autapses_held]" if preserve_autapses else ""),
        version=connectome.version,
    )
