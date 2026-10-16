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
    from collections.abc import Collection

    import numpy as np

_DEFAULT_SWAPS_PER_EDGE = 10
_MAX_ATTEMPTS_PER_TARGET = 100  # safety cap: give up mixing rather than loop forever
_MIN_UNDIRECTED_SWAP_NODES = 4  # an undirected swap needs four distinct nodes (no self-loop)
_ALT_PAIRING_PROB = 0.5  # coin flip between the two undirected rewirings, for mixing
# coin flip for which of a focal swap's two new pairs carries the focal count
_FOCAL_SIDE_PROB = 0.5


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


def _undirected_double_edge_swap(  # noqa: PLR0913 - the swap plus its two narrowing sets
    edges: list[tuple[str, str]],
    weights: dict[tuple[str, str], int],
    rng: np.random.Generator,
    target_swaps: int,
    *,
    forbidden: Collection[tuple[str, str]] = (),
    focal: Collection[int] | None = None,
) -> int:
    """Rewire an undirected (canonical ``a<b``) edge list in place, preserving every node's degree.

    Each accepted swap takes two edges on four distinct nodes ``{a,b}, {c,d}`` to one of the two
    alternative pairings (``{a,c},{b,d}`` or ``{a,d},{b,c}``), rejecting duplicates. Requiring four
    distinct nodes avoids self-loops; degree is conserved by construction.

    ``forbidden`` names pairs held outside ``edges`` that a swap must not create, so a held edge is
    never duplicated. ``focal``, if given, is a set of positions in ``edges``: each focal position
    in turn is swapped once with a partner drawn from the other positions, retried until a swap is
    accepted or the attempt cap is reached, so each focal edge moves once and displaces one
    partner; ``target_swaps`` is then ignored. Returns the accepted swap count.
    """
    edge_set = set(edges) | set(forbidden)
    n = len(edges)
    if focal is not None:
        return _focal_swaps(edges, weights, rng, edge_set, sorted(focal))
    accepted = 0
    attempts = 0
    max_attempts = target_swaps * _MAX_ATTEMPTS_PER_TARGET
    while accepted < target_swaps and attempts < max_attempts:
        attempts += 1
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i != j and _try_undirected_swap(edges, weights, rng, edge_set, i, j):
            accepted += 1
    if accepted < target_swaps:
        logger.warning(
            "Undirected (gap-junction) rewiring reached only %d/%d swaps in %d attempts.",
            accepted,
            target_swaps,
            attempts,
        )
    return accepted


def _try_undirected_swap(  # noqa: PLR0913 - one swap reads the whole edge state
    edges: list[tuple[str, str]],
    weights: dict[tuple[str, str], int],
    rng: np.random.Generator,
    edge_set: set[tuple[str, str]],
    i: int,
    j: int,
) -> bool:
    """Try one degree-preserving swap of positions ``i`` and ``j``; return whether it was made."""
    a, b = edges[i]
    c, d = edges[j]
    if len({a, b, c, d}) < _MIN_UNDIRECTED_SWAP_NODES:  # four distinct nodes -> no self-loop
        return False
    p, q = ((a, c), (b, d)) if rng.random() < _ALT_PAIRING_PROB else ((a, d), (b, c))
    new_i = (p[0], p[1]) if p[0] < p[1] else (p[1], p[0])
    new_j = (q[0], q[1]) if q[0] < q[1] else (q[1], q[0])
    if new_i in edge_set or new_j in edge_set:
        return False
    old_i, old_j = edges[i], edges[j]
    edge_set.discard(old_i)
    edge_set.discard(old_j)
    edges[i], edges[j] = new_i, new_j
    edge_set.add(new_i)
    edge_set.add(new_j)
    weights[new_i] = weights.pop(old_i)
    weights[new_j] = weights.pop(old_j)
    return True


def _focal_swaps(
    edges: list[tuple[str, str]],
    weights: dict[tuple[str, str], int],
    rng: np.random.Generator,
    edge_set: set[tuple[str, str]],
    focal: list[int],
) -> int:
    """Swap each focal position once, in order, with a partner no earlier swap has used.

    A swap keeps each edge's first-sorted endpoint on the pair its weight moves to, so a single
    swap would always leave a focal count on that endpoint. After each accepted swap a fair coin
    decides which of the two new pairs carries the focal count, so it lands on a pair containing
    either of its endpoints with equal chance.
    """
    focal_set = set(focal)
    available = [k for k in range(len(edges)) if k not in focal_set]
    accepted = 0
    for i in focal:
        for _ in range(_MAX_ATTEMPTS_PER_TARGET):
            if not available:
                break
            slot = int(rng.integers(0, len(available)))
            j = available[slot]
            if _try_undirected_swap(edges, weights, rng, edge_set, i, j):
                if rng.random() < _FOCAL_SIDE_PROB:
                    weights[edges[i]], weights[edges[j]] = weights[edges[j]], weights[edges[i]]
                available.pop(slot)
                accepted += 1
                break
    if accepted < len(focal):
        logger.warning("Focal gap swap moved %d of %d focal pairs.", accepted, len(focal))
    return accepted


def _rewire_gaps(  # noqa: PLR0913 - the narrowed swap's inputs, passed through
    gap_edges: list[tuple[str, str]],
    gap_weights: dict[tuple[str, str], int],
    rng: np.random.Generator,
    swaps_per_edge: int,
    hold_gap_pairs: Collection[tuple[str, str]] | None,
    focal_gap_pairs: Collection[tuple[str, str]] | None,
    report: dict[str, int] | None,
) -> list[tuple[str, str]]:
    """Run the undirected swap, narrowed by a held or a focal set, and return the new pair list.

    With neither set this is the plain swap over every pair, drawing exactly as before the sets
    existed.
    """
    input_pairs = set(gap_edges)
    held = set(hold_gap_pairs) if hold_gap_pairs is not None else set()
    focal_pairs = set(focal_gap_pairs) if focal_gap_pairs is not None else set()
    unknown = (held | focal_pairs) - input_pairs
    if unknown:
        msg = f"gap pairs absent from the input: {sorted(unknown)}"
        raise ValueError(msg)
    swappable = [edge for edge in gap_edges if edge not in held]
    focal = None
    target = swaps_per_edge * len(swappable)
    if focal_gap_pairs is not None:
        focal = {k for k, edge in enumerate(swappable) if edge in focal_pairs}
        target = len(focal)
    accepted = _undirected_double_edge_swap(
        swappable,
        gap_weights,
        rng,
        target,
        forbidden=held,
        focal=focal,
    )
    result = swappable + [edge for edge in gap_edges if edge in held]
    if report is not None:
        report.update(accepted=accepted, target=target, moved=len(input_pairs - set(result)))
    return result


def rewire_degree_preserving(  # noqa: PLR0913 - each keyword narrows what the null rewires
    connectome: Connectome,
    rng: np.random.Generator,
    swaps_per_edge: int = _DEFAULT_SWAPS_PER_EDGE,
    *,
    rewire_gap_junctions: bool = True,
    preserve_autapses: bool = False,
    hold_boundary: tuple[Collection[str], Collection[str]] | None = None,
    rewire_chemical: bool = True,
    origins: dict[tuple[str, str], tuple[str, str]] | None = None,
    hold_gap_pairs: Collection[tuple[str, str]] | None = None,
    focal_gap_pairs: Collection[tuple[str, str]] | None = None,
    gap_swap_report: dict[str, int] | None = None,
) -> Connectome:
    """Return a degree-preserving rewired copy of ``connectome``.

    The neuron set and ordering are untouched (so downstream index stability, per-post fan-in, and
    hence the strict-mask / weight-init scale / gap-junction normalisation are preserved); only the
    chemical and gap-junction edge sets are rewired by independent seeded double-edge-swaps, each
    running ``swaps_per_edge * |E|`` accepted swaps for mixing. Deterministic given ``rng``'s seed.

    Four options narrow what the null changes:

    * ``rewire_gap_junctions=False`` skips the undirected swap, so gap junctions keep their pairs
      and counts, and every neuron keeps its total gap-junction strength.
    * ``preserve_autapses=True`` takes the self-loops out of the directed swap and puts them back
      unchanged, so every autapse keeps its count. In- and out-degree stay exact, since each
      autapse contributes one of each to its own neuron.
    * ``hold_boundary=(sensory, motor)`` takes out of the directed swap every chemical edge
      leaving a neuron in ``sensory`` or entering a neuron in ``motor``, and puts them back
      unchanged. Only the interior -- edges from a non-sensory neuron to a non-motor one -- is
      rewired. A swap of two interior edges yields interior edges, so it can never recreate a
      held one, and degrees stay exact. Every chemical route of one or two hops from a sensory
      neuron to a motor neuron is kept, since its first edge leaves a sensory neuron and its last
      enters a motor neuron.

    * ``rewire_chemical=False`` skips the directed swap, so the chemical graph, autapses included,
      is the input's exactly; with ``rewire_gap_junctions`` left true the null differs in its gap
      junctions alone, placement and each neuron's total gap strength together.

    Two further options narrow the gap-junction swap, and need ``rewire_gap_junctions``:

    * ``hold_gap_pairs`` takes those gap pairs out of the undirected swap and puts them back with
      their counts; the rest are swapped among themselves, never onto a held pair.
    * ``focal_gap_pairs`` swaps each of those pairs once, in turn, with a partner drawn from the
      pairs no earlier swap has used, so each focal pair's count moves to a new pair containing
      either of its endpoints with equal chance, and the other pairs move only as the one
      partner each displaces; ``swaps_per_edge`` does not apply.

    ``gap_swap_report``, if given, receives the undirected swap's ``accepted`` and ``target``
    counts, and ``moved``: the number of the input's gap pairs absent from the result.

    ``origins``, if given, is filled with each rewired chemical edge's origin: the input edge whose
    place it took. A swap edits the edge list in place, keeping each edge's pre-synaptic neuron and
    moving its target, so an edge's origin shares its pre-synaptic neuron, and its count is the
    origin's. Edges the swap does not touch map to themselves. Recording origins draws nothing.

    At their defaults the options leave the draws, and so the rewired graph, exactly as they were.
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
    held: list[tuple[str, str]] = []
    if hold_boundary is not None:
        sensory, motor = frozenset(hold_boundary[0]), frozenset(hold_boundary[1])
        held = [edge for edge in chem_edges if edge[0] in sensory or edge[1] in motor]
        chem_edges = [
            edge for edge in chem_edges if edge[0] not in sensory and edge[1] not in motor
        ]
    slots = list(chem_edges)
    if rewire_chemical:
        _directed_double_edge_swap(chem_edges, chem_weights, rng, swaps_per_edge * len(chem_edges))
    if origins is not None:
        origins.update(zip(chem_edges, slots, strict=True))
        origins.update((edge, edge) for edge in autapses + held)
    chem_edges += autapses + held

    gap_weights = {(g.neuron_a, g.neuron_b): g.weight for g in connectome.gap_junctions}
    if (hold_gap_pairs is not None or focal_gap_pairs is not None) and not rewire_gap_junctions:
        msg = "hold_gap_pairs and focal_gap_pairs narrow the gap swap and need rewire_gap_junctions"
        raise ValueError(msg)
    if hold_gap_pairs is not None and focal_gap_pairs is not None:
        msg = "hold_gap_pairs and focal_gap_pairs are alternatives; pass one"
        raise ValueError(msg)
    if rewire_gap_junctions:
        gap_edges = _rewire_gaps(
            gap_edges,
            gap_weights,
            rng,
            swaps_per_edge,
            hold_gap_pairs,
            focal_gap_pairs,
            gap_swap_report,
        )

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
        + ("[autapses_held]" if preserve_autapses else "")
        + ("[boundary_held]" if hold_boundary is not None else "")
        + ("" if rewire_chemical else "[chemical_held]")
        + ("[gap_pairs_held]" if hold_gap_pairs is not None else "")
        + ("[gap_pairs_focal]" if focal_gap_pairs is not None else ""),
        version=connectome.version,
    )
