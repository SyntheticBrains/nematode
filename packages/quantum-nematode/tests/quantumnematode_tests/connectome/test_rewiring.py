"""Tests for the degree-preserving rewired-null connectome utility."""

from collections import Counter

import numpy as np
import pytest
from quantumnematode.connectome.model import (
    ChemicalSynapse,
    Connectome,
    GapJunction,
    Neuron,
)
from quantumnematode.connectome.rewiring import rewire_degree_preserving

_N = 20  # neurons n00..n19 - large enough that distinct seeds reliably diverge


def _synthetic_connectome() -> Connectome:
    """Build a regular synthetic connectome: chemical out/in-degree 2, gap degree 2, all neurons."""
    names = [f"n{i:02d}" for i in range(_N)]
    neurons = {name: Neuron(name=name, cell_class="interneuron") for name in names}
    # Directed chemical: i -> i+1 and i -> i+7 (mod N) - every node out-degree 2, in-degree 2.
    chem = [
        ChemicalSynapse(pre=names[i], post=names[(i + step) % _N], weight=1)
        for i in range(_N)
        for step in (1, 7)
    ]
    # Undirected gap: {i, i+3} canonicalised - every node degree 2.
    gap = []
    for i in range(_N):
        a, b = names[i], names[(i + 3) % _N]
        lo, hi = (a, b) if a < b else (b, a)
        gap.append(GapJunction(neuron_a=lo, neuron_b=hi, weight=1))
    return Connectome(
        neurons=neurons,
        chemical_synapses=chem,
        gap_junctions=gap,
        source="synthetic",
        version="test",
    )


def _chem_out(c: Connectome) -> Counter:
    return Counter(s.pre for s in c.chemical_synapses)


def _chem_in(c: Connectome) -> Counter:
    return Counter(s.post for s in c.chemical_synapses)


def _gap_deg(c: Connectome) -> Counter:
    d: Counter = Counter()
    for g in c.gap_junctions:
        d[g.neuron_a] += 1
        d[g.neuron_b] += 1
    return d


def test_preserves_chemical_in_out_degree():
    """Every neuron's chemical out-degree and in-degree are unchanged by rewiring."""
    wild = _synthetic_connectome()
    rewired = rewire_degree_preserving(wild, np.random.default_rng(0))
    assert _chem_out(rewired) == _chem_out(wild)
    assert _chem_in(rewired) == _chem_in(wild)


def test_preserves_gap_degree():
    """Every neuron's gap-junction degree is unchanged by rewiring."""
    wild = _synthetic_connectome()
    rewired = rewire_degree_preserving(wild, np.random.default_rng(0))
    assert _gap_deg(rewired) == _gap_deg(wild)


def test_no_self_loops_or_duplicates():
    """The rewired graph is simple: no self-loops, no duplicate edges."""
    rewired = rewire_degree_preserving(_synthetic_connectome(), np.random.default_rng(3))
    chem_pairs = [(s.pre, s.post) for s in rewired.chemical_synapses]
    assert all(p != q for p, q in chem_pairs)
    assert len(chem_pairs) == len(set(chem_pairs))
    gap_pairs = [(g.neuron_a, g.neuron_b) for g in rewired.gap_junctions]
    assert all(a != b for a, b in gap_pairs)  # GapJunction validator also enforces a < b
    assert len(gap_pairs) == len(set(gap_pairs))


def test_preserves_neuron_set_and_edge_counts():
    """Node set + ordering and edge counts are preserved (only which pairs connect changes)."""
    wild = _synthetic_connectome()
    rewired = rewire_degree_preserving(wild, np.random.default_rng(1))
    assert list(rewired.neurons) == list(wild.neurons)
    assert len(rewired.chemical_synapses) == len(wild.chemical_synapses)
    assert len(rewired.gap_junctions) == len(wild.gap_junctions)
    assert rewired.source == "synthetic+rewired_degree_preserving"


def test_actually_rewires():
    """Rewiring changes the connected pairs (it is not a no-op on a mixable graph)."""
    wild = _synthetic_connectome()
    rewired = rewire_degree_preserving(wild, np.random.default_rng(0))
    wild_pairs = {(s.pre, s.post) for s in wild.chemical_synapses}
    rewired_pairs = {(s.pre, s.post) for s in rewired.chemical_synapses}
    assert wild_pairs != rewired_pairs


def test_deterministic_under_same_seed():
    """Same seed -> identical rewiring; different seeds -> different rewiring."""
    wild = _synthetic_connectome()
    a = rewire_degree_preserving(wild, np.random.default_rng(7))
    b = rewire_degree_preserving(wild, np.random.default_rng(7))
    c = rewire_degree_preserving(wild, np.random.default_rng(8))
    a_pairs = [(s.pre, s.post) for s in a.chemical_synapses]
    b_pairs = [(s.pre, s.post) for s in b.chemical_synapses]
    c_pairs = [(s.pre, s.post) for s in c.chemical_synapses]
    assert a_pairs == b_pairs
    assert a_pairs != c_pairs


def test_output_lists_are_sorted():
    """The rewired Connectome keeps the model's deterministic sorted edge order."""
    rewired = rewire_degree_preserving(_synthetic_connectome(), np.random.default_rng(2))
    chem = [(s.pre, s.post) for s in rewired.chemical_synapses]
    gap = [(g.neuron_a, g.neuron_b) for g in rewired.gap_junctions]
    assert chem == sorted(chem)
    assert gap == sorted(gap)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_degree_preserved_across_seeds(seed):
    """Degree preservation holds for every seed (chemical + gap), not just one draw."""
    wild = _synthetic_connectome()
    rewired = rewire_degree_preserving(wild, np.random.default_rng(seed))
    assert _chem_out(rewired) == _chem_out(wild)
    assert _chem_in(rewired) == _chem_in(wild)
    assert _gap_deg(rewired) == _gap_deg(wild)


def _irregular_connectome() -> Connectome:
    """Build a connectome with heterogeneous, distinct in/out degrees (catches in/out confusion)."""
    n = 12
    names = [f"m{i:02d}" for i in range(n)]
    neurons = {name: Neuron(name=name, cell_class="interneuron") for name in names}
    chem, seen = [], set()
    for i in range(n):
        for k in range(1, (i % 3) + 2):  # out-degree 1..3, varies by node
            e = (names[i], names[(i + k) % n])
            if e[0] != e[1] and e not in seen:
                seen.add(e)
                chem.append(ChemicalSynapse(pre=e[0], post=e[1], weight=1))
    gap, gseen = [], set()
    for i in range(0, n, 2):  # only even nodes seed a gap edge -> heterogeneous gap degree
        a, b = sorted((names[i], names[(i + 3) % n]))
        if a != b and (a, b) not in gseen:
            gseen.add((a, b))
            gap.append(GapJunction(neuron_a=a, neuron_b=b, weight=1))
    return Connectome(
        neurons=neurons,
        chemical_synapses=chem,
        gap_junctions=gap,
        source="synthetic-irregular",
        version="test",
    )


def test_degree_preserved_on_irregular_graph():
    """On a non-regular graph, in- and out-degree are preserved *independently*, and edges move."""
    wild = _irregular_connectome()
    assert len(set(_chem_out(wild).values())) > 1  # the fixture is genuinely irregular
    rewired = rewire_degree_preserving(wild, np.random.default_rng(0))
    assert _chem_out(rewired) == _chem_out(wild)  # out-degree per node preserved
    assert _chem_in(rewired) == _chem_in(wild)  # in-degree per node preserved (independently)
    assert _gap_deg(rewired) == _gap_deg(wild)
    wild_pairs = {(s.pre, s.post) for s in wild.chemical_synapses}
    rewired_pairs = {(s.pre, s.post) for s in rewired.chemical_synapses}
    assert wild_pairs != rewired_pairs  # non-trivial: edges actually moved


def test_rejects_duplicate_input_edges():
    """A non-simple input (parallel edges) is rejected rather than silently corrupting weights."""
    wild = _synthetic_connectome()
    dup = wild.chemical_synapses[0]
    bad = wild.model_copy(update={"chemical_synapses": [*wild.chemical_synapses, dup]})
    with pytest.raises(ValueError, match="simple connectome"):
        rewire_degree_preserving(bad, np.random.default_rng(0))
