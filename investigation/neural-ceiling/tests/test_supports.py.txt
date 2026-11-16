"""null-R's reciprocity-preserving swaps and sparse-ER's random supports, on small graphs."""

from collections import Counter

import numpy as np
import pytest

from ceiling import config as C
from ceiling import supports


def random_support(n: int, p: float, seed: int, reciprocal_bias: float = 0.4) -> supports.Support:
    rng = np.random.default_rng(seed)
    counts = {}
    for r in range(n):
        for c in range(n):
            if r != c and rng.random() < p:
                counts[(r, c)] = [float(rng.integers(1, 5)), 0.0]
    for (r, c) in list(counts):  # add reverses so a good share of entries are reciprocal
        if rng.random() < reciprocal_bias:
            counts.setdefault((c, r), [0.0, 0.0])
            g = float(rng.integers(1, 3))
            counts[(r, c)][1] = counts[(c, r)][1] = g
    return supports._from_counts("toy", n, counts)


def degrees(s: supports.Support) -> tuple[np.ndarray, np.ndarray]:
    return np.bincount(s.rows, minlength=s.n), np.bincount(s.cols, minlength=s.n)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_null_r_preserves_counts_degrees_and_reciprocity(seed):
    """Section 5, null-R: entries, every in- and out-degree and the reciprocal pairs are exact."""
    base = random_support(40, 0.08, seed)
    out = supports.null_r(base, 9781 + seed)
    assert out.size == base.size
    for a, b in zip(degrees(base), degrees(out)):
        assert np.array_equal(a, b)
    assert out.reciprocal_entries() == base.reciprocal_entries()
    assert np.all(out.rows != out.cols)
    assert len(out.keys()) == out.size  # no duplicates
    rep = out.report
    assert all(rep["checks"].values())
    assert 0.0 <= rep["fraction_moved"] <= 1.0
    assert rep["fraction_moved"] == pytest.approx(1 - len(base.keys() & out.keys()) / base.size)
    assert rep["accepted"]["undirected"] > 0 and rep["accepted"]["directed"] > 0
    assert rep["attempted_per_entry"] >= C.NULL_R_SWAPS_PER_ENTRY
    # counts travel with their entries: the multiset of (chemical, gap) counts is unchanged
    assert Counter(zip(base.chem, base.gap)) == Counter(zip(out.chem, out.gap))


def test_null_r_moves_most_entries_and_is_deterministic():
    base = random_support(60, 0.08, 7)
    a, b = supports.null_r(base, 9790), supports.null_r(base, 9790)
    assert a.keys() == b.keys()
    assert a.keys() != base.keys()
    assert a.report["fraction_moved"] >= C.NULL_R_MIN_MOVED


def test_null_r_extends_when_under_the_moved_bar(monkeypatch):
    """A draw moving under the bar continues, up to 30 attempted swaps per entry."""
    monkeypatch.setattr(C, "NULL_R_MIN_MOVED", 1.01)  # unreachable: must run to the cap
    out = supports.null_r(random_support(30, 0.1, 4), 9785)
    assert out.report["attempted_per_entry"] == C.NULL_R_MAX_SWAPS_PER_ENTRY
    assert not out.report["reached_min_moved"]


def test_directed_swap_never_creates_a_reciprocal_pair():
    # a purely directed graph (no reciprocal pairs) must stay purely directed
    base = random_support(30, 0.1, 5, reciprocal_bias=0.0)
    base_pairs = base.reciprocal_entries()
    out = supports.null_r(base, 9786)
    assert out.reciprocal_entries() == base_pairs


def test_sparse_er_size_and_diagonal():
    """Section 5, sparse-ER: the masked support's size, off the diagonal, no repeats."""
    s = supports.sparse_er(500, 50, 9801)
    assert s.size == 500
    assert np.all(s.rows != s.cols)
    assert len(s.keys()) == 500
    assert s.rows.max() < 50 and s.cols.max() < 50
    assert supports.sparse_er(500, 50, 9801).keys() == s.keys()
    assert supports.sparse_er(500, 50, 9802).keys() != s.keys()


def test_sparse_er_can_fill_every_off_diagonal_entry():
    s = supports.sparse_er(12 * 11, 12, 9803)
    assert s.keys() == {(r, c) for r in range(12) for c in range(12) if r != c}


def test_union_of_chemical_and_gap_entries():
    """The anatomical support: chemical pre -> post is (post, pre); a gap is both ways."""
    class Syn:
        def __init__(self, pre, post, weight): self.pre, self.post, self.weight = pre, post, weight

    class Gap:
        def __init__(self, a, b, weight): self.neuron_a, self.neuron_b, self.weight = a, b, weight

    class Conn:
        chemical_synapses = [Syn("A", "B", 3), Syn("B", "B", 2), Syn("C", "A", 1)]
        gap_junctions = [Gap("A", "B", 4)]

    s = supports.from_connectome(Conn(), ["A", "B", "C"])
    assert s.keys() == {(1, 0), (0, 1), (0, 2)}  # the autapse B -> B is dropped
    k = dict(zip(zip(s.rows.tolist(), s.cols.tolist()), zip(s.chem, s.gap)))
    assert k[(1, 0)] == (3.0, 4.0) and k[(0, 1)] == (0.0, 4.0) and k[(0, 2)] == (1.0, 0.0)
    chem = np.zeros((3, 3), bool)
    chem[0, 1] = chem[1, 1] = chem[2, 0] = True
    gap = np.zeros((3, 3), bool)
    gap[0, 1] = True
    assert supports.from_masks(chem, gap, "m").keys() == s.keys()
