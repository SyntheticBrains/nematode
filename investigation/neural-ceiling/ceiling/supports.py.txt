"""Coupling supports over the 302 neurons, in W's orientation: W[target, source].

A chemical synapse pre -> post is the entry (post, pre); a gap junction a - b is both (a, b) and
(b, a). The diagonal is always empty. Each entry carries its chemical and gap counts, which only the
masked-prior arm reads.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np

from ceiling import config as C


@dataclass
class Support:
    name: str
    n: int
    rows: np.ndarray  # target
    cols: np.ndarray  # source
    chem: np.ndarray  # chemical count of source -> target
    gap: np.ndarray  # gap count of the pair
    report: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.rows)

    def keys(self) -> set[tuple[int, int]]:
        return set(zip(self.rows.tolist(), self.cols.tolist()))

    def reciprocal_entries(self) -> int:
        keys = self.keys()
        return sum((c, r) in keys for r, c in keys)


def _from_counts(name: str, n: int, counts: dict[tuple[int, int], list[float]]) -> Support:
    keys = sorted(k for k in counts if k[0] != k[1])
    return Support(
        name=name,
        n=n,
        rows=np.array([k[0] for k in keys], dtype=np.int64),
        cols=np.array([k[1] for k in keys], dtype=np.int64),
        chem=np.array([counts[k][0] for k in keys], dtype=np.float64),
        gap=np.array([counts[k][1] for k in keys], dtype=np.float64),
    )


def from_connectome(conn, names302: list[str], name: str = "masked") -> Support:
    """The union of directed chemical entries and gap entries both ways, off the diagonal."""
    position = {nm: k for k, nm in enumerate(names302)}
    counts: dict[tuple[int, int], list[float]] = {}
    for syn in conn.chemical_synapses:
        key = (position[syn.post], position[syn.pre])
        counts.setdefault(key, [0.0, 0.0])[0] += float(syn.weight)
    for gj in conn.gap_junctions:
        a, b = position[gj.neuron_a], position[gj.neuron_b]
        for key in ((a, b), (b, a)):
            counts.setdefault(key, [0.0, 0.0])[1] += float(gj.weight)
    return _from_counts(name, len(names302), counts)


def from_masks(chem_pre_post: np.ndarray, gap: np.ndarray, name: str) -> Support:
    """The union support of a built brain's chemical mask [pre, post] and gap matrix."""
    chem_pre_post = np.asarray(chem_pre_post, dtype=bool)
    gap = np.asarray(gap, dtype=bool)
    union = chem_pre_post.T | gap | gap.T
    np.fill_diagonal(union, False)
    rows, cols = np.nonzero(union)
    zeros = np.zeros(len(rows))
    return Support(name, len(union), rows.astype(np.int64), cols.astype(np.int64), zeros, zeros.copy())


def anatomical(names302: list[str] | None = None) -> Support:
    from ceiling import inputs

    conn = inputs.connectome()
    return from_connectome(conn, names302 or inputs.connectome_names(conn), "masked")


# ---- null-D: the package's degree-preserving rewirings, as H0 built them --------------------------


def _probe():
    if str(C.PROBE_DIR) not in sys.path:
        sys.path.insert(0, str(C.PROBE_DIR))
    import c2_encoder_probe

    return c2_encoder_probe


def brain_support(seed: int | None) -> Support:
    """Build the encoded brain H0 built (wild type when seed is None) and read its union support."""
    probe = _probe()
    if seed is None:
        brain = probe.encoded_brain((), **probe.PROCEDURE_SETTINGS)
    else:
        brain = probe.encoded_brain((), **probe.PROCEDURE_SETTINGS,
                                    wiring="rewired_degree_preserving", rewire_seed=seed)
    top = brain.topology
    if list(top.neuron_names) != sorted(top.neuron_names) or len(top.neuron_names) != C.N_CONNECTOME:
        raise ValueError("unexpected brain neuron order")
    chem = top.m_chem.to(dtype=bool).numpy().copy()
    gap = top.g_gap.detach().numpy() != 0
    return from_masks(chem, gap, "built" if seed is None else f"nullD:{seed}")


def null_d(seed: int) -> Support:
    """null-D at one seed: counts from rewire_degree_preserving, support asserted equal to the brain's."""
    from quantumnematode.connectome.rewiring import rewire_degree_preserving

    from ceiling import inputs

    conn = inputs.connectome()
    names = inputs.connectome_names(conn)
    rewired = rewire_degree_preserving(conn, np.random.default_rng(seed), origins={})
    support = from_connectome(rewired, names, f"nullD:{seed}")
    built = brain_support(seed)
    if support.keys() != built.keys():
        raise AssertionError(f"null-D {seed}: counted support differs from the built brain's")
    support.report = {"size": support.size, "matches_built_brain": True}
    return support


# ---- null-R: reciprocity-preserving rewiring of a support ----------------------------------------


def _in_out(keys: set[tuple[int, int]], n: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.zeros(n, dtype=np.int64)
    cols = np.zeros(n, dtype=np.int64)
    for r, c in keys:
        rows[r] += 1
        cols[c] += 1
    return rows, cols


def _reciprocal_pairs(keys: set[tuple[int, int]]) -> int:
    return sum((c, r) in keys for r, c in keys) // 2


class _Rewirer:
    """Undirected swaps on the reciprocal pairs, directed swaps on the rest; counts travel."""

    def __init__(self, base: Support, rng: np.random.Generator):
        self.rng = rng
        self.keys = base.keys()
        self.counts = {(int(r), int(c)): (float(a), float(b))
                       for r, c, a, b in zip(base.rows, base.cols, base.chem, base.gap)}
        self.pairs = sorted((r, c) for r, c in self.keys if r < c and (c, r) in self.keys)
        self.directed = sorted((r, c) for r, c in self.keys if (c, r) not in self.keys)
        self.accepted = {"undirected": 0, "directed": 0}
        self.attempted = {"undirected": 0, "directed": 0}

    def _move(self, old: tuple[int, int], new: tuple[int, int], counts: dict) -> None:
        counts[new] = self.counts.pop(old)

    def undirected(self) -> None:
        self.attempted["undirected"] += 1
        n = len(self.pairs)
        i, j = int(self.rng.integers(0, n)), int(self.rng.integers(0, n))
        if i == j:
            return
        a, b = self.pairs[i]
        c, d = self.pairs[j]
        if self.rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4:
            return
        new = [(a, d), (d, a), (c, b), (b, c)]
        if any(k in self.keys for k in new):
            return  # a duplicate (which also covers a new reciprocal pair on a directed entry)
        old = [(a, b), (b, a), (c, d), (d, c)]
        moved: dict = {}
        for o, nw in zip(old, new):
            self._move(o, nw, moved)
        self.counts.update(moved)
        self.keys.difference_update(old)
        self.keys.update(new)
        self.pairs[i] = (min(a, d), max(a, d))
        self.pairs[j] = (min(b, c), max(b, c))
        self.accepted["undirected"] += 1

    def directed_swap(self) -> None:
        self.attempted["directed"] += 1
        n = len(self.directed)
        i, j = int(self.rng.integers(0, n)), int(self.rng.integers(0, n))
        if i == j:
            return
        (r1, c1), (r2, c2) = self.directed[i], self.directed[j]
        new_i, new_j = (r2, c1), (r1, c2)  # each source keeps its count; targets exchange
        if r2 == c1 or r1 == c2:
            return  # self-connection
        if new_i in self.keys or new_j in self.keys:
            return  # duplicate
        if (c1, r2) in self.keys or (c2, r1) in self.keys:
            return  # a new reciprocal pair
        moved: dict = {}
        self._move((r1, c1), new_i, moved)
        self._move((r2, c2), new_j, moved)
        self.counts.update(moved)
        self.keys.difference_update([(r1, c1), (r2, c2)])
        self.keys.update([new_i, new_j])
        self.directed[i], self.directed[j] = new_i, new_j
        self.accepted["directed"] += 1

    def round(self, per_entry: int) -> None:
        if C.NULL_R_ORDER != "undirected_then_directed":
            raise NotImplementedError(C.NULL_R_ORDER)
        if len(self.pairs) > 1:
            for _ in range(per_entry * 2 * len(self.pairs)):
                self.undirected()
        if len(self.directed) > 1:
            for _ in range(per_entry * len(self.directed)):
                self.directed_swap()


def null_r(base: Support, seed: int) -> Support:
    """One reciprocity-preserving draw, with its three invariants asserted and its report."""
    rng = np.random.default_rng(seed)
    rewirer = _Rewirer(base, rng)
    original = base.keys()
    rewirer.round(C.NULL_R_SWAPS_PER_ENTRY)
    per_entry = C.NULL_R_SWAPS_PER_ENTRY

    def moved() -> float:
        return 1.0 - len(original & rewirer.keys) / len(original)

    while moved() < C.NULL_R_MIN_MOVED and per_entry < C.NULL_R_MAX_SWAPS_PER_ENTRY:
        step = min(C.NULL_R_EXTENSION_ROUND, C.NULL_R_MAX_SWAPS_PER_ENTRY - per_entry)
        rewirer.round(step)
        per_entry += step
    keys = rewirer.keys
    counts = {k: list(v) for k, v in rewirer.counts.items()}
    out = _from_counts(f"nullR:{seed}", base.n, counts)
    in0, out0 = _in_out(original, base.n)
    in1, out1 = _in_out(keys, base.n)
    checks = {
        "entries": len(keys) == len(original) == out.size,
        "degrees": bool((in0 == in1).all() and (out0 == out1).all()),
        "reciprocal_pairs": _reciprocal_pairs(keys) == _reciprocal_pairs(original),
        "zero_diagonal": all(r != c for r, c in keys),
        "no_duplicates": len(set(zip(out.rows.tolist(), out.cols.tolist()))) == out.size,
    }
    if not all(checks.values()):
        raise AssertionError(f"null-R {seed}: invariant failed {checks}")
    out.report = {"seed": seed, "attempted_per_entry": per_entry, "accepted": dict(rewirer.accepted),
                  "attempted": dict(rewirer.attempted), "fraction_moved": moved(),
                  "reached_min_moved": moved() >= C.NULL_R_MIN_MOVED, "checks": checks,
                  "reciprocal_pairs": _reciprocal_pairs(keys)}
    return out


# ---- sparse-ER -----------------------------------------------------------------------------------


def sparse_er(size: int, n: int, seed: int) -> Support:
    """A uniformly random support of `size` off-diagonal entries over n nodes."""
    if C.ER_DRAW != "choice_without_replacement":
        raise NotImplementedError(C.ER_DRAW)
    flat = np.random.default_rng(seed).choice(n * (n - 1), size=size, replace=False)
    rows = flat // (n - 1)
    offset = flat % (n - 1)
    cols = offset + (offset >= rows)  # skip the diagonal
    order = np.lexsort((cols, rows))
    rows, cols = rows[order].astype(np.int64), cols[order].astype(np.int64)
    zeros = np.zeros(size)
    return Support(f"er:{seed}", n, rows, cols, zeros, zeros.copy(), {"seed": seed, "size": size})
