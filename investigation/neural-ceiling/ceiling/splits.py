"""Classes, folds, the three splits' hidden and visible sets, inner folds and the entry-set rules.

Matrices are [responder, stimulated]: a held-out stimulation hides a column, a held-out neuron a
row. The entry-set functions take finite-entry patterns only, so they can be tested on synthetic
patterns; the real ones are called by the guarded scorer, which alone reads S's pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ceiling import config as C


def measured_classes(names154: list[str]) -> list[str]:
    """Each measured neuron's class among the 302-neuron classes (not classes of the 154 alone)."""
    from quantumnematode.connectome.neuron_classes import neuron_classes

    full = neuron_classes()
    if len(full) != C.N_CONNECTOME:
        raise ValueError("expected classes for all 302 neurons")
    return [full[name] for name in names154]


@dataclass(frozen=True)
class Layout:
    """The classes of the measured neurons and their outer folds."""

    classes: tuple[str, ...]  # class of each of the 154
    class_names: tuple[str, ...]  # the distinct classes, sorted (bootstrap index order)
    order: tuple[str, ...]  # the classes after the permutation
    class_index: np.ndarray  # each neuron's position in class_names
    neuron_fold: np.ndarray  # each neuron's outer fold

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def in_fold(self, fold: int) -> np.ndarray:
        return self.neuron_fold == fold


def make_layout(classes154: list[str], seed: int = C.SEED_FOLDS, n_folds: int = C.N_FOLDS) -> Layout:
    """Permute the distinct classes with default_rng(seed) and deal them round-robin into folds."""
    if C.CLASS_ORDER_BEFORE_PERMUTATION != "sorted":
        raise NotImplementedError(C.CLASS_ORDER_BEFORE_PERMUTATION)
    names = tuple(sorted(set(classes154)))
    perm = np.random.default_rng(seed).permutation(len(names))
    order = tuple(names[k] for k in perm)
    fold_of = {c: pos % n_folds for pos, c in enumerate(order)}
    index_of = {c: k for k, c in enumerate(names)}
    return Layout(
        classes=tuple(classes154),
        class_names=names,
        order=order,
        class_index=np.array([index_of[c] for c in classes154], dtype=np.int64),
        neuron_fold=np.array([fold_of[c] for c in classes154], dtype=np.int64),
    )


def off_diagonal(n: int = C.N_MEASURED) -> np.ndarray:
    return ~np.eye(n, dtype=bool)


def hidden_mask(split: str, in_fold: np.ndarray | None, n: int | None = None) -> np.ndarray:
    n = n if n is not None else (C.N_MEASURED if in_fold is None else len(in_fold))
    if split == "stimulations":
        return np.broadcast_to(in_fold[None, :], (n, n)).copy()
    if split == "neurons":
        return np.broadcast_to(in_fold[:, None], (n, n)).copy()
    if split == "worms":
        return np.zeros((n, n), dtype=bool)
    raise ValueError(split)


def visible_mask(t_finite: np.ndarray, split: str, in_fold: np.ndarray | None) -> np.ndarray:
    """Entries a fit may see: T finite, off the diagonal, not hidden."""
    n = len(t_finite)
    return t_finite & off_diagonal(n) & ~hidden_mask(split, in_fold, n)


# ---- Inner folds --------------------------------------------------------------------------------


def inner_class_folds(layout: Layout, outer_fold: int, n_inner: int = C.N_INNER) -> list[np.ndarray]:
    """The visible classes, in the outer permutation's order, dealt round-robin into inner folds."""
    if C.INNER_CLASS_FOLDS != "round_robin_in_outer_order":
        raise NotImplementedError(C.INNER_CLASS_FOLDS)
    visible = [c for pos, c in enumerate(layout.order) if pos % C.N_FOLDS != outer_fold]
    inner_of = {c: rank % n_inner for rank, c in enumerate(visible)}
    labels = np.array([inner_of.get(c, -1) for c in layout.classes])
    return [labels == g for g in range(n_inner)]


def inner_problems(split: str, t_finite: np.ndarray, layout: Layout | None, outer_fold: int | None,
                   worms_seed: int = C.SEED_WORMS_INNER) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train, validation) masks for each inner fold of one outer unit."""
    off = off_diagonal(len(t_finite))
    if split == "worms":
        return worms_inner_problems(t_finite, worms_seed)
    outer = layout.in_fold(outer_fold)
    visible = visible_mask(t_finite, split, outer)
    problems = []
    for inner in inner_class_folds(layout, outer_fold):
        inner_hidden = hidden_mask(split, inner)
        if split == "stimulations":
            within = inner[:, None]  # responder in the same inner fold
        else:
            within = inner[None, :]  # stimulated neuron in the same inner fold
        train = visible & ~inner_hidden
        validation = t_finite & off & inner_hidden & ~within
        problems.append((train, validation))
    return problems


def worms_inner_problems(t_finite: np.ndarray, seed: int = C.SEED_WORMS_INNER,
                         n_inner: int = C.N_INNER) -> list[tuple[np.ndarray, np.ndarray]]:
    """Random entries of T: a permutation of its finite off-diagonal entries, position mod 3."""
    if C.WORMS_INNER_FOLDS != "permutation_mod_3":
        raise NotImplementedError(C.WORMS_INNER_FOLDS)
    visible = t_finite & off_diagonal(len(t_finite))
    flat = np.flatnonzero(visible)  # row-major
    perm = np.random.default_rng(seed).permutation(len(flat))
    label = np.empty(len(flat), dtype=np.int64)
    label[perm] = np.arange(len(flat)) % n_inner
    problems = []
    for g in range(n_inner):
        validation = np.zeros(visible.size, dtype=bool)
        validation[flat[label == g]] = True
        validation = validation.reshape(visible.shape)
        problems.append((visible & ~validation, validation))
    return problems


# ---- Entry sets (section 4) ---------------------------------------------------------------------


def entry_set_stimulations(t_finite: np.ndarray, s_finite: np.ndarray, in_fold: np.ndarray) -> np.ndarray:
    """S finite in the fold's columns, responder outside the fold, floor-T and floor-ME defined."""
    off = off_diagonal(len(t_finite))
    visible = t_finite & off & ~in_fold[None, :]
    transpose_defined = visible.T  # T[j, i] finite and visible
    me_defined = visible.any(axis=1)[:, None]  # responder has a visible response
    return s_finite & off & in_fold[None, :] & ~in_fold[:, None] & transpose_defined & me_defined


def entry_set_neurons(t_finite: np.ndarray, s_finite: np.ndarray, in_fold: np.ndarray) -> np.ndarray:
    """S finite in the fold's rows, stimulated neuron outside the fold, column with a finite T."""
    off = off_diagonal(len(t_finite))
    column_has_t = (t_finite & off).any(axis=0)[None, :]
    entries = s_finite & off & in_fold[:, None] & ~in_fold[None, :] & column_has_t
    if C.NEURON_SPLIT_REQUIRES_TRANSPOSE:
        visible = t_finite & off & ~in_fold[:, None]
        entries &= visible.T
    return entries


def entry_set_worms(t_finite: np.ndarray, s_finite: np.ndarray) -> np.ndarray:
    """Every finite S entry with a finite T counterpart."""
    return s_finite & t_finite & off_diagonal(len(t_finite))


def entry_sets(t_finite: np.ndarray, s_finite: np.ndarray, layout: Layout) -> dict[str, list[np.ndarray]]:
    """Per split, the per-fold entry masks (held-out worms has one)."""
    out: dict[str, list[np.ndarray]] = {"stimulations": [], "neurons": []}
    for f in range(C.N_FOLDS):
        fold = layout.in_fold(f)
        out["stimulations"].append(entry_set_stimulations(t_finite, s_finite, fold))
        out["neurons"].append(entry_set_neurons(t_finite, s_finite, fold))
    out["worms"] = [entry_set_worms(t_finite, s_finite)]
    return out


def entry_set_report(t_finite: np.ndarray, s_finite: np.ndarray, layout: Layout) -> dict:
    """Each split's size and what each rule excluded, per fold."""
    off = off_diagonal(len(t_finite))
    sets = entry_sets(t_finite, s_finite, layout)
    report: dict = {}
    for split in C.HELD_OUT_SPLITS:
        folds = []
        for f in range(C.N_FOLDS):
            fold = layout.in_fold(f)
            hidden = hidden_mask(split, fold)
            within = (fold[:, None] & fold[None, :])
            base = s_finite & off & hidden
            folds.append({"s_finite_hidden": int(base.sum()),
                          "within_fold_excluded": int((base & within).sum()),
                          "other_rules_excluded": int((base & ~within).sum() - sets[split][f].sum()),
                          "size": int(sets[split][f].sum())})
        report[split] = {"size": sum(x["size"] for x in folds), "folds": folds}
    report["worms"] = {"size": int(sets["worms"][0].sum()),
                       "s_finite": int((s_finite & off).sum()),
                       "t_not_finite_excluded": int((s_finite & off & ~t_finite).sum())}
    return report


def unit_class(split: str, layout: Layout, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """The bootstrap unit of each entry: the responder class on held-out neurons, else stimulated."""
    return layout.class_index[rows] if split == "neurons" else layout.class_index[cols]
