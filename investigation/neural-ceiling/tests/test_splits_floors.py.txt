"""Folds from the 302-neuron classes, the entry-set rules, and the floors, on synthetic patterns."""

import json

import numpy as np
import pytest

from ceiling import config as C
from ceiling import floors, splits

# ---- folds ----------------------------------------------------------------------------------------


def test_folds_use_the_302_neuron_classes():
    """Section 4: the 154's classes are the 302-neuron classes restricted to them (74), not the
    classes the 154 would form on their own."""
    from quantumnematode.connectome.neuron_classes import neuron_classes

    names = json.loads(C.NAMES_PATH.read_text())
    classes = splits.measured_classes(names)
    assert len(set(classes)) == 74
    own = neuron_classes(names)
    assert len(set(own.values())) != 74  # pooling the 154 alone would give other classes
    full = neuron_classes()
    assert classes == [full[n] for n in names]


def test_fold_assignment_is_deterministic_round_robin():
    classes = [f"c{k % 74:02d}" for k in range(154)]
    a, b = splits.make_layout(classes), splits.make_layout(classes)
    assert np.array_equal(a.neuron_fold, b.neuron_fold)
    names = sorted(set(classes))
    perm = np.random.default_rng(C.SEED_FOLDS).permutation(74)
    expected = {names[k]: pos % 5 for pos, k in enumerate(perm)}
    assert [expected[c] for c in classes] == a.neuron_fold.tolist()
    sizes = sorted(np.bincount([expected[c] for c in names]).tolist())
    assert sizes == [14, 15, 15, 15, 15]
    # every member of a class shares its fold
    for c in names:
        members = [i for i, x in enumerate(classes) if x == c]
        assert len(set(a.neuron_fold[members])) == 1


def test_inner_class_folds_cover_the_visible_classes():
    classes = [f"c{k % 20:02d}" for k in range(60)]
    layout = splits.make_layout(classes)
    inner = splits.inner_class_folds(layout, 2)
    union = np.any(np.stack(inner), axis=0)
    assert np.array_equal(union, ~layout.in_fold(2))
    assert np.sum(np.stack(inner), axis=0).max() == 1
    visible = [c for pos, c in enumerate(layout.order) if pos % 5 != 2]
    expected = {c: rank % 3 for rank, c in enumerate(visible)}
    for g, mask in enumerate(inner):
        assert {classes[i] for i in np.flatnonzero(mask)} == {c for c, v in expected.items() if v == g}


def test_worms_inner_folds_partition_finite_entries():
    rng = np.random.default_rng(0)
    t_finite = rng.random((12, 12)) > 0.2
    problems = splits.worms_inner_problems(t_finite)
    finite = t_finite & splits.off_diagonal(12)
    vals = np.stack([v for _, v in problems])
    assert np.array_equal(vals.sum(0) == 1, finite) and vals.sum(0).max() == 1
    for train, val in problems:
        assert not (train & val).any() and np.array_equal(train | val, finite)
    again = splits.worms_inner_problems(t_finite)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(problems, again))


def test_inner_problems_hide_inner_classes_and_exclude_within_inner_pairs():
    classes = [f"c{k % 15:02d}" for k in range(30)]
    layout = splits.make_layout(classes)
    t_finite = np.ones((30, 30), bool)
    np.fill_diagonal(t_finite, False)
    for split, axis in (("stimulations", 1), ("neurons", 0)):
        outer = layout.in_fold(0)
        for (train, val), inner in zip(splits.inner_problems(split, t_finite, layout, 0),
                                       splits.inner_class_folds(layout, 0)):
            hidden = splits.hidden_mask(split, inner) | splits.hidden_mask(split, outer)
            assert not (train & hidden).any()
            assert not (val & (inner[:, None] & inner[None, :])).any()
            covered = val.any(axis=1 - axis)
            assert np.array_equal(covered, inner)


# ---- entry sets -----------------------------------------------------------------------------------


def toy(n=10, seed=0):
    classes = [f"k{i // 2}" for i in range(n)]  # pairs share a class
    layout = splits.make_layout(classes)
    rng = np.random.default_rng(seed)
    t_finite = rng.random((n, n)) > 0.15
    s_finite = rng.random((n, n)) > 0.3
    np.fill_diagonal(t_finite, False)
    np.fill_diagonal(s_finite, False)
    return layout, t_finite, s_finite


def test_stimulation_entry_set_rules():
    """S finite in a hidden column; responder outside the fold; T[j, i] finite; floor-ME defined."""
    layout, t, s = toy()
    t[:, 0] = False  # neuron 0's column: no finite T, so its row's transposes are undefined
    for f in range(C.N_FOLDS):
        fold = layout.in_fold(f)
        got = splits.entry_set_stimulations(t, s, fold)
        for i in range(10):
            for j in range(10):
                visible_row = (t[i] & ~fold).any()
                want = (s[i, j] and i != j and fold[j] and not fold[i] and t[j, i] and visible_row)
                assert got[i, j] == want


def test_within_fold_pairs_are_excluded():
    layout, t, s = toy()
    t[:] = True
    s[:] = True
    np.fill_diagonal(t, False)
    np.fill_diagonal(s, False)
    for f in range(C.N_FOLDS):
        fold = layout.in_fold(f)
        within = fold[:, None] & fold[None, :]
        assert not (splits.entry_set_stimulations(t, s, fold) & within).any()
        assert not (splits.entry_set_neurons(t, s, fold) & within).any()
        assert (splits.entry_set_stimulations(t, s, fold) == (fold[None, :] & ~fold[:, None])).all()


def test_neuron_entry_set_rules():
    """Amendment 1, item 5: S finite in a hidden row; stimulated neuron outside the fold; floor-ME
    defined from visible rows only; floor-T (T[j, i] visible and finite) defined."""
    layout, t, s = toy(seed=2)
    t[:, 4] = False  # a DB1-like column
    for f in range(C.N_FOLDS):
        fold = layout.in_fold(f)
        got = splits.entry_set_neurons(t, s, fold)
        assert not got[:, 4].any()
        for i in range(10):
            for j in range(10):
                me = (t[:, j] & ~fold & (np.arange(10) != j)).any()
                want = s[i, j] and i != j and fold[i] and not fold[j] and me and t[j, i]
                assert got[i, j] == want


def test_neuron_floor_me_ignores_hidden_rows():
    """A column whose only finite entries lie in the hidden rows has no floor-ME there."""
    layout, t, s = toy(seed=4)
    fold = layout.in_fold(0)
    j = int(np.flatnonzero(~fold)[0])
    t[:, j] = False
    t[np.flatnonzero(fold), j] = True  # finite only in the fold's (hidden) rows
    t[j, j] = False
    assert not splits.entry_set_neurons(t, s, fold)[:, j].any()


def test_every_reference_is_defined_on_every_split_entry_set():
    """On a T-like pattern (an all-NaN column, scattered gaps), each reference is defined on every
    entry of each split's set, so every arm is scored on the same entries."""
    from ceiling import stages

    K = C.N_MEASURED
    rng = np.random.default_rng(3)
    layout = splits.make_layout([f"c{k % 74:02d}" for k in range(K)])
    T = rng.normal(size=(K, K))
    T[rng.random((K, K)) < 0.07] = np.nan
    T[:, 58] = np.nan
    np.fill_diagonal(T, np.nan)
    s_fin = rng.random((K, K)) < 0.47
    np.fill_diagonal(s_fin, False)
    sets = splits.entry_sets(np.isfinite(T), s_fin, layout)
    mech = {"H0": rng.normal(size=(K, K)), "C.2": rng.normal(size=(K, K))}
    refs, _ = stages.reference_pool(T, layout, [u for sp in C.SPLITS for u in
                                                ([(sp, None)] if sp == "worms" else [(sp, f) for f in range(5)])],
                                    mech)
    for split in C.SPLITS:
        entries = np.any(np.stack(sets[split]), axis=0)
        assert entries.sum() > 0
        for name, pred in refs[split].items():
            assert np.isfinite(pred[entries]).all(), (split, name)


def test_worm_entry_set_and_report():
    layout, t, s = toy(seed=3)
    got = splits.entry_set_worms(t, s)
    assert np.array_equal(got, s & t)
    report = splits.entry_set_report(t, s, layout)
    assert report["worms"]["size"] == int((s & t).sum())
    total = sum(fl["size"] for fl in report["stimulations"]["folds"])
    assert total == report["stimulations"]["size"]
    for fl in report["stimulations"]["folds"]:
        assert fl["s_finite_hidden"] == fl["size"] + fl["within_fold_excluded"] + fl["other_rules_excluded"]


# ---- floors ---------------------------------------------------------------------------------------


def test_floor_me_and_floor_t_read_only_visible_entries():
    rng = np.random.default_rng(1)
    T = rng.normal(size=(6, 6))
    np.fill_diagonal(T, np.nan)
    in_fold = np.array([False, True, False, False, True, False])
    visible = splits.visible_mask(np.isfinite(T), "stimulations", in_fold)
    me = floors.floor_me(T, visible, "stimulations")
    for i in range(6):
        cols = [j for j in range(6) if j != i and not in_fold[j]]
        assert np.allclose(me[i], np.mean(T[i, cols]))
    ft = floors.floor_t(T, visible, "stimulations")
    assert ft[2, 1] == T[1, 2]  # (2, 1): responder 2 outside the fold reads visible column 2
    assert np.isnan(ft[1, 4])  # (1, 4): within the fold, T[4, 1] is in hidden column 1
    rows_visible = splits.visible_mask(np.isfinite(T), "neurons", in_fold)
    me_rows = floors.floor_me(T, rows_visible, "neurons")
    for j in range(6):
        rows = [i for i in range(6) if i != j and not in_fold[i]]
        assert np.allclose(me_rows[:, j], np.mean(T[rows, j]))


def test_floor_me_on_held_out_worms():
    T = np.arange(16, dtype=float).reshape(4, 4)
    np.fill_diagonal(T, np.nan)
    visible = np.isfinite(T)
    me = floors.floor_me(T, visible, "worms")
    grand = np.nanmean(T)
    assert np.isclose(me[0, 1], np.nanmean(T[0]) + np.nanmean(T[:, 1]) - grand)
    assert floors.floor_t(T, visible, "worms") is None  # floor-T is a held-out-split arm only


def test_calibration_excludes_entries_whose_transpose_is_hidden():
    rng = np.random.default_rng(4)
    T = rng.normal(size=(8, 8))
    np.fill_diagonal(T, np.nan)
    in_fold = np.zeros(8, bool)
    in_fold[[2, 5]] = True
    visible = splits.visible_mask(np.isfinite(T), "stimulations", in_fold)
    refs = floors.references(T, visible, "stimulations", {"H0": rng.normal(size=(8, 8))})
    # usable calibration entries: visible, with responder outside the fold (T[j, i] visible)
    usable = visible & ~in_fold[:, None]
    assert refs["floor-C"]["n_fit"] == int(usable.sum())
    assert refs["H0-cal"]["n_fit"] == int(usable.sum())
    assert len(refs["H0-cal"]["coef"]) == 4  # a + b H0 + c floor-ME + d floor-T
    X = np.stack([np.ones(usable.sum()), refs["floor-ME"]["pred"][usable], refs["floor-T"]["pred"][usable]], 1)
    coef, *_ = np.linalg.lstsq(X, T[usable], rcond=None)
    assert np.allclose(coef, refs["floor-C"]["coef"])


def test_held_out_worm_references_drop_the_transpose_term():
    rng = np.random.default_rng(5)
    T = rng.normal(size=(8, 8))
    np.fill_diagonal(T, np.nan)
    refs = floors.references(T, np.isfinite(T), "worms", {"H0": rng.normal(size=(8, 8))})
    assert "floor-T" not in refs
    assert len(refs["floor-C"]["coef"]) == 2 and len(refs["H0-cal"]["coef"]) == 3


@pytest.mark.parametrize("split", ["stimulations", "neurons", "worms"])
def test_hidden_and_visible_masks(split):
    in_fold = np.array([True, False, False, True])
    t_finite = np.ones((4, 4), bool)
    vis = splits.visible_mask(t_finite, split, None if split == "worms" else in_fold)
    assert not np.diag(vis).any()
    if split == "stimulations":
        assert not vis[:, in_fold].any()
    if split == "neurons":
        assert not vis[in_fold].any()
