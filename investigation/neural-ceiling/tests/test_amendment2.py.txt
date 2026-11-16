"""Amendment 2: an arm whose prediction is constant on a split's entry set is unreadable there.

Synthetic data only. The constant here is the one oracle P's collapsed tied fits returned: the
training mean on every entry, up to rounding.
"""

import math

import numpy as np

from ceiling import bootstrap, readings, report, splits
from ceiling import config as C

K = C.N_MEASURED
CLASSES = [f"c{k % 74:02d}" for k in range(K)]
MEAN = 0.22544775225168356  # the training mean P's collapsed tied predicted


def collapsed(shape, seed=0):
    """The training mean with rounding noise of a few ulps, as a collapsed fit returns it."""
    rng = np.random.default_rng(seed)
    return MEAN + rng.integers(-2, 3, size=shape) * np.spacing(MEAN)


def setup(seed=1):
    layout = splits.make_layout(CLASSES)
    rng = np.random.default_rng(seed)
    target = rng.normal(size=(K, K))
    entries = rng.random((K, K)) < 0.3
    np.fill_diagonal(entries, False)
    counts = bootstrap.resample_counts(layout.n_classes, n_boot=50)
    return layout, rng, target, entries, counts


def test_rounding_noise_gives_nan_corr_a_spurious_value():
    """The failure the amendment closes: nan_corr divides by a near-zero spread."""
    _, _, target, entries, _ = setup()
    r = bootstrap.r_point(target[entries], collapsed(target.shape)[entries])
    assert not math.isfinite(r) or abs(r) > 0  # a number with no meaning, or infinite


def test_a_constant_prediction_is_unreadable():
    layout, rng, target, entries, counts = setup()
    preds = {"masked": target + rng.normal(size=(K, K)), "tied": collapsed((K, K)),
             "zeros": np.zeros((K, K))}
    for split in ("stimulations", "neurons", "worms"):
        sc = readings.score_split(target, preds, entries, split, layout, counts)
        assert sc["masked"]["readable"] and math.isfinite(sc["masked"]["r"])
        for arm in ("tied", "zeros"):
            assert sc[arm]["readable"] is False
            assert sc[arm]["why"] == C.CONSTANT_PREDICTION
            assert "r" not in sc[arm]


def test_a_small_but_real_prediction_stays_readable():
    """The test is on rounding-level spread, not on small predictions."""
    layout, rng, target, entries, counts = setup()
    small = MEAN + 1e-6 * (target + rng.normal(size=(K, K)))
    sc = readings.score_split(target, {"small": small}, entries, "stimulations", layout, counts)
    assert sc["small"]["readable"] and 0.5 < sc["small"]["r"] < 0.9


def test_the_test_reads_the_prediction_alone():
    """The same prediction is readable or not whatever the target, so S cannot decide it."""
    layout, rng, target, entries, counts = setup()
    pred = {"tied": collapsed((K, K)), "masked": target + rng.normal(size=(K, K))}
    for scale in (1e-12, 1.0, 1e12):
        sc = readings.score_split(scale * target, pred, entries, "worms", layout, counts)
        assert sc["tied"]["readable"] is False and sc["masked"]["readable"] is True


def test_one_collapsed_fold_makes_the_arm_unreadable():
    """A collapsed fold is an unreadable fold fit, and an arm needs all of its fold fits."""
    layout, rng, target, entries, counts = setup()
    folds = [entries & (layout.neuron_fold[None, :] == f) for f in range(C.N_FOLDS)]
    good = target + rng.normal(size=(K, K))
    one = good.copy()
    one[folds[1]] = collapsed((K, K))[folds[1]]  # one fold collapsed, the others not
    sc = readings.score_split(target, {"good": good, "one": one}, entries, "stimulations", layout,
                              counts, folds)
    assert sc["good"]["readable"] and all(math.isfinite(r) for r in sc["good"]["per_fold"])
    assert sc["one"]["readable"] is False and sc["one"]["why"] == C.CONSTANT_PREDICTION


def test_every_fold_at_its_own_mean_is_unreadable():
    """Each fold collapsed to a different constant (its own training mean): the pooled prediction
    is not constant, but it carries only fold means, so the arm is unreadable."""
    layout, rng, target, entries, counts = setup()
    folds = [entries & (layout.neuron_fold[None, :] == f) for f in range(C.N_FOLDS)]
    pred = np.zeros((K, K))
    for f, mask in enumerate(folds):
        pred[mask] = collapsed((K, K), seed=f)[mask] + 0.01 * f
    vec = pred[entries]
    assert not readings.is_constant(vec)  # the pooled-only rule would have missed it
    sc = readings.score_split(target, {"tied": pred}, entries, "stimulations", layout, counts, folds)
    assert sc["tied"]["readable"] is False


def test_a_collapsed_tied_makes_its_contrasts_unreadable():
    """R-A's tied cell and R-C's masked-tied contrast become unreadable (amendment 1, item 7),
    and R-C is then read on sparse-ER alone."""
    layout, rng, target, entries, counts = setup()
    plan = {"null-D": [], "null-R": [], "sparse-ER": [f"er:{k}" for k in range(5)]}
    preds = {"masked": target + rng.normal(size=(K, K)), "tied": collapsed((K, K)),
             "floor-C": target + 2 * rng.normal(size=(K, K)), "T": target + 3 * rng.normal(size=(K, K))}
    preds |= {f"er:{k}": target + 4 * rng.normal(size=(K, K)) for k in range(5)}
    for split in ("stimulations", "worms"):
        sc = readings.score_split(target, preds, entries, split, layout, counts)
        rd = report.split_readings(sc, split, plan, ("floor-C",))
        if split == "stimulations":
            assert rd["R-A"]["tied"] is None and rd["R-A"]["masked"] is not None
        else:
            assert rd["R-C"]["tied"] is None and rd["R-C"]["sparse-ER"] is not None


def test_a_collapsed_masked_makes_its_split_unreadable():
    layout, rng, target, entries, counts = setup()
    plan = {"null-D": [], "null-R": [], "sparse-ER": []}
    preds = {"masked": collapsed((K, K)), "floor-C": target + rng.normal(size=(K, K))}
    sc = readings.score_split(target, preds, entries, "stimulations", layout, counts)
    rd = report.split_readings(sc, "stimulations", plan, ("floor-C",))
    assert rd["masked_readable"] is False and rd["R-A"]["masked"] is None
