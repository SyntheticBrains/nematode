"""Section 6's fitting rules, section 7's noise and calibration and gates, section 8's bootstrap."""

import math

import numpy as np
import pytest
import torch

from ceiling import bootstrap, fitting, models, oracles, supports
from ceiling import config as C

# ---- fitting --------------------------------------------------------------------------------------


def toy_problem(seed=0, n=12, k=8, nnz=30):
    s = supports.sparse_er(nnz, n, seed)
    measured = np.arange(k)
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        M = models.SparseSeries(s.rows, s.cols, n, measured, torch.tensor(rng.normal(0, 0.4, nnz))).response().numpy()
    T = 3.0 + 2.0 * M + rng.normal(0, 0.1, M.shape)
    np.fill_diagonal(T, np.nan)
    return s, measured, T


def test_standardise_uses_the_visible_entries():
    _, _, T = toy_problem()
    mask = np.isfinite(T)
    mask[:, :2] = False
    y, mu, sigma = fitting.standardise(T, mask)
    assert abs(y[mask].mean()) < 1e-12 and abs(y[mask].std() - 1) < 1e-12
    assert mu == pytest.approx(T[mask].mean())


def test_fit_returns_raw_units_and_flags_the_cap():
    s, measured, T = toy_problem()
    mask = np.isfinite(T)
    build = lambda: models.SparseSeries(s.rows, s.cols, s.n, measured)  # noqa: E731
    capped = fitting.fit(build, T, mask, 1e-3, 1e-2, max_iter=2)
    assert capped.at_cap and capped.readable and capped.n_iter == 2
    full = fitting.fit(build, T, mask, 1e-4, 1e-3, max_iter=500)
    assert full.readable and not full.refit
    assert np.corrcoef(full.pred[mask], T[mask])[0, 1] > 0.9  # raw units, close to T
    assert abs(full.pred[mask].mean() - T[mask].mean()) < 0.5
    assert sum(full.shares["covariance_share"]) == pytest.approx(1.0)
    assert len(full.moved) == 1 and full.moved[0] > 0


def test_non_finite_loss_is_refit_once_at_the_next_lambda(monkeypatch):
    s, measured, T = toy_problem()
    mask = np.isfinite(T)
    build = lambda: models.SparseSeries(s.rows, s.cols, s.n, measured)  # noqa: E731
    real = fitting._fit_once
    seen = []

    def flaky(model, y, m, lam, it):
        seen.append(lam)
        return (math.nan, math.nan, 1) if lam == 1e-3 else real(model, y, m, lam, it)

    monkeypatch.setattr(fitting, "_fit_once", flaky)
    out = fitting.fit(build, T, mask, 1e-3, 1e-2, max_iter=20)
    assert seen == [1e-3, 1e-2] and out.refit and out.readable and out.lam == 1e-2
    monkeypatch.setattr(fitting, "_fit_once", lambda *a: (math.nan, math.nan, 1))
    dead = fitting.fit(build, T, mask, 1e-3, 1e-2, max_iter=20)
    assert not dead.readable and dead.refit and dead.pred is None
    assert fitting.validation_mse(dead.pred, T, mask, 1.0) == math.inf


def test_lambda_choice_and_edge_extension():
    grid = list(C.LAMBDA_GRID)
    assert fitting.choose({1e-4: 3.0, 1e-3: 1.0, 1e-2: 1.0, 1e-1: 2.0, 1.0: math.inf}) == 1e-3
    assert fitting.choose({1e-4: math.inf}) is None
    assert fitting.edge_extension(1e-4, grid) == pytest.approx(1e-5)
    assert fitting.edge_extension(1.0, grid) == pytest.approx(10.0)
    assert fitting.edge_extension(1e-2, grid) is None
    assert fitting.next_lambda(1e-2, grid) == 1e-1
    assert fitting.next_lambda(1.0, grid) == pytest.approx(10.0)
    assert fitting.next_lambda(1e-5, sorted(grid + [1e-5])) == 1e-4


# ---- oracles --------------------------------------------------------------------------------------


def test_count_model_reproduces_the_registered_value():
    out = oracles.count_model(0.173)
    assert out["ceiling"] == pytest.approx(0.34, abs=0.005)
    assert out["bracket"] == [0.173, pytest.approx(math.sqrt(0.173))]
    assert math.sqrt(out["rel_S"] * out["rel_T"]) == pytest.approx(0.173)
    assert math.isnan(oracles.count_model(-0.1)["ceiling"])


def test_calibration_falls_back_per_estimate():
    keep = oracles.calibration({"rel_T": 0.3, "rel_S": 0.2, "r_TS": 0.173})
    assert keep["rel_T"] == 0.3 and keep["rel_S"] == 0.2 and not keep["notes"]
    swap = oracles.calibration({"rel_T": 0.3, "rel_S": 1.4, "r_TS": 0.173})
    assert swap["rel_T"] == 0.3
    assert swap["rel_S"] == pytest.approx(oracles.count_model(0.173)["rel_S"])
    assert len(swap["notes"]) == 1


def noisy_inputs(seed=0, n=20):
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=(n, n))
    residual = rng.standard_t(3, size=(n, n))
    residual[:, 4] = np.nan  # a column with nothing to resample
    residual[rng.random((n, n)) < 0.1] = np.nan
    t_pat = rng.random((n, n)) > 0.1
    s_pat = rng.random((n, n)) > 0.5
    return truth, residual, t_pat, s_pat


def test_residual_blocks_lie_on_the_pattern_and_come_from_one_column():
    truth, residual, _, s_pat = noisy_inputs()
    noise = oracles.residual_blocks(residual, s_pat, np.random.default_rng(9773))
    assert np.array_equal(np.isfinite(noise), s_pat)
    for j in range(noise.shape[1]):
        col = noise[s_pat[:, j], j]
        sources = [k for k in range(residual.shape[1]) if np.isfinite(residual[:, k]).any()]
        # every value in the target column is a value of one source column
        assert any(set(col.tolist()) <= set(residual[np.isfinite(residual[:, k]), k].tolist())
                   for k in sources)


def test_synthetic_pair_hits_the_registered_reliabilities():
    truth, residual, t_pat, s_pat = noisy_inputs(1)
    t_syn, s_syn, report = oracles.synthetic_pair(truth, residual, t_pat, s_pat, 0.26, 0.12,
                                                  np.random.default_rng(9775))
    off = ~np.eye(len(truth), dtype=bool)
    assert np.array_equal(np.isfinite(t_syn), t_pat & off)
    assert np.array_equal(np.isfinite(s_syn), s_pat & off)
    for name, rel, data in (("T", 0.26, t_syn), ("S", 0.12, s_syn)):
        pattern = np.isfinite(data)
        assert np.var(truth[pattern]) / np.var(data[pattern]) == pytest.approx(rel, rel=1e-10)
        assert report[name]["achieved_rel"] == pytest.approx(rel, rel=1e-10)
    again = oracles.synthetic_pair(truth, residual, t_pat, s_pat, 0.26, 0.12, np.random.default_rng(9775))
    assert np.array_equal(again[0], t_syn, equal_nan=True)


def test_known_answer_is_noise_free_on_the_patterns():
    truth, _, t_pat, s_pat = noisy_inputs(2)
    t_syn, s_syn = oracles.known_answer(truth, t_pat, s_pat)
    both = np.isfinite(t_syn) & np.isfinite(s_syn)
    assert np.array_equal(t_syn[both], s_syn[both]) and np.array_equal(t_syn[both], truth[both])


def test_gates():
    assert oracles.k_gate(0.85, [1e-5])["passes"]
    k = oracles.k_gate(0.5, [1e-5])
    assert not k["passes"] and k["diagnosis"] == "the channel is not identified for this family"
    assert oracles.k_gate(0.5, [0.3])["diagnosis"] == "the optimiser stopped"
    good = {"readable_draws": 19, "rank": 1, "diff": 0.05, "ci": (0.01, 0.1)}
    bad = dict(good, rank=2)
    draw = lambda d, r: {"m": 0.02, "null-D": d, "null-R": r}  # noqa: E731
    assert oracles.p_split_gate([draw(good, good), draw(good, good), draw(bad, good)])["passes"]
    assert not oracles.p_split_gate([draw(good, good), draw(bad, good), draw(good, bad)])["passes"]
    assert not oracles.p_split_gate([draw(dict(good, readable_draws=16), good)] * 3)["passes"]
    line = {"diff": 0.02, "ci": (0.001, 0.04)}
    assert oracles.p_worm_gate([line, line, None])["passes"]
    assert not oracles.p_worm_gate([line, {"diff": 0.02, "ci": (-0.01, 0.04)}, None])["passes"]
    first = {"m": 0.05, "null-D": {"rank": 1, "diff": 0.0}}
    big = {"m": 0.05, "null-D": {"rank": 4, "diff": 0.06}}
    neither = {"m": 0.05, "null-D": {"rank": 4, "diff": 0.01}}
    assert oracles.n_gate([first, big, neither])["fails"]
    assert not oracles.n_gate([first, neither, neither])["fails"]
    assert oracles.r_att([0.2, 0.3, 0.25])["r_att"] == 0.25


# ---- bootstrap ------------------------------------------------------------------------------------


def test_nan_corr_is_pearson_over_finite_pairs():
    rng = np.random.default_rng(0)
    y, a = rng.normal(size=200), rng.normal(size=200)
    y[::7] = np.nan
    keep = np.isfinite(y)
    assert bootstrap.r_point(y, a) == pytest.approx(np.corrcoef(y[keep], a[keep])[0, 1])


def test_resampled_r_is_the_weighted_pearson_of_each_resample():
    rng = np.random.default_rng(1)
    n, classes = 300, 12
    unit = rng.integers(0, classes, n)
    y = rng.normal(size=n)
    a = y + rng.normal(size=n)
    counts = bootstrap.resample_counts(classes, n_boot=5, seed=3)
    got = bootstrap.resampled_r(y, a, unit, counts)
    for b in range(5):
        w = counts[b][unit]
        my, ma = np.average(y, weights=w), np.average(a, weights=w)
        cov = np.average((y - my) * (a - ma), weights=w)
        expect = cov / math.sqrt(np.average((y - my) ** 2, weights=w) * np.average((a - ma) ** 2, weights=w))
        assert got[b] == pytest.approx(expect)
    assert np.all(counts.sum(1) == classes)
    assert np.array_equal(counts, bootstrap.resample_counts(classes, n_boot=5, seed=3))
