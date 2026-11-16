"""The truncated series against dense matrix powers, its gradient, and the arms' parameterisations."""

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

from ceiling import models, supports


def small_case(n=14, nnz=40, k=6, seed=0):
    s = supports.sparse_er(nnz, n, seed)
    rng = np.random.default_rng(seed)
    measured = np.sort(rng.choice(n, k, replace=False))
    values = torch.tensor(rng.normal(0, 0.4, nnz), dtype=torch.float64)
    return s, measured, values


def dense_series(W, measured):
    M = W + W @ W + W @ W @ W + W @ W @ W @ W
    return M[np.ix_(measured, measured)]


def test_sparse_series_matches_dense_powers():
    s, measured, values = small_case()
    model = models.SparseSeries(s.rows, s.cols, s.n, measured, values)
    W = np.zeros((s.n, s.n))
    W[s.rows, s.cols] = values.numpy()
    with torch.no_grad():
        got = model.response().numpy()
        terms = [t.numpy() for t in model.terms()]
    assert np.allclose(got, dense_series(W, measured), atol=1e-12)
    powers = [np.linalg.matrix_power(W, p)[np.ix_(measured, measured)] for p in range(1, 5)]
    for t, p in zip(terms, powers):
        assert np.allclose(t, p, atol=1e-12)


def test_sparse_series_gradient():
    s, measured, _ = small_case(n=10, nnz=25, k=5, seed=3)

    def response(values):
        return models.SparseSeries(s.rows, s.cols, s.n, measured, values).response()

    # SparseSeries clones its initial values, so check through a differentiable wrapper
    def through(values):
        m = models.SparseSeries(s.rows, s.cols, s.n, measured)
        m.values = values
        return m.response()

    v = torch.randn(len(s.rows), dtype=torch.float64, requires_grad=True)
    assert gradcheck(through, (v,), eps=1e-6, atol=1e-6)
    assert torch.allclose(response(v.detach()), through(v.detach()))


def test_diagonal_entries_are_refused():
    with pytest.raises(ValueError):
        models.SparseSeries(np.array([0, 1]), np.array([0, 2]), 3, np.array([0, 1]))


def test_tied_dense_untied_have_zero_diagonal_and_series():
    torch.manual_seed(0)
    for model in (models.Tied(8, 3, seed=1, sd=0.3), models.Untied(8, 3, seed=1, sd=0.3), models.Dense(8)):
        with torch.no_grad():
            if isinstance(model, models.Dense):
                model.W.copy_(torch.randn(8, 8, dtype=torch.float64) * 0.2)
            terms = model.terms()
            W = terms[0].numpy()
            assert np.allclose(np.diag(W), 0)
            expect = W + W @ W + W @ W @ W + W @ W @ W @ W
            assert np.allclose(model.response().numpy(), expect)


def test_parameter_counts_and_matched_rank():
    assert models.Tied(154, 28).n_params == 154 * 28 + 28 * 28 == 5096
    assert models.matched_tied_rank(5103) == 28
    assert models.Dense(154).n_params == 154 * 153
    assert models.Untied(154, 17).n_params == 2 * 154 * 17


def test_tied_initialisation_is_seeded_and_nonzero():
    a, b = models.Tied(10, 3), models.Tied(10, 3)
    assert torch.equal(a.U, b.U) and torch.equal(a.R, b.R)
    assert a.R.abs().min() > 0  # at R = 0 the gradient on U is exactly zero


def test_masked_prior_coupling_and_ridge_on_delta_only():
    s, measured, _ = small_case()
    chem = np.arange(s.size, dtype=float)
    gap = np.ones(s.size)
    m = models.MaskedPrior(s.rows, s.cols, s.n, measured, chem, gap)
    with torch.no_grad():
        m.scales.copy_(torch.tensor([0.5, -0.25], dtype=torch.float64))
    expected = 0.5 * np.log1p(chem) - 0.25 * np.log1p(gap)
    assert np.allclose(m.coupling().detach().numpy(), expected)
    assert m.penalised_values().numel() == s.size  # delta only, not the two scales


def test_dense_ridge_counts_off_diagonal_entries_only():
    assert models.Dense(6).penalised_values().numel() == 30


def test_variance_shares_sum_to_one():
    s, measured, values = small_case(seed=5)
    model = models.SparseSeries(s.rows, s.cols, s.n, measured, values)
    shares = models.variance_shares(model.terms())
    assert sum(shares["covariance_share"]) == pytest.approx(1.0)
    assert len(shares["variance_ratio"]) == 4
