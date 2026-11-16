"""Section 5's identification argument, noise-free, with the registered loss and optimiser.

The targets are in the model's own units with a spread near 1 (the standardisation is tested
apart): a held-out column is predictable only through paths that pass through it (masked) or
through the neuron's visible row (tied); an untied factorisation leaves it to the initialisation.
"""

import numpy as np
import torch

from ceiling import fitting, models, supports

HIDE = [3, 11]


def held_out(k):
    visible = ~np.eye(k, dtype=bool)
    visible[:, HIDE] = False
    hidden = np.zeros_like(visible)
    hidden[:, HIDE] = True
    hidden &= ~np.eye(k, dtype=bool)
    hidden[np.ix_(HIDE, HIDE)] = False  # within-fold pairs are not scored
    return visible, hidden


def fit_model(model, M, visible, iters=3000):
    _, data, _ = fitting._fit_once(model, M, visible, 1e-10, iters)
    with torch.no_grad():
        return model.response().numpy(), data


def r(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def tied_truth(k=20, rank=3, seed=1):
    rng = np.random.default_rng(seed)
    truth = models.Tied(k, rank)
    with torch.no_grad():
        truth.U.copy_(torch.tensor(rng.normal(0, 0.5, (k, rank))))
        truth.R.copy_(torch.tensor(rng.normal(0, 0.8, (rank, rank))))
        return truth.response().numpy()


def test_tied_recovers_a_held_out_column_and_untied_does_not():
    """Tied predicts an unseen column through the neuron's visible row (r near 1); untied fits the
    visible entries as well, but its held-out column depends on its initialisation (the gauge)."""
    k, rank = 20, 3
    M = tied_truth(k, rank)
    visible, hidden = held_out(k)
    pred, data = fit_model(models.Tied(k, rank), M, visible)
    assert data < 1e-7
    assert r(pred[hidden], M[hidden]) > 0.999

    untied = [fit_model(models.Untied(k, rank, seed=s), M, visible) for s in (1, 2)]
    for p, d in untied:
        assert d < 1e-2  # the visible entries are fitted
        assert r(p[hidden], M[hidden]) < 0.9
    spread = np.abs(untied[0][0][hidden] - untied[1][0][hidden]).max() / M[hidden].std()
    assert spread > 1.0  # two equally good visible fits disagree on the unseen column


def sparse_truth(seed, inputs_to_hidden: bool, n=20, nnz=50, sd=0.9):
    rng = np.random.default_rng(seed)
    keys = supports.sparse_er(nnz, n, seed).keys()
    if inputs_to_hidden:
        for j in HIDE:
            sources = [c for c in range(n) if c not in HIDE and c != j]
            keys |= {(j, int(c)) for c in rng.choice(sources, 3, replace=False)}
    else:
        keys = {(a, b) for a, b in keys if a not in HIDE}  # the hidden neurons receive nothing
    keys = sorted(keys)
    rows, cols = np.array([a for a, _ in keys]), np.array([b for _, b in keys])
    values = torch.tensor(rng.normal(0, sd, len(keys)))
    with torch.no_grad():
        M = models.SparseSeries(rows, cols, n, np.arange(n), values).response().numpy()
    return rows, cols, M


def test_masked_recovers_a_held_out_column_through_its_paths():
    """A held-out column whose neuron receives visible inputs is recovered exactly."""
    rows, cols, M = sparse_truth(6, inputs_to_hidden=True)
    visible, hidden = held_out(20)
    pred, data = fit_model(models.SparseSeries(rows, cols, 20, np.arange(20)), M, visible)
    assert data < 1e-8
    assert r(pred[hidden], M[hidden]) > 0.999


def test_masked_cannot_reach_a_column_with_no_path_through_it():
    """With no inputs into the held-out neurons, their outputs get no gradient: the column stays at
    its initial zero while the visible entries are fitted."""
    rows, cols, M = sparse_truth(6, inputs_to_hidden=False)
    visible, hidden = held_out(20)
    pred, data = fit_model(models.SparseSeries(rows, cols, 20, np.arange(20)), M, visible)
    assert data < 1e-8
    assert np.abs(M[hidden]).max() > 0.1
    assert np.all(pred[hidden] == 0.0)
