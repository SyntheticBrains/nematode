"""The arms' coupling models. Every arm's response is the truncated series M = W + W^2 + W^3 + W^4,
read on the measured neurons, with W's diagonal zero.

The sparse arms hold one value per support entry over all n nodes and multiply by index (a gather
and a scatter per product), so a product costs one multiply-add per entry and measured column. The
tied, dense and untied arms hold W over the measured neurons only.
"""

from __future__ import annotations

import numpy as np
import torch

from ceiling import config as C

DTYPE = torch.float64


class SeriesModel:
    """A parameterised W and its series terms on the measured neurons."""

    kind = "base"

    def __init__(self) -> None:
        self.params: list[torch.Tensor] = []
        self.penalised: list[torch.Tensor] = []

    def penalised_values(self) -> torch.Tensor:
        """The parameters the ridge averages over, flattened."""
        return torch.cat([p.reshape(-1) for p in self.penalised])

    def terms(self) -> list[torch.Tensor]:
        raise NotImplementedError

    def response(self) -> torch.Tensor:
        return torch.stack(self.terms()).sum(0)

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.params))

    def snapshot(self) -> list[torch.Tensor]:
        return [p.detach().clone() for p in self.params]


def _dense_terms(W: torch.Tensor, n_terms: int = C.SERIES_TERMS) -> list[torch.Tensor]:
    out = [W]
    for _ in range(n_terms - 1):
        out.append(out[-1] @ W)
    return out


def _off_diagonal(n: int) -> torch.Tensor:
    return 1.0 - torch.eye(n, dtype=DTYPE)


class SparseSeries(SeriesModel):
    """W free on a support over n nodes (W[target, source]); M read on `measured`."""

    kind = "sparse"

    def __init__(self, rows: np.ndarray, cols: np.ndarray, n: int, measured: np.ndarray,
                 values: torch.Tensor | None = None) -> None:
        super().__init__()
        if np.any(np.asarray(rows) == np.asarray(cols)):
            raise ValueError("a support entry lies on the diagonal")
        self.n = n
        self.rows = torch.as_tensor(np.asarray(rows), dtype=torch.long)
        self.cols = torch.as_tensor(np.asarray(cols), dtype=torch.long)
        self.measured = torch.as_tensor(np.asarray(measured), dtype=torch.long)
        pos = -np.ones(n, dtype=np.int64)
        pos[np.asarray(measured)] = np.arange(len(measured))
        col_pos = pos[np.asarray(cols)]
        self.first = torch.as_tensor(np.flatnonzero(col_pos >= 0), dtype=torch.long)
        self.first_rows = self.rows[self.first]
        self.first_cols = torch.as_tensor(col_pos[col_pos >= 0], dtype=torch.long)
        init = torch.zeros(len(rows), dtype=DTYPE) if values is None else values
        self.values = init.clone().to(DTYPE).requires_grad_(True)
        self.params = [self.values]
        self.penalised = [self.values]

    def coupling(self) -> torch.Tensor:
        return self.values

    def terms(self) -> list[torch.Tensor]:
        v = self.coupling()
        k = len(self.measured)
        x = torch.zeros(self.n, k, dtype=DTYPE).index_put(
            (self.first_rows, self.first_cols), v[self.first], accumulate=True)
        out = [x]
        for _ in range(C.SERIES_TERMS - 1):
            x = torch.zeros(self.n, k, dtype=DTYPE).index_add(0, self.rows, v[:, None] * x[self.cols])
            out.append(x)
        return [t[self.measured] for t in out]


class MaskedPrior(SparseSeries):
    """W = s_c log(1 + chemical count) + s_g log(1 + gap count) + delta on a support; ridge on delta."""

    kind = "prior"

    def __init__(self, rows, cols, n, measured, chem: np.ndarray, gap: np.ndarray) -> None:
        super().__init__(rows, cols, n, measured)
        self.log_chem = torch.as_tensor(np.log1p(np.asarray(chem, dtype=np.float64)), dtype=DTYPE)
        self.log_gap = torch.as_tensor(np.log1p(np.asarray(gap, dtype=np.float64)), dtype=DTYPE)
        self.scales = torch.zeros(2, dtype=DTYPE, requires_grad=True)
        self.delta = self.values
        self.params = [self.scales, self.delta]
        self.penalised = [self.delta]

    def coupling(self) -> torch.Tensor:
        return self.scales[0] * self.log_chem + self.scales[1] * self.log_gap + self.delta


class Dense(SeriesModel):
    """W dense over the measured neurons, zero diagonal."""

    kind = "dense"

    def __init__(self, k: int) -> None:
        super().__init__()
        if C.DENSE_INIT != "zeros":
            raise NotImplementedError(C.DENSE_INIT)
        self.mask = _off_diagonal(k)
        self.W = torch.zeros(k, k, dtype=DTYPE, requires_grad=True)
        self.params = [self.W]
        self.penalised = [self.W]

    def penalised_values(self) -> torch.Tensor:
        return self.W[self.mask.bool()]

    @property
    def n_params(self) -> int:
        k = self.W.shape[0]
        return k * (k - 1)

    def terms(self) -> list[torch.Tensor]:
        return _dense_terms(self.W * self.mask)


class Tied(SeriesModel):
    """W = U R U^T over the measured neurons (one embedding per neuron in both roles), zero diagonal."""

    kind = "tied"

    def __init__(self, k: int, rank: int, seed: int = C.SEED_TIED_INIT, sd: float = C.INIT_SD) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        U = rng.normal(0.0, sd, (k, rank))
        R = rng.normal(0.0, sd, (rank, rank))
        self.U = torch.tensor(U, dtype=DTYPE, requires_grad=True)
        self.R = torch.tensor(R, dtype=DTYPE, requires_grad=True)
        self.mask = _off_diagonal(k)
        self.params = [self.U, self.R]
        self.penalised = [self.U, self.R]

    def terms(self) -> list[torch.Tensor]:
        return _dense_terms((self.U @ self.R @ self.U.T) * self.mask)


class Untied(SeriesModel):
    """W = U V^T over the measured neurons, zero diagonal (description, held-out worms only)."""

    kind = "untied"

    def __init__(self, k: int, rank: int, seed: int = C.UNTIED_INIT_SEED, sd: float = C.INIT_SD) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        U = rng.normal(0.0, sd, (k, rank))
        V = rng.normal(0.0, sd, (k, rank))
        self.U = torch.tensor(U, dtype=DTYPE, requires_grad=True)
        self.V = torch.tensor(V, dtype=DTYPE, requires_grad=True)
        self.mask = _off_diagonal(k)
        self.params = [self.U, self.V]
        self.penalised = [self.U, self.V]

    def terms(self) -> list[torch.Tensor]:
        return _dense_terms((self.U @ self.V.T) * self.mask)


def matched_tied_rank(n_params: int, k: int = C.N_MEASURED) -> int:
    """The rank whose tied parameter count k*r + r^2 is closest to n_params (ties to the lower)."""
    best = min(range(1, k + 1), key=lambda r: (abs(k * r + r * r - n_params), r))
    return best


def variance_shares(terms: list[torch.Tensor]) -> dict[str, list[float]]:
    """Each hop's share of M's variance over the off-diagonal entries."""
    with torch.no_grad():
        k = terms[0].shape[0]
        off = ~torch.eye(k, dtype=torch.bool)
        parts = torch.stack([t[off] for t in terms])
        m = parts.sum(0)
        var_m = torch.var(m, unbiased=False)
        centred = parts - parts.mean(1, keepdim=True)
        cov = (centred * (m - m.mean())).mean(1)
        var = centred.pow(2).mean(1)
        if not torch.isfinite(var_m) or var_m <= 0:
            nan = [float("nan")] * len(terms)
            return {"covariance_share": nan, "variance_ratio": nan}
        return {"covariance_share": (cov / var_m).tolist(), "variance_ratio": (var / var_m).tolist()}
