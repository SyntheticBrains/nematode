"""One fit: standardise T by the fit's visible entries, minimise MSE + lambda * mean(theta^2) by
L-BFGS (strong Wolfe, history 20), and apply the refit and readability rules. Also the lambda
choice with its one-time edge extension.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from ceiling import config as C
from ceiling.models import DTYPE, SeriesModel, variance_shares


@dataclass
class FitResult:
    lam_requested: float
    lam: float  # the lambda the kept fit used
    readable: bool
    refit: bool
    at_cap: bool
    n_iter: int
    loss: float  # the objective at the end
    data_loss: float  # visible-entry MSE, standardised units
    mu: float
    sigma: float
    n_params: int
    seconds: float
    moved: list[float] = field(default_factory=list)  # max |change| of each parameter tensor
    shares: dict = field(default_factory=dict)
    pred: np.ndarray | None = None  # raw units, measured x measured

    def summary(self) -> dict:
        out = asdict(self)
        out.pop("pred")
        return out


def standardise(T: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = T[mask]
    mu, sigma = float(values.mean()), float(values.std())
    if not sigma > 0:
        raise ValueError("visible entries have no spread")
    return (T - mu) / sigma, mu, sigma


def _fit_once(model: SeriesModel, y: np.ndarray, mask: np.ndarray, lam: float,
              max_iter: int) -> tuple[float, float, int]:
    flat = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long)
    target = torch.as_tensor(y.reshape(-1)[np.flatnonzero(mask)], dtype=DTYPE)
    optimiser = torch.optim.LBFGS(
        model.params, lr=1.0, max_iter=max_iter, max_eval=max_iter * C.LBFGS_MAX_EVAL_PER_ITER,
        tolerance_grad=C.LBFGS_TOLERANCE_GRAD, tolerance_change=C.LOSS_CHANGE_TOL,
        history_size=C.LBFGS_HISTORY, line_search_fn="strong_wolfe")

    def objective() -> tuple[torch.Tensor, torch.Tensor]:
        pred = model.response().reshape(-1)[flat]
        data = torch.mean((pred - target) ** 2)
        if C.RIDGE_MEAN != "joint":
            raise NotImplementedError(C.RIDGE_MEAN)
        return data + lam * torch.mean(model.penalised_values() ** 2), data

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss, _ = objective()
        loss.backward()
        return loss

    optimiser.step(closure)
    state = optimiser.state[optimiser._params[0]]
    with torch.no_grad():
        loss, data = objective()
    return float(loss), float(data), int(state.get("n_iter", 0))


def fit(build: Callable[[], SeriesModel], T: np.ndarray, mask: np.ndarray, lam: float,
        next_lam: float, max_iter: int = C.MAX_ITER) -> FitResult:
    """Fit once at lam; a non-finite loss is refit once at next_lam; still non-finite, unreadable."""
    y, mu, sigma = standardise(T, mask)
    start = time.perf_counter()
    attempts = 0
    for lam_used in (lam, next_lam):
        attempts += 1
        model = build()
        before = model.snapshot()
        loss, data, n_iter = _fit_once(model, y, mask, lam_used, max_iter)
        if math.isfinite(loss):
            break
    refit = attempts == 2
    readable = math.isfinite(loss)
    result = FitResult(lam_requested=lam, lam=lam_used, readable=readable, refit=refit,
                       at_cap=n_iter >= max_iter, n_iter=n_iter, loss=loss, data_loss=data,
                       mu=mu, sigma=sigma, n_params=model.n_params,
                       seconds=time.perf_counter() - start)
    if readable:
        with torch.no_grad():
            terms = model.terms()
            pred = torch.stack(terms).sum(0).numpy()
        result.pred = mu + sigma * pred
        result.shares = variance_shares(terms)
        result.moved = [float((p.detach() - b).abs().max()) for p, b in zip(model.params, before)]
    return result


def validation_mse(pred_raw: np.ndarray | None, T: np.ndarray, validation: np.ndarray,
                   unit_var: float) -> float:
    """Validation MSE in the outer unit's standardised units; +inf for an unreadable fit."""
    if pred_raw is None:
        return math.inf
    diff = pred_raw[validation] - T[validation]
    value = float(np.mean(diff ** 2) / unit_var)
    return value if math.isfinite(value) else math.inf


# ---- lambda ------------------------------------------------------------------------------------


def next_lambda(lam: float, grid: list[float]) -> float:
    larger = [g for g in sorted(grid) if g > lam * (1 + 1e-12)]
    return larger[0] if larger else lam * 10.0


def edge_extension(choice: float, grid: list[float]) -> float | None:
    """The decade beyond the edge the choice landed on, or None if it is interior."""
    grid = sorted(grid)
    if math.isclose(choice, grid[0]):
        return grid[0] / 10.0
    if math.isclose(choice, grid[-1]):
        return grid[-1] * 10.0
    return None


def choose(losses: dict[float, float]) -> float | None:
    """The lambda with the smallest summed loss (the smaller lambda on a tie); None if none finite."""
    finite = {lam: v for lam, v in losses.items() if math.isfinite(v)}
    if not finite:
        return None
    return min(sorted(finite), key=lambda lam: finite[lam])
