"""The scorer (Creamer's nan_corr, as ported for Logbook 134) and the class bootstrap.

A resample draws the classes with replacement; an entry whose class is drawn c times counts c
times, so each resample's r is a weighted Pearson correlation. It is computed from per-class sums,
which makes 2,000 resamples of every arm cheap.
"""

from __future__ import annotations

import numpy as np
import scipy.stats

from ceiling import config as C


def nan_corr(y_true, y_hat, alpha=0.05, mean_sub=True):
    """Ported verbatim from Creamer's metrics.py (bba43302), lines 109-136."""
    y_true = y_true.reshape(-1)
    y_hat = y_hat.reshape(-1)

    mask = ~np.isnan(y_true) & ~np.isnan(y_hat)
    y_true = y_true[mask]
    y_hat = y_hat[mask]

    if mean_sub:
        y_true = y_true - np.mean(y_true)
        y_hat = y_hat - np.mean(y_hat)

    y_true_std = np.std(y_true, ddof=0)
    y_hat_std = np.std(y_hat, ddof=0)

    corr = np.mean(y_true * y_hat) / y_true_std / y_hat_std

    # now estimate the confidence intervals for the correlation
    n = y_true.shape[0]
    z_a = scipy.stats.norm.ppf(1 - alpha / 2)
    z_r = np.log((1 + corr) / (1 - corr)) / 2
    l = z_r - (z_a / np.sqrt(n - 3))  # noqa: E741 - verbatim
    u = z_r + (z_a / np.sqrt(n - 3))
    ci_l = (np.exp(2 * l) - 1) / (np.exp(2 * l) + 1)
    ci_u = (np.exp(2 * u) - 1) / (np.exp(2 * u) + 1)
    ci = [np.abs(ci_l - corr), ci_u - corr]

    return corr, ci


def r_point(y: np.ndarray, pred: np.ndarray) -> float:
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(nan_corr(np.asarray(y, float), np.asarray(pred, float))[0])


def resample_counts(n_classes: int, n_boot: int = C.N_BOOT, seed: int = C.SEED_BOOTSTRAP) -> np.ndarray:
    """(n_boot, n_classes): how often each class is drawn in each resample."""
    if C.BOOTSTRAP_DRAW != "integers_once":
        raise NotImplementedError(C.BOOTSTRAP_DRAW)
    draws = np.random.default_rng(seed).integers(0, n_classes, size=(n_boot, n_classes))
    counts = np.zeros((n_boot, n_classes))
    np.add.at(counts, (np.repeat(np.arange(n_boot), n_classes), draws.reshape(-1)), 1.0)
    return counts


def _class_sums(values: list[np.ndarray], unit: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((n_classes, len(values)))
    for k, v in enumerate(values):
        out[:, k] = np.bincount(unit, weights=v, minlength=n_classes)
    return out


def resampled_r(y: np.ndarray, pred: np.ndarray, unit: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """r of pred against y in every resample (entries where either is NaN are dropped, as nan_corr
    drops them)."""
    keep = np.isfinite(y) & np.isfinite(pred)
    y, a, unit = y[keep], pred[keep], unit[keep]
    y = y - y.mean()
    a = a - a.mean()
    sums = _class_sums([np.ones_like(y), y, a, y * y, a * a, a * y], unit, counts.shape[1])
    n, sy, sa, syy, saa, say = (counts @ sums).T
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = say / n - sy * sa / n ** 2
        vy = syy / n - (sy / n) ** 2
        va = saa / n - (sa / n) ** 2
        return cov / np.sqrt(vy * va)


def interval(values: np.ndarray, level: float = C.CI_LEVEL) -> tuple[float, float]:
    if C.BOOT_INTERVAL != "percentile":
        raise NotImplementedError(C.BOOT_INTERVAL)
    tail = 100 * (1 - level) / 2
    lo, hi = np.nanpercentile(values, [tail, 100 - tail])
    return float(lo), float(hi)


def resampled_moments(x: np.ndarray, y: np.ndarray, unit: np.ndarray, counts: np.ndarray) -> dict:
    """Per resample: Var(x), Var(y) and Cov(x, y) (population moments) over entries finite in both."""
    keep = np.isfinite(x) & np.isfinite(y)
    x, y, unit = x[keep], y[keep], unit[keep]
    x = x - x.mean()
    y = y - y.mean()
    sums = _class_sums([np.ones_like(x), x, y, x * x, y * y, x * y], unit, counts.shape[1])
    n, sx, sy, sxx, syy, sxy = (counts @ sums).T
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"var_x": sxx / n - (sx / n) ** 2, "var_y": syy / n - (sy / n) ** 2,
                "cov": sxy / n - sx * sy / n ** 2}
