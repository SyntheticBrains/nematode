"""The zero-parameter floors and the calibrated references (floor-C, H0-cal, C.2-cal).

Every floor reads T only through the unit's visible entries, so a prediction never reads a hidden
entry. floor-T built from the visible matrix is undefined (NaN) wherever T[j, i] is hidden or
missing, which is what drops within-fold pairs from the calibration fits.
"""

from __future__ import annotations

import numpy as np

from ceiling import config as C


def visible_T(T: np.ndarray, visible: np.ndarray) -> np.ndarray:
    return np.where(visible, T, np.nan)


def _nanmean(x: np.ndarray, axis: int) -> np.ndarray:
    counts = np.isfinite(x).sum(axis)
    sums = np.nansum(x, axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def floor_me(T: np.ndarray, visible: np.ndarray, split: str) -> np.ndarray:
    """Main effects: the responder's mean visible response (held-out columns), the stimulated
    neuron's mean visible effect (held-out rows), or row + column - grand mean (held-out worms)."""
    Tv = visible_T(T, visible)
    n = len(T)
    if split == "stimulations":
        return np.repeat(_nanmean(Tv, 1)[:, None], n, axis=1)
    if split == "neurons":
        return np.repeat(_nanmean(Tv, 0)[None, :], n, axis=0)
    if split == "worms":
        grand = float(np.nanmean(Tv))
        return _nanmean(Tv, 1)[:, None] + _nanmean(Tv, 0)[None, :] - grand
    raise ValueError(split)


def floor_t(T: np.ndarray, visible: np.ndarray, split: str) -> np.ndarray | None:
    """The transpose: entry (i, j) predicted by T[j, i]; None where floor-T is not an arm."""
    if split == "worms" and not C.FLOOR_T_ON_HELD_OUT_WORMS:
        return None
    return visible_T(T, visible).T.copy()


def _design(features: list[np.ndarray]) -> np.ndarray:
    return np.stack([np.ones_like(features[0])] + features, axis=-1)


def calibrate(T: np.ndarray, visible: np.ndarray, features: list[np.ndarray]) -> dict:
    """Least squares of T on [1, features] over the visible entries where every feature is defined;
    returns the coefficients and the prediction everywhere the features are defined."""
    X = _design(features)
    usable = visible & np.all(np.isfinite(X), axis=-1)
    coef, *_ = np.linalg.lstsq(X[usable], T[usable], rcond=None)
    pred = X @ coef
    pred[~np.all(np.isfinite(X), axis=-1)] = np.nan
    return {"coef": coef.tolist(), "n_fit": int(usable.sum()), "pred": pred}


def references(T: np.ndarray, visible: np.ndarray, split: str,
               mechanistic: dict[str, np.ndarray] | None = None) -> dict[str, dict]:
    """floor-ME, floor-T, floor-C and, given the zero-shot responses, X and X-cal for each."""
    me = floor_me(T, visible, split)
    ft = floor_t(T, visible, split)
    floors = [me] if ft is None else [me, ft]
    out: dict[str, dict] = {"floor-ME": {"pred": me, "coef": [], "n_fit": 0}}
    if ft is not None:
        out["floor-T"] = {"pred": ft, "coef": [], "n_fit": 0}
    out["floor-C"] = calibrate(T, visible, floors)
    for name, response in (mechanistic or {}).items():
        out[name] = {"pred": response, "coef": [], "n_fit": 0}
        out[f"{name}-cal"] = calibrate(T, visible, [response] + floors)
    return out
