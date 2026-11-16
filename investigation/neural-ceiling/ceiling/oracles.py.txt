"""Section 7: synthetic data sets for the oracles, the noise calibration, and the gates.

A synthetic pair (T_syn, S_syn) is a truth M* placed on T's and S's finite-entry patterns plus
column-block residual noise, scaled so that Var(M*) / Var(T_syn) = rel_T and
Var(M*) / Var(S_syn) = rel_S on those patterns.
"""

from __future__ import annotations

import math

import numpy as np

from ceiling import config as C
from ceiling.readings import family_passes

# ---- Calibration (sections 3 and 7) --------------------------------------------------------------


def count_model(r_ts: float, k: float = C.COUNT_MODEL_K) -> dict:
    """Equal per-animal signal and noise, T averaging k times S's animals: the reliabilities that
    reproduce r(T, S), the ceiling sqrt(rel_S), and the bracket [r, sqrt(r)]."""
    if not 0 < r_ts < 1:
        return {"rel_S": math.nan, "rel_T": math.nan, "ceiling": math.nan, "bracket": [math.nan] * 2}
    a, b, c = 1 / k, 1 + 1 / k, 1 - 1 / r_ts ** 2
    y = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)  # noise-to-signal over S's animal count
    rel_s, rel_t = 1 / (1 + y), 1 / (1 + y / k)
    return {"rel_S": rel_s, "rel_T": rel_t, "ceiling": math.sqrt(rel_s),
            "bracket": [r_ts, math.sqrt(r_ts)]}


def calibration(section3: dict) -> dict:
    """The reliabilities the oracles use: section 3's, or the count model's where one falls
    outside (0, 1)."""
    if C.CALIBRATION_FALLBACK != "per_estimate":
        raise NotImplementedError(C.CALIBRATION_FALLBACK)
    model = count_model(section3["r_TS"])
    out, notes = {}, []
    for key in ("rel_T", "rel_S"):
        value = section3[key]
        if 0 < value < 1:
            out[key] = value
        else:
            out[key] = model[key]
            notes.append(f"{key} = {value} lies outside (0, 1); the count model's {model[key]} is used")
    out["notes"] = notes
    return out


# ---- Noise ---------------------------------------------------------------------------------------


def residual_blocks(residual: np.ndarray, pattern: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Column-block resampled residuals on `pattern` (NaN elsewhere)."""
    if C.RESIDUAL_BLOCK_FILL != "aligned_then_pool":
        raise NotImplementedError(C.RESIDUAL_BLOCK_FILL)
    finite = np.isfinite(residual)
    sources = np.flatnonzero(finite.any(axis=0))
    noise = np.full(residual.shape, np.nan)
    for j in range(residual.shape[1]):
        k = sources[int(rng.integers(len(sources)))]
        need = pattern[:, j]
        column = residual[:, k]
        aligned = need & finite[:, k]
        noise[aligned, j] = column[aligned]
        fill = need & ~finite[:, k]
        if fill.any():
            noise[fill, j] = rng.choice(column[finite[:, k]], size=int(fill.sum()), replace=True)
    return noise


def noise_scale(truth: np.ndarray, noise: np.ndarray, pattern: np.ndarray, rel: float) -> float:
    """a > 0 with Var(M*) / Var(M* + a e) = rel on the pattern (exact, covariance kept)."""
    if C.NOISE_SCALE_RULE != "exact_quadratic":
        raise NotImplementedError(C.NOISE_SCALE_RULE)
    m, e = truth[pattern], noise[pattern]
    vm, ve = float(np.var(m)), float(np.var(e))
    cme = float(np.mean((m - m.mean()) * (e - e.mean())))
    target = vm * (1 / rel - 1)
    return (-cme + math.sqrt(cme * cme + ve * target)) / ve


def place(truth: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    out = np.where(pattern, truth, np.nan)
    np.fill_diagonal(out, np.nan)
    return out


def synthetic_pair(truth: np.ndarray, residual: np.ndarray, t_pattern: np.ndarray,
                   s_pattern: np.ndarray, rel_t: float, rel_s: float,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    """T_syn then S_syn, each with its own column-block noise draw from one generator."""
    out, report = [], {}
    for name, pattern, rel in (("T", t_pattern, rel_t), ("S", s_pattern, rel_s)):
        pattern = pattern & ~np.eye(len(pattern), dtype=bool)
        noise = residual_blocks(residual, pattern, rng)
        a = noise_scale(truth, noise, pattern, rel)
        data = place(truth + a * np.nan_to_num(noise), pattern)
        achieved = float(np.var(truth[pattern]) / np.var(data[pattern]))
        report[name] = {"scale": a, "target_rel": rel, "achieved_rel": achieved}
        out.append(data)
    return out[0], out[1], report


def known_answer(truth: np.ndarray, t_pattern: np.ndarray, s_pattern: np.ndarray):
    """K: T_syn = S_syn = M*, on T's and S's patterns."""
    if not C.K_ON_PATTERNS:
        full = place(truth, np.ones_like(t_pattern))
        return full, full.copy()
    return place(truth, t_pattern), place(truth, s_pattern)


def synthetic_ceiling(t_syn: np.ndarray, s_syn: np.ndarray) -> float:
    """sqrt(Cov(T_syn, S_syn) / Var(S_syn)) on their overlap."""
    both = np.isfinite(t_syn) & np.isfinite(s_syn)
    t, s = t_syn[both], s_syn[both]
    rel = float(np.mean((t - t.mean()) * (s - s.mean())) / np.var(s))
    return math.sqrt(rel) if rel > 0 else math.nan


# ---- Gates ---------------------------------------------------------------------------------------


def k_gate(r_masked: float, data_losses: list[float]) -> dict:
    """K on one held-out split: r >= 0.8; on a failure, near-zero loss (not identified) or high
    loss (the optimiser stopped)."""
    passes = bool(math.isfinite(r_masked) and r_masked >= C.K_BAR)
    worst = max(data_losses) if data_losses else math.nan
    diagnosis = None
    if not passes:
        diagnosis = ("the channel is not identified for this family" if worst <= C.K_NEAR_ZERO_LOSS
                     else "the optimiser stopped")
    return {"passes": passes, "r": r_masked, "max_visible_loss": worst, "diagnosis": diagnosis}


def p_split_gate(draws: list[dict]) -> dict:
    """P on one held-out split: masked meets R-B's bar against both families in >= 2 of 3 draws.
    Each draw holds its m and the two family summaries."""
    met = []
    for d in draws:
        ok = all(fam["readable_draws"] >= C.NULL_READABLE_MIN and family_passes(fam, d["m"])
                 for fam in (d["null-D"], d["null-R"]))
        met.append(bool(ok))
    return {"passes": sum(met) >= C.P_MIN_DRAWS, "met_by_draw": met}


def p_worm_gate(draws: list[dict | None]) -> dict:
    """P's held-out-worm line: masked - r(T_syn, S_syn) above 0 with its interval above 0."""
    met = [bool(d is not None and d["diff"] > 0 and d["ci"][0] > 0) for d in draws]
    return {"passes": sum(met) >= C.P_MIN_DRAWS, "met_by_draw": met}


def n_gate(truths: list[dict]) -> dict:
    """N on one held-out split: biased when, in >= 2 truths, masked is first of the 19 or exceeds
    the 18 draws' median by at least m. Each truth holds its m and its family summary."""
    biased = []
    for t in truths:
        fam = t["null-D"]
        ok = "rank" in fam and (fam["rank"] == 1 or fam["diff"] >= t["m"])
        biased.append(bool(ok))
    return {"fails": sum(biased) >= C.N_FAIL_MIN_TRUTHS, "biased_by_truth": biased}


def r_att(values: list[float]) -> dict:
    finite = [v for v in values if math.isfinite(v)]
    return {"r_att": float(np.median(finite)) if finite else math.nan, "draws": values,
            "spread": [min(finite), max(finite)] if finite else [math.nan, math.nan]}
