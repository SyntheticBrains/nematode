"""Red Queen analysis primitives for the co-evolution loop.

Five pure functions operating on numpy arrays / floats only — no `Genome`,
`FitnessFunction`, or env dependencies. Designed to be called by both the
post-hoc aggregator and any in-loop diagnostic that wants a quick
readout on a per-generation trait or fitness series.

The five primitives:

1. :func:`phenotypic_cycling` — autocorrelation peak detection within a
   bounded lag range. Used by the verdict gate's cycling criterion to detect
   oscillation in trait/fitness series.
2. :func:`trait_escalation` — windowed linear regression of a per-gen
   trait-mean series; reports slope sign + significance. Used by the
   verdict gate's escalation criterion for monotone trait drift.
3. :func:`fitness_lag` — cross-correlation between two series with a
   `max_lag` parameter; returns the lag at which |correlation| peaks.
4. :func:`coupled_rate` — Pearson correlation between per-generation
   deltas of two series; +1 means lock-step change, 0 means independent.
5. :func:`generality` — single-scalar summary of a held-out probe matrix
   (gens by opponents) indicating whether elite performance generalises
   uniformly (positive) or self-play overfits (zero / negative).

All functions return either a scalar or a small dict; none mutate their
inputs. Zero-variance / pathological inputs return `NaN` rather than
raising — the aggregator turns NaN into a "no signal" cell rather than
aborting the whole run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


# Default p-value threshold matching the verdict gate's significance bar.
_DEFAULT_P_THRESHOLD = 0.05

# Minimum series length to produce a meaningful detection. Below this the
# cycling / escalation tests cannot reach significance regardless of the
# underlying signal — gating saves callers from spurious "False" returns
# at trivial input lengths.
_MIN_SERIES_LENGTH = 10

# Default lag-range window for `phenotypic_cycling` per the verdict
# gate ("autocorrelation peak at lag in [3, 15] generations").
_DEFAULT_LAG_RANGE = (3, 15)

# Default gen-window for `trait_escalation` per the verdict gate
# ("regression over gens 5..30, skipping TPE bootstrap noise").
_DEFAULT_GEN_WINDOW = (5, 30)


# ---------------------------------------------------------------------------
# 1. Phenotypic cycling
# ---------------------------------------------------------------------------


def phenotypic_cycling(
    series: Sequence[float] | np.ndarray,
    *,
    lag_range: tuple[int, int] = _DEFAULT_LAG_RANGE,
    p_threshold: float = _DEFAULT_P_THRESHOLD,
) -> dict[str, Any]:
    """Detect periodic oscillation via autocorrelation peak in a lag range.

    Approach: compute the unbiased autocorrelation of the mean-centered
    series, scan lags in ``lag_range`` for the peak |correlation|, and
    return its lag + a permutation-based p-value (random-shuffle null
    distribution). The series-length and lag-range guards ensure the
    returned dict is always well-formed even on degenerate inputs.

    Parameters
    ----------
    series
        Per-generation values (trait mean, fitness, etc.). Must be
        1-D and finite. Length must be > ``lag_range[1]`` for the
        peak to be defined.
    lag_range
        Inclusive ``(low, high)`` lag bounds in generations. Defaults
        to ``(3, 15)`` per the verdict gate.
    p_threshold
        Significance threshold for `cycling_detected`. A series is
        flagged as cycling iff the peak |correlation|'s permutation
        p-value is strictly below this threshold.

    Returns
    -------
    dict[str, Any]
        ``{"cycling_detected": bool, "dominant_period": int | None,
        "p_value": float}``. ``dominant_period`` is the lag at the
        peak (only meaningful when ``cycling_detected=True``); set to
        ``None`` when no peak is found within the lag range.
    """
    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim != 1:
        msg = f"phenotypic_cycling: series must be 1-D, got shape {arr.shape}"
        raise ValueError(msg)
    low, high = lag_range
    if low < 1 or high <= low:
        msg = f"lag_range must satisfy 1 <= low < high, got {lag_range}"
        raise ValueError(msg)

    n = arr.size
    # Need at least `high + 1` samples to compute autocorr at lag `high`,
    # plus a small buffer so the peak detection has signal to work with.
    min_required = max(high + 1, _MIN_SERIES_LENGTH)
    if n < min_required:
        return {
            "cycling_detected": False,
            "dominant_period": None,
            "p_value": float("nan"),
        }

    # Detrend before autocorrelation. Without detrending a monotone
    # ramp produces a near-1 autocorrelation at every lag (consecutive
    # values are similar), which the peak detector would happily flag
    # as "cycling at lag low" — a false positive on the canonical
    # rejection case ("Rejects Pure Linear Trend"). OLS-detrend
    # subtracts the linear best-fit so the residuals carry only the
    # higher-frequency content where genuine cycling lives.
    x_axis = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(x_axis, arr, deg=1)
    centered = arr - (intercept + slope * x_axis)
    variance = float((centered**2).sum())
    raw_variance = float(((arr - arr.mean()) ** 2).sum())
    # Two-stage variance gate:
    #
    # 1. Absolute floor scaled to the series magnitude. Catches
    #    "tiny-amplitude oscillations on a large constant" — e.g.
    #    `[1.0, 1.0+1e-15, 1.0-1e-15, ...]` where raw_variance is
    #    machine-epsilon-noise around the mean (~1e-29) and ratio
    #    `variance / raw_variance` is order-1 → would bypass the
    #    rel-floor below and falsely flag cycling on numerical noise.
    #    The absolute floor is `eps * (1 + mean^2) * n` with
    #    `eps = 1e-20`: well below any realistic per-element fitness
    #    noise (typical fitness values are O(0.1)-O(1), so the comparable
    #    noise floor is around 1e-12 absolute, i.e. raw_variance
    #    around 1e-23 for an n=40 series).
    # 2. Relative floor (residual variance < 1e-12 x raw variance):
    #    catches pure-linear series whose detrend residuals are
    #    numerical-noise (~1e-28) but whose raw variance is large
    #    (~1e+3 for a `[0..40]` ramp). Without this gate the
    #    autocorrelation of the noise-only residuals would
    #    spuriously flag low-lag cycling.
    abs_variance_eps = 1e-20
    abs_variance_floor = abs_variance_eps * (1.0 + arr.mean() ** 2) * n
    rel_variance_floor = 1e-12
    if (
        variance <= 0.0
        or raw_variance <= 0.0
        or raw_variance < abs_variance_floor
        or variance / raw_variance < rel_variance_floor
    ):
        # Constant, near-constant, or numerically-pure-linear series
        # — no oscillatory residual that's not numerical noise.
        return {
            "cycling_detected": False,
            "dominant_period": None,
            "p_value": float("nan"),
        }

    # Track the peak POSITIVE autocorrelation (period repetition), not
    # peak |autocorr|. A pure sine of period P has autocorr cos(2π·lag/P),
    # so the first true repetition lag is P (corr = +1), but |corr| also
    # peaks at P/2 with corr = -1. Using positive-only matches the
    # spec's "dominant period" semantics.
    lags = np.arange(low, high + 1)
    autocorrs = np.array(
        [
            float((centered[: n - lag] * centered[lag:]).sum() / ((n - lag) * variance / n))
            for lag in lags
        ],
    )

    peak_idx = int(np.argmax(autocorrs))
    peak_lag = int(lags[peak_idx])
    peak_corr = float(autocorrs[peak_idx])
    if peak_corr <= 0.0:
        # No positive peak in range → no plausible repetition.
        return {
            "cycling_detected": False,
            "dominant_period": None,
            "p_value": float("nan"),
        }

    # Permutation null: shuffle the centered series many times, take the
    # max positive autocorrelation within the same lag range, count how
    # often the null peak exceeds the observed peak. Cheap (autocorr at
    # fixed lags is O(n) per shuffle), and distribution-free, which
    # suits noisy short series better than a parametric Lomb-Scargle.
    n_perms = 500
    rng = np.random.default_rng(seed=0)  # fixed seed → deterministic p-value
    null_max = np.empty(n_perms)
    for i in range(n_perms):
        shuffled = rng.permutation(centered)
        shuffled_max = max(
            float((shuffled[: n - lag] * shuffled[lag:]).sum() / ((n - lag) * variance / n))
            for lag in lags
        )
        null_max[i] = shuffled_max
    # +1 in numerator + denominator is the standard Monte-Carlo p-value
    # correction so the result is never exactly zero for finite n_perms.
    p_value = float((1 + (null_max >= peak_corr).sum()) / (1 + n_perms))

    return {
        "cycling_detected": bool(p_value < p_threshold),
        "dominant_period": peak_lag,
        "p_value": p_value,
    }


# ---------------------------------------------------------------------------
# 2. Trait escalation
# ---------------------------------------------------------------------------


def trait_escalation(
    values: Sequence[float] | np.ndarray,
    *,
    gen_window: tuple[int, int] = _DEFAULT_GEN_WINDOW,
    p_threshold: float = _DEFAULT_P_THRESHOLD,
) -> dict[str, Any]:
    """Fit a linear regression on the per-generation trait-mean series.

    The regression is fit over generations ``gen_window[0]..gen_window[1]``
    (half-open: includes ``gen_window[0]``, excludes ``gen_window[1]+1``)
    so the default ``(5, 30)`` skips the optimiser-bootstrap noise of
    the first 5 generations (where TPE-style optimisers are still
    accumulating their initial sample distribution).

    Significance is reported as a two-sided t-test on the slope; the
    `escalation_detected` flag fires iff the slope is significant AND
    has a sign — the gate is direction-agnostic (escalation in either
    direction qualifies; the verdict caller filters by expected sign).

    Parameters
    ----------
    values
        Per-generation trait series (one float per generation, indexed
        from generation 0).
    gen_window
        Inclusive-low, inclusive-high generation indices for the
        regression. The default ``(5, 30)`` matches the verdict gate.
    p_threshold
        Significance threshold for `escalation_detected`.

    Returns
    -------
    dict[str, Any]
        ``{"escalation_detected": bool, "slope": float,
        "slope_sign": int, "slope_se": float, "p_value": float}``.
        ``slope_sign`` is `+1`, `-1`, or `0` (zero only on degenerate
        all-flat inputs).
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        msg = f"trait_escalation: values must be 1-D, got shape {arr.shape}"
        raise ValueError(msg)
    low, high = gen_window
    if low < 0 or high <= low:
        msg = f"gen_window must satisfy 0 <= low < high, got {gen_window}"
        raise ValueError(msg)

    # Slice out the requested window. Series shorter than `low` cannot
    # produce a regression — return a nan-shaped result.
    if arr.size <= low:
        return {
            "escalation_detected": False,
            "slope": float("nan"),
            "slope_sign": 0,
            "slope_se": float("nan"),
            "p_value": float("nan"),
        }
    upper = min(high, arr.size - 1)
    y = arr[low : upper + 1]
    x = np.arange(low, upper + 1, dtype=np.float64)

    if y.size < 3:  # noqa: PLR2004 — t-test on slope needs >=3 samples
        return {
            "escalation_detected": False,
            "slope": float("nan"),
            "slope_sign": 0,
            "slope_se": float("nan"),
            "p_value": float("nan"),
        }
    if np.std(y) == 0.0:
        return {
            "escalation_detected": False,
            "slope": 0.0,
            "slope_sign": 0,
            "slope_se": 0.0,
            "p_value": 1.0,
        }

    # Closed-form OLS slope + standard error. Avoids a scipy dependency
    # for what's a textbook regression on a tiny window.
    x_mean = x.mean()
    y_mean = y.mean()
    sxx = float(((x - x_mean) ** 2).sum())
    sxy = float(((x - x_mean) * (y - y_mean)).sum())
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    residuals = y - (intercept + slope * x)
    n = y.size
    # Degrees of freedom = n - 2 (slope + intercept).
    sigma_squared = float((residuals**2).sum() / (n - 2))
    slope_se = float(np.sqrt(sigma_squared / sxx)) if sxx > 0 else float("nan")

    # Two-sided t-test on slope. We avoid scipy and use the survival
    # function of the standard normal as the asymptotic approximation
    # (n in our window is up to 26, large enough that the t-distribution
    # is close to normal for verdict-gate purposes; the aggregator can
    # report the t statistic itself if a reviewer wants tighter inference).
    if slope_se <= 0.0 or not np.isfinite(slope_se):
        # Zero-residual perfectly-linear fit: the slope is *exactly*
        # known, so p_value = 0 if slope is non-zero and 1 if slope is
        # zero (no signal). Anything else is the degenerate insignificant
        # case (we have no inference available).
        p_value = 0.0 if slope != 0.0 else 1.0
    else:
        t_stat = slope / slope_se
        # Two-sided p-value via the standard normal CDF (asymptotic).
        from math import erf, sqrt

        p_value = float(2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t_stat) / sqrt(2.0)))))

    slope_sign = int(np.sign(slope))
    detected = bool(p_value < p_threshold and slope_sign != 0)
    return {
        "escalation_detected": detected,
        "slope": float(slope),
        "slope_sign": slope_sign,
        "slope_se": slope_se,
        "p_value": p_value,
    }


# ---------------------------------------------------------------------------
# 3. Fitness lag
# ---------------------------------------------------------------------------


def fitness_lag(
    series_a: Sequence[float] | np.ndarray,
    series_b: Sequence[float] | np.ndarray,
    *,
    max_lag: int = 20,
) -> float:
    """Return the lag at which the cross-correlation between two series peaks.

    Lag convention: positive means ``series_b`` is shifted forward
    relative to ``series_a`` (i.e. ``series_b[g] = series_a[g - lag]``
    is the matched-up version). Scanned over
    ``[-max_lag, +max_lag]``.

    Tracks peak POSITIVE correlation (where the two series align in
    phase). Anti-phase coupling is reported by :func:`coupled_rate`,
    not here — for periodic inputs the |correlation| would peak at
    half-period whether or not the series actually align, which would
    confuse "where do they line up?" with "where are they anti-aligned?".

    Parameters
    ----------
    series_a, series_b
        Equal-length 1-D series.
    max_lag
        Maximum absolute lag scanned. Must be < min(len(a), len(b)).

    Returns
    -------
    float
        Best lag (integer-valued but typed `float` so callers can
        compare against `NaN` for trivial inputs without a separate
        sentinel). Returns `NaN` when either series has zero variance
        or when `max_lag` is too large for the input length.
    """
    a = np.asarray(series_a, dtype=np.float64)
    b = np.asarray(series_b, dtype=np.float64)
    if a.shape != b.shape:
        msg = f"fitness_lag: series shapes must match, got {a.shape} vs {b.shape}"
        raise ValueError(msg)
    if a.ndim != 1:
        msg = f"fitness_lag: series must be 1-D, got shape {a.shape}"
        raise ValueError(msg)
    n = a.size
    if max_lag < 0 or max_lag >= n:
        return float("nan")
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")

    a_centered = a - a.mean()
    b_centered = b - b.mean()
    norm = float(np.sqrt((a_centered**2).sum() * (b_centered**2).sum()))
    if norm <= 0.0:
        return float("nan")

    # Track peak POSITIVE correlation (where the two series align in
    # phase), NOT peak |correlation|. For periodic series the
    # anti-phase peak at half-period would otherwise dominate the true
    # alignment lag — e.g. two sines at period 12 with shift 4: peak
    # +corr is at lag +4 (the actual shift), but |corr| also peaks at
    # lag +/-2 (half-period anti-phase). Anti-phase coupling is a
    # separate concept captured by `coupled_rate` (returns -1 for
    # anti-coupled deltas); `fitness_lag` reports the in-phase shift.
    best_lag = 0
    best_corr = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            corr = float((a_centered[: n - lag] * b_centered[lag:]).sum() / norm)
        else:
            shift = -lag
            corr = float((a_centered[shift:] * b_centered[: n - shift]).sum() / norm)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return float(best_lag)


# ---------------------------------------------------------------------------
# 4. Coupled rate
# ---------------------------------------------------------------------------


def coupled_rate(
    trait_a: Sequence[float] | np.ndarray,
    trait_b: Sequence[float] | np.ndarray,
) -> float:
    """Return the Pearson correlation of generation-to-generation deltas.

    Computes ``delta_a[g] = trait_a[g+1] - trait_a[g]`` (and similarly
    for ``b``), then returns ``Pearson(delta_a, delta_b)``. A value
    near +1 means the two sides change in lock-step; near 0 means
    independent change; near -1 means anti-coupled change. Returns
    `NaN` when either delta series has zero variance.

    Parameters
    ----------
    trait_a, trait_b
        Equal-length 1-D series of length >= 2.

    Returns
    -------
    float
        Coupling coefficient in `[-1, 1]` or `NaN`.
    """
    a = np.asarray(trait_a, dtype=np.float64)
    b = np.asarray(trait_b, dtype=np.float64)
    if a.shape != b.shape:
        msg = f"coupled_rate: series shapes must match, got {a.shape} vs {b.shape}"
        raise ValueError(msg)
    if a.ndim != 1:
        msg = f"coupled_rate: series must be 1-D, got shape {a.shape}"
        raise ValueError(msg)
    if a.size < 2:  # noqa: PLR2004 — need at least 2 samples for one delta
        return float("nan")

    da = np.diff(a)
    db = np.diff(b)
    sa = np.std(da)
    sb = np.std(db)
    if sa == 0.0 or sb == 0.0:
        return float("nan")
    # `np.corrcoef` returns the 2x2 correlation matrix; the off-diagonal
    # is the Pearson coefficient.
    return float(np.corrcoef(da, db)[0, 1])


# ---------------------------------------------------------------------------
# 5. Generality
# ---------------------------------------------------------------------------


def generality(probe_series: np.ndarray) -> float:
    """Summarise a (gens, opponents) probe matrix as a single scalar.

    The probe matrix has one row per generation at which the held-out
    probe fired and one column per held-out opponent. The scalar
    summary indicates whether the elite improves uniformly across
    opponents (positive, near +1) or only on training opponents while
    held-out fitness stays flat or declines (zero, near 0, or
    negative).

    Algorithm: for each opponent column, fit a linear regression of
    fitness on generation; take the mean slope across opponents,
    normalised by the maximum |slope| across opponents (so the result
    is bounded in `[-1, 1]`). When all opponents have zero slope (no
    movement), returns 0.0.

    Parameters
    ----------
    probe_series
        2-D array of shape ``(num_probes, num_opponents)``. Each row is
        one probe (corresponding to one generation at which the probe
        fired); each column is one held-out opponent.

    Returns
    -------
    float
        Generality scalar in `[-1, 1]`, or `NaN` for fewer than 2 probe
        rows (slope is undefined).
    """
    arr = np.asarray(probe_series, dtype=np.float64)
    expected_ndim = 2
    if arr.ndim != expected_ndim:
        msg = f"generality: probe_series must be 2-D, got shape {arr.shape}"
        raise ValueError(msg)
    n_probes, n_opponents = arr.shape
    if n_probes < 2:  # noqa: PLR2004 — slope is undefined for <2 probe rows
        return float("nan")
    if n_opponents == 0:
        return float("nan")

    x = np.arange(n_probes, dtype=np.float64)
    x_mean = x.mean()
    sxx = float(((x - x_mean) ** 2).sum())
    if sxx <= 0.0:  # pragma: no cover — n_probes>=2 guarantees variance
        return float("nan")
    slopes = np.empty(n_opponents)
    for j in range(n_opponents):
        y = arr[:, j]
        y_mean = y.mean()
        sxy = float(((x - x_mean) * (y - y_mean)).sum())
        slopes[j] = sxy / sxx

    max_abs_slope = float(np.max(np.abs(slopes)))
    if max_abs_slope <= 0.0:
        return 0.0
    return float(slopes.mean() / max_abs_slope)
