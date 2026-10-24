"""Grade a worm's lawn and search statistics against real worms' between-date spread.

Real and simulated tracks are read through the same functions:

1. **Windows.** Positions sampled each second are read at the roaming/dwelling classifier's 5-s
   step and cut into 10-s windows (:func:`animal_windows`). A window is on the lawn when every 1-s
   sample in its span is, off the lawn when none is, and unclassified when any sample is missing.
   Its state comes from the classifier, decoded within each run of on-lawn windows.
2. **Tallies.** Each animal's windows reduce to a vector of sums (:data:`TALLY_FIELDS`), so a
   population's statistics are ratios of its animals' summed tallies and a resample is a sum.
3. **Statistics** (:func:`lawn_statistics`, :func:`local_search_statistics`): the on-lawn roaming
   fraction; censoring-aware bout means; sustained exits per on-lawn hour; roaming enrichment
   30-180 s before exits; time on the lawn; and the local-search decay and early/late ratio, both
   free of the reorientation detector's level.
4. **Bands.** A statistic's band at a population of n animals comes from a cluster bootstrap over
   recording dates (:func:`cluster_bootstrap`): its centre is the statistic pooled over every
   reference animal, its scale the bootstrap's SD, holding both the date-to-date spread and the
   sampling noise of n animals.
5. **The rule.** ``z = (s - centre) / scale``, capped. A family of statistics passes when every
   ``|z|`` is within a Bonferroni bar for a 95% family-wise rate (:func:`z_star`). A mutant's
   difference from its own wild type must also take the reference's sign wherever the reference
   resolves one. The distance is the weighted RMS of the capped z's (:func:`distance`).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np

from quantumnematode.benchmark.readouts import (
    OFF_FOOD_DETECTOR,
    fit_local_search,
    local_search_curve,
    reversal_onsets,
)
from quantumnematode.benchmark.references import boundary_distance
from quantumnematode.validation import roaming_dwelling as rd

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from quantumnematode.report.dtypes import BehaviourStep

SAMPLE_S = 1.0
WINDOW_S = rd.WINDOW_SECONDS
SAMPLES_PER_WINDOW = round(WINDOW_S / SAMPLE_S)
# A 40-min recording sampled each second: the real tracks' length and the lawn assays' window.
LAWN_SAMPLES = 2400
Z_CAP = 10.0
FAMILY_ALPHA = 0.05
# The direction a reference resolves: its centre's sign, where centre +/- 1.96 scale excludes zero.
DIRECTION_Z = NormalDist().inv_cdf(0.975)

TALLY_FIELDS = (
    "animals",
    "windows",
    "on_windows",
    "classified",
    "roaming",
    "roaming_time_s",
    "roaming_completed",
    "dwelling_time_s",
    "dwelling_completed",
    "exits",
    "band_roaming",
    "band_windows",
)
_F = {name: k for k, name in enumerate(TALLY_FIELDS)}

LAWN_STATISTICS = (
    "roaming_fraction",
    "roaming_bout_s",
    "dwelling_bout_s",
    "exits_per_on_lawn_hour",
    "pre_exit_enrichment",
    "time_on_lawn",
)
UNIFORM_STATISTICS = ("roaming_fraction", "roaming_bout_s", "dwelling_bout_s")
LOCAL_SEARCH_STATISTICS = ("gamma_per_min", "early_late_ratio")
FAMILIES: dict[str, tuple[str, ...]] = {
    "small_lawn": LAWN_STATISTICS,
    "uniform_lawn": UNIFORM_STATISTICS,
    "local_search": LOCAL_SEARCH_STATISTICS,
}
# Distance weights: each assay weighs a third, split equally among its statistics. The fitted
# distance leaves local search out and weighs the other two a half each.
FIT_FAMILIES = ("small_lawn", "uniform_lawn")
MUTANT_LAWN_STATISTICS = ("roaming_fraction", "exits_per_on_lawn_hour")

# The off-food phase read for local search: 45 min from removal, sampled each second.
SEARCH_SAMPLES = 2700
# Local search, in minutes after removal from food: 2-min bins from 8 to 44 min, fitted from the
# wild type's peak bin with t0 at that bin's centre; early is 8-14 min, late 34-44 min.
SEARCH_BIN_MIN = 2.0
SEARCH_FIRST_BIN_MIN = 8.0
SEARCH_LAST_BIN_MIN = 42.0
SEARCH_T0_MIN = 9.0
EARLY_BINS_MIN = (8.0, 10.0, 12.0)
LATE_BINS_MIN = (34.0, 36.0, 38.0, 40.0, 42.0)


def z_star(statistics: int, alpha: float = FAMILY_ALPHA) -> float:
    """Return the two-sided Bonferroni bar on ``|z|`` for a family of ``statistics`` tests."""
    return NormalDist().inv_cdf(1.0 - alpha / (2 * statistics))


# ---------------------------------------------------------------------------------------------
# Windows and tallies
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AnimalWindows:
    """One animal's 10-s windows: state, wholly on or off the lawn, and free of missing samples."""

    states: np.ndarray
    on: np.ndarray
    off: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class WindowInputs:
    """One animal's 10-s window measures and lawn masks, before classification."""

    speed: np.ndarray
    angular: np.ndarray
    on: np.ndarray
    off: np.ndarray
    valid: np.ndarray


def window_inputs(x: np.ndarray, y: np.ndarray, on_lawn: np.ndarray) -> WindowInputs:
    """Measure 10-s windows from positions sampled each second, with their lawn masks.

    ``on_lawn`` marks each sample on the lawn. Window ``k`` spans samples ``10k`` to ``10k + 10``,
    the samples its two 5-s steps read; it is on the lawn when every one of them is, off when none
    is, and neither, unclassified, when any is missing.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    lawn = np.asarray(on_lawn, dtype=bool)
    speed, angular = rd.window_measures(*rd.downsample_to_reference(x, y, SAMPLE_S))
    n = len(speed)
    present = np.isfinite(x) & np.isfinite(y)
    on = np.zeros(n, dtype=bool)
    off = np.zeros(n, dtype=bool)
    valid = np.zeros(n, dtype=bool)
    for k in range(n):
        span = slice(SAMPLES_PER_WINDOW * k, SAMPLES_PER_WINDOW * (k + 1) + 1)
        if not present[span].all():
            continue
        valid[k] = True
        on[k] = bool(lawn[span].all())
        off[k] = not lawn[span].any()
    on &= np.isfinite(speed)
    return WindowInputs(speed=speed, angular=angular, on=on, off=off, valid=valid)


def animal_windows(
    x: np.ndarray,
    y: np.ndarray,
    on_lawn: np.ndarray,
    classifier: rd.GaussianClassifier,
) -> AnimalWindows:
    """Cut positions sampled each second into classified 10-s windows, as :func:`window_inputs`."""
    w = window_inputs(x, y, on_lawn)
    states = classifier.states(w.speed, w.angular, w.on)
    return AnimalWindows(states=states, on=w.on, off=w.off, valid=w.valid)


def _bouts(states: np.ndarray, state: int) -> tuple[float, int]:
    """Return a state's total time and its completed bouts, within runs of classified windows.

    A bout is a maximal run of one state inside one run of classified windows; it is completed
    when the other state follows it in the same run, and censored when the run ends.
    """
    time_s, completed = 0.0, 0
    for start, stop in rd.on_food_runs(states != rd.OFF_FOOD):
        run = states[start:stop]
        edges = np.flatnonzero(np.diff(run)) + 1
        bounds = np.concatenate(([0], edges, [len(run)]))
        for a, b in itertools.pairwise(bounds):
            if run[a] != state:
                continue
            time_s += (b - a) * WINDOW_S
            completed += int(b < len(run))
    return time_s, completed


def tally(windows: AnimalWindows) -> np.ndarray:
    """Return one animal's tally: the sums every lawn statistic is a ratio of."""
    states = windows.states
    classified = states != rd.OFF_FOOD
    exits = rd.lawn_exits(windows.on, windows.off)
    band = rd.pre_exit_band(states, exits)
    roaming_time, roaming_done = _bouts(states, rd.ROAMING)
    dwelling_time, dwelling_done = _bouts(states, rd.DWELLING)
    out = np.zeros(len(TALLY_FIELDS))
    out[_F["animals"]] = 1
    out[_F["windows"]] = windows.valid.sum()
    out[_F["on_windows"]] = windows.on.sum()
    out[_F["classified"]] = classified.sum()
    out[_F["roaming"]] = np.sum(states == rd.ROAMING)
    out[_F["roaming_time_s"]] = roaming_time
    out[_F["roaming_completed"]] = roaming_done
    out[_F["dwelling_time_s"]] = dwelling_time
    out[_F["dwelling_completed"]] = dwelling_done
    out[_F["exits"]] = len(exits)
    out[_F["band_roaming"]] = np.sum(band == rd.ROAMING)
    out[_F["band_windows"]] = len(band)
    return out


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else math.nan


def lawn_statistics(total: np.ndarray) -> dict[str, float]:
    """Return the lawn statistics of a population from its summed tally; NaN where undefined."""
    t = {name: float(total[k]) for name, k in _F.items()}
    roaming = _ratio(t["roaming"], t["classified"])
    band = _ratio(t["band_roaming"], t["band_windows"])
    return {
        "roaming_fraction": roaming,
        "roaming_bout_s": _ratio(t["roaming_time_s"], t["roaming_completed"]),
        "dwelling_bout_s": _ratio(t["dwelling_time_s"], t["dwelling_completed"]),
        "exits_per_on_lawn_hour": _ratio(t["exits"], t["on_windows"] * WINDOW_S / 3600.0),
        "pre_exit_enrichment": band - roaming,
        "time_on_lawn": _ratio(t["on_windows"], t["windows"]),
    }


# ---------------------------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------------------------


def real_on_lawn(
    x: np.ndarray,
    y: np.ndarray,
    radii: np.ndarray,
    *,
    uniform: bool = False,
) -> np.ndarray:
    """Return which of a real worm's samples are on the lawn.

    On a small lawn, a sample is on it when the midbody lies inside the lawn's boundary. A uniform
    well has no boundary, and its bacteria cover it, so every sample there is on the lawn.
    """
    if uniform:
        return np.ones(len(x), dtype=bool)
    with np.errstate(invalid="ignore"):
        return boundary_distance(x, y, radii) > 0


def analysis_steps(
    steps: Sequence[BehaviourStep],
    analysis: tuple[int, int],
    samples: int,
) -> list[BehaviourStep]:
    """Return the first ``samples`` steps of an assay's analysis window.

    Raises
    ------
    ValueError
        If the window is shorter than ``samples``: the reference's length cannot be read from it.
    """
    first, stop = analysis
    if stop - first < samples or len(steps) < first + samples:
        msg = (
            f"the analysis window holds {min(stop, len(steps)) - first} steps, shorter than the "
            f"reference's {samples}"
        )
        raise ValueError(msg)
    return list(steps[first : first + samples])


def simulated_windows(
    steps: Sequence[BehaviourStep],
    classifier: rd.GaussianClassifier,
    *,
    uniform: bool = False,
) -> AnimalWindows:
    """Classify a simulated worm's 1-s steps; on a uniform well every sample is on the lawn."""
    x = np.array([s.x for s in steps], dtype=float)
    y = np.array([s.y for s in steps], dtype=float)
    lawn = (
        np.ones(len(steps), dtype=bool) if uniform else np.array([bool(s.on_lawn) for s in steps])
    )
    return animal_windows(x, y, lawn, classifier)


# ---------------------------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------------------------


def local_search_statistics(bin_starts_min: np.ndarray, means: np.ndarray) -> dict[str, float]:
    """Return the decay and early/late ratio of mean onsets per 2-min bin after removal.

    ``bin_starts_min`` are each bin's start in minutes after removal. Every bin from 8 to 42 min
    must be present and finite; otherwise both statistics are NaN.
    """
    starts = np.asarray(bin_starts_min, dtype=float)
    values = np.asarray(means, dtype=float)
    wanted = np.arange(
        SEARCH_FIRST_BIN_MIN,
        SEARCH_LAST_BIN_MIN + SEARCH_BIN_MIN / 2,
        SEARCH_BIN_MIN,
    )
    by_start = {round(float(s), 6): float(v) for s, v in zip(starts, values, strict=True)}
    picked = np.array([by_start.get(round(float(s), 6), math.nan) for s in wanted])
    if not np.all(np.isfinite(picked)):
        return {"gamma_per_min": math.nan, "early_late_ratio": math.nan}
    fit = fit_local_search(wanted + SEARCH_BIN_MIN / 2, picked, SEARCH_T0_MIN)
    early = np.mean([by_start[round(s, 6)] for s in EARLY_BINS_MIN])
    late = np.mean([by_start[round(s, 6)] for s in LATE_BINS_MIN])
    return {"gamma_per_min": float(fit.gamma), "early_late_ratio": _ratio(early, late)}


def simulated_search(populations: Sequence[Sequence[BehaviourStep]]) -> dict[str, float]:
    """Return a simulated population's local-search statistics, from each worm's positions alone.

    Each worm's steps start at removal from food. Reversal-initiated reorientations are found by
    the off-food detector, counted per 2-min bin over animals, and fitted as the real curve is.
    """
    onsets, valid = [], []
    for steps in populations:
        x = np.array([s.x for s in steps], dtype=float)
        y = np.array([s.y for s in steps], dtype=float)
        onsets.append(reversal_onsets(x, y, OFF_FOOD_DETECTOR))
        valid.append(np.isfinite(np.diff(x)) & np.isfinite(np.diff(y)))
    curve = local_search_curve(onsets, valid, bin_s=120, start_s=0)
    return local_search_statistics(curve.starts_s / 60.0, curve.mean)


# ---------------------------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Band:
    """A statistic's band: centre, scale, and whether the scale was inflated for a thin arm."""

    centre: float
    scale: float
    inflated: bool = False

    def direction(self) -> int:
        """Return the sign the reference resolves, or 0 where centre +/- 1.96 scale holds zero."""
        if not (math.isfinite(self.centre) and math.isfinite(self.scale)):
            return 0
        if abs(self.centre) > DIRECTION_Z * self.scale:
            return 1 if self.centre > 0 else -1
        return 0


def _draw(
    sizes: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Draw dates with replacement until ``n`` animals; return (date, animals taken) pairs."""
    drawn: list[tuple[int, int]] = []
    count = 0
    while count < n:
        d = int(rng.integers(len(sizes)))
        take = min(int(sizes[d]), n - count)
        drawn.append((d, take))
        count += take
    return drawn


def _subset(tallies: np.ndarray, take: int, rng: np.random.Generator) -> np.ndarray:
    if take >= len(tallies):
        return tallies.sum(axis=0)
    return tallies[rng.choice(len(tallies), size=take, replace=False)].sum(axis=0)


def cluster_bootstrap(
    per_date: Sequence[np.ndarray],
    n: int,
    statistic: Callable[[np.ndarray], Mapping[str, float]],
    resamples: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Resample populations of ``n`` animals by date; return each statistic's resampled values.

    ``per_date`` holds each date's animals' tallies, ``(animals, fields)``. A resample draws dates
    with replacement, taking all of each drawn date's animals until ``n`` is reached and a random
    subset of the last.
    """
    sizes = np.array([len(t) for t in per_date])
    out: dict[str, list[float]] = {}
    for _ in range(resamples):
        total = sum(
            (_subset(per_date[d], take, rng) for d, take in _draw(sizes, n, rng)),
            start=np.zeros(len(TALLY_FIELDS)),
        )
        for name, value in statistic(total).items():
            out.setdefault(name, []).append(value)
    return {name: np.asarray(values) for name, values in out.items()}


def paired_bootstrap(
    pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    n: int,
    statistic: Callable[[np.ndarray], Mapping[str, float]],
    resamples: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Resample a mutant's difference from its same-date wild type, by paired date.

    ``pairs`` holds each paired date's (mutant tallies, wild-type tallies). A resample draws dates
    until ``n`` mutant animals are taken, subsampling the last date's mutants, and takes every
    wild-type animal of each drawn date.
    """
    sizes = np.array([len(m) for m, _ in pairs])
    out: dict[str, list[float]] = {}
    for _ in range(resamples):
        mutant = np.zeros(len(TALLY_FIELDS))
        wild = np.zeros(len(TALLY_FIELDS))
        for d, take in _draw(sizes, n, rng):
            mutant += _subset(pairs[d][0], take, rng)
            wild += pairs[d][1].sum(axis=0)
        m, w = statistic(mutant), statistic(wild)
        for name in m:
            out.setdefault(name, []).append(m[name] - w[name])
    return {name: np.asarray(values) for name, values in out.items()}


def scale_of(values: np.ndarray) -> float:
    """Return a bootstrap's SD over its finite values; NaN if fewer than two."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else math.nan


# ---------------------------------------------------------------------------------------------
# The rule and the distance
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Reading:
    """One graded statistic: its value, band, capped z, and whether it passes."""

    value: float
    band: Band
    z: float
    passes: bool


def z_score(value: float, band: Band) -> float:
    """Return ``(value - centre) / scale`` capped at ``Z_CAP``; an undefined value is at the cap."""
    if not (math.isfinite(value) and math.isfinite(band.centre) and band.scale > 0):
        return Z_CAP
    z = (value - band.centre) / band.scale
    return float(max(-Z_CAP, min(Z_CAP, z)))


def grade(
    values: Mapping[str, float],
    bands: Mapping[str, Band],
    bar: float,
    *,
    directions: bool = False,
) -> dict[str, Reading]:
    """Grade each statistic in ``bands``; with ``directions``, a resolved sign must also match.

    An undefined value fails, entering at the cap.
    """
    out = {}
    for name, band in bands.items():
        value = float(values.get(name, math.nan))
        z = z_score(value, band)
        passes = math.isfinite(value) and abs(z) <= bar
        sign = band.direction()
        if directions and sign != 0 and math.isfinite(value):
            passes = passes and np.sign(value) == sign
        out[name] = Reading(value=value, band=band, z=z, passes=bool(passes))
    return out


def weights(families: Sequence[str]) -> dict[tuple[str, str], float]:
    """Return each (family, statistic)'s weight: families equal, split equally within each."""
    return {
        (family, statistic): 1.0 / (len(families) * len(FAMILIES[family]))
        for family in families
        for statistic in FAMILIES[family]
    }


def distance(readings: Mapping[str, Mapping[str, Reading]], families: Sequence[str]) -> float:
    """Return the weighted RMS of the capped z's over ``families``."""
    w = weights(families)
    total = sum(w.values())
    return math.sqrt(sum(wt * readings[f][s].z ** 2 for (f, s), wt in w.items()) / total)
