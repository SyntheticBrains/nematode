"""Build the benchmark's bands at any size from the real tracks' band cohorts.

Each band is a centre and a scale. The centre is the band cohorts' pooled statistic. The scale is
calibrated from the spread between recording dates, by cluster bootstrap over dates, at the
population's own size. Each band draws from its own generator, seeded by the band and its size.
Only band-cohort rows are read: the held-out dates are never touched.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats as scipy_stats

from wormbench import classifier as rd
from wormbench import data
from wormbench import grading as g
from wormbench import references as refs

if TYPE_CHECKING:
    from collections.abc import Callable


WILD_TYPE = "PD1074_od2_Data"


UNIFORM = "PD1074_od2_uniformlawn_Data"


MUTANTS = {
    "tph-1": ("MT15434_tph1mg280_od2_Data", "N2_od2_Data"),
    "pdfr-1": ("CX14295_pdfr1ok3425_od2_Data", WILD_TYPE),
    "tax-4": ("CX13078_tax4p678_od2_Data", WILD_TYPE),
}


THIN_DATES = 5


# Each band draws from its own generator, seeded by the band and its size, so a band at n does not
# depend on which bands were built before it.
BAND_INDEX = {"small_lawn": 0, "uniform_lawn": 1, "tph-1": 2, "pdfr-1": 3, "tax-4": 4}


DIRECTION_ALPHA = 0.05


RESAMPLES = 2000


BAND_SEED = 2030


def band_rows(tracks: refs.ScheerTracks, arm: str) -> np.ndarray:
    """Return an arm's rows on band dates only: the only rows the builder reads."""
    rows = tracks.arm(arm)
    return rows[tracks.side[rows] == refs.BANDS]


def per_date_tallies(
    tracks: refs.ScheerTracks,
    arm: str,
    classifier: rd.GaussianClassifier,
    *,
    uniform: bool = False,
) -> dict[int, np.ndarray]:
    """Return each band date's animals' tallies, ``(animals, fields)``."""
    out: dict[int, list[np.ndarray]] = {}
    for i in band_rows(tracks, arm):
        lawn = g.real_on_lawn(
            tracks.x[i],
            tracks.y[i],
            tracks.boundary_radius[i],
            uniform=uniform,
        )
        windows = g.animal_windows(tracks.x[i], tracks.y[i], lawn, classifier)
        out.setdefault(int(tracks.date[i]), []).append(g.tally(windows))
    return {d: np.vstack(v) for d, v in sorted(out.items())}


def _band_dict(band: g.Band) -> dict[str, Any]:
    return {"centre": band.centre, "scale": band.scale, "inflated": band.inflated}


def lawn_bands(  # noqa: PLR0913 - the arm, its statistics, the size, the stream, two widenings
    per_date: dict[int, np.ndarray],
    statistics: tuple[str, ...],
    n: int,
    rng: np.random.Generator,
    *,
    floor: dict[str, float] | None = None,
    kappa: dict[str, float] | None = None,
    statistic: Callable[[np.ndarray], dict[str, float]] = g.lawn_statistics,
) -> dict[str, g.Band]:
    """Return a lawn arm's bands at n, by ``statistic`` (the lawn statistics unless given).

    The scale is ``kappa * sqrt(sd_n^2 + sd_centre^2)``: the bootstrap SD of a population of n, and
    of the pooled centre over the band dates at their full size. With ``floor``, a scale is at least
    the floor's.
    """
    everyone = np.vstack(list(per_date.values()))
    pooled = statistic(everyone.sum(axis=0))
    boot = g.cluster_bootstrap(list(per_date.values()), n, statistic, RESAMPLES, rng)
    centre = g.cluster_bootstrap(
        list(per_date.values()),
        len(everyone),
        statistic,
        RESAMPLES,
        rng,
    )
    out = {}
    for name in statistics:
        factor = 1.0 if kappa is None else kappa[name]
        scale = factor * math.hypot(g.scale_of(boot[name]), g.scale_of(centre[name]))
        inflated = False
        if floor is not None and not scale >= floor[name]:
            scale, inflated = floor[name], True
        out[name] = g.Band(centre=pooled[name], scale=scale, inflated=inflated)
    return out


def mutant_bands(  # noqa: PLR0913 - both arms, the size, the stream and two widenings
    mutant: dict[int, np.ndarray],
    control: dict[int, np.ndarray],
    n: int,
    rng: np.random.Generator,
    wild_type_scale: dict[str, float],
    kappa: dict[str, float],
) -> tuple[dict[str, g.Band], list[int]]:
    """Return a mutant's Δ bands on its paired band dates at n, and those dates.

    The scale is the wild type's ``kappa`` times the root sum of squares of the paired bootstrap's
    SD at n and at the mutant's full paired size; a thin arm's is at least √2 times the wild type's.
    """
    dates = sorted(set(mutant) & set(control))
    pairs = [(mutant[d], control[d]) for d in dates]
    mutants = np.vstack([p[0] for p in pairs])
    m = g.lawn_statistics(mutants.sum(axis=0))
    w = g.lawn_statistics(np.vstack([p[1] for p in pairs]).sum(axis=0))
    boot = g.paired_bootstrap(pairs, n, g.lawn_statistics, RESAMPLES, rng)
    centre = g.paired_bootstrap(pairs, len(mutants), g.lawn_statistics, RESAMPLES, rng)
    out = {}
    for name in g.MUTANT_LAWN_STATISTICS:
        scale = kappa[name] * math.hypot(g.scale_of(boot[name]), g.scale_of(centre[name]))
        inflated = False
        if len(dates) < THIN_DATES and not scale >= math.sqrt(2) * wild_type_scale[name]:
            scale, inflated = math.sqrt(2) * wild_type_scale[name], True
        out[name] = g.Band(
            centre=m[name] - w[name],
            scale=scale,
            inflated=inflated,
            direction=paired_direction(pairs, name)["direction"],
        )
    return out, dates


def paired_direction(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    statistic: str,
) -> dict[str, Any]:
    """Return a two-sided paired t-test of a mutant's per-date difference from its wild type.

    Each paired date gives one difference, the mutant's pooled statistic minus the wild type's; a
    date where either is undefined is left out. The direction is the mean difference's sign where
    |t| exceeds ``t_{k-1}(1 - alpha/2)`` over k dates, and 0 otherwise or with fewer than two.
    """
    diffs = []
    for mutant, wild in pairs:
        d = (
            g.lawn_statistics(mutant.sum(axis=0))[statistic]
            - g.lawn_statistics(
                wild.sum(axis=0),
            )[statistic]
        )
        if math.isfinite(d):
            diffs.append(d)
    k = len(diffs)
    if k < 2:  # noqa: PLR2004 - a t-test needs two dates
        return {"dates": k, "mean": math.nan, "t": math.nan, "critical": math.nan, "direction": 0}
    values = np.asarray(diffs)
    se = float(values.std(ddof=1)) / math.sqrt(k)
    t_value = float(values.mean()) / se if se > 0 else math.inf * np.sign(values.mean())
    critical = float(scipy_stats.t.ppf(1.0 - DIRECTION_ALPHA / 2, k - 1))
    resolved = abs(t_value) > critical
    return {
        "dates": k,
        "mean": float(values.mean()),
        "t": t_value,
        "critical": critical,
        "direction": int(np.sign(values.mean())) if resolved else 0,
    }


def band_rng(band: str, n: int) -> np.random.Generator:
    """Return the generator one band at size n draws from."""
    return np.random.default_rng([BAND_SEED, BAND_INDEX[band], n])


class BandBuilder:
    """Build any family's or mutant's bands at any size, each on its own generator.

    The store's per-date tallies are classified once and cached. The uniform well's floor and a
    thin mutant's inflation read the small lawn's scale at the same size, so the small lawn is built
    first; it is cached per size too.
    """

    def __init__(
        self,
        tracks: refs.ScheerTracks | None = None,
        kappa: dict[str, float] | None = None,
    ) -> None:
        self._tracks = tracks
        self._kappa = kappa
        self._classifier = rd.GaussianClassifier(
            hmm=rd.load_calibrated_hmm(data.path("benchmark", "classifier.json")),
        )
        self._tallies: dict[tuple[str, bool], dict[int, np.ndarray]] = {}
        self._small: dict[int, dict[str, g.Band]] = {}

    @property
    def tracks(self) -> refs.ScheerTracks:
        """The Scheer store, loaded on first use."""
        if self._tracks is None:
            self._tracks = refs.load_scheer_tracks()
        return self._tracks

    @property
    def kappa(self) -> dict[str, float]:
        """The calibration factors, from v1's record shipped with the package, unless given."""
        if self._kappa is None:
            self._kappa = data.package_record("kappa_v1.json")["kappa"]
        return self._kappa

    def tallies(self, arm: str, *, uniform: bool = False) -> dict[int, np.ndarray]:
        """Return an arm's per-band-date tallies, classified once."""
        key = (arm, uniform)
        if key not in self._tallies:
            self._tallies[key] = per_date_tallies(
                self.tracks,
                arm,
                self._classifier,
                uniform=uniform,
            )
        return self._tallies[key]

    def small_lawn(self, n: int) -> dict[str, g.Band]:
        """Return the small lawn's bands at n."""
        if n not in self._small:
            self._small[n] = lawn_bands(
                self.tallies(WILD_TYPE),
                g.LAWN_STATISTICS,
                n,
                band_rng("small_lawn", n),
                kappa=self.kappa,
            )
        return self._small[n]

    def uniform_lawn(self, n: int) -> dict[str, g.Band]:
        """Return the uniform well's bands at n, floored at the small lawn's where it is thin."""
        uniform = self.tallies(UNIFORM, uniform=True)
        small = self.small_lawn(n)
        floor = {name: small[name].scale for name in g.UNIFORM_STATISTICS}
        return lawn_bands(
            uniform,
            g.UNIFORM_STATISTICS,
            n,
            band_rng("uniform_lawn", n),
            floor=floor if len(uniform) < THIN_DATES else None,
            kappa=self.kappa,
        )

    def mutant(self, name: str, n: int) -> tuple[dict[str, g.Band], list[int]]:
        """Return a mutant's Δ bands at n and its paired band dates."""
        arm, control = MUTANTS[name]
        small = self.small_lawn(n)
        return mutant_bands(
            self.tallies(arm),
            self.tallies(control),
            n,
            band_rng(name, n),
            {s: small[s].scale for s in g.MUTANT_LAWN_STATISTICS},
            self.kappa,
        )

    def directions(self, name: str) -> dict[str, dict[str, Any]]:
        """Return each mutant statistic's paired t-test over its paired band dates."""
        arm, control = MUTANTS[name]
        mutant, wild = self.tallies(arm), self.tallies(control)
        pairs = [(mutant[d], wild[d]) for d in sorted(set(mutant) & set(wild))]
        return {s: paired_direction(pairs, s) for s in g.MUTANT_LAWN_STATISTICS}


KINEMATICS_INDEX = 5  # the kinematics family's band generator, after v1's


def per_date_kinematic_tallies(
    tracks: refs.ScheerTracks,
    arm: str,
    classifier: rd.GaussianClassifier,
    *,
    uniform: bool = False,
) -> dict[int, np.ndarray]:
    """Return each band date's animals' kinematic tallies, ``(animals, fields)``."""
    out: dict[int, list[np.ndarray]] = {}
    for i in band_rows(tracks, arm):
        lawn = g.real_on_lawn(tracks.x[i], tracks.y[i], tracks.boundary_radius[i], uniform=uniform)
        windows = g.animal_windows(tracks.x[i], tracks.y[i], lawn, classifier)
        out.setdefault(int(tracks.date[i]), []).append(
            g.kinematic_tally(windows, tracks.x[i], tracks.y[i], lawn),
        )
    return {d: np.vstack(v) for d, v in sorted(out.items())}


ACTIVITY_INDEX = 6  # the activity family's band generator, after kinematics


def per_date_activity_tallies(
    tracks: refs.ScheerTracks,
    arm: str,
    rows: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Return each date's animals' activity tallies on the small lawn, ``(animals, fields)``.

    By default over the arm's band rows; ``rows`` names other animals, as a control's units do.
    """
    out: dict[int, list[np.ndarray]] = {}
    for i in band_rows(tracks, arm) if rows is None else rows:
        lawn = g.real_on_lawn(tracks.x[i], tracks.y[i], tracks.boundary_radius[i])
        out.setdefault(int(tracks.date[i]), []).append(
            g.activity_tally(tracks.x[i], tracks.y[i], lawn),
        )
    return {d: np.vstack(v) for d, v in sorted(out.items())}
