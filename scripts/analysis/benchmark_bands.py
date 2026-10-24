#!/usr/bin/env python
r"""Build the benchmark's bands from the band cohorts, check the rule on them, and write both.

Every statistic is computed by :mod:`quantumnematode.benchmark.grading` from the vendored stores,
on band dates and band batches only; no held-out date or batch is read.

**Arms** (Scheer & Bargmann 2023, ``data/benchmark/scheer2023_tracks.npz``):

- the small lawn: the OD 2 wild type (PD1074), six statistics;
- the uniform lawn: PD1074's uniform-lawn arm, read with every sample on the lawn, three
  statistics, inflated to the small-lawn wild type's scale at the same n (two band dates);
- mutants against their same-date wild type, on paired band dates: tph-1 against N2, pdfr-1 and
  tax-4 against PD1074; Δ roaming fraction and Δ exits per on-lawn hour, inflated to √2 times the
  OD 2 wild type's scale where fewer than five paired band dates exist.

**Bands** are a cluster bootstrap over dates at each graded size (``default_rng(2030)``, 2,000
resamples). The centre is the statistic pooled over every band animal. The scale is
``kappa * sqrt(sd_n^2 + sd_centre^2)``: the SD of a population of n, and of the pooled centre over
the band dates at their full size. ``kappa`` is a per-statistic calibration factor: the SD of that
statistic's z over 200 half-splits of the OD 2 band dates (``default_rng(2033)``), floored at 1. It
scales the small lawn, the uniform lawn (before its floor) and the mutants' Δ (before theirs).

**Local search** (López-Cruz 2019, ``data/benchmark/lopezcruz2019_reorientations.json``): a batch is
one distinct date string among the ``wt`` rows, the pooled Fig. 1C row aside; where two rows share
one, the larger n is kept. pdfr-1's same-date control batch stays in the bands; the others are
split a third held out by date (``default_rng(2031 + k)``, the first draw whose held-out share of
animals lies in 0.28-0.38, the Scheer split's wild-type rule). The band's centre is the band
batches' mean; its scale their SD widened to a Student-t prediction interval for one new batch.
pdfr-1's Δ against its control has √2 times that scale.

**The rule's own checks** (``default_rng(2034)``, 200 half-splits, kappa fixed), gated before
anything is written:

- the small lawn: bands at n = 200 from half the band dates grade one 200-animal population drawn
  from the other half; joint pass rate over six statistics at the wild-type bar >= 0.90;
- local search: a band from half the band batches grades each batch of the other half alone;
  per-batch joint pass rate >= 0.90.

This is the registered retry. Attempt 1 (no centre term, no kappa, a plain SD for local search,
checks on ``default_rng(2032)``) failed both checks; its record is beside this one.

Usage::

    uv run python scripts/analysis/benchmark_bands.py --out data/benchmark/bands.json \
        --record docs/experiments/logbooks/supporting/109-benchmark-grading/bands_record.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from quantumnematode.benchmark import grading as g
from scipy import stats as scipy_stats
from quantumnematode.benchmark import references as refs
from quantumnematode.validation import roaming_dwelling as rd

DESCRIPTION = "Build the benchmark's bands from the band cohorts, and check the rule on them."
_REPO = Path(__file__).resolve().parents[2]
CLASSIFIER_PATH = _REPO / "data" / "benchmark" / "classifier.json"
LOPEZCRUZ_SPLIT_PATH = _REPO / "data" / "benchmark" / "lopezcruz2019_split.json"
PIROUETTES_PATH = _REPO / "data" / "chemotaxis" / "pierce_shimomura1999_pirouettes.json"
LOPEZCRUZ_REFERENCE_PATH = _REPO / "data" / "benchmark" / "lopezcruz2019_reference.json"

WILD_TYPE = "PD1074_od2_Data"
UNIFORM = "PD1074_od2_uniformlawn_Data"
DENSITY = {"OD 1": "PD1074_od1_Data", "OD 2": WILD_TYPE, "OD 4": "PD1074_od4_Data"}
MUTANTS = {
    "tph-1": ("MT15434_tph1mg280_od2_Data", "N2_od2_Data"),
    "pdfr-1": ("CX14295_pdfr1ok3425_od2_Data", WILD_TYPE),
    "tax-4": ("CX13078_tax4p678_od2_Data", WILD_TYPE),
}
# The roadmap's expected mutant directions, reported beside the references' own.
EXPECTED = {
    "tph-1": {"roaming_fraction": "more", "exits_per_on_lawn_hour": "more"},
    "pdfr-1": {
        "roaming_fraction": "less",
        "exits_per_on_lawn_hour": "normal",
        "gamma_per_min": "more search",
        "early_late_ratio": "more search",
    },
    "tax-4": {"roaming_fraction": "less", "exits_per_on_lawn_hour": "normal"},
}
PDFR1_ROWS = (91, 90)  # pdfr-1(ok3425) and its same-date wild type, López-Cruz 2019 Fig. 2A
THIN_DATES = 5
SIZES = (64, 200)
RESAMPLES = 2000
BAND_SEED = 2030
LC_SPLIT_SEED = 2031
LC_SPLIT_DRAWS = 10_000
LC_HELD_OUT_PROBABILITY = 1.0 / 3.0
LC_HELD_OUT_SHARE = (0.28, 0.38)
# Attempt 1's check seed; the registered retry measures kappa on one seed and re-reads the check on
# another, so the factor is never fitted to the splits that judge it.
ATTEMPT1_CHECK_SEED = 2032
KAPPA_SEED = 2033
CHECK_SEED = 2034
CHECK_SPLITS = 200
CHECK_SIZE = 200
CHECK_GATE = 0.90
WILD_TYPE_STATISTICS = sum(len(s) for s in g.FAMILIES.values())
FLAVELL_2013 = {
    "dwelling_tau_s": 482.0,
    "roaming_taus_s": [75.0, 520.0],
    "source": "Flavell et al. 2013, Cell 10.1016/j.cell.2013.08.001; published values, not "
    "simulated and not graded",
}


# ---------------------------------------------------------------------------------------------
# The Scheer arms
# ---------------------------------------------------------------------------------------------


def band_rows(tracks: refs.ScheerTracks, arm: str) -> np.ndarray:
    """Return an arm's rows on band dates only: the only rows this script reads."""
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


def lawn_bands(
    per_date: dict[int, np.ndarray],
    statistics: tuple[str, ...],
    n: int,
    rng: np.random.Generator,
    *,
    floor: dict[str, float] | None = None,
    kappa: dict[str, float] | None = None,
) -> dict[str, g.Band]:
    """Return a lawn arm's bands at n.

    The scale is ``kappa * sqrt(sd_n^2 + sd_centre^2)``: the bootstrap SD of a population of n, and
    of the pooled centre over the band dates at their full size. With ``floor``, a scale is at least
    the floor's.
    """
    everyone = np.vstack(list(per_date.values()))
    pooled = g.lawn_statistics(everyone.sum(axis=0))
    boot = g.cluster_bootstrap(list(per_date.values()), n, g.lawn_statistics, RESAMPLES, rng)
    centre = g.cluster_bootstrap(
        list(per_date.values()),
        len(everyone),
        g.lawn_statistics,
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


def mutant_bands(
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
        out[name] = g.Band(centre=m[name] - w[name], scale=scale, inflated=inflated)
    return out, dates


# ---------------------------------------------------------------------------------------------
# López-Cruz
# ---------------------------------------------------------------------------------------------


def lopezcruz_batches(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the wild-type batches by date string, the larger n kept where rows share one."""
    out: dict[str, dict[str, Any]] = {}
    for row in data["rows"]:
        if row["genotype"] != "wt" or row["dates"] == "many":
            continue
        kept = out.get(row["dates"])
        if kept is None or row["n"] > kept["n"]:
            out[row["dates"]] = row
    return dict(sorted(out.items()))


def search_statistics(row: dict[str, Any]) -> dict[str, float]:
    """Return a López-Cruz row's γ and early/late ratio; recording bin j starts 2j + 4 min after removal."""
    means = np.array([math.nan if v is None else float(v) for v in row["mean"]])
    starts = np.arange(len(means)) * g.SEARCH_BIN_MIN + 4.0
    return g.local_search_statistics(starts, means)


def split_batches(batches: dict[str, dict[str, Any]], kept: str) -> dict[str, Any]:
    """Split the batches other than ``kept`` a third held out, by the Scheer wild type's rule."""
    names = [b for b in batches if b != kept]
    sizes = np.array([batches[b]["n"] for b in names], dtype=float)
    for k in range(LC_SPLIT_DRAWS):
        held = np.random.default_rng(LC_SPLIT_SEED + k).random(len(names)) < LC_HELD_OUT_PROBABILITY
        share = float(sizes[held].sum() / sizes.sum())
        if LC_HELD_OUT_SHARE[0] <= share <= LC_HELD_OUT_SHARE[1]:
            sides = {b: (refs.HELD_OUT if h else refs.BANDS) for b, h in zip(names, held, strict=True)}
            sides[kept] = refs.BANDS
            return {
                "seed": LC_SPLIT_SEED,
                "draw": k,
                "held_out_probability": LC_HELD_OUT_PROBABILITY,
                "held_out_share_bounds": list(LC_HELD_OUT_SHARE),
                "held_out_share": share,
                "kept_in_bands": {kept: "pdfr-1's same-date control"},
                "batches": len(batches),
                "sides": dict(sorted(sides.items())),
                "drawn_from": "each batch's date string and n; no behavioural field",
            }
    msg = f"no draw of {LC_SPLIT_DRAWS} met the share bounds"
    raise RuntimeError(msg)


def prediction_factor(k: int) -> float:
    """Return the Student-t prediction interval's width for one new batch, in units of z*.

    With k batches it is ``sqrt(1 + 1/k) * t_{k-1}(1 - alpha/2m) / z*(m)`` for the wild type's m
    statistics, so the common bar grades a batch at the prediction interval's level.
    """
    q = 1.0 - g.FAMILY_ALPHA / (2 * WILD_TYPE_STATISTICS)
    return math.sqrt(1.0 + 1.0 / k) * float(scipy_stats.t.ppf(q, k - 1)) / g.z_star(
        WILD_TYPE_STATISTICS,
    )


def search_band(stats: list[dict[str, float]]) -> dict[str, g.Band]:
    """Return the band batches' mean, and their SD widened to a prediction interval."""
    out = {}
    for name in g.LOCAL_SEARCH_STATISTICS:
        values = np.array([s[name] for s in stats])
        values = values[np.isfinite(values)]
        scale = g.scale_of(values) * prediction_factor(len(values))
        out[name] = g.Band(centre=float(values.mean()), scale=scale)
    return out


# ---------------------------------------------------------------------------------------------
# The rule's own checks
# ---------------------------------------------------------------------------------------------


def _half_split(
    per_date: dict[int, np.ndarray],
    rng: np.random.Generator,
    kappa: dict[str, float] | None,
) -> tuple[dict[str, float], dict[str, g.Band]]:
    """Draw one half-split: bands at N from one half, and a population of N from the other."""
    dates = list(per_date)
    order = rng.permutation(len(dates))
    half = len(dates) // 2
    build = {dates[k]: per_date[dates[k]] for k in order[:half]}
    held = [per_date[dates[k]] for k in order[half:]]
    bands = lawn_bands(build, g.LAWN_STATISTICS, CHECK_SIZE, rng, kappa=kappa)
    population = g.cluster_bootstrap(held, CHECK_SIZE, g.lawn_statistics, 1, rng)
    return {name: float(v[0]) for name, v in population.items()}, bands


def measure_kappa(per_date: dict[int, np.ndarray]) -> dict[str, Any]:
    """Return each statistic's calibration factor: its z's SD over half-splits, floored at 1."""
    rng = np.random.default_rng(KAPPA_SEED)
    zs: dict[str, list[float]] = {name: [] for name in g.LAWN_STATISTICS}
    for _ in range(CHECK_SPLITS):
        values, bands = _half_split(per_date, rng, None)
        for name in g.LAWN_STATISTICS:
            zs[name].append(g.z_score(values[name], bands[name]))
    sd = {name: float(np.std(v, ddof=1)) for name, v in zs.items()}
    return {
        "seed": KAPPA_SEED,
        "splits": CHECK_SPLITS,
        "size": CHECK_SIZE,
        "z_sd": sd,
        "kappa": {name: max(1.0, s) for name, s in sd.items()},
    }


def check_small_lawn(
    per_date: dict[int, np.ndarray],
    bar: float,
    kappa: dict[str, float],
) -> dict[str, Any]:
    """Grade a 200-animal population from one half of the dates against bands from the other."""
    rng = np.random.default_rng(CHECK_SEED)
    passes, failures = 0, {name: 0 for name in g.LAWN_STATISTICS}
    for _ in range(CHECK_SPLITS):
        values, bands = _half_split(per_date, rng, kappa)
        readings = g.grade(values, bands, bar)
        passes += all(r.passes for r in readings.values())
        for name, r in readings.items():
            failures[name] += not r.passes
    rate = passes / CHECK_SPLITS
    return {
        "seed": CHECK_SEED,
        "splits": CHECK_SPLITS,
        "size": CHECK_SIZE,
        "bar": bar,
        "joint_pass_rate": rate,
        "failures_by_statistic": failures,
        "gate": CHECK_GATE,
        "passes": rate >= CHECK_GATE,
    }


def check_local_search(stats: list[dict[str, float]], bar: float) -> dict[str, Any]:
    """Grade each batch of one half of the band batches against a band from the other half."""
    rng = np.random.default_rng(CHECK_SEED)
    graded = passes = 0
    for _ in range(CHECK_SPLITS):
        order = rng.permutation(len(stats))
        half = len(stats) // 2
        band = search_band([stats[k] for k in order[:half]])
        for k in order[half:]:
            readings = g.grade(stats[k], band, bar)
            graded += 1
            passes += all(r.passes for r in readings.values())
    rate = passes / graded
    return {
        "seed": CHECK_SEED,
        "splits": CHECK_SPLITS,
        "bar": bar,
        "graded_batches": graded,
        "joint_pass_rate": rate,
        "gate": CHECK_GATE,
        "passes": rate >= CHECK_GATE,
    }


# ---------------------------------------------------------------------------------------------
# Reported beside the grading
# ---------------------------------------------------------------------------------------------


def window_agreement(tracks: refs.ScheerTracks) -> dict[str, float]:
    """Return how often the geometric rule agrees with the authors' in-lawn runs and flag."""
    rows = band_rows(tracks, WILD_TYPE)
    flag = tracks.flag("midbody_in_lawn")
    runs_same = runs_total = seconds_same = seconds_total = 0
    per_second = round(refs.SCHEER_FRAMES_PER_SECOND)
    for i in rows:
        lawn = g.real_on_lawn(tracks.x[i], tracks.y[i], tracks.boundary_radius[i])
        inputs = g.window_inputs(tracks.x[i], tracks.y[i], lawn)
        valid = inputs.valid
        runs = np.asarray(tracks.in_lawn_run[i][: len(valid)], dtype=bool)
        runs_same += int(np.sum(inputs.on[valid] == runs[valid]))
        runs_total += int(valid.sum())
        present = np.isfinite(tracks.x[i])
        in_flag = flag[i][::per_second][: len(lawn)]
        seconds_same += int(np.sum(lawn[present] == in_flag[present]))
        seconds_total += int(present.sum())
    return {
        "windows_agreeing_with_in_lawn_runs": runs_same / runs_total,
        "seconds_agreeing_with_midbody_in_lawn": seconds_same / seconds_total,
    }


def authors_roaming(tracks: refs.ScheerTracks, arm: str) -> float:
    """Return the authors' labels' on-lawn roaming fraction over an arm's band animals."""
    labels = tracks.labels[band_rows(tracks, arm)]
    return float(np.sum(labels == 1) / np.sum(labels >= 0))


def density_rows(
    tracks: refs.ScheerTracks,
    classifier: rd.GaussianClassifier,
    wild: dict[int, np.ndarray],
) -> dict[str, Any]:
    """Return the roaming fraction at each density on its band dates, and on the shared date."""
    out: dict[str, Any] = {}
    per = {
        label: wild if arm == WILD_TYPE else per_date_tallies(tracks, arm, classifier)
        for label, arm in DENSITY.items()
    }
    shared = sorted(set.intersection(*(set(p) for p in per.values())))
    for label, dates in per.items():
        everything = g.lawn_statistics(np.vstack(list(dates.values())).sum(axis=0))
        out[label] = {
            "band_dates": len(dates),
            "animals": int(sum(len(v) for v in dates.values())),
            "roaming_fraction": everything["roaming_fraction"],
            "roaming_fraction_on_shared_dates": g.lawn_statistics(
                np.vstack([dates[d] for d in shared]).sum(axis=0),
            )["roaming_fraction"]
            if shared
            else None,
        }
    out["shared_band_dates"] = shared
    return out


def _finite(value: Any) -> Any:  # noqa: ANN401 - any JSON-shaped value
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


# ---------------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------------


def build(tracks: refs.ScheerTracks, lopezcruz: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:  # noqa: PLR0915 - one pass, in the order the record reads
    """Return the bands file and the record beside it, or raise if a check fails."""
    classifier = rd.GaussianClassifier(hmm=rd.load_calibrated_hmm(CLASSIFIER_PATH))
    rng = np.random.default_rng(BAND_SEED)
    wt_bar = g.z_star(WILD_TYPE_STATISTICS)

    wild = per_date_tallies(tracks, WILD_TYPE, classifier)
    uniform = per_date_tallies(tracks, UNIFORM, classifier, uniform=True)
    controls = {arm: per_date_tallies(tracks, arm, classifier) for arm in {c for _, c in MUTANTS.values()} - {WILD_TYPE}}
    controls[WILD_TYPE] = wild
    mutant_tallies = {name: per_date_tallies(tracks, arm, classifier) for name, (arm, _) in MUTANTS.items()}

    batches = lopezcruz_batches(lopezcruz)
    rows = {r["row"]: r for r in lopezcruz["rows"]}
    kept = rows[PDFR1_ROWS[1]]["dates"]
    split = split_batches(batches, kept)
    band_batches = [b for b, side in split["sides"].items() if side == refs.BANDS]
    batch_stats = {b: search_statistics(batches[b]) for b in band_batches}
    search = search_band(list(batch_stats.values()))

    calibration = measure_kappa(wild)
    kappa = calibration["kappa"]
    checks = {
        "small_lawn": check_small_lawn(wild, wt_bar, kappa),
        "local_search": check_local_search(list(batch_stats.values()), wt_bar),
    }

    sizes: dict[str, Any] = {}
    mutant_dates: dict[str, list[int]] = {}
    for n in SIZES:
        small = lawn_bands(wild, g.LAWN_STATISTICS, n, rng, kappa=kappa)
        floor = {name: small[name].scale for name in g.UNIFORM_STATISTICS}
        thin = len(uniform) < THIN_DATES
        uni = lawn_bands(
            uniform,
            g.UNIFORM_STATISTICS,
            n,
            rng,
            floor=floor if thin else None,
            kappa=kappa,
        )
        mutants: dict[str, Any] = {}
        for name, (arm, control) in MUTANTS.items():
            mutant = mutant_tallies[name]
            wt_scale = {s: small[s].scale for s in g.MUTANT_LAWN_STATISTICS}
            bands, dates = mutant_bands(mutant, controls[control], n, rng, wt_scale, kappa)
            mutant_dates[name] = dates
            mutants[name] = bands
        pdfr1, control = (search_statistics(rows[r]) for r in PDFR1_ROWS)
        for s in g.LOCAL_SEARCH_STATISTICS:
            mutants["pdfr-1"][s] = g.Band(
                centre=pdfr1[s] - control[s],
                scale=math.sqrt(2) * search[s].scale,
                inflated=True,
            )
        sizes[str(n)] = {
            "small_lawn": {k: _band_dict(v) for k, v in small.items()},
            "uniform_lawn": {k: _band_dict(v) for k, v in uni.items()},
            "local_search": {k: _band_dict(v) for k, v in search.items()},
            "mutants": {
                m: {k: {**_band_dict(v), "direction": v.direction()} for k, v in bands.items()}
                for m, bands in mutants.items()
            },
        }

    bars = {
        "wild_type": wt_bar,
        **{m: g.z_star(len(sizes[str(SIZES[-1])]["mutants"][m])) for m in MUTANTS},
    }
    bands_file = {
        "description": "The benchmark's bands, from band cohorts and band batches only.",
        "classifier": str(CLASSIFIER_PATH.relative_to(_REPO)),
        "sizes": list(SIZES),
        "families": {f: list(s) for f, s in g.FAMILIES.items()},
        "fit_families": list(g.FIT_FAMILIES),
        "z_cap": g.Z_CAP,
        "bars": bars,
        "weights": {f"{f}/{s}": w for (f, s), w in g.weights(list(g.FAMILIES)).items()},
        "fit_weights": {f"{f}/{s}": w for (f, s), w in g.weights(list(g.FIT_FAMILIES)).items()},
        "arms": {
            "small_lawn": {"arm": WILD_TYPE, "band_dates": len(wild), "animals": int(sum(map(len, wild.values())))},
            "uniform_lawn": {"arm": UNIFORM, "band_dates": len(uniform), "animals": int(sum(map(len, uniform.values()))), "inflated_to": "small_lawn"},
            "local_search": {"source": "López-Cruz et al. 2019 wild-type batches", "band_batches": band_batches},
            "mutants": {
                m: {"arm": arm, "control": control, "paired_band_dates": mutant_dates[m]}
                for m, (arm, control) in MUTANTS.items()
            },
        },
        "bands": sizes,
        "w_minimum_rule": "Read in B.4: split the held-out OD 2 wild-type dates into halves by date "
        "1,000 times; for each split take z = (s_A - s_B) / scale, the band's scale at a half's "
        "size, and form D over the wild-type statistics; W's minimum is that distribution's 95th "
        "percentile.",
    }
    lc_reference = json.loads(LOPEZCRUZ_REFERENCE_PATH.read_text())
    pirouettes = json.loads(PIROUETTES_PATH.read_text())
    record = {
        "description": DESCRIPTION,
        "attempt": 2,
        "attempt1": "attempt1_rule_check.json, beside this record",
        "calibration": calibration,
        "local_search_prediction_factor": prediction_factor(len(band_batches)),
        "checks": checks,
        "lopezcruz_split": str(LOPEZCRUZ_SPLIT_PATH.relative_to(_REPO)),
        "lopezcruz_band_batches": batch_stats,
        "reported": {
            "geometric_rule": window_agreement(tracks),
            "authors_labels_roaming_fraction": {
                "small_lawn": authors_roaming(tracks, WILD_TYPE),
                "uniform_lawn": authors_roaming(tracks, UNIFORM),
            },
            "ours_roaming_fraction": {
                "small_lawn": sizes[str(SIZES[-1])]["small_lawn"]["roaming_fraction"]["centre"],
                "uniform_lawn": sizes[str(SIZES[-1])]["uniform_lawn"]["roaming_fraction"]["centre"],
            },
            "density": density_rows(tracks, classifier, wild),
            "flavell_2013": FLAVELL_2013,
            "pooled_curve_gamma": {
                "value": lc_reference["reference"]["gamma_per_min"],
                "note": "B.2's fit of the pooled Fig. 1C curve; the band's centre is the mean of the "
                "band batches' fits",
                "band_centre": search["gamma_per_min"].centre,
            },
            "quintile_ratio_reference": pirouettes["reference_ratio"],
            "expected_directions": {
                m: {
                    s: {
                        "expected": EXPECTED[m].get(s),
                        "reference_centre": sizes[str(SIZES[-1])]["mutants"][m][s]["centre"],
                        "reference_scale": sizes[str(SIZES[-1])]["mutants"][m][s]["scale"],
                        "resolved_direction": sizes[str(SIZES[-1])]["mutants"][m][s]["direction"],
                    }
                    for s in sizes[str(SIZES[-1])]["mutants"][m]
                }
                for m in MUTANTS
            },
            "limits": [
                "Real dwelling speeds carry tracking noise of a few µm per sample; simulated ones "
                "carry none.",
                "The bout means are total time over completed bouts, the exponential estimator; "
                "roaming bouts are double-exponential.",
                "López-Cruz's band is the spread of single batches (n about 55), conservative for "
                "simulated populations of 200.",
                "The uniform lawn has two band dates and cannot be halved, so the rule's check "
                "leaves it out.",
            ],
        },
    }
    return _finite(bands_file), _finite({**record, "split": split})


def main() -> int:
    """CLI: build, check, and write the bands, the López-Cruz split and the record."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    bands, record = build(refs.load_scheer_tracks(), refs.load_lopezcruz())
    print(json.dumps(record["checks"], indent=2))
    failed = [name for name, check in record["checks"].items() if not check["passes"]]
    args.record.parent.mkdir(parents=True, exist_ok=True)
    if failed:
        args.record.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        print(f"The rule's own check failed for {failed}; the bands were not written.", file=sys.stderr)
        return 1
    split = record.pop("split")
    LOPEZCRUZ_SPLIT_PATH.write_text(json.dumps(split, indent=2, allow_nan=False) + "\n")
    args.out.write_text(json.dumps(bands, indent=2, allow_nan=False) + "\n")
    args.record.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
