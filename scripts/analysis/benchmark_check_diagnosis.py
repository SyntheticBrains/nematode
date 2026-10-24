#!/usr/bin/env python
r"""Diagnose attempt 1's failed rule checks, on band data only, as description.

Attempt 1 of the benchmark's bands failed both of the rule's own checks (the small lawn's joint
pass rate 0.80, local search's 0.86, gates 0.90). This script reads why, on the band cohorts and
band batches alone, under attempt 1's rule:

- **the small lawn**: over 200 half-splits of the OD 2 band dates (``default_rng(2032)``), the SD of
  each statistic's z when a 200-animal population from one half is graded against bands from the
  other (1.0 if the scale were right), and the joint pass rate with the half-band centre's own
  bootstrap SD added to the scale;
- **local search**: over 200 half-splits of the band batches, the per-batch joint pass rate under a
  Student-t prediction interval in place of the plain SD.

It changes nothing; its numbers are in
``docs/experiments/logbooks/supporting/109-benchmark-grading/attempt1_rule_check.json``.

Usage::

    uv run python scripts/analysis/benchmark_check_diagnosis.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from quantumnematode.benchmark import grading as g
from quantumnematode.benchmark import references as refs
from quantumnematode.validation import roaming_dwelling as rd
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_bands as bb

SEED = 2032
SPLITS = 200
SIZE = 200
CENTRE_RESAMPLES = 400


def _attempt1_lawn_bands(
    per_date: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, g.Band]:
    """Return attempt 1's bands: the pooled centre and the bootstrap SD at n alone."""
    pooled = g.lawn_statistics(np.vstack(list(per_date.values())).sum(axis=0))
    boot = g.cluster_bootstrap(list(per_date.values()), SIZE, g.lawn_statistics, bb.RESAMPLES, rng)
    return {s: g.Band(pooled[s], g.scale_of(boot[s])) for s in g.LAWN_STATISTICS}


def small_lawn(wild: dict[int, np.ndarray], bar: float) -> dict[str, object]:
    """Return the z SDs and the pass rates with and without the centre's uncertainty."""
    rng = np.random.default_rng(SEED)
    dates = list(wild)
    registered = corrected = 0
    zs = []
    for _ in range(SPLITS):
        order = rng.permutation(len(dates))
        half = len(dates) // 2
        build = {dates[k]: wild[dates[k]] for k in order[:half]}
        held = [wild[dates[k]] for k in order[half:]]
        bands = _attempt1_lawn_bands(build, rng)
        size = int(sum(len(v) for v in build.values()))
        centre = g.cluster_bootstrap(
            list(build.values()),
            size,
            g.lawn_statistics,
            CENTRE_RESAMPLES,
            rng,
        )
        population = g.cluster_bootstrap(held, SIZE, g.lawn_statistics, 1, rng)
        values = {k: float(v[0]) for k, v in population.items()}
        first = g.grade(values, bands, bar)
        registered += all(r.passes for r in first.values())
        widened = {
            k: g.Band(b.centre, math.hypot(b.scale, g.scale_of(centre[k])))
            for k, b in bands.items()
        }
        corrected += all(r.passes for r in g.grade(values, widened, bar).values())
        zs.append([first[k].z for k in g.LAWN_STATISTICS])
    sd = np.asarray(zs).std(axis=0)
    return {
        "joint_pass_registered": registered / SPLITS,
        "joint_pass_with_centre_uncertainty": corrected / SPLITS,
        "z_sd_per_statistic": dict(zip(g.LAWN_STATISTICS, sd.round(2).tolist(), strict=True)),
    }


def local_search(bar: float) -> dict[str, object]:
    """Return the per-batch pass rates under the plain SD and a t prediction interval."""
    data = refs.load_lopezcruz()
    batches = bb.lopezcruz_batches(data)
    rows = {r["row"]: r for r in data["rows"]}
    split = bb.split_batches(batches, rows[bb.PDFR1_ROWS[1]]["dates"])
    band = [b for b, side in split["sides"].items() if side == refs.BANDS]
    values = [bb.search_statistics(batches[b]) for b in band]
    rng = np.random.default_rng(SEED)
    registered = predicted = graded = 0
    zs = []
    for _ in range(SPLITS):
        order = rng.permutation(len(values))
        half = len(values) // 2
        built = [values[k] for k in order[:half]]
        plain = {}
        for name in g.LOCAL_SEARCH_STATISTICS:
            column = np.array([v[name] for v in built])
            plain[name] = g.Band(float(column.mean()), g.scale_of(column))
        k = len(built)
        t_bar = float(scipy_stats.t.ppf(1 - g.FAMILY_ALPHA / 22, k - 1)) * math.sqrt(1 + 1 / k)
        for j in order[half:]:
            first = g.grade(values[j], plain, bar)
            registered += all(r.passes for r in first.values())
            predicted += all(r.passes for r in g.grade(values[j], plain, t_bar).values())
            graded += 1
            zs.append([first[s].z for s in g.LOCAL_SEARCH_STATISTICS])
    return {
        "batches": len(batches),
        "band_batches": len(band),
        "joint_pass_registered": registered / graded,
        "joint_pass_t_prediction_interval": predicted / graded,
        "z_sd": np.asarray(zs).std(axis=0).round(2).tolist(),
    }


def main() -> int:
    """CLI: print the diagnosis."""
    tracks = refs.load_scheer_tracks()
    classifier = rd.GaussianClassifier(hmm=rd.load_calibrated_hmm(bb.CLASSIFIER_PATH))
    wild = bb.per_date_tallies(tracks, bb.WILD_TYPE, classifier)
    bar = g.z_star(bb.WILD_TYPE_STATISTICS)
    print(
        json.dumps(
            {"small_lawn": small_lawn(wild, bar), "local_search": local_search(bar)},
            indent=2,
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
