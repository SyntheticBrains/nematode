"""The vendored Scheer store: it reproduces D.1's windows, re-draws its split, loads with cohorts.

Covers the benchmark-references spec's scenarios "The re-derived windows match", "The split
re-draws identically" (on the vendored counts), "Tracks load with their cohorts" and "Every
vendored set names its licence".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from quantumnematode.benchmark import references as refs
from quantumnematode.validation import roaming_dwelling as rd

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))

import extract_benchmark_references as ex  # noqa: E402  # pyright: ignore[reportMissingImports]

_WINDOWS = _REPO / "data" / "roaming_dwelling" / "scheer2023_windows.npz"
_SAMPLES_PER_STEP = int(rd.STEP_SECONDS)  # 1-s samples per 5-s step
# Each coordinate is rounded within 0.5 um, so a step's displacement moves by at most sqrt(2) um.
_SPEED_BOUND_MM_S = np.sqrt(2) * 1e-3 / rd.STEP_SECONDS + 1e-7


@pytest.fixture(scope="module")
def tracks() -> refs.ScheerTracks:
    """Load the vendored store once for the module."""
    return refs.load_scheer_tracks()


def test_the_re_derived_windows_match(tracks: refs.ScheerTracks) -> None:
    """The 1-s store reproduces the vendored 5-s windows on the Fig. 1 animals it matches."""
    with np.load(_WINDOWS) as w:
        windows = {k: w[k] for k in w.files}
    rows = tracks.arm(ex.WILD_TYPE_OD2)
    rows = rows[tracks.fig1_row[rows] >= 0]
    assert len(rows) >= 0.95 * windows["speed"].shape[0]
    hmm = rd.load_calibrated_hmm()
    classifier = rd.GaussianClassifier(hmm=hmm)
    on_windows = emission_same = state_same = 0
    for i in rows:
        j = tracks.fig1_row[i]
        x = tracks.x[i][::_SAMPLES_PER_STEP]
        y = tracks.y[i][::_SAMPLES_PER_STEP]
        measured_speed, measured_angular = rd.window_measures(x, y)
        n = len(windows["speed"][j])
        # A trailing partial window has no measure; D.1 stored it as NaN.
        speed, angular = np.full(n, np.nan), np.full(n, np.nan)
        speed[: len(measured_speed)] = measured_speed[:n]
        angular[: len(measured_angular)] = measured_angular[:n]
        on_food = tracks.in_lawn_run[i][:n] & np.isfinite(speed)
        assert np.array_equal(on_food, windows["on_food"][j])
        assert np.array_equal(tracks.out_lawn_run[i][:n], windows["off_lawn"][j])
        assert np.array_equal(tracks.labels[i][:n], windows["labels"][j])
        finite = np.isfinite(windows["speed"][j])
        assert np.array_equal(np.isfinite(speed), finite)
        assert np.all(np.abs(speed[finite] - windows["speed"][j][finite]) <= _SPEED_BOUND_MM_S)
        if not on_food.any():
            continue
        ours = rd.window_features(speed[on_food], angular[on_food])
        theirs = rd.window_features(windows["speed"][j][on_food], windows["angular"][j][on_food])
        emission_same += int(
            np.sum(hmm.log_likelihood(ours).argmax(1) == hmm.log_likelihood(theirs).argmax(1)),
        )
        decoded = classifier.states(speed, angular, on_food)
        original = classifier.states(windows["speed"][j], windows["angular"][j], on_food)
        state_same += int(np.sum(decoded[on_food] == original[on_food]))
        on_windows += int(on_food.sum())
    assert emission_same / on_windows >= 0.995
    assert state_same / on_windows >= 0.99


def test_the_vendored_split_re_draws(tracks: refs.ScheerTracks) -> None:
    """The split rule, re-run on the store's dates and arms, gives the vendored split."""
    vendored = json.loads(refs.SCHEER_SPLIT_PATH.read_text())
    redrawn = ex.draw_split(tracks.date, tracks.genotype, tracks.genotypes.index(ex.WILD_TYPE_OD2))
    assert redrawn["sides"] == vendored["sides"]
    assert redrawn["k"] == vendored["k"]


def test_tracks_load_with_their_cohorts(tracks: refs.ScheerTracks) -> None:
    """Every animal carries its genotype, date and side; positions are mm with NaN gaps."""
    assert set(tracks.genotypes) == set(ex.SCHEER_ARMS)
    assert set(np.unique(tracks.side).tolist()) == {refs.BANDS, refs.HELD_OUT}
    assert tracks.x.shape[1] == 2400
    assert tracks.frame_flags.shape[1] == 7200
    assert np.isnan(tracks.x).any()
    assert np.nanmax(np.abs(tracks.x)) < 40.0
    for name in ex.SCHEER_ARMS:
        assert len(tracks.arm(name)) > 0
    uniform = tracks.arm("PD1074_od2_uniformlawn_Data")
    if np.all(tracks.origin[uniform] == refs.ORIGIN_FIRST_POSITION):
        assert np.all(np.isnan(tracks.boundary_radius[uniform]))


def test_every_vendored_set_names_its_licence() -> None:
    """Each data file under data/benchmark has a provenance entry with its licence and checksum."""
    provenance = (refs.BENCHMARK_DATA / "PROVENANCE.md").read_text()
    sections = provenance.split("\n## ")[1:]
    named = {s.split("\n", 1)[0].strip("` ") for s in sections}
    files = {p.name for p in refs.BENCHMARK_DATA.iterdir() if p.suffix in {".npz", ".json"}}
    assert files <= {n for name in named for n in name.replace("`", "").split(" and ")}
    for section in sections:
        assert "**Licence:**" in section
        assert "SHA-256" in section or "MD5" in section
