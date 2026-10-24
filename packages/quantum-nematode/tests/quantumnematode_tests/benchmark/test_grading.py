"""The benchmark's grading rule: windows, tallies, statistics, bands, the pass rule and distance.

Covers the benchmark-grading spec's scenarios "A worm on the boundary", "The uniform well has no
boundary", "The small lawn reads its last 40 minutes", "A bout cut by leaving is censored", "A worm
that never leaves", "A band grows as the population shrinks", "Normal leaving is an equivalence
band", "A wrong direction fails inside the band", "A worm at every centre" and "The gate is read on
dates the fit never saw"; and the assay-protocols scenario "The uniform well stays on food".
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from quantumnematode.benchmark import grading as g
from quantumnematode.benchmark.readouts import OFF_FOOD_DETECTOR
from quantumnematode.report.dtypes import BehaviourStep
from quantumnematode.validation import roaming_dwelling as rd

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
import benchmark_bands as bb  # noqa: E402  # pyright: ignore[reportMissingImports]
import benchmark_classifier as bc  # noqa: E402  # pyright: ignore[reportMissingImports]
import benchmark_grade as bg  # noqa: E402  # pyright: ignore[reportMissingImports]

_CLASSIFIER = rd.GaussianClassifier(hmm=rd.load_calibrated_hmm(bb.CLASSIFIER_PATH))


def _track(seconds: int, speed: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(seconds + 1, dtype=float)
    return speed * t, np.zeros_like(t)


def test_a_worm_on_the_boundary() -> None:
    """One sample just outside makes a window neither wholly on nor wholly off the lawn."""
    x, y = _track(40)
    lawn = np.ones(len(x), dtype=bool)
    lawn[15] = False
    w = g.window_inputs(x, y, lawn)
    assert w.on.tolist() == [True, False, True, True]
    assert not w.off.any()


def test_a_missing_sample_leaves_its_window_unclassified() -> None:
    """A window with a missing position is neither on, off nor valid."""
    x, y = _track(40)
    x[25] = np.nan
    w = g.window_inputs(x, y, np.ones(len(x), dtype=bool))
    assert w.valid.tolist() == [True, True, False, True]
    assert w.on.tolist() == [True, True, False, True]


def test_the_uniform_well_has_no_boundary() -> None:
    """With the boundary's radii missing, every sample of the uniform well is on the lawn."""
    x, y = _track(40)
    radii = np.full(64, np.nan)
    assert not g.real_on_lawn(x, y, radii).any()
    assert g.real_on_lawn(x, y, radii, uniform=True).all()
    windows = g.animal_windows(x, y, g.real_on_lawn(x, y, radii, uniform=True), _CLASSIFIER)
    assert (windows.states != rd.OFF_FOOD).all()


def test_a_bout_cut_by_leaving_is_censored() -> None:
    """A 300-s dwelling bout that ends at the lawn's edge adds time but no completed bout."""
    states = np.array([rd.DWELLING] * 30 + [rd.OFF_FOOD] * 3 + [rd.ROAMING] * 2 + [rd.DWELLING])
    time_s, completed = g._bouts(states, rd.DWELLING)
    assert time_s == pytest.approx(310.0)
    assert completed == 0
    roaming_time, roaming_done = g._bouts(states, rd.ROAMING)
    assert (roaming_time, roaming_done) == (20.0, 1)


def _windows(states: list[int], on: list[bool], off: list[bool]) -> g.AnimalWindows:
    return g.AnimalWindows(
        states=np.array(states),
        on=np.array(on),
        off=np.array(off),
        valid=np.ones(len(states), dtype=bool),
    )


def test_a_worm_that_never_leaves() -> None:
    """No exit: the exit rate is 0 and graded; the pre-exit enrichment cannot be computed and fails."""
    w = _windows([rd.DWELLING] * 20, [True] * 20, [False] * 20)
    stats = g.lawn_statistics(g.tally(w))
    assert stats["exits_per_on_lawn_hour"] == 0.0
    assert math.isnan(stats["pre_exit_enrichment"])
    readings = g.grade(stats, {"pre_exit_enrichment": g.Band(0.3, 0.07)}, 2.84)
    assert not readings["pre_exit_enrichment"].passes
    assert readings["pre_exit_enrichment"].z == g.Z_CAP


def test_the_small_lawn_reads_its_last_40_minutes() -> None:
    """The reader takes the analysis window's first 2,400 steps, and refuses a shorter window."""
    steps = [
        BehaviourStep(
            step=i,
            x=0.0,
            y=0.0,
            heading_rad=0.0,
            concentration=0.0,
            dc_dt=0.0,
            grad_dir=0.0,
            grad_strength=0.0,
        )
        for i in range(3600)
    ]
    read = g.analysis_steps(steps, (1200, 3600), g.LAWN_SAMPLES)
    assert (read[0].step, read[-1].step, len(read)) == (1200, 3599, 2400)
    with pytest.raises(ValueError, match="shorter than the reference's 2400"):
        g.analysis_steps(steps, (1300, 3600), g.LAWN_SAMPLES)


def test_the_real_tracks_give_239_windows() -> None:
    """A 40-min track at 1 s gives 239 complete windows, real or simulated alike."""
    x, y = _track(g.LAWN_SAMPLES - 1)
    assert len(g.window_inputs(x, y, np.ones(len(x), dtype=bool)).on) == 239


def test_a_band_grows_as_the_population_shrinks() -> None:
    """The cluster bootstrap's scale at 64 animals exceeds its scale at 200."""
    rng = np.random.default_rng(0)
    per_date = []
    for _ in range(20):
        roaming = rng.uniform(0.05, 0.4)
        animals = np.zeros((25, len(g.TALLY_FIELDS)))
        animals[:, g._F["animals"]] = 1
        animals[:, g._F["classified"]] = 200
        animals[:, g._F["roaming"]] = rng.binomial(200, roaming, size=25)
        per_date.append(animals)
    scales = [
        g.scale_of(
            g.cluster_bootstrap(per_date, n, g.lawn_statistics, 400, np.random.default_rng(1))[
                "roaming_fraction"
            ],
        )
        for n in (64, 200)
    ]
    assert scales[0] > scales[1]


def test_normal_leaving_is_an_equivalence_band() -> None:
    """A reference Δ whose interval holds zero is graded by magnitude alone, either sign."""
    band = g.Band(centre=0.04, scale=0.27)
    assert band.direction() == 0
    for value in (-0.3, 0.3):
        assert g.grade({"d": value}, {"d": band}, 2.24, directions=True)["d"].passes


def test_a_wrong_direction_fails_inside_the_band() -> None:
    """A positive Δ fails against a reference resolved below zero, though |z| is within the bar."""
    band = g.Band(centre=-0.116, scale=0.045)
    assert band.direction() == -1
    reading = g.grade({"d": 0.01}, {"d": band}, 2.84, directions=True)["d"]
    assert abs(reading.z) <= 2.84
    assert not reading.passes
    assert g.grade({"d": -0.05}, {"d": band}, 2.84, directions=True)["d"].passes


def test_a_thin_arms_direction_reads_the_inflated_scale() -> None:
    """A direction is resolved from centre +/- 1.96 scale, so widening the scale can unresolve it."""
    assert g.Band(centre=0.1, scale=0.04).direction() == 1
    assert g.Band(centre=0.1, scale=0.06, inflated=True).direction() == 0


def test_a_worm_at_every_centre() -> None:
    """Every statistic at its centre gives D = D_fit = 0."""
    bands = {f: {s: g.Band(1.0, 0.5) for s in stats} for f, stats in g.FAMILIES.items()}
    readings = {f: g.grade(dict.fromkeys(stats, 1.0), bands[f], 2.84) for f, stats in g.FAMILIES.items()}
    assert g.distance(readings, list(g.FAMILIES)) == 0.0
    assert g.distance(readings, list(g.FIT_FAMILIES)) == 0.0


def test_the_bars_and_weights() -> None:
    """The wild type's bar is 2.84 over 11 statistics; mutants' 2.24 and 2.50; weights sum to 1."""
    assert round(g.z_star(11), 2) == 2.84
    assert round(g.z_star(2), 2) == 2.24
    assert round(g.z_star(4), 2) == 2.50
    assert sum(g.weights(list(g.FAMILIES)).values()) == pytest.approx(1.0)
    w = g.weights(list(g.FIT_FAMILIES))
    assert w[("small_lawn", "roaming_fraction")] == pytest.approx(1 / 12)
    assert w[("uniform_lawn", "roaming_fraction")] == pytest.approx(1 / 6)


def test_the_local_search_statistics_need_every_bin() -> None:
    """A missing bin between 8 and 44 min leaves both statistics undefined."""
    starts = np.arange(0, 46, 2.0)
    means = 2.5 * np.exp(-0.11 * (starts + 1 - 9)) + 0.4
    full = g.local_search_statistics(starts, means)
    assert full["gamma_per_min"] == pytest.approx(0.11, abs=1e-3)
    means[10] = np.nan
    assert math.isnan(g.local_search_statistics(starts, means)["gamma_per_min"])


def test_the_registered_detector_is_the_calibrated_one() -> None:
    """The off-food detector's parameters are the ones its calibration chose."""
    record = json.loads(
        (_REPO / "docs/experiments/logbooks/supporting/107-benchmark-readouts/detector.json").read_text(),
    )
    chosen = record["chosen"]
    assert (
        OFF_FOOD_DETECTOR.theta_deg,
        OFF_FOOD_DETECTOR.window_s,
        OFF_FOOD_DETECTOR.kappa,
        OFF_FOOD_DETECTOR.refractory_s,
        OFF_FOOD_DETECTOR.retrace,
    ) == (
        chosen["theta_deg"],
        chosen["window_s"],
        chosen["kappa"],
        chosen["refractory_s"],
        chosen["retrace"],
    )


# ---------------------------------------------------------------------------------------------
# The scripts: no held-out data, the vendored records
# ---------------------------------------------------------------------------------------------


def test_the_gate_is_read_on_dates_the_fit_never_saw() -> None:
    """The classifier's fit and gate dates are disjoint band dates, and its gate passed."""
    record = json.loads(bb.CLASSIFIER_PATH.read_text())
    fit, gate = set(record["split"]["fit_dates"]), set(record["split"]["gate_dates"])
    assert not fit & gate
    split = json.loads((_REPO / "data/benchmark/scheer2023_split.json").read_text())["sides"]
    assert all(split[str(d)] == "bands" for d in fit | gate)
    assert record["gate"]["passes"]
    assert record["gate"]["kappa"] >= bc.KAPPA_GATE
    assert bc.split_dates(sorted(fit | gate)) == (sorted(fit), sorted(gate))


def test_the_bands_read_no_held_out_date_or_batch() -> None:
    """Every date and batch behind the bands is on the band side of its split."""
    bands = json.loads((_REPO / "data/benchmark/bands.json").read_text())
    split = json.loads((_REPO / "data/benchmark/scheer2023_split.json").read_text())["sides"]
    for mutant in bands["arms"]["mutants"].values():
        assert all(split[str(d)] == "bands" for d in mutant["paired_band_dates"])
    lc = json.loads(bb.LOPEZCRUZ_SPLIT_PATH.read_text())
    assert all(lc["sides"][b] == "bands" for b in bands["arms"]["local_search"]["band_batches"])
    pdfr1_control = {r["row"]: r for r in bb.refs.load_lopezcruz()["rows"]}[90]["dates"]
    assert lc["sides"][pdfr1_control] == "bands"


def test_band_rows_skip_every_held_out_animal() -> None:
    """The bands script's only row selector returns band-side animals alone."""
    tracks = bb.refs.load_scheer_tracks()
    for arm in (bb.WILD_TYPE, bb.UNIFORM, *(a for a, _ in bb.MUTANTS.values())):
        rows = bb.band_rows(tracks, arm)
        assert len(rows)
        assert (tracks.side[rows] == bb.refs.BANDS).all()


def test_the_lopezcruz_split_keeps_the_pdfr1_control() -> None:
    """The López-Cruz draw never holds out pdfr-1's same-date control, and re-draws exactly."""
    data = bb.refs.load_lopezcruz()
    batches = bb.lopezcruz_batches(data)
    kept = {r["row"]: r for r in data["rows"]}[90]["dates"]
    split = bb.split_batches(batches, kept)
    assert split["sides"][kept] == "bands"
    assert split == {
        k: v
        for k, v in json.loads(bb.LOPEZCRUZ_SPLIT_PATH.read_text()).items()
        if k in split
    }


# ---------------------------------------------------------------------------------------------
# The uniform well and an end-to-end grade
# ---------------------------------------------------------------------------------------------


def test_the_uniform_well_stays_on_food() -> None:
    """Every step of the uniform well is on the lawn, in both runners, rim included."""
    from quantumnematode.env.hazard_controllers import ConstantHazards
    from quantumnematode.evaluation.batched import BatchedEvaluator
    from quantumnematode.evaluation.controllers import BatchedConstantHazards
    from quantumnematode.evaluation.reference import run_reference
    from quantumnematode.utils.config_loader import load_simulation_config

    path = bg.FORAGING / f"{bg.ARMS['mlp']}_assay_uniform_lawn_scheer.yml"
    config = load_simulation_config(str(path)).model_copy(update={"max_steps": 600})
    fast = (0.2, (0.02, 0.01, 0.01))
    trajectories = BatchedEvaluator(config, BatchedConstantHazards(*fast), [5991, 5992]).run()
    for i in range(2):
        assert all(s.on_lawn for s in trajectories.to_behaviour_steps(i))
    episode = run_reference(config, ConstantHazards(*fast), 5991, 600)
    assert all(s.on_lawn for s in episode.behaviour)


@pytest.mark.slow
def test_a_batched_population_is_graded_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nine smoke animals run every assay and come back graded on all 11 statistics."""
    bands = bg.load_bands()
    monkeypatch.setattr(bg, "bands_at", lambda b, _n: b["bands"]["64"])
    populations = bg.batched_populations("mlp", "constant", list(range(5991, 6000)))
    result = bg.grade_populations(populations, bands)
    assert result["n"] == 9
    assert set(result["families"]) == set(g.FAMILIES)
    assert sum(len(v) for v in result["families"].values()) == 11
    assert result["D"] is not None
    assert result["D_fit"] is not None
