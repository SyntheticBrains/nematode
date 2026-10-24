"""The benchmark's trajectory readouts on synthetic tracks.

Covers the benchmark-readouts spec's scenarios "A simulated track is read without its labels", "A
forward turn is not a reversal", "Missing positions", "A track in pixels" and "Units cancel".
"""

from __future__ import annotations

import numpy as np
import pytest
from quantumnematode.benchmark import readouts as ro


def _path(headings_deg: list[float], step: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Positions each second from a list of per-second headings."""
    radians = np.radians(headings_deg)
    x = np.concatenate(([0.0], np.cumsum(step * np.cos(radians))))
    y = np.concatenate(([0.0], np.cumsum(step * np.sin(radians))))
    return x, y


def test_a_reversal_is_one_onset() -> None:
    """A straight run that backs along its own path gives one onset where it turns back."""
    x, y = _path([0.0] * 10 + [180.0] * 3 + [0.0] * 10)
    assert ro.reversal_onsets(x, y).tolist() == [10]


def test_a_forward_turn_is_not_a_reversal() -> None:
    """A sharp turn that runs off to one side is not a reversal once retrace is on."""
    x, y = _path([0.0] * 10 + [90.0, 150.0] + [150.0] * 8)
    assert ro.reversal_onsets(x, y, ro.DetectorParams(theta_deg=100, retrace=False)).size == 1
    assert ro.reversal_onsets(x, y, ro.DetectorParams(theta_deg=100, retrace=True)).size == 0
    back = _path([0.0] * 10 + [180.0] * 3 + [0.0] * 5)
    assert ro.reversal_onsets(*back, ro.DetectorParams(retrace=True)).tolist() == [10]


def test_resuming_after_a_long_reversal_is_not_a_second_onset() -> None:
    """A six-second reversal, then the old heading again, is one onset."""
    x, y = _path([0.0] * 10 + [180.0] * 6 + [0.0] * 10)
    assert ro.reversal_onsets(x, y, ro.DetectorParams(refractory_s=2)).tolist() == [10]


def test_missing_positions_never_carry_an_onset() -> None:
    """A step touching a missing position is not a candidate, and rates are per valid second."""
    x, y = _path([0.0] * 10 + [180.0] * 3 + [0.0] * 5)
    x[10] = np.nan
    assert 9 not in ro.reversal_onsets(x, y).tolist()
    assert 10 not in ro.reversal_onsets(x, y).tolist()
    assert ro.valid_seconds(x, y).sum() == len(x) - 1 - 2


def test_pixels_and_millimetres_read_alike() -> None:
    """Scaling a track by any factor, with the floor off, leaves its onsets unchanged."""
    rng = np.random.default_rng(0)
    headings = np.cumsum(rng.normal(0, 30, 300))
    headings[rng.random(300) < 0.03] += 180
    x, y = _path(headings.tolist())
    mm = ro.reversal_onsets(x, y, floor=None)
    px = ro.reversal_onsets(x * 116.3, y * 116.3, floor=None)
    assert np.array_equal(mm, px)
    assert mm.size > 0


def test_the_bout_rule_at_two_frame_rates() -> None:
    """At 3 frames/s one-frame gaps merge and lone frames drop; at 30 frames/s, the same."""
    flag = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1], dtype=bool)
    assert ro.reference_onsets(flag, 3.0).tolist() == pytest.approx([1 / 3, 11 / 3])
    long = np.repeat(flag, 10)
    assert ro.reference_onsets(long, 30.0).tolist() == pytest.approx([1 / 3, 11 / 3])


def test_matching_is_one_to_one() -> None:
    """Each reference onset matches at most one detection within the tolerance."""
    assert ro.match_onsets(np.array([1, 2, 10]), np.array([1.5, 30.0])) == 1
    assert ro.match_onsets(np.array([1, 5]), np.array([1.4, 5.9])) == 2


def test_the_fit_recovers_known_parameters() -> None:
    """A noiseless alpha * exp(-gamma * (t - t0)) + beta is recovered."""
    t = np.arange(9.0, 45.0, 2.0)
    values = 2.5 * np.exp(-0.115 * (t - 9.0)) + 0.4
    fit = ro.fit_local_search(t, values, 9.0)
    assert fit.gamma == pytest.approx(0.115, rel=1e-4)
    assert fit.alpha == pytest.approx(2.5, rel=1e-4)
    assert fit.beta == pytest.approx(0.4, rel=1e-4)


def test_the_curve_counts_only_present_bins() -> None:
    """A bin an animal is mostly missing from is not counted for it."""
    present = np.ones(600, dtype=bool)
    gappy = present.copy()
    gappy[130:240] = False
    curve = ro.local_search_curve([np.array([10, 200]), np.array([10, 200])], [present, gappy])
    assert curve.n.tolist() == [2, 1, 2, 2, 2]
    assert curve.mean[0] == 1.0


def test_units_cancel_in_the_quintile_ratio() -> None:
    """Rescaling dC/dt by a positive factor leaves the ratio unchanged."""
    rng = np.random.default_rng(1)
    dcdt = rng.normal(0, 1, 2000)
    onsets = np.flatnonzero(rng.random(2000) < np.where(dcdt < 0, 0.1, 0.02))
    present = np.ones(2000, dtype=bool)
    ratio = ro.quintile_ratio([onsets], [dcdt], [present])
    assert ratio == ro.quintile_ratio([onsets], [dcdt * 1e-3], [present])
    assert ratio > 2.0


def test_a_simulated_one_second_capture_classifies_through_the_five_second_path() -> None:
    """Covers "A simulated 1-s capture": classified at 1 s, it gives the 5-s capture's states.

    On Flavell's uniform lawn the worm is always on food, so the two differ only in how the
    positions are sampled, which ``downsample_to_reference`` makes the same.
    """
    import sys
    from dataclasses import replace
    from pathlib import Path

    import quantumnematode.brain  # noqa: F401
    from quantumnematode.env.hazard_controllers import PirouetteHazards  # noqa: F401
    from quantumnematode.evaluation.batched import BatchedEvaluator
    from quantumnematode.evaluation.controllers import BatchedPirouetteHazards
    from quantumnematode.utils.config_loader import load_simulation_config
    from quantumnematode.validation import roaming_dwelling as rd

    repo = Path(__file__).resolve().parents[5]
    sys.path.insert(0, str(repo / "scripts" / "analysis"))
    import lawn_states as ls  # pyright: ignore[reportMissingImports]

    path = (
        repo
        / "configs/scenarios/foraging"
        / (
            "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_internal_s1_primitives"
            "_assay_uniform_lawn_flavell.yml"
        )
    )
    config = load_simulation_config(str(path)).model_copy(update={"max_steps": 601})
    trajectories = BatchedEvaluator(
        config,
        BatchedPirouetteHazards(0.15, (0.02, 0.01, 0.01), beta=3.0),
        [7601],
    ).run()
    steps = trajectories.to_behaviour_steps(0)
    assert config.environment is not None
    assert config.environment.foraging is not None
    lawns = config.environment.foraging.lawns
    assert lawns is not None
    classifier = rd.GaussianClassifier(hmm=rd.load_calibrated_hmm())
    one = ls.episode_windows(steps, classifier, lawns.intake_fraction, 1.0, step_seconds=1.0)
    every_fifth = [replace(s, step=k) for k, s in enumerate(steps[::5])]
    five = ls.episode_windows(every_fifth, classifier, lawns.intake_fraction, 1.0, step_seconds=5.0)
    assert np.array_equal(one.states, five.states)
    assert (one.states != rd.OFF_FOOD).all()
