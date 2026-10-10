"""The roaming/dwelling instrument: the measures, the line, the smoothing model, and agreement.

Covers the realworm-behavioural-validation scenarios "Real and simulated worms are read alike" (one
set of functions and parameters for both) and "Off-food windows are not classified".
"""

from __future__ import annotations

import numpy as np
import pytest
from quantumnematode.validation import roaming_dwelling as rd


def _track(headings_deg: list[float], step_mm: float) -> tuple[np.ndarray, np.ndarray]:
    angles = np.radians(np.asarray(headings_deg, dtype=float))
    x = np.concatenate([[0.0], np.cumsum(step_mm * np.cos(angles))])
    y = np.concatenate([[0.0], np.cumsum(step_mm * np.sin(angles))])
    return x, y


class TestMeasures:
    def test_a_straight_run(self) -> None:
        """Steps of 0.5 mm in 5 s are 0.1 mm/s, with no turn."""
        speed, angular = rd.window_measures(*_track([0.0] * 8, 0.5))
        assert speed == pytest.approx([0.1] * 4)
        assert np.nanmax(angular) == pytest.approx(0.0, abs=1e-6)

    def test_a_reversal_is_a_half_turn(self) -> None:
        """Doubling back turns 180 degrees between two steps."""
        _, turn = rd.step_measures(*_track([0.0, 180.0], 0.5))
        assert np.isnan(turn[0])
        assert turn[1] == pytest.approx(180.0)

    def test_angular_speed_is_degrees_per_second(self) -> None:
        """A steady 30-degree turn per 5 s step is 6 degrees/s."""
        _, angular = rd.window_measures(*_track([30.0 * k for k in range(9)], 0.5))
        assert angular[1:] == pytest.approx([6.0] * 3)

    def test_a_still_worm_has_no_turn(self) -> None:
        """Zero displacement gives zero speed and an unmeasurable turn."""
        speed, angular = rd.window_measures(np.zeros(5), np.zeros(5))
        assert np.all(speed == 0.0)
        assert np.all(np.isnan(angular))


class TestLine:
    def test_fast_and_straight_is_roaming(self) -> None:
        """Roaming where speed times the slope exceeds the angular speed; still is dwelling."""
        speed = np.array([0.12, 0.02, 0.0])
        angular = np.array([2.0, 20.0, np.nan])
        assert rd.roaming_observations(speed, angular, slope=100.0).tolist() == [1, 0, 0]


class TestModel:
    def test_the_reference_model_loads_with_roaming_as_state_one(self) -> None:
        """The vendored model is a proper two-state model whose roaming state emits roaming."""
        hmm = rd.load_reference_hmm()
        for row in (np.exp(hmm.log_transitions), np.exp(hmm.log_emissions)):
            assert row.sum(axis=1) == pytest.approx([1.0, 1.0])
        assert np.exp(hmm.log_emissions)[rd.ROAMING, 1] > np.exp(hmm.log_emissions)[rd.DWELLING, 1]

    def test_viterbi_recovers_sampled_states(self) -> None:
        """Decoding a sequence drawn from the reference model recovers most of its states."""
        hmm = rd.load_reference_hmm()
        states, obs = hmm.sample(5000, np.random.default_rng(1))
        assert np.mean(hmm.viterbi(obs) == states) > 0.85

    def test_a_lone_roaming_window_inside_dwelling_is_smoothed_away(self) -> None:
        """One roaming observation amid dwelling is not enough to switch states."""
        obs = np.array([0] * 10 + [1] + [0] * 10)
        assert set(rd.load_reference_hmm().viterbi(obs)) == {rd.DWELLING}


class TestClassifier:
    def test_off_food_windows_are_not_classified(self) -> None:
        """Off food a window is OFF_FOOD; each on-food run is decoded on its own."""
        classifier = rd.Classifier(slope=100.0, hmm=rd.load_reference_hmm())
        speed = np.array([0.15] * 6 + [0.0] * 2 + [0.15] * 6)
        angular = np.zeros(14)
        on_food = np.array([True] * 6 + [False] * 2 + [True] * 6)
        states = classifier.states(speed, angular, on_food)
        assert states[6:8].tolist() == [rd.OFF_FOOD, rd.OFF_FOOD]
        assert set(states[:6]) == set(states[8:]) == {rd.ROAMING}

    def test_on_food_runs(self) -> None:
        """Runs are the contiguous stretches of True."""
        assert rd.on_food_runs(np.array([1, 1, 0, 1, 0, 0, 1], dtype=bool)) == [
            (0, 2),
            (3, 4),
            (6, 7),
        ]


class TestCalibration:
    def test_agreement_and_kappa(self) -> None:
        """Perfect agreement has kappa 1; off-food windows are left out."""
        labels = np.array([0, 1, 1, 0, rd.OFF_FOOD])
        result = rd.agreement(labels, labels)
        assert result["n"] == 4
        assert result["accuracy"] == 1.0
        assert result["kappa"] == pytest.approx(1.0)

    def test_the_slope_that_reproduces_the_labels_is_found(self) -> None:
        """Labels made with slope 200 are best reproduced by the candidate nearest 200."""
        rng = np.random.default_rng(0)
        hmm = rd.load_reference_hmm()
        tracks = []
        for _ in range(20):
            speed = rng.uniform(0.0, 0.2, size=60)
            angular = rng.uniform(0.0, 30.0, size=60)
            on = np.ones(60, dtype=bool)
            labels = rd.Classifier(slope=200.0, hmm=hmm).states(speed, angular, on)
            tracks.append((speed, angular, on, labels))
        slope, kappa = rd.calibrate_slope(tracks, hmm, np.array([50.0, 100.0, 200.0, 400.0]))
        assert slope == 200.0
        assert kappa == pytest.approx(1.0)
