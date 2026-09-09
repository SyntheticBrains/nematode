"""The positive control's task: closed-form floor and optimum, and no leak of the answer."""

from __future__ import annotations

import numpy as np
import pytest
from quantumnematode.plasticity.positive_control import ContextualAssociation


def _task() -> ContextualAssociation:
    return ContextualAssociation.default()


class TestTheClosedForms:
    def test_the_floor_is_the_target_variance_plus_the_noise_cost(self) -> None:
        task = _task()
        noise = 0.37
        assert task.cue_blind_floor(noise) == pytest.approx(
            -float(np.var(task.targets)) - noise**2,
        )

    def test_the_optimum_is_the_noise_cost_alone(self) -> None:
        assert _task().optimum(0.37) == pytest.approx(-(0.37**2))

    def test_the_gap_is_exactly_the_target_variance(self) -> None:
        # The noise term sits on both sides, so what a learner can win is Var[t] and nothing else.
        task = _task()
        for noise in (0.0, 0.1, 0.37, 1.0):
            assert task.gap(noise) == pytest.approx(float(np.var(task.targets)))

    def test_no_cue_blind_action_beats_the_floor(self) -> None:
        # Checked numerically rather than asserted: the floor is the control's foundation.
        task = _task()
        noise = 0.37
        floor = task.cue_blind_floor(noise)
        for constant in np.linspace(-2.0, 2.0, 401):
            expected = float(np.mean([-((constant - t) ** 2) for t in task.targets])) - noise**2
            assert expected <= floor + 1e-9

    def test_the_optimum_is_reached_only_by_using_the_cue(self) -> None:
        task = _task()
        noise = 0.0
        perfect = float(
            np.mean([task.reward(c, float(task.targets[c])) for c in range(task.n_cues)]),
        )
        assert perfect == pytest.approx(task.optimum(noise))


class TestTheObservation:
    def test_it_carries_the_cue(self) -> None:
        task = _task()
        for cue in range(task.n_cues):
            observation = task.observation(cue)
            assert observation.sum() == pytest.approx(1.0)
            assert observation[cue] == pytest.approx(1.0)

    def test_it_does_not_carry_the_target(self) -> None:
        # The association is discoverable only from reward: two tasks with different targets
        # produce identical observations.
        one = ContextualAssociation(targets=np.array([-1.0, 0.0, 1.0]))
        other = ContextualAssociation(targets=np.array([0.5, -0.5, 0.25]))
        for cue in range(3):
            assert np.array_equal(one.observation(cue), other.observation(cue))


class TestSampling:
    def test_it_is_uniform_and_seeded(self) -> None:
        task = _task()
        rng = np.random.default_rng(11)
        draws = [task.sample_cue(rng) for _ in range(8000)]
        counts = np.bincount(draws, minlength=task.n_cues)
        assert counts.min() > 8000 / task.n_cues * 0.9
        assert np.array_equal(
            [task.sample_cue(np.random.default_rng(3)) for _ in range(5)],
            [task.sample_cue(np.random.default_rng(3)) for _ in range(5)],
        )

    def test_the_reward_peaks_at_the_target(self) -> None:
        task = _task()
        for cue in range(task.n_cues):
            best = task.reward(cue, float(task.targets[cue]))
            assert best == pytest.approx(0.0)
            assert task.reward(cue, float(task.targets[cue]) + 0.3) < best
