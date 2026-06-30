"""Tests for the bit-memory delayed-match-to-cue phase machine."""

import numpy as np
import pytest
from quantumnematode.env.bit_memory import BitMemoryPhase, BitMemoryTask


def _task(*, trials=2, cue=2, delay=3, response=1, reward=1.0, penalty=0.0, seed=0):  # noqa: PLR0913
    return BitMemoryTask(
        trials_per_episode=trials,
        cue_steps=cue,
        delay_steps=delay,
        response_steps=response,
        reward_correct=reward,
        penalty_wrong=penalty,
        rng=np.random.default_rng(seed),
    )


class TestPhaseStructure:
    """Phase derivation + the cue/go observation signals."""

    def test_trial_span(self):
        assert _task(cue=2, delay=8, response=1).trial_span == 11

    def test_phases_land_on_boundaries(self):
        """cue=2, delay=3, response=1 -> steps [CUE,CUE,DELAY,DELAY,DELAY,RESPONSE]."""
        task = _task(cue=2, delay=3, response=1)
        expected = [
            BitMemoryPhase.CUE,
            BitMemoryPhase.CUE,
            BitMemoryPhase.DELAY,
            BitMemoryPhase.DELAY,
            BitMemoryPhase.DELAY,
            BitMemoryPhase.RESPONSE,
        ]
        seen = []
        for _ in range(task.trial_span):
            seen.append(task.phase)
            task.advance()
        assert seen == expected

    def test_zero_delay_collapses(self):
        """delay=0 means the cue phase is followed immediately by the response phase."""
        task = _task(cue=1, delay=0, response=1)
        assert task.phase == BitMemoryPhase.CUE
        task.advance()
        assert task.phase == BitMemoryPhase.RESPONSE

    def test_signals_cue_only_in_cue_phase(self):
        """The cue rides the cue channel only during the cue phase; go only in response."""
        task = _task(cue=1, delay=1, response=1)
        task._cue = 1.0
        assert task.signals() == (1.0, 0.0)  # cue phase
        task.advance()
        assert task.signals() == (0.0, 0.0)  # delay: cue withheld, no go
        task.advance()
        assert task.signals() == (0.0, 1.0)  # response: cue still withheld, go on

    def test_signals_zero_when_done(self):
        task = _task(trials=1, cue=1, delay=0, response=1)
        for _ in range(task.trial_span):
            task.advance()
        assert task.done
        assert task.signals() == (0.0, 0.0)


class TestResponseScoring:
    """Response scoring + the deferred reward, via sign(turn)."""

    def test_correct_response_scores_reward(self):
        task = _task(cue=1, delay=0, response=1, reward=1.0, penalty=0.5)
        task._cue = 1.0
        task.advance()  # -> response phase
        assert task.phase == BitMemoryPhase.RESPONSE
        task.record_response(turn=0.7)  # sign(+) == cue (+1) -> correct
        assert task.take_reward() == pytest.approx(1.0)
        assert task.take_reward() == 0.0  # popped once

    def test_wrong_response_scores_penalty(self):
        task = _task(cue=1, delay=0, response=1, reward=1.0, penalty=0.5)
        task._cue = 1.0
        task.advance()  # -> response
        task.record_response(turn=-0.3)  # sign(-) != cue (+1) -> wrong
        assert task.take_reward() == pytest.approx(-0.5)

    def test_no_scoring_outside_response_phase(self):
        task = _task(cue=2, delay=2, response=1)
        task.record_response(turn=0.9)  # cue phase -> no-op
        assert task.take_reward() == 0.0
        assert task.num_responses == 0


class TestEpisodeProgress:
    """Trial roll-over, done, and the success-rate metric."""

    def test_done_after_all_trials(self):
        task = _task(trials=3, cue=1, delay=1, response=1)
        steps = task.trial_span * 3
        for _ in range(steps):
            assert not task.done
            task.advance()
        assert task.done

    def test_cue_resamples_each_trial_reproducibly(self):
        """Two tasks with the same seed produce the same per-trial cue sequence."""
        a, b = _task(trials=5, seed=42), _task(trials=5, seed=42)
        cues_a, cues_b = [], []
        for _ in range(5):
            cues_a.append(a.current_cue)
            cues_b.append(b.current_cue)
            for _ in range(a.trial_span):
                a.advance()
            for _ in range(b.trial_span):
                b.advance()
        assert cues_a == cues_b
        assert set(cues_a) <= {-1.0, 1.0}

    def test_success_rate_tracks_correct_responses(self):
        task = _task(trials=4, cue=1, delay=0, response=1, reward=1.0, penalty=0.0)
        # Drive 4 trials; respond correctly on the first 3, wrong on the 4th.
        for i in range(4):
            task.advance()  # cue -> response (delay 0)
            cue = task.current_cue
            turn = cue if i < 3 else -cue  # match for first 3, mismatch on the 4th
            task.record_response(turn=turn)
            task.advance()  # response -> next trial
        assert task.num_responses == 4
        assert task.cue_match_success_rate == pytest.approx(0.75)
