"""Tests for the associative-memory delayed-match phase machine (with reversal)."""

import numpy as np
from quantumnematode.env.associative_memory import AssociativeMemoryPhase, AssociativeMemoryTask


def _task(*, trials=2, cond=1, delay=2, response=1, reversal=0.5, reward=1.0, penalty=0.0, seed=0):  # noqa: PLR0913
    return AssociativeMemoryTask(
        trials_per_episode=trials,
        cond_steps_per_cue=cond,
        delay_steps=delay,
        response_steps=response,
        reversal_prob=reversal,
        reward_correct=reward,
        penalty_wrong=penalty,
        rng=np.random.default_rng(seed),
    )


def _walk_to_response(task):
    while task.phase != AssociativeMemoryPhase.RESPONSE:
        task.advance()


class TestPhaseStructure:
    """Phase derivation across conditioning/reversal/delay/response."""

    def test_reversal_trial_phase_sequence(self):
        """cond=1 (2 cues), reversed: [COND, COND, REV, REV, DELAY, DELAY, RESPONSE]."""
        task = _task(cond=1, delay=2, response=1, reversal=1.0)  # forced reversal
        seen = [task.phase]
        for _ in range(task._trial_span() - 1):
            task.advance()
            seen.append(task.phase)
        P = AssociativeMemoryPhase  # noqa: N806 - enum alias for a compact expected-sequence list
        assert seen == [
            P.CONDITIONING,
            P.CONDITIONING,
            P.REVERSAL,
            P.REVERSAL,
            P.DELAY,
            P.DELAY,
            P.RESPONSE,
        ]

    def test_non_reversal_trial_has_no_reversal_phase(self):
        """With reversal_prob=0 the trial is conditioning -> delay -> response only."""
        task = _task(cond=1, delay=2, response=1, reversal=0.0)
        phases = set()
        for _ in range(task._trial_span()):
            phases.add(task.phase)
            task.advance()
        assert AssociativeMemoryPhase.REVERSAL not in phases

    def test_zero_delay_collapses(self):
        """delay=0 (no reversal): conditioning is followed immediately by the response phase."""
        task = _task(cond=1, delay=0, response=1, reversal=0.0)
        task.advance()  # past cue A
        task.advance()  # past cue B -> response
        assert task.phase == AssociativeMemoryPhase.RESPONSE


class TestSignals:
    """The cue-identity + outcome + go observation channels."""

    def test_conditioning_binds_cue_to_outcome(self):
        """Each conditioning step exposes a cue-identity with its valence; exactly one is +1."""
        task = _task(cond=1, reversal=0.0)
        outcomes = []
        for _ in range(2):  # two conditioning steps
            cue, outcome, go = task.signals()
            assert cue in (-1.0, 1.0)
            assert go == 0.0
            outcomes.append(outcome)
            task.advance()
        assert sorted(outcomes) == [-1.0, 1.0]  # exactly one rewarded

    def test_reversal_flips_the_outcomes(self):
        """The reversal block re-presents the cues with flipped outcomes."""
        task = _task(cond=1, reversal=1.0)
        cond = [task.signals()[:2] for _ in _advance_n(task, 2)]  # (cue, outcome) x2
        rev = [task.signals()[:2] for _ in _advance_n(task, 2)]
        # same cues, flipped outcomes
        assert {c[0] for c in cond} == {c[0] for c in rev}
        for c in cond:
            match = next(r for r in rev if r[0] == c[0])
            assert match[1] == -c[1]

    def test_withheld_in_delay_and_response(self):
        """Cue + outcome are exactly 0 in delay/response; go=1 only in response."""
        task = _task(cond=1, delay=2, response=1, reversal=0.0)
        _walk_to_response(task)  # advance through conditioning + delay
        # step back is not possible; re-walk collecting delay/response signals
        task = _task(cond=1, delay=2, response=1, reversal=0.0)
        for _ in range(task._trial_span()):
            cue, outcome, go = task.signals()
            if task.phase in (AssociativeMemoryPhase.DELAY, AssociativeMemoryPhase.RESPONSE):
                assert cue == 0.0
                assert outcome == 0.0
            if task.phase == AssociativeMemoryPhase.RESPONSE:
                assert go == 1.0
            else:
                assert go == 0.0
            task.advance()

    def test_done_signals_are_zero(self):
        """Once every trial completes, all channels read 0."""
        task = _task(trials=1, cond=1, delay=0, response=1, reversal=0.0)
        for _ in range(task._trial_span()):
            task.advance()
        assert task.done
        assert task.signals() == (0.0, 0.0, 0.0)


class TestScoring:
    """Reward + accuracy, including the reversal update demand."""

    def test_updated_response_is_correct_on_reversal(self):
        """Responding with the post-reversal (current) cue scores correct."""
        task = _task(cond=1, reversal=1.0)
        _walk_to_response(task)
        task.record_response(turn=task.current_rewarded_cue)
        assert task.take_reward() == 1.0
        assert task.response_accuracy == 1.0

    def test_hold_only_response_fails_on_reversal(self):
        """A hold-only policy (responds the initial cue) is wrong on a reversal trial."""
        task = _task(cond=1, reversal=1.0)
        initial = -task.current_rewarded_cue  # current is flipped from initial on a reversal trial
        _walk_to_response(task)
        task.record_response(turn=initial)
        assert task.take_reward() == 0.0
        assert task.response_accuracy == 0.0

    def test_no_reward_outside_response_phase(self):
        """record_response is a no-op in conditioning/reversal/delay."""
        task = _task(cond=1, reversal=1.0)
        task.record_response(turn=1.0)  # in conditioning
        assert task.take_reward() == 0.0
        assert task.num_responses == 0

    def test_reversal_and_non_reversal_split(self):
        """Accuracy splits by reversal vs non-reversal trials."""
        task = _task(trials=6, cond=1, delay=0, response=1, reversal=0.5)
        for _ in range(task._trial_span() * 6 + 6):
            if task.done:
                break
            if task.phase == AssociativeMemoryPhase.RESPONSE:
                task.record_response(turn=task.current_rewarded_cue)  # always correct
            task.advance()
        assert task.response_accuracy == 1.0
        assert task.reversal_accuracy in (0.0, 1.0)  # 1.0 if any reversal trials occurred
        assert task.non_reversal_accuracy in (0.0, 1.0)


class TestEpisodeLifecycle:
    """Reset + reproducibility."""

    def test_reset_clears_records(self):
        task = _task(cond=1, delay=0, response=1, reversal=0.5)
        _walk_to_response(task)
        task.record_response(turn=1.0)
        task.reset()
        assert task.num_responses == 0
        assert not task.done

    def test_same_seed_reproduces_trial_sequence(self):
        """Same seed -> identical rewarded-cue + reversal sequence."""
        a, b = _task(seed=3), _task(seed=3)
        for _ in range(20):
            assert a.current_rewarded_cue == b.current_rewarded_cue
            a.advance()
            b.advance()


def _advance_n(task, n):
    """Yield n times, advancing the task after each yield (helper for signal collection)."""
    for _ in range(n):
        yield
        task.advance()
