"""Chemosensory associative-memory probe phase machine.

A deliberately-artificial working-memory probe, the biological remember-and-use twin of the
bit-memory control. Per trial two cues are presented in a conditioning phase with opposite
outcomes (one rewarded); with probability ``reversal_prob`` a reversal block re-presents them
with *flipped* outcomes; after a delay the agent must give a binary readout of the **current**
(post-reversal) rewarded cue. The reversal makes the demand working-memory *update* (overwrite a
held association on new evidence): a memoryless policy is at chance, and a policy that only
retains the initial association without updating is at chance on the reversal fraction.

Pure phase-machine logic (no env/agent imports) so it is unit-testable in isolation: the
environment owns an instance and exposes its cue/outcome/go signals; the episode runner drives
``record_response`` / ``take_reward`` / ``advance``. Cues are encoded A = +1 / B = -1; 0 on the
cue/outcome channels unambiguously means "withheld" (delay/response).
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

_CUE_A = 1.0
_CUE_B = -1.0


class AssociativeMemoryPhase(StrEnum):
    """The phases of an associative-memory trial (reversal only on reversal trials)."""

    CONDITIONING = "conditioning"
    REVERSAL = "reversal"
    DELAY = "delay"
    RESPONSE = "response"


class AssociativeMemoryTask:
    """Stateful phase machine for the associative-memory delayed-match task.

    The phase is *derived* from the step index within the trial, so a zero-length delay
    collapses correctly. The per-trial span varies (reversal trials add a reversal block), so
    ``advance`` rolls to the next trial at the current trial's own span.

    Parameters
    ----------
    trials_per_episode, cond_steps_per_cue, delay_steps, response_steps
        Trial structure. Each of the two cues is shown for ``cond_steps_per_cue`` steps in the
        conditioning block (and again in the reversal block on reversal trials).
    reversal_prob
        Per-trial probability of a reversal block (flipped outcomes → the rewarded cue flips).
    reward_correct, penalty_wrong
        Response-phase reward for a matching / non-matching action.
    rng
        The environment RNG, so per-trial sampling is reproducible under the environment seed.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        trials_per_episode: int,
        cond_steps_per_cue: int,
        delay_steps: int,
        response_steps: int,
        reversal_prob: float,
        reward_correct: float,
        penalty_wrong: float,
        rng: np.random.Generator,
    ) -> None:
        self._trials_per_episode = trials_per_episode
        self._cond_steps_per_cue = cond_steps_per_cue
        self._delay_steps = delay_steps
        self._response_steps = response_steps
        self._reversal_prob = reversal_prob
        self._reward_correct = reward_correct
        self._penalty_wrong = penalty_wrong
        self._rng = rng
        self.reset()

    @property
    def _cond_len(self) -> int:
        """Steps in the conditioning (or reversal) block: both cues, ``cond_steps_per_cue`` each."""
        return 2 * self._cond_steps_per_cue

    def _trial_span(self) -> int:
        """Return the *current* trial's step count (adds the reversal block on reversal trials)."""
        rev = self._cond_len if self._reversed else 0
        return self._cond_len + rev + self._delay_steps + self._response_steps

    def rebind_rng(self, rng: np.random.Generator) -> None:
        """Rebind the RNG — used when the environment is recreated between runs."""
        self._rng = rng

    def reset(self) -> None:
        """Reset to the first trial's conditioning phase and clear records (per-episode)."""
        self._trial = 0
        self._step_in_trial = 0
        self._pending_reward: float | None = None
        self._responses: list[bool] = []
        self._response_was_reversal: list[bool] = []
        self._sample_trial()

    def _sample_trial(self) -> None:
        # Rewarded cue, presentation order, and reversal all sampled per trial from the env RNG.
        half = 0.5
        self._initial_rewarded = _CUE_A if self._rng.random() < half else _CUE_B
        self._order = [_CUE_A, _CUE_B] if self._rng.random() < half else [_CUE_B, _CUE_A]
        self._reversed = self._rng.random() < self._reversal_prob
        self._current_rewarded = (
            -self._initial_rewarded if self._reversed else self._initial_rewarded
        )

    @property
    def phase(self) -> AssociativeMemoryPhase:
        """The current phase, derived from the step index within the trial."""
        step = self._step_in_trial
        cond_end = self._cond_len
        rev_end = cond_end + (self._cond_len if self._reversed else 0)
        delay_end = rev_end + self._delay_steps
        if step < cond_end:
            return AssociativeMemoryPhase.CONDITIONING
        if step < rev_end:
            return AssociativeMemoryPhase.REVERSAL
        if step < delay_end:
            return AssociativeMemoryPhase.DELAY
        return AssociativeMemoryPhase.RESPONSE

    @property
    def current_rewarded_cue(self) -> float:
        """The current trial's rewarded cue-identity (post-reversal if reversed)."""
        return self._current_rewarded

    @property
    def done(self) -> bool:
        """True once every trial has completed."""
        return self._trial >= self._trials_per_episode

    def signals(self) -> tuple[float, float, float]:
        """Return ``(cue_signal, outcome_signal, go_signal)`` for the current step.

        Cue-identity + outcome are present only during the conditioning and reversal blocks
        (0 during delay/response, so a memoryless policy cannot recover the association); the
        outcome is flipped during the reversal block. The go-signal is 1 only during the
        response phase. When the task is done, all are 0.
        """
        if self.done:
            return 0.0, 0.0, 0.0
        phase = self.phase
        if phase in (AssociativeMemoryPhase.CONDITIONING, AssociativeMemoryPhase.REVERSAL):
            in_block = self._step_in_trial
            if phase == AssociativeMemoryPhase.REVERSAL:
                in_block -= self._cond_len
            presented = self._order[in_block // self._cond_steps_per_cue]
            # Outcome +1 for the rewarded cue of *this* block, -1 for the other. In the reversal
            # block the rewarded cue is flipped, so the outcomes are flipped vs conditioning.
            rewarded = (
                self._current_rewarded
                if phase == AssociativeMemoryPhase.REVERSAL
                else self._initial_rewarded
            )
            outcome = 1.0 if presented == rewarded else -1.0
            return presented, outcome, 0.0
        go = 1.0 if phase == AssociativeMemoryPhase.RESPONSE else 0.0
        return 0.0, 0.0, go

    def record_response(self, turn: float) -> None:
        """Score the agent's action against the current rewarded cue, if this is a response step.

        The binary response is ``sign(turn)`` (continuous turn component, or a discrete
        left/right mapped to -1/+1), interpreted as the remembered rewarded cue-identity. On a
        response step the score is queued as the next ``take_reward`` and the correctness recorded
        (with whether the trial reversed, for the reversal/non-reversal split). Outside the
        response phase this is a no-op.
        """
        if self.done or self.phase != AssociativeMemoryPhase.RESPONSE:
            return
        chosen = _CUE_A if turn >= 0.0 else _CUE_B
        correct = chosen == self._current_rewarded
        self._pending_reward = self._reward_correct if correct else -self._penalty_wrong
        self._responses.append(correct)
        self._response_was_reversal.append(self._reversed)

    def take_reward(self) -> float:
        """Pop the pending response score (0.0 if none) — the reward for the prior response."""
        reward = self._pending_reward if self._pending_reward is not None else 0.0
        self._pending_reward = None
        return reward

    def advance(self) -> None:
        """Advance one step; roll to the next trial (new association) at the trial boundary."""
        self._step_in_trial += 1
        if self._step_in_trial >= self._trial_span():
            self._trial += 1
            self._step_in_trial = 0
            self._sample_trial()

    @property
    def response_accuracy(self) -> float:
        """Fraction of scored responses matching the current rewarded cue (chance = 0.5)."""
        if not self._responses:
            return 0.0
        return sum(self._responses) / len(self._responses)

    def _split_accuracy(self, *, reversal: bool) -> float:
        vals = [
            c
            for c, r in zip(self._responses, self._response_was_reversal, strict=True)
            if r == reversal
        ]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def reversal_accuracy(self) -> float:
        """Accuracy on the reversal-trial responses (the update demand; hold-only → chance)."""
        return self._split_accuracy(reversal=True)

    @property
    def non_reversal_accuracy(self) -> float:
        """Accuracy on the non-reversal-trial responses."""
        return self._split_accuracy(reversal=False)

    @property
    def num_responses(self) -> int:
        """Number of scored response steps so far."""
        return len(self._responses)
