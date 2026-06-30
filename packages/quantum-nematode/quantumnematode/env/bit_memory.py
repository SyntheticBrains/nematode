"""Bit-memory delayed-match-to-cue positive-control phase machine.

A deliberately-artificial working-memory probe. Per trial a binary cue is presented during
a cue phase, withheld across a delay phase, then on a go-signalled response phase the agent
must act on the *remembered* cue. A memoryless policy is pinned at chance; a
recurrent/attention policy can solve it.

This module is pure phase-machine logic (no env/agent imports) so it is unit-testable in
isolation: the environment owns an instance and exposes its cue/go signals; the episode
runner drives ``record_response`` / ``take_reward`` / ``advance``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class BitMemoryPhase(StrEnum):
    """The three phases of a bit-memory trial."""

    CUE = "cue"
    DELAY = "delay"
    RESPONSE = "response"


class BitMemoryTask:
    """Stateful phase machine for the bit-memory delayed-match-to-cue task.

    The phase is *derived* from the step index within the trial (rather than tracked via
    entry/exit transitions), so a zero-length delay correctly collapses to no delay.

    Parameters
    ----------
    trials_per_episode, cue_steps, delay_steps, response_steps
        Trial structure (steps per phase). ``cue_steps``/``response_steps`` are >= 1;
        ``delay_steps`` may be 0.
    reward_correct, penalty_wrong
        Response-phase reward for a matching / non-matching action.
    rng
        The environment RNG (``numpy`` Generator) used to sample the per-trial cue, so
        runs are reproducible under the environment seed.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        trials_per_episode: int,
        cue_steps: int,
        delay_steps: int,
        response_steps: int,
        reward_correct: float,
        penalty_wrong: float,
        rng: np.random.Generator,
    ) -> None:
        self._trials_per_episode = trials_per_episode
        self._cue_steps = cue_steps
        self._delay_steps = delay_steps
        self._response_steps = response_steps
        self._reward_correct = reward_correct
        self._penalty_wrong = penalty_wrong
        self._rng = rng
        self.reset()

    @property
    def trial_span(self) -> int:
        """Steps per trial (cue + delay + response)."""
        return self._cue_steps + self._delay_steps + self._response_steps

    def rebind_rng(self, rng: np.random.Generator) -> None:
        """Rebind the cue RNG — used when the environment is recreated between runs.

        The next ``reset`` (at episode start) samples the first cue from the rebound RNG, so
        per-run seeding flows through to the cue sequence.
        """
        self._rng = rng

    def reset(self) -> None:
        """Reset to the first trial's cue phase and clear records (per-episode boundary)."""
        self._trial = 0
        self._step_in_trial = 0
        self._pending_reward: float | None = None
        self._responses: list[bool] = []
        self._sample_cue()

    def _sample_cue(self) -> None:
        # Binary cue encoded as -1 / +1 so that 0 unambiguously means "no cue"
        # (the withheld value during delay/response is distinct from both cue values).
        self._cue = 1.0 if self._rng.random() < 0.5 else -1.0  # noqa: PLR2004

    @property
    def phase(self) -> BitMemoryPhase:
        """The current phase, derived from the step index within the trial."""
        if self._step_in_trial < self._cue_steps:
            return BitMemoryPhase.CUE
        if self._step_in_trial < self._cue_steps + self._delay_steps:
            return BitMemoryPhase.DELAY
        return BitMemoryPhase.RESPONSE

    @property
    def current_cue(self) -> float:
        """The current trial's cue value (-1 / +1)."""
        return self._cue

    @property
    def done(self) -> bool:
        """True once every trial has completed."""
        return self._trial >= self._trials_per_episode

    def signals(self) -> tuple[float, float]:
        """Return ``(cue_signal, go_signal)`` for the current step's observation.

        The cue is present only during the cue phase (0 during delay/response, so a
        memoryless policy cannot recover it); the go-signal is 1 only during the response
        phase. When the task is done, both are 0.
        """
        if self.done:
            return 0.0, 0.0
        cue = self._cue if self.phase == BitMemoryPhase.CUE else 0.0
        go = 1.0 if self.phase == BitMemoryPhase.RESPONSE else 0.0
        return cue, go

    def record_response(self, turn: float) -> None:
        """Score the agent's action against the cue, if this is a response step.

        The binary response is ``sign(turn)`` (the continuous turn component, or a discrete
        left/right mapped to -1/+1). On a response step the score is queued as the next
        ``take_reward`` value and the correctness recorded for the per-episode metric.
        Outside the response phase this is a no-op.
        """
        if self.done or self.phase != BitMemoryPhase.RESPONSE:
            return
        chosen = 1.0 if turn >= 0.0 else -1.0
        correct = chosen == self._cue
        self._pending_reward = self._reward_correct if correct else -self._penalty_wrong
        self._responses.append(correct)

    def take_reward(self) -> float:
        """Pop the pending response score (0.0 if none) — the reward for the prior response.

        Consumed at the top of the runner step and fed to ``run_brain`` (which treats it as
        the reward for the previous action), mirroring the foraging reward timing.
        """
        reward = self._pending_reward if self._pending_reward is not None else 0.0
        self._pending_reward = None
        return reward

    def advance(self) -> None:
        """Advance one step; roll to the next trial (new cue) at the trial boundary."""
        self._step_in_trial += 1
        if self._step_in_trial >= self.trial_span:
            self._trial += 1
            self._step_in_trial = 0
            self._sample_cue()

    @property
    def cue_match_success_rate(self) -> float:
        """Fraction of scored response steps where the action matched the cue (chance = 0.5)."""
        if not self._responses:
            return 0.0
        return sum(self._responses) / len(self._responses)

    @property
    def num_responses(self) -> int:
        """Number of scored response steps so far."""
        return len(self._responses)
