"""A control that can measure the eligibility horizon.

The failure these pin is an instrument that cannot see what it was built to measure. The control
is one step by design -- it resets the trace every trial, which is what removes the horizon
confound from the undelayed question -- so the horizon needs a delay. And a delay whose
intervening steps add nothing to the trace measures nothing either: the trace at reward time is
then a scalar multiple of the credited step's, and a rule normalising its trace by a running RMS
divides exactly that out. So the filler must drive the plastic layer, and the quantity under test
is the credited step's *share*, not its magnitude.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from quantumnematode.plasticity.positive_control import ContextualAssociation

_NOISE = float(np.exp(-1.0))


@pytest.fixture
def task() -> ContextualAssociation:
    """Build the control's default task."""
    return ContextualAssociation.default()


class TestTheFiller:
    def test_it_is_nonzero(self, task: ContextualAssociation) -> None:
        # A zero filler adds nothing to the trace, and a pure scalar decay is divided out by the
        # rule's trace normalisation -- the delay would then measure nothing at any length.
        assert float(np.abs(task.filler()).sum()) > 0.0

    def test_it_carries_nothing_about_the_cue(self, task: ContextualAssociation) -> None:
        # Identical on every trial, and flat across channels, so no cue is distinguishable in it.
        assert len(set(task.filler().tolist())) == 1
        assert np.array_equal(task.filler(), task.filler())

    def test_it_keeps_the_observation_width(self, task: ContextualAssociation) -> None:
        # A separate "no cue" channel would widen the input and change the network's
        # initialisation, so a delay of zero would no longer be the undelayed control.
        assert task.filler().shape == task.observation(0).shape

    def test_it_is_distinguishable_from_every_cue(self, task: ContextualAssociation) -> None:
        for cue in range(task.n_cues):
            assert not np.array_equal(task.filler(), task.observation(cue))


class TestTheBoundsDoNotMoveWithTheDelay:
    def test_the_floor_and_optimum_depend_only_on_targets_and_noise(
        self,
        task: ContextualAssociation,
    ) -> None:
        # There is no delay argument to either, which is the point: a delayed arm is scored
        # against the same bounds and is directly comparable with the committed one.
        assert task.cue_blind_floor(_NOISE) == pytest.approx(
            -float(np.var(task.targets)) - _NOISE**2,
        )
        assert task.optimum(_NOISE) == pytest.approx(-(_NOISE**2))
        assert task.gap(_NOISE) == pytest.approx(float(np.var(task.targets)))


class TestTheNominalCreditRatio:
    def test_zero_delay_credits_the_scored_step_entirely(
        self,
        task: ContextualAssociation,
    ) -> None:
        assert task.nominal_credit_ratio(0.9, 0) == pytest.approx(1.0)

    def test_it_falls_with_the_delay(self, task: ContextualAssociation) -> None:
        shares = [task.nominal_credit_ratio(0.9, d) for d in (0, 2, 5, 10, 20)]
        assert all(b < a for a, b in pairwise(shares))

    def test_a_longer_horizon_credits_the_scored_step_more(
        self,
        task: ContextualAssociation,
    ) -> None:
        # What the horizon grid asks: does raising trace_decay recover the nominal credit ratio?
        assert task.nominal_credit_ratio(0.999, 10) > task.nominal_credit_ratio(0.9, 10)

    def test_it_is_a_share_not_a_magnitude(self, task: ContextualAssociation) -> None:
        # The quantity that survives trace normalisation. A pure decay would be divided out; the
        # share is what a delay actually changes, so it is what the control measures.
        for delay in (0, 2, 5, 10):
            assert 0.0 < task.nominal_credit_ratio(0.9, delay) <= 1.0

    def test_a_negative_delay_is_refused(self, task: ContextualAssociation) -> None:
        # Unguarded it returns 1.0, which reads as "the scored step keeps everything" — the
        # opposite of what a caller passing a negative delay could possibly mean.
        with pytest.raises(ValueError, match="non-negative"):
            task.nominal_credit_ratio(0.9, -1)

    def test_it_is_nominal_not_a_measured_tensor_share(
        self,
        task: ContextualAssociation,
    ) -> None:
        # It weights every step equally and asks only what the decay does. With no decay at all
        # it is exactly 1/(delay+1), which no measurement of outer-product norms would be.
        assert task.nominal_credit_ratio(1.0, 9) == pytest.approx(0.1)

    def test_a_perfectly_retentive_trace_still_dilutes(
        self,
        task: ContextualAssociation,
    ) -> None:
        # Even with no decay at all the credited step is one term among delay + 1, so dilution is
        # a property of the delay rather than of the decay. This is why the control works.
        assert task.nominal_credit_ratio(1.0, 3) == pytest.approx(1 / 4)
