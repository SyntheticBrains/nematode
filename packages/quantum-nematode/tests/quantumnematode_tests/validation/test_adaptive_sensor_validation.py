"""Tests for the step-input adaptation-transient validation (the T6 acceptance gate)."""

from __future__ import annotations

from quantumnematode.agent.adaptive_sensor import (
    READOUT_CONTRAST,
    READOUT_FOLD_CHANGE,
    READOUT_LOG,
)
from quantumnematode.validation.adaptive_sensor import (
    run_step_input,
    weber_invariance_peaks,
    weber_spread,
)


class TestAdaptationTransient:
    def test_contrast_peaks_then_relaxes(self) -> None:
        """The contrast readout spikes on the step then relaxes back to baseline."""
        res = run_step_input(READOUT_CONTRAST, background=0.1, step=0.2, alpha=0.1)
        assert res.peak > 0.2  # a clear response on the step
        assert res.relaxation_ratio < 0.1  # relaxes back ~to baseline (adaptation)

    def test_fold_change_response_is_transient(self) -> None:
        """Fold-change gives a derivative impulse on the step, then decays."""
        res = run_step_input(READOUT_FOLD_CHANGE, background=0.1, step=0.2, alpha=0.1)
        assert res.peak > 0.0
        assert res.relaxation_ratio < 0.2  # impulse decays after the step

    def test_log_baseline_does_not_adapt(self) -> None:
        """The log baseline is a sustained level shift — it does NOT relax (no adaptation)."""
        res = run_step_input(READOUT_LOG, background=0.1, step=0.2, alpha=0.1)
        assert res.relaxation_ratio > 0.9  # sustained, no relaxation


class TestWeberInvariance:
    def test_contrast_is_background_invariant(self) -> None:
        """Same relative step at different backgrounds → ~constant contrast peak (Weber)."""
        peaks = weber_invariance_peaks(READOUT_CONTRAST, relative_step=2.0)
        assert weber_spread(peaks) < 0.05  # tightly invariant

    def test_log_baseline_is_not_invariant(self) -> None:
        """The log baseline's peak grows with the absolute background (not Weber)."""
        peaks = weber_invariance_peaks(READOUT_LOG, relative_step=2.0)
        assert weber_spread(peaks) > 0.5  # strongly background-dependent

    def test_adaptive_more_invariant_than_baseline(self) -> None:
        """The load-bearing comparison: adaptive coding beats the log baseline on Weber."""
        adaptive = weber_spread(weber_invariance_peaks(READOUT_CONTRAST, relative_step=2.0))
        baseline = weber_spread(weber_invariance_peaks(READOUT_LOG, relative_step=2.0))
        assert adaptive < baseline
