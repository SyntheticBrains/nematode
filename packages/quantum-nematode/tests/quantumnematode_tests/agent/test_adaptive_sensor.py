"""Unit tests for the adaptive-threshold / biphasic chemosensory sensor (Rung-2)."""

from __future__ import annotations

import math

import pytest
from quantumnematode.agent.adaptive_sensor import (
    READOUT_CONTRAST,
    READOUT_FOLD_CHANGE,
    READOUT_LOG,
    AdaptiveChemosensor,
    LeakyIntegrator,
    fold_change_readout,
    log_concentration_readout,
    magnitude_contrast_readout,
)


class TestLeakyIntegrator:
    def test_seeds_on_first_sample(self) -> None:
        li = LeakyIntegrator(alpha=0.1)
        assert li.background == 0.0  # before any sample
        assert li.update(5.0) == 5.0  # first sample seeds the background exactly
        assert li.background == 5.0

    def test_tracks_toward_sustained_value(self) -> None:
        li = LeakyIntegrator(alpha=0.5)
        li.update(0.0)  # seed at 0
        for _ in range(50):
            li.update(1.0)
        assert li.background == pytest.approx(1.0, abs=1e-6)  # converges to the input

    def test_decay_formula(self) -> None:
        li = LeakyIntegrator(alpha=0.25)
        li.update(0.0)  # B=0
        b = li.update(1.0)  # B = 0.75*0 + 0.25*1
        assert b == pytest.approx(0.25)

    def test_reset_clears(self) -> None:
        li = LeakyIntegrator(alpha=0.1)
        li.update(3.0)
        li.reset()
        assert li.background == 0.0
        assert li.update(7.0) == 7.0  # re-seeds


class TestReadouts:
    def test_fold_change_regularised_at_zero(self) -> None:
        # Raw (dC/dt)/C would divide by zero; (dC/dt)/(C+eps) stays finite.
        val = fold_change_readout(derivative=0.5, concentration=0.0, eps=1e-3)
        assert math.isfinite(val)
        assert val == pytest.approx(0.5 / 1e-3)

    def test_fold_change_is_relative_to_concentration(self) -> None:
        # Same derivative at a higher concentration yields a smaller fold-change.
        low = fold_change_readout(0.1, concentration=0.1, eps=1e-3)
        high = fold_change_readout(0.1, concentration=1.0, eps=1e-3)
        assert high < low

    def test_magnitude_contrast_sign(self) -> None:
        assert magnitude_contrast_readout(1.0, background=0.2, eps=1e-3) > 0  # rising
        assert magnitude_contrast_readout(0.2, background=1.0, eps=1e-3) < 0  # falling
        assert magnitude_contrast_readout(0.5, background=0.5, eps=1e-3) == pytest.approx(
            0.0,
            abs=1e-3,
        )

    def test_magnitude_contrast_bounded(self) -> None:
        v = magnitude_contrast_readout(1.0, background=0.0, eps=1e-3)
        assert -1.0 <= v <= 1.0

    def test_log_readout(self) -> None:
        assert log_concentration_readout(0.0) == 0.0
        assert log_concentration_readout(math.e - 1.0) == pytest.approx(1.0)


class TestAdaptiveChemosensor:
    def test_fold_change_reshapes_derivative_only(self) -> None:
        s = AdaptiveChemosensor(readout=READOUT_FOLD_CHANGE, alpha=0.1, eps=1e-3)
        strength, derivative = s.adapt(concentration=0.5, derivative=0.2)
        assert strength == 0.5  # magnitude channel untouched
        assert derivative == pytest.approx(0.2 / (0.5 + 1e-3))  # derivative reshaped

    def test_contrast_reshapes_strength_only(self) -> None:
        s = AdaptiveChemosensor(readout=READOUT_CONTRAST, alpha=1.0, eps=1e-3)
        # alpha=1.0 → background equals the current sample, so contrast ~ 0 on first call.
        s.adapt(concentration=0.2, derivative=0.9)  # seed background at 0.2
        strength, derivative = s.adapt(concentration=1.0, derivative=0.9)
        assert derivative == 0.9  # derivative channel untouched
        # background after seed(0.2)+update(1.0) at alpha=1.0 is 1.0 → contrast ~ 0
        assert strength == pytest.approx(0.0, abs=1e-2)

    def test_log_reshapes_strength(self) -> None:
        s = AdaptiveChemosensor(readout=READOUT_LOG)
        strength, derivative = s.adapt(concentration=math.e - 1.0, derivative=0.3)
        assert strength == pytest.approx(1.0)
        assert derivative == 0.3

    def test_reset_clears_background(self) -> None:
        s = AdaptiveChemosensor(readout=READOUT_CONTRAST, alpha=0.5)
        s.adapt(concentration=1.0, derivative=0.0)
        s.reset()
        # After reset the next sample re-seeds → contrast against itself ~ 0.
        strength, _ = s.adapt(concentration=0.8, derivative=0.0)
        assert strength == pytest.approx(0.0, abs=1e-2)

    def test_unknown_readout_clears_last_readout(self) -> None:
        # Defensive no-op path: an unknown readout must not leak a stale cached value.
        s = AdaptiveChemosensor(readout=READOUT_CONTRAST, alpha=0.5)
        s.adapt(concentration=0.8, derivative=0.1)
        assert s.last_readout is not None
        s.readout = "bogus"
        strength, derivative = s.adapt(concentration=0.5, derivative=0.2)
        assert (strength, derivative) == (0.5, 0.2)  # passthrough
        assert s.last_readout is None


class TestAgentWiring:
    """The sensing config plumbs the adaptive sensor onto the agent (or not)."""

    def _brain(self):
        from quantumnematode.brain.arch.qvarcircuit import (
            QVarCircuitBrain,
            QVarCircuitBrainConfig,
        )
        from quantumnematode.brain.modules import ModuleName

        return QVarCircuitBrain(
            config=QVarCircuitBrainConfig(
                modules={ModuleName.FOOD_CHEMOTAXIS: [0, 1]},
                num_layers=1,
            ),
            shots=50,
        )

    def test_disabled_by_default_no_sensor(self) -> None:
        from quantumnematode.agent import QuantumNematodeAgent

        agent = QuantumNematodeAgent(brain=self._brain())
        assert agent._adaptive_food is None

    def test_enabled_constructs_sensor_with_config(self) -> None:
        from quantumnematode.agent import QuantumNematodeAgent
        from quantumnematode.utils.config_loader import SensingConfig

        agent = QuantumNematodeAgent(
            brain=self._brain(),
            sensing_config=SensingConfig(
                adaptive_chemosensor_enabled=True,
                adaptive_chemosensor_readout=READOUT_CONTRAST,
                adaptive_chemosensor_alpha=0.2,
                adaptive_chemosensor_epsilon=1e-2,
            ),
        )
        sensor = agent._adaptive_food
        assert isinstance(sensor, AdaptiveChemosensor)
        assert sensor.readout == READOUT_CONTRAST
        assert sensor.alpha == 0.2
        assert sensor.eps == 1e-2
