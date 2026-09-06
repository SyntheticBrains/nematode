"""The three-factor rule's substrate-invariant scaling.

Two opt-in switches: a bounded, scale-free modulator ``tanh(delta / sigma)``
and a per-tensor trace normalisation ``E / rho``, each driven by a
bias-corrected running RMS. Off, the rule must not move a bit (the frozen
reference test proves that separately); on, the claims pinned here are the
ones a matched-rule comparison across substrates rests on.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, cast

import pytest
import torch
from pydantic import ValidationError
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch._mlp_topology import MLPTopology
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
    ConnectomeTopology,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.mlpppo import MLPPPOBrainConfig
from quantumnematode.brain.modules import ModuleName
from quantumnematode.learning_rules import ScalingOptions, ThreeFactorRule
from quantumnematode.learning_rules.three_factor import (
    MODULATOR_CENTRE_KEY,
    MODULATOR_KEY,
    MODULATOR_SCALE_KEY,
    PREDICTION_ERROR_KEY,
    TRACE_SCALE_KEY,
    ThreeFactorBatch,
    record_plasticity_report,
)
from torch import nn

if TYPE_CHECKING:
    from quantumnematode.brain.arch._rule import RuleStepReport
    from quantumnematode.brain.arch._topology import BrainTopology

_SEED = 7331
_ETA = 0.05
_FLOOR = 1e-6
_RATE = 0.1


def _topology(trace_scale: float = 1.0) -> ConnectomeTopology:
    brain = ConnectomePPOBrain(
        config=ConnectomePPOBrainConfig(
            seed=_SEED,
            action_mode="continuous",
            learning_rule="three_factor",
            enable_activity_traces=True,
        ),
        device=DeviceType.CPU,
    )
    topo = brain.topology
    torch.manual_seed(_SEED + 1)
    with torch.no_grad():
        topo.activity_traces.copy_(
            trace_scale * topo.apply_weight_mask(torch.randn_like(topo.activity_traces)),
        )
    return topo


def _mlp_topology(trace_scale: float = 1.0) -> MLPTopology:
    torch.manual_seed(_SEED + 2)
    actor = nn.Sequential(
        nn.Linear(6, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )
    topo = MLPTopology(actor, enable_activity_traces=True, trace_decay=0.9)
    with torch.no_grad():
        for trace in topo.eligibility_traces:
            trace.copy_(trace_scale * torch.randn_like(trace))
    return topo


def _rule(topology: ConnectomeTopology | MLPTopology, **overrides: object) -> ThreeFactorRule:
    kwargs: dict[str, object] = {
        "plasticity_rate": _ETA,
        "weight_decay": 0.0,
        "weight_bound": 100.0,
        "baseline_rate": 0.0,  # delta == reward, so tests can name the prediction error directly
        "freeze_updates": False,
        "modulated": True,
    }
    kwargs.update(overrides)
    return ThreeFactorRule(topology, device=torch.device("cpu"), **kwargs)  # type: ignore[arg-type]


def _step(rule: ThreeFactorRule, reward: float) -> RuleStepReport:
    """Step a rule over the topology it was built on."""
    return rule.step(cast("BrainTopology", rule._topology), ThreeFactorBatch(reward=reward))


def _scaling(**overrides: object) -> ScalingOptions:
    kwargs: dict[str, object] = {"scale_rate": _RATE, "scale_floor": _FLOOR}
    kwargs.update(overrides)
    return ScalingOptions(**kwargs)  # type: ignore[arg-type]


def _corrected_rms(observations: list[float], rate: float) -> float:
    """Compute an independent bias-corrected running RMS, for checking the rule's."""
    ema = 0.0
    for value in observations:
        ema = (1.0 - rate) * ema + rate * value * value
    return math.sqrt(ema / (1.0 - (1.0 - rate) ** len(observations)))


def _corrected_mean(observations: list[float], rate: float) -> float:
    """Compute an independent bias-corrected running mean from a zero prior."""
    if not observations:
        return 0.0
    ema = 0.0
    for value in observations:
        ema = (1.0 - rate) * ema + rate * value
    return ema / (1.0 - (1.0 - rate) ** len(observations))


def _masked_rms(trace: torch.Tensor, mask: torch.Tensor) -> float:
    return float(trace[mask.to(torch.bool)].square().mean().sqrt().item())


class TestDefaultsAreOff:
    def test_options_default_off(self) -> None:
        assert ScalingOptions() == ScalingOptions(
            normalise_modulator=False,
            normalise_trace=False,
            scale_rate=0.01,
            scale_floor=1e-6,
        )

    @pytest.mark.parametrize(
        "bad",
        [{"scale_rate": 0.0}, {"scale_rate": 1.5}, {"scale_floor": 0.0}, {"scale_floor": -1e-3}],
    )
    def test_direct_construction_is_bounded_too(self, bad: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="scale_"):
            ScalingOptions(**bad)  # type: ignore[arg-type]

    def test_default_config_builds_a_rule_with_both_switches_off(self) -> None:
        brain = ConnectomePPOBrain(
            config=ConnectomePPOBrainConfig(
                seed=_SEED,
                action_mode="continuous",
                learning_rule="three_factor",
                enable_activity_traces=True,
            ),
            device=DeviceType.CPU,
        )
        assert isinstance(brain._rule, ThreeFactorRule)
        assert brain._rule.scaling == ScalingOptions()

    def test_off_reports_nan_scales_and_touches_no_estimator(self) -> None:
        rule = _rule(_topology())
        report = _step(rule, 1.5)
        assert math.isnan(report.extra[MODULATOR_SCALE_KEY])
        assert math.isnan(report.extra[TRACE_SCALE_KEY])
        assert math.isnan(report.extra[MODULATOR_CENTRE_KEY])
        assert rule.modulator_centre.count == 0
        assert report.extra[MODULATOR_KEY] == report.extra[PREDICTION_ERROR_KEY]
        assert rule.modulator_scale.count == 0
        assert all(scale.count == 0 for scale in rule.trace_scales)


class TestBoundedModulator:
    def test_first_step_scale_is_the_first_error_and_the_modulator_is_tanh_of_one(self) -> None:
        rule = _rule(_topology(), scaling=_scaling(normalise_modulator=True))
        report = _step(rule, -10.0)
        assert report.extra[PREDICTION_ERROR_KEY] == -10.0
        assert report.extra[MODULATOR_SCALE_KEY] == pytest.approx(10.0)
        assert report.extra[MODULATOR_KEY] == pytest.approx(math.tanh(-1.0))
        assert report.extra[MODULATOR_CENTRE_KEY] == 0.0  # zero prior: the first step is uncentred

    def test_modulator_is_tanh_of_delta_over_the_pre_update_scale(self) -> None:
        rewards = [0.05, -0.02, -10.0, 0.04, 0.5]
        rule = _rule(_topology(), scaling=_scaling(normalise_modulator=True))
        compressed: list[float] = []
        for index, reward in enumerate(rewards):
            report = _step(rule, reward)
            expected_scale = (
                abs(rewards[0]) if index == 0 else _corrected_rms(rewards[:index], _RATE)
            )
            expected_scale = max(expected_scale, _FLOOR)
            assert report.extra[MODULATOR_SCALE_KEY] == pytest.approx(expected_scale)
            compressed.append(math.tanh(reward / expected_scale))
            expected_centre = _corrected_mean(compressed[:-1], _RATE)
            assert report.extra[MODULATOR_CENTRE_KEY] == pytest.approx(expected_centre)
            assert report.extra[MODULATOR_KEY] == pytest.approx(compressed[-1] - expected_centre)
            assert -2.0 <= report.extra[MODULATOR_KEY] <= 2.0

    def test_modulator_is_bounded_however_large_the_error(self) -> None:
        rule = _rule(_topology(), scaling=_scaling(normalise_modulator=True))
        _step(rule, 0.01)
        report = _step(rule, -1e6)
        # Compressed to -1, then centred by the first step's tanh(1).
        assert report.extra[MODULATOR_KEY] == pytest.approx(-1.0 - math.tanh(1.0))
        assert -2.0 <= report.extra[MODULATOR_KEY] <= 2.0

    def test_floor_applies_to_a_zero_first_error(self) -> None:
        rule = _rule(_topology(), scaling=_scaling(normalise_modulator=True))
        report = _step(rule, 0.0)
        assert report.extra[MODULATOR_SCALE_KEY] == _FLOOR
        assert report.extra[MODULATOR_KEY] == 0.0

    def test_raw_delta_is_still_reported_and_the_baseline_is_untouched(self) -> None:
        rule = _rule(_topology(), baseline_rate=0.5, scaling=_scaling(normalise_modulator=True))
        report = _step(rule, 2.0)
        assert report.extra[PREDICTION_ERROR_KEY] == 2.0
        assert rule.baseline == pytest.approx(1.0)


class TestCentring:
    """The centred modulator is zero-mean where the bare compression is not."""

    _PERIOD = 203
    _FOODS = 3
    _RATE_SLOW = 0.01

    def _period(self) -> list[float]:
        # Three +2 foods, one -10 death, and small steps sized so the raw period sums to zero.
        small = (10.0 - 2.0 * self._FOODS) / (self._PERIOD - self._FOODS - 1)
        rewards = [small] * self._PERIOD
        for position in (40, 110, 170):
            rewards[position] = 2.0
        rewards[self._PERIOD - 1] = -10.0
        assert abs(sum(rewards)) < 1e-9
        return rewards

    def test_zero_mean_on_a_periodic_skewed_stream(self) -> None:
        rule = _rule(
            _topology(),
            freeze_updates=True,  # the modulator is what is under test; skip the weight writes
            scaling=_scaling(normalise_modulator=True, scale_rate=self._RATE_SLOW),
        )
        period = self._period()
        warm_up, measured = 30, 10
        for _ in range(warm_up):
            for reward in period:
                _step(rule, reward)
        centred, uncentred = [], []
        for _ in range(measured):
            for reward in period:
                report = _step(rule, reward)
                centred.append(report.extra[MODULATOR_KEY])
                uncentred.append(report.extra[MODULATOR_KEY] + report.extra[MODULATOR_CENTRE_KEY])
        assert abs(sum(centred) / len(centred)) < 0.005
        assert abs(sum(uncentred) / len(uncentred)) > 0.005  # the centring is load-bearing
        assert sum(uncentred) > 0  # and the bias is the positive drift the probe showed

    def test_frozen_arm_advances_the_centre(self) -> None:
        topo = _topology()
        before = topo.w_chem.detach().clone()
        rule = _rule(topo, freeze_updates=True, scaling=_scaling(normalise_modulator=True))
        _step(rule, 1.0)
        report = _step(rule, 2.0)
        assert torch.equal(topo.w_chem, before)
        assert rule.modulator_centre.count == 2
        assert report.extra[MODULATOR_CENTRE_KEY] == pytest.approx(math.tanh(1.0))

    def test_unmodulated_arm_reports_the_centre_but_uses_one(self) -> None:
        rule = _rule(_topology(), modulated=False, scaling=_scaling(normalise_modulator=True))
        _step(rule, 1.0)
        report = _step(rule, -3.0)
        assert report.extra[MODULATOR_KEY] == 1.0
        assert report.extra[MODULATOR_CENTRE_KEY] == pytest.approx(math.tanh(1.0))
        assert rule.modulator_centre.count == 2

    def test_centre_is_recorded(self) -> None:
        rule = _rule(_topology(), scaling=_scaling(normalise_modulator=True))
        report = _step(rule, 0.3)
        history = BrainHistoryData()
        record_plasticity_report(history, report, reward=0.3)
        assert history.plasticity_modulator_centre == [report.extra[MODULATOR_CENTRE_KEY]]


class TestTraceNormalisation:
    def test_first_step_divides_by_the_masked_rms_of_the_trace(self) -> None:
        topo = _topology()
        rho = _masked_rms(topo.activity_traces, topo.m_chem)
        before = topo.w_chem.detach().clone()
        trace = topo.activity_traces.detach().clone()
        rule = _rule(topo, modulated=False, scaling=_scaling(normalise_trace=True))
        report = rule.step(topo, ThreeFactorBatch(reward=0.0))
        assert report.extra[TRACE_SCALE_KEY] == pytest.approx(rho)
        expected = before + _ETA * trace / rho
        assert torch.allclose(topo.w_chem, expected)

    def test_hebbian_step_is_invariant_to_the_trace_scale(self) -> None:
        small, large = _topology(trace_scale=1.0), _topology(trace_scale=1000.0)
        assert torch.equal(small.w_chem, large.w_chem)
        for topo in (small, large):
            rule = _rule(topo, modulated=False, scaling=_scaling(normalise_trace=True))
            rule.step(topo, ThreeFactorBatch(reward=0.0))
        assert torch.allclose(small.w_chem, large.w_chem, atol=1e-6)

    def test_each_tensor_carries_its_own_scale(self) -> None:
        topo = _mlp_topology()
        with torch.no_grad():
            topo.eligibility_traces[0].mul_(100.0)
        rule = _rule(topo, modulated=False, scaling=_scaling(normalise_trace=True))
        rule.step(topo, ThreeFactorBatch(reward=0.0))
        scales = [scale.current() for scale in rule.trace_scales]
        values = [s for s in scales if s is not None]
        assert len(values) == len(scales)
        assert values[0] > 10 * max(values[1:])
        for trace, scale in zip(topo.eligibility_traces, values, strict=True):
            assert scale == pytest.approx(float(trace.square().mean().sqrt().item()))

    def test_scale_is_over_the_edge_set_only(self) -> None:
        topo = _topology()
        rule = _rule(topo, modulated=False, scaling=_scaling(normalise_trace=True))
        rule.step(topo, ThreeFactorBatch(reward=0.0))
        masked = _masked_rms(topo.activity_traces, topo.m_chem)
        dense = float(topo.activity_traces.square().mean().sqrt().item())
        assert rule.trace_scales[0].current() == pytest.approx(masked)
        assert masked > dense  # the off-edge zeros would have diluted it

    def test_a_zero_trace_neither_updates_the_scale_nor_counts(self) -> None:
        topo = _topology()
        with torch.no_grad():
            topo.activity_traces.zero_()
        rule = _rule(topo, modulated=False, scaling=_scaling(normalise_trace=True))
        report = rule.step(topo, ThreeFactorBatch(reward=0.0))
        assert rule.trace_scales[0].count == 0
        assert report.extra[TRACE_SCALE_KEY] == _FLOOR
        # The first non-zero trace then counts fully.
        torch.manual_seed(_SEED + 3)
        with torch.no_grad():
            topo.activity_traces.copy_(
                topo.apply_weight_mask(torch.randn_like(topo.activity_traces)),
            )
        rho = _masked_rms(topo.activity_traces, topo.m_chem)
        report = rule.step(topo, ThreeFactorBatch(reward=0.0))
        assert rule.trace_scales[0].count == 1
        assert report.extra[TRACE_SCALE_KEY] == pytest.approx(rho)

    def test_scale_is_bias_corrected_over_steps(self) -> None:
        topo = _topology()
        rule = _rule(topo, modulated=False, scaling=_scaling(normalise_trace=True))
        observed: list[float] = []
        for step in range(4):
            torch.manual_seed(_SEED + 10 + step)
            with torch.no_grad():
                topo.activity_traces.copy_(
                    (1.0 + step) * topo.apply_weight_mask(torch.randn_like(topo.activity_traces)),
                )
            observed.append(_masked_rms(topo.activity_traces, topo.m_chem))
            rule.step(topo, ThreeFactorBatch(reward=0.0))
        assert rule.trace_scales[0].current() == pytest.approx(_corrected_rms(observed, _RATE))


class TestFreezeAndUnmodulated:
    def test_frozen_arm_advances_the_scales_but_writes_nothing(self) -> None:
        topo = _topology()
        before = topo.w_chem.detach().clone()
        rule = _rule(
            topo,
            freeze_updates=True,
            scaling=_scaling(normalise_modulator=True, normalise_trace=True),
        )
        report = rule.step(topo, ThreeFactorBatch(reward=3.0))
        assert torch.equal(topo.w_chem, before)
        assert rule.modulator_scale.count == 1
        assert rule.trace_scales[0].count == 1
        assert report.extra[MODULATOR_SCALE_KEY] == pytest.approx(3.0)
        assert not math.isnan(report.extra[TRACE_SCALE_KEY])

    def test_unmodulated_modulator_is_one_while_the_scale_is_still_reported(self) -> None:
        topo = _topology()
        rho = _masked_rms(topo.activity_traces, topo.m_chem)
        before = topo.w_chem.detach().clone()
        trace = topo.activity_traces.detach().clone()
        rule = _rule(
            topo,
            modulated=False,
            scaling=_scaling(normalise_modulator=True, normalise_trace=True),
        )
        report = rule.step(topo, ThreeFactorBatch(reward=-10.0))
        assert report.extra[MODULATOR_KEY] == 1.0
        assert report.extra[MODULATOR_SCALE_KEY] == pytest.approx(10.0)
        assert torch.allclose(topo.w_chem, before + _ETA * trace / rho)


class TestTelemetry:
    def test_keys_are_present_and_recorded(self) -> None:
        rule = _rule(_topology(), scaling=_scaling(normalise_modulator=True, normalise_trace=True))
        report = _step(rule, 0.7)
        for key in (MODULATOR_KEY, MODULATOR_SCALE_KEY, TRACE_SCALE_KEY):
            assert key in report.extra
        history = BrainHistoryData()
        record_plasticity_report(history, report, reward=0.7)
        assert history.plasticity_modulator == [report.extra[MODULATOR_KEY]]
        assert history.plasticity_modulator_scale == [report.extra[MODULATOR_SCALE_KEY]]
        assert history.plasticity_trace_scale == [report.extra[TRACE_SCALE_KEY]]
        assert history.rewards == [0.7]


class TestMatchedAcrossSubstrates:
    def test_rms_hebbian_step_per_unit_modulator_is_eta_on_both(self) -> None:
        """Traces three orders of magnitude apart land the same RMS step under normalisation."""
        connectome = _topology(trace_scale=1.0)
        mlp = _mlp_topology(trace_scale=1e-3)
        for topo in (connectome, mlp):
            befores = [w.detach().clone() for w in topo.plastic_weights]
            rule = _rule(topo, modulated=False, scaling=_scaling(normalise_trace=True))
            rule.step(topo, ThreeFactorBatch(reward=0.0))
            steps = [
                (w.detach() - b)[m.to(torch.bool)]
                for w, b, m in zip(topo.plastic_weights, befores, topo.plastic_masks, strict=True)
            ]
            for step in steps:
                assert float(step.square().mean().sqrt().item()) == pytest.approx(_ETA, rel=1e-4)

    def test_without_normalisation_the_same_traces_give_steps_three_orders_apart(self) -> None:
        connectome = _topology(trace_scale=1.0)
        mlp = _mlp_topology(trace_scale=1e-3)
        magnitudes = []
        for topo in (connectome, mlp):
            befores = [w.detach().clone() for w in topo.plastic_weights]
            rule = _rule(topo, modulated=False)
            rule.step(topo, ThreeFactorBatch(reward=0.0))
            deltas = torch.cat(
                [
                    (w.detach() - b)[m.to(torch.bool)]
                    for w, b, m in zip(
                        topo.plastic_weights,
                        befores,
                        topo.plastic_masks,
                        strict=True,
                    )
                ],
            )
            magnitudes.append(float(deltas.square().mean().sqrt().item()))
        assert magnitudes[0] > 100 * magnitudes[1]


class TestConfigFields:
    @pytest.mark.parametrize(
        ("config_cls", "required"),
        [
            (ConnectomePPOBrainConfig, {}),
            (MLPPPOBrainConfig, {"sensory_modules": [ModuleName.FOOD_CHEMOTAXIS]}),
        ],
    )
    def test_defaults_are_shared(self, config_cls: type, required: dict[str, object]) -> None:
        config = config_cls.model_validate(
            {"learning_rule": "three_factor", "enable_activity_traces": True, **required},
        )
        assert config.plasticity_normalise_modulator is False
        assert config.plasticity_normalise_trace is False
        assert config.plasticity_scale_rate == 0.01
        assert config.plasticity_scale_floor == 1e-6

    @pytest.mark.parametrize(
        "bad",
        [
            {"plasticity_scale_rate": 0.0},
            {"plasticity_scale_rate": 1.5},
            {"plasticity_scale_floor": 0.0},
            {"plasticity_scale_floor": -1e-3},
        ],
    )
    def test_bounds_fail_at_load(self, bad: dict[str, float]) -> None:
        with pytest.raises(ValidationError):
            ConnectomePPOBrainConfig.model_validate(
                {"learning_rule": "three_factor", "enable_activity_traces": True, **bad},
            )

    def test_brain_passes_the_fields_to_the_rule(self) -> None:
        brain = ConnectomePPOBrain(
            config=ConnectomePPOBrainConfig(
                seed=_SEED,
                action_mode="continuous",
                learning_rule="three_factor",
                enable_activity_traces=True,
                plasticity_normalise_modulator=True,
                plasticity_normalise_trace=True,
                plasticity_scale_rate=0.2,
                plasticity_scale_floor=1e-3,
            ),
            device=DeviceType.CPU,
        )
        assert isinstance(brain._rule, ThreeFactorRule)
        assert brain._rule.scaling == ScalingOptions(
            normalise_modulator=True,
            normalise_trace=True,
            scale_rate=0.2,
            scale_floor=1e-3,
        )


def test_deepcopied_rule_state_is_independent() -> None:
    """Copying a rule copies its scales, and the copies diverge freely: plain rule state."""
    rule = _rule(_topology(), scaling=_scaling(normalise_modulator=True))
    _step(rule, 1.0)
    other = copy.deepcopy(rule)
    _step(other, 5.0)
    assert rule.modulator_scale.count == 1
    assert other.modulator_scale.count == 2
