"""The three-factor rule's consolidation mechanisms.

Three brakes, one selected at a time. The minimal rule writes at a magnitude
set by its normalised modulator and trace, not by how good the policy is, so
none of these is a tuning of the rate: the anchor opposes departure from where
the weights were, rigidity makes repeatedly-written synapses harder to write,
and the oracle stops writing once an externally-supplied success rate is met.
Off, the rule must not move a bit.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import pytest
import torch
from pydantic import ValidationError
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
    ConnectomeTopology,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.learning_rules import ConsolidationOptions, ScalingOptions, ThreeFactorRule
from quantumnematode.learning_rules.three_factor import (
    ANCHOR_DEPARTURE_KEY,
    RATE_MULTIPLIER_KEY,
    RIGIDITY_KEY,
    ThreeFactorBatch,
    record_plasticity_report,
)

if TYPE_CHECKING:
    from quantumnematode.brain.arch._rule import RuleStepReport
    from quantumnematode.brain.arch._topology import BrainTopology, PlasticTopology

_SEED = 4242
_ETA = 0.05


def _topology(seed: int = _SEED) -> ConnectomeTopology:
    brain = ConnectomePPOBrain(
        config=ConnectomePPOBrainConfig(
            seed=seed,
            action_mode="continuous",
            learning_rule="three_factor",
            enable_activity_traces=True,
        ),
        device=DeviceType.CPU,
    )
    topo = brain.topology
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        topo.activity_traces.copy_(
            topo.apply_weight_mask(torch.randn_like(topo.activity_traces)),
        )
    return topo


def _rule(topology: ConnectomeTopology, **overrides: object) -> ThreeFactorRule:
    kwargs: dict[str, object] = {
        "plasticity_rate": _ETA,
        "weight_decay": 0.0,
        "weight_bound": 100.0,
        "baseline_rate": 0.0,  # delta == reward, so a test can name the modulator directly
        "freeze_updates": False,
        "modulated": True,
    }
    kwargs.update(overrides)
    return ThreeFactorRule(topology, device=torch.device("cpu"), **kwargs)  # type: ignore[arg-type]


def _step(rule: ThreeFactorRule, reward: float) -> RuleStepReport:
    return rule.step(cast("BrainTopology", rule._topology), ThreeFactorBatch(reward=reward))


def _edges(topology: PlasticTopology) -> torch.Tensor:
    return topology.plastic_masks[0].to(torch.bool)


class TestDefaultsAreOff:
    def test_options_default_to_no_mechanism(self) -> None:
        assert ConsolidationOptions().mechanism == "none"
        assert ConsolidationOptions().anchor_stiffness == 0.0
        assert ConsolidationOptions().rigidity_growth == 0.0
        # The reference never closes the gate, so a screen pins its own.
        assert ConsolidationOptions().oracle_reference == 1.0

    def test_no_state_is_allocated(self) -> None:
        rule = _rule(_topology())
        assert rule._anchors == []
        assert rule._rigidity == []

    def test_trajectory_is_bit_identical(self) -> None:
        plain = _rule(_topology())
        consolidated = _rule(_topology(), consolidation=ConsolidationOptions())
        for reward in (1.0, -2.0, 0.5, 3.0):
            _step(plain, reward)
            _step(consolidated, reward)
        for left, right in zip(
            plain._topology.plastic_weights,
            consolidated._topology.plastic_weights,
            strict=True,
        ):
            assert torch.equal(left, right)

    def test_telemetry_reports_inactive_mechanisms(self) -> None:
        report = _step(_rule(_topology()), 1.0)
        assert report.extra[RATE_MULTIPLIER_KEY] == 1.0
        assert math.isnan(report.extra[ANCHOR_DEPARTURE_KEY])
        assert math.isnan(report.extra[RIGIDITY_KEY])


class TestInertMechanismsAreRejected:
    @pytest.mark.parametrize(
        ("mechanism", "field"),
        [
            ("anchor", "plasticity_anchor_stiffness"),
            ("rigidity", "plasticity_rigidity_growth"),
            ("rigidity", "plasticity_rigidity_strength"),
        ],
    )
    def test_a_named_brake_that_would_not_brake_fails_at_load(
        self,
        mechanism: str,
        field: str,
    ) -> None:
        keys: dict[str, object] = {
            "plasticity_anchor_stiffness": 0.5,
            "plasticity_rigidity_growth": 0.5,
            "plasticity_rigidity_strength": 0.5,
        }
        keys[field] = 0.0
        with pytest.raises(ValidationError, match=field):
            ConnectomePPOBrainConfig(
                learning_rule="three_factor",
                enable_activity_traces=True,
                plasticity_consolidation=mechanism,  # type: ignore[arg-type]
                **keys,  # type: ignore[arg-type]
            )

    def test_a_configured_mechanism_loads(self) -> None:
        config = ConnectomePPOBrainConfig(
            learning_rule="three_factor",
            enable_activity_traces=True,
            plasticity_consolidation="anchor",
            plasticity_anchor_stiffness=0.5,
        )
        assert config.plasticity_consolidation == "anchor"

    def test_the_oracle_needs_no_parameter_to_be_live(self) -> None:
        config = ConnectomePPOBrainConfig(
            learning_rule="three_factor",
            enable_activity_traces=True,
            plasticity_consolidation="oracle",
        )
        assert config.plasticity_oracle_reference == 1.0

    @pytest.mark.parametrize(
        "options",
        [
            {"mechanism": "nonsense"},
            {"anchor_rate": 1.0},
            {"rigidity_decay": -0.1},
            {"oracle_reference": 0.0},
            {"oracle_rate": 1.5},
            {"anchor_stiffness": -1.0},
        ],
    )
    def test_direct_construction_is_held_to_the_same_bounds(self, options: dict) -> None:
        with pytest.raises(ValueError, match=r"."):
            ConsolidationOptions(**options)


class TestElasticAnchor:
    def _anchored(self, **overrides: object) -> ThreeFactorRule:
        options: dict[str, object] = {
            "mechanism": "anchor",
            "anchor_rate": 0.0,
            "anchor_stiffness": 1.0,
        }
        options.update(overrides)
        return _rule(_topology(), consolidation=ConsolidationOptions(**options))  # type: ignore[arg-type]

    def test_the_anchor_starts_at_the_constructed_weights(self) -> None:
        rule = self._anchored()
        for weight, anchor in zip(rule._topology.plastic_weights, rule._anchors, strict=True):
            assert torch.equal(weight, anchor)

    def test_a_zero_rate_holds_the_anchor_where_it_started(self) -> None:
        rule = self._anchored(anchor_rate=0.0)
        start = [a.clone() for a in rule._anchors]
        for reward in (1.0, 2.0, -1.0, 4.0):
            _step(rule, reward)
        assert not torch.equal(rule._topology.plastic_weights[0], start[0])
        for anchor, began in zip(rule._anchors, start, strict=True):
            assert torch.equal(anchor, began)

    def test_a_positive_rate_moves_the_anchor_by_that_fraction(self) -> None:
        rate = 0.25
        rule = self._anchored(anchor_rate=rate, anchor_stiffness=0.0)
        before_w = rule._topology.plastic_weights[0].clone()
        before_a = rule._anchors[0].clone()
        _step(rule, 1.0)
        after_w = rule._topology.plastic_weights[0]
        expected = before_a + rate * (after_w - before_a)
        assert torch.allclose(rule._anchors[0], expected, atol=1e-7)
        assert not torch.equal(after_w, before_w)

    def test_the_restoring_term_opposes_departure(self) -> None:
        # A weight held away from a fixed anchor, with no Hebbian drive: the
        # only term left is the restoring one, and it must point back.
        rule = self._anchored(anchor_rate=0.0, anchor_stiffness=0.5)
        weight = rule._topology.plastic_weights[0]
        edges = _edges(rule._topology)
        with torch.no_grad():
            weight.add_(torch.where(edges, torch.full_like(weight, 0.5), torch.zeros_like(weight)))
        departed = weight.clone()
        rule._topology.reset_traces()  # no Hebbian term this step
        _step(rule, 0.0)
        moved = (weight - departed)[edges]
        # Every edge moved back toward the anchor, by eta * stiffness * departure.
        assert torch.all(moved < 0)
        assert torch.allclose(moved, torch.full_like(moved, -_ETA * 0.5 * 0.5), atol=1e-6)

    def test_nothing_is_written_off_the_edge_set(self) -> None:
        rule = self._anchored(anchor_rate=0.01, anchor_stiffness=1.0)
        weight = rule._topology.plastic_weights[0]
        edges = _edges(rule._topology)
        before = weight.clone()
        for reward in (1.0, -1.0, 2.0):
            _step(rule, reward)
        assert torch.equal(weight[~edges], before[~edges])

    def test_departure_telemetry_tracks_the_gap(self) -> None:
        rule = self._anchored(anchor_rate=0.0, anchor_stiffness=0.0)
        first = _step(rule, 0.0).extra[ANCHOR_DEPARTURE_KEY]
        for reward in (3.0, 3.0, 3.0):
            report = _step(rule, reward)
        assert first == pytest.approx(0.0, abs=1e-12)
        assert report.extra[ANCHOR_DEPARTURE_KEY] > 0.0
        assert math.isnan(report.extra[RIGIDITY_KEY])


class TestReinforcedRigidity:
    def _rigid(self, **overrides: object) -> ThreeFactorRule:
        options: dict[str, object] = {
            "mechanism": "rigidity",
            "rigidity_growth": 1.0,
            "rigidity_decay": 0.0,
            "rigidity_strength": 10.0,
        }
        options.update(overrides)
        return _rule(_topology(), consolidation=ConsolidationOptions(**options))  # type: ignore[arg-type]

    def test_rigidity_starts_at_zero(self) -> None:
        rule = self._rigid()
        assert all(float(c.abs().max().item()) == 0.0 for c in rule._rigidity)

    def test_it_grows_where_a_positive_modulator_met_a_large_trace(self) -> None:
        rule = self._rigid()
        _step(rule, 2.0)
        trace = rule._topology.eligibility_traces[0]
        edges = _edges(rule._topology)
        grown = rule._rigidity[0]
        assert torch.allclose(grown[edges], (2.0 * trace.abs())[edges], atol=1e-6)

    def test_a_negative_modulator_grows_nothing(self) -> None:
        rule = self._rigid()
        _step(rule, -2.0)
        assert float(rule._rigidity[0].abs().max().item()) == 0.0

    def test_the_divisor_uses_the_pre_growth_value(self) -> None:
        # One step from zero rigidity must move the weights exactly as the
        # unbraked rule does: the step is not charged for what it creates.
        braked = self._rigid()
        plain = _rule(_topology())
        _step(braked, 1.0)
        _step(plain, 1.0)
        assert torch.allclose(
            braked._topology.plastic_weights[0],
            plain._topology.plastic_weights[0],
            atol=1e-7,
        )
        assert float(braked._rigidity[0].abs().max().item()) > 0.0

    def test_a_later_step_is_smaller_than_the_unbraked_one(self) -> None:
        braked = self._rigid()
        plain = _rule(_topology())
        for _ in range(4):
            _step(braked, 1.0)
            _step(plain, 1.0)
        edges = _edges(braked._topology)
        start = _topology().plastic_weights[0].detach().clone()
        braked_move = (braked._topology.plastic_weights[0].detach() - start)[edges].abs().sum()
        plain_move = (plain._topology.plastic_weights[0].detach() - start)[edges].abs().sum()
        assert float(braked_move) < float(plain_move)

    def test_it_decays_toward_zero_without_reinforcement(self) -> None:
        decay = 0.5
        rule = self._rigid(rigidity_decay=decay)
        _step(rule, 2.0)
        grown = float(rule._rigidity[0].abs().sum().item())
        rule._topology.reset_traces()  # no trace, so growth is zero
        _step(rule, 2.0)
        assert float(rule._rigidity[0].abs().sum().item()) == pytest.approx(
            grown * (1.0 - decay),
            rel=1e-6,
        )

    def test_growth_is_measured_on_the_normalised_trace(self) -> None:
        # With trace normalisation on, growth uses the trace as the update saw
        # it, so a pinned growth rate means the same thing on every substrate.
        rule = _rule(
            _topology(),
            scaling=ScalingOptions(normalise_trace=True, scale_rate=1.0),
            consolidation=ConsolidationOptions(
                mechanism="rigidity",
                rigidity_growth=1.0,
                rigidity_strength=1.0,
            ),
        )
        trace = rule._topology.eligibility_traces[0].clone()
        edges = _edges(rule._topology)
        report = _step(rule, 1.0)
        divisor = report.extra["plasticity_trace_scale"]
        assert divisor > 0.0
        assert torch.allclose(
            rule._rigidity[0][edges],
            (trace.abs() / divisor)[edges],
            atol=1e-6,
        )

    def test_the_multiplier_telemetry_falls_below_one(self) -> None:
        rule = self._rigid()
        first = _step(rule, 2.0).extra[RATE_MULTIPLIER_KEY]
        later = _step(rule, 2.0).extra[RATE_MULTIPLIER_KEY]
        assert first == pytest.approx(1.0)
        assert later < 1.0
        assert math.isnan(_step(rule, 2.0).extra[ANCHOR_DEPARTURE_KEY])


class TestOracleGate:
    def _gated(self, **overrides: object) -> ThreeFactorRule:
        options: dict[str, object] = {
            "mechanism": "oracle",
            "oracle_reference": 0.4,
            "oracle_rate": 1.0,  # one episode moves the estimate the whole way
        }
        options.update(overrides)
        return _rule(_topology(), consolidation=ConsolidationOptions(**options))  # type: ignore[arg-type]

    def test_it_starts_ungated(self) -> None:
        rule = self._gated()
        assert _step(rule, 1.0).extra[RATE_MULTIPLIER_KEY] == pytest.approx(1.0)

    def test_reaching_the_reference_stops_every_write(self) -> None:
        rule = self._gated()
        rule.observe_episode(success=True)  # rate 1.0, so the estimate is now 1.0 > 0.4
        before = [w.clone() for w in rule._topology.plastic_weights]
        report = _step(rule, 5.0)
        assert report.extra[RATE_MULTIPLIER_KEY] == 0.0
        for weight, began in zip(rule._topology.plastic_weights, before, strict=True):
            assert torch.equal(weight, began)

    def test_far_below_the_reference_the_rate_is_unchanged(self) -> None:
        rule = self._gated(oracle_rate=0.01)
        rule.observe_episode(success=False)
        plain = _rule(_topology())
        _step(rule, 1.0)
        _step(plain, 1.0)
        assert torch.allclose(
            rule._topology.plastic_weights[0],
            plain._topology.plastic_weights[0],
            atol=1e-7,
        )

    def test_the_default_reference_never_closes_the_gate(self) -> None:
        rule = _rule(_topology(), consolidation=ConsolidationOptions(mechanism="oracle"))
        for _ in range(50):
            rule.observe_episode(success=True)
        assert _step(rule, 1.0).extra[RATE_MULTIPLIER_KEY] > 0.0

    def test_no_other_mechanism_reads_the_flag(self) -> None:
        for options in (
            ConsolidationOptions(),
            ConsolidationOptions(mechanism="anchor", anchor_stiffness=1.0),
        ):
            rule = _rule(_topology(), consolidation=options)
            rule.observe_episode(success=True)
            assert rule.success_rate == 0.0

    def test_a_missing_flag_is_ignored(self) -> None:
        rule = self._gated()
        rule.observe_episode(success=None)
        assert rule.success_rate == 0.0

    def test_the_brain_hook_forwards_the_flag(self) -> None:
        brain = ConnectomePPOBrain(
            config=ConnectomePPOBrainConfig(
                seed=_SEED,
                action_mode="continuous",
                learning_rule="three_factor",
                enable_activity_traces=True,
                plasticity_consolidation="oracle",
                plasticity_oracle_reference=0.4,
                plasticity_oracle_rate=1.0,
            ),
            device=DeviceType.CPU,
        )
        brain.post_process_episode(episode_success=True)
        assert cast("ThreeFactorRule", brain._rule).success_rate == pytest.approx(1.0)

    def test_the_hook_is_harmless_under_the_gradient_rule(self) -> None:
        brain = ConnectomePPOBrain(
            config=ConnectomePPOBrainConfig(seed=_SEED, action_mode="continuous"),
            device=DeviceType.CPU,
        )
        brain.post_process_episode(episode_success=True)


class TestStateFollowsALoadedPolicy:
    def test_the_anchor_re_anchors_and_its_first_pull_is_zero(self) -> None:
        rule = _rule(
            _topology(),
            consolidation=ConsolidationOptions(
                mechanism="anchor",
                anchor_rate=0.0,
                anchor_stiffness=1.0,
            ),
        )
        weight = rule._topology.plastic_weights[0]
        edges = _edges(rule._topology)
        with torch.no_grad():  # stand in for a loaded clone
            weight.copy_(torch.where(edges, weight + 1.5, weight))
        rule.reset_state()
        assert torch.equal(rule._anchors[0], weight)
        loaded = weight.clone()
        _step(rule, 0.0)  # traces were cleared by the reset, so only the anchor could move it
        assert torch.equal(weight, loaded)

    def test_rigidity_does_not_survive_a_load(self) -> None:
        rule = _rule(
            _topology(),
            consolidation=ConsolidationOptions(
                mechanism="rigidity",
                rigidity_growth=1.0,
                rigidity_strength=1.0,
            ),
        )
        _step(rule, 2.0)
        assert float(rule._rigidity[0].abs().max().item()) > 0.0
        rule.reset_state()
        assert float(rule._rigidity[0].abs().max().item()) == 0.0

    def test_the_oracle_estimate_does_not_survive_a_load(self) -> None:
        rule = _rule(
            _topology(),
            consolidation=ConsolidationOptions(mechanism="oracle", oracle_rate=1.0),
        )
        rule.observe_episode(success=True)
        rule.reset_state()
        assert rule.success_rate == 0.0

    def test_consolidation_state_is_not_persisted_with_weights(self) -> None:
        brain = ConnectomePPOBrain(
            config=ConnectomePPOBrainConfig(
                seed=_SEED,
                action_mode="continuous",
                learning_rule="three_factor",
                enable_activity_traces=True,
                plasticity_consolidation="rigidity",
                plasticity_rigidity_growth=1.0,
                plasticity_rigidity_strength=1.0,
            ),
            device=DeviceType.CPU,
        )
        components = brain.get_weight_components()
        flat = str(sorted(components))
        assert "rigidity" not in flat
        assert "anchor" not in flat


class TestFrozenArmsStayComparable:
    def test_a_freeze_writes_nothing_but_still_reports(self) -> None:
        frozen = _rule(
            _topology(),
            freeze_updates=True,
            consolidation=ConsolidationOptions(
                mechanism="rigidity",
                rigidity_growth=1.0,
                rigidity_strength=1.0,
            ),
        )
        before = frozen._topology.plastic_weights[0].clone()
        first = _step(frozen, 2.0)
        second = _step(frozen, 2.0)
        assert torch.equal(frozen._topology.plastic_weights[0], before)
        # Reported at the state each step was taken at: zero entering the
        # first, grown by the time the second is taken, so the frozen arm's
        # telemetry tracks what the plastic arm's would.
        assert first.extra[RIGIDITY_KEY] == 0.0
        assert second.extra[RIGIDITY_KEY] > 0.0


class TestCompositionWithTheOtherTerms:
    def test_the_bound_still_holds_last(self) -> None:
        bound = 0.05
        rule = _rule(
            _topology(),
            weight_bound=bound,
            weight_decay=0.01,
            homeostasis=True,
            consolidation=ConsolidationOptions(
                mechanism="anchor",
                anchor_rate=0.01,
                anchor_stiffness=1.0,
            ),
        )
        for reward in (5.0, -5.0, 5.0, 5.0):
            _step(rule, reward)
        assert float(rule._topology.plastic_weights[0].abs().max().item()) <= bound + 1e-7

    def test_grounded_signs_survive_a_consolidated_update(self) -> None:
        topo = _topology()
        signs = torch.sign(topo.plastic_weights[0].detach()).to(torch.int8)
        rule = _rule(
            topo,
            homeostasis=True,
            synapse_signs=[signs],
            consolidation=ConsolidationOptions(
                mechanism="rigidity",
                rigidity_growth=1.0,
                rigidity_strength=1.0,
            ),
        )
        for reward in (4.0, -4.0, 4.0):
            _step(rule, reward)
        weight = rule._topology.plastic_weights[0].detach()
        grounded = signs != 0
        assert not bool(((weight * signs.to(weight.dtype)) < 0)[grounded].any())


class TestTelemetryIsRecorded:
    def test_the_shared_recorder_carries_every_key(self) -> None:
        history = BrainHistoryData()
        rule = _rule(
            _topology(),
            consolidation=ConsolidationOptions(
                mechanism="anchor",
                anchor_rate=0.01,
                anchor_stiffness=1.0,
            ),
        )
        record_plasticity_report(history, _step(rule, 1.0), reward=1.0)
        assert len(history.plasticity_rate_multiplier) == 1
        assert len(history.plasticity_anchor_departure) == 1
        assert len(history.plasticity_rigidity) == 1
