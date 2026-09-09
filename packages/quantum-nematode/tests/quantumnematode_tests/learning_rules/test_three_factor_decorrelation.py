"""The three-factor rule's decorrelating terms.

The minimal update potentiates a synapse where pre- and post-synaptic activity
agree in sign and depresses it where they disagree, so on a mostly-excitatory
network the dominant loop is positive feedback. Two terms oppose it: one negates
the update where the atlas grounds a synapse inhibitory, the other subtracts the
classic normalisation and needs no identity. Off, the rule must not move a bit.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

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
from quantumnematode.learning_rules import DecorrelationOptions, ThreeFactorRule
from quantumnematode.learning_rules.three_factor import (
    DECORRELATION_SHARE_KEY,
    ThreeFactorBatch,
    record_plasticity_report,
)
from torch import nn

if TYPE_CHECKING:
    from quantumnematode.brain.arch._rule import RuleStepReport
    from quantumnematode.brain.arch._topology import BrainTopology, PlasticTopology

_SEED = 8613
_ETA = 0.05


def _brain(**overrides: object) -> ConnectomePPOBrain:
    config: dict[str, object] = {
        "seed": _SEED,
        "action_mode": "continuous",
        "learning_rule": "three_factor",
        "enable_activity_traces": True,
    }
    config.update(overrides)
    return ConnectomePPOBrain(
        config=ConnectomePPOBrainConfig(**config),  # type: ignore[arg-type]
        device=DeviceType.CPU,
    )


def _topology(**overrides: object) -> ConnectomeTopology:
    topo = _brain(**overrides).topology
    torch.manual_seed(_SEED + 1)
    with torch.no_grad():
        topo.activity_traces.copy_(topo.apply_weight_mask(torch.randn_like(topo.activity_traces)))
        topo.prev_activity.copy_(torch.randn(topo.prev_activity.shape[0]))
    return topo


def _rule(topology: ConnectomeTopology | MLPTopology, **overrides: object) -> ThreeFactorRule:
    kwargs: dict[str, object] = {
        "plasticity_rate": _ETA,
        "weight_decay": 0.0,
        "weight_bound": 100.0,
        "baseline_rate": 0.0,
        "freeze_updates": False,
        "modulated": True,
    }
    kwargs.update(overrides)
    return ThreeFactorRule(topology, device=torch.device("cpu"), **kwargs)  # type: ignore[arg-type]


def _step(rule: ThreeFactorRule, reward: float) -> RuleStepReport:
    return rule.step(cast("BrainTopology", rule._topology), ThreeFactorBatch(reward=reward))


def _signs_of(topology: ConnectomeTopology) -> list[torch.Tensor]:
    return [cast("torch.Tensor", topology.chem_sign)]


def _edges(topology: PlasticTopology) -> torch.Tensor:
    return topology.plastic_masks[0].to(torch.bool)


class TestDefaultsAreOff:
    def test_options_default_to_no_term(self) -> None:
        assert DecorrelationOptions().mechanism == "none"
        assert DecorrelationOptions().oja_coefficient == 0.0

    def test_trajectory_is_bit_identical(self) -> None:
        plain = _rule(_topology())
        with_options = _rule(_topology(), decorrelation=DecorrelationOptions())
        for reward in (1.0, -2.0, 0.5, 3.0):
            _step(plain, reward)
            _step(with_options, reward)
        assert torch.equal(
            plain._topology.plastic_weights[0],
            with_options._topology.plastic_weights[0],
        )

    def test_the_share_is_zero(self) -> None:
        assert _step(_rule(_topology()), 1.0).extra[DECORRELATION_SHARE_KEY] == 0.0


class TestConfigurationRefusals:
    def test_oja_without_a_coefficient_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="plasticity_oja_coefficient"):
            ConnectomePPOBrainConfig(
                learning_rule="three_factor",
                enable_activity_traces=True,
                plasticity_decorrelation="oja",
            )

    def test_the_sign_keyed_variant_needs_grounded_signs(self) -> None:
        with pytest.raises(ValidationError, match="synapse_signs='atlas'"):
            ConnectomePPOBrainConfig(
                learning_rule="three_factor",
                enable_activity_traces=True,
                plasticity_decorrelation="anti_hebbian_inhibitory",
            )

    def test_the_construction_guard_catches_a_copied_config(self) -> None:
        # `model_copy` skips validators, which is how the campaign runner derives arms.
        config = ConnectomePPOBrainConfig(
            seed=_SEED,
            action_mode="continuous",
            learning_rule="three_factor",
            enable_activity_traces=True,
            synapse_signs="atlas",
            plasticity_decorrelation="anti_hebbian_inhibitory",
        ).model_copy(update={"synapse_signs": "random"})
        with pytest.raises(ValueError, match="synapse_signs='atlas'"):
            ConnectomePPOBrain(config=config, device=DeviceType.CPU)

    def test_the_dense_substrate_refuses_the_sign_keyed_variant(self) -> None:
        with pytest.raises(ValidationError, match="not available on this substrate"):
            MLPPPOBrainConfig(
                sensory_modules=[next(iter(ModuleName))],
                plasticity_decorrelation="anti_hebbian_inhibitory",
            )

    @pytest.mark.parametrize(
        "options",
        [{"mechanism": "nonsense"}, {"mechanism": "oja"}, {"oja_coefficient": -1.0}],
    )
    def test_direct_construction_is_held_to_the_same_bounds(self, options: dict) -> None:
        with pytest.raises(ValueError, match=r"."):
            DecorrelationOptions(**options)

    def test_the_rule_refuses_the_variant_without_signs(self) -> None:
        with pytest.raises(ValueError, match="no inhibitory synapse to key on"):
            _rule(
                _topology(),
                decorrelation=DecorrelationOptions(mechanism="anti_hebbian_inhibitory"),
            )


class TestAntiHebbianInhibitory:
    def _pair(self) -> tuple[ThreeFactorRule, ThreeFactorRule]:
        """Build the variant and the plain rule over identical grounded substrates."""
        plain_topo = _topology(synapse_signs="atlas")
        variant_topo = _topology(synapse_signs="atlas")
        plain = _rule(plain_topo, synapse_signs=_signs_of(plain_topo), enforce_signs=False)
        variant = _rule(
            variant_topo,
            synapse_signs=_signs_of(variant_topo),
            enforce_signs=False,
            decorrelation=DecorrelationOptions(mechanism="anti_hebbian_inhibitory"),
        )
        return plain, variant

    def test_inhibitory_entries_take_the_opposite_update(self) -> None:
        plain, variant = self._pair()
        before = plain._topology.plastic_weights[0].detach().clone()
        _step(plain, 2.0)
        _step(variant, 2.0)
        variant_topo = cast("ConnectomeTopology", variant._topology)
        signs = cast("torch.Tensor", variant_topo.chem_sign)
        inhibitory = (signs < 0) & _edges(variant_topo)
        plain_delta = plain._topology.plastic_weights[0].detach() - before
        variant_delta = variant._topology.plastic_weights[0].detach() - before
        assert inhibitory.any()
        assert torch.allclose(
            variant_delta[inhibitory],
            -plain_delta[inhibitory],
            atol=1e-7,
        )

    def test_excitatory_and_ungrounded_entries_are_untouched(self) -> None:
        plain, variant = self._pair()
        before = plain._topology.plastic_weights[0].detach().clone()
        _step(plain, 2.0)
        _step(variant, 2.0)
        variant_topo = cast("ConnectomeTopology", variant._topology)
        signs = cast("torch.Tensor", variant_topo.chem_sign)
        rest = (signs >= 0) & _edges(variant_topo)
        plain_delta = plain._topology.plastic_weights[0].detach() - before
        variant_delta = variant._topology.plastic_weights[0].detach() - before
        assert torch.allclose(variant_delta[rest], plain_delta[rest], atol=1e-7)

    def test_the_update_is_redirected_not_resized(self) -> None:
        plain, variant = self._pair()
        before = plain._topology.plastic_weights[0].detach().clone()
        _step(plain, 2.0)
        _step(variant, 2.0)
        edges = _edges(variant._topology)
        plain_magnitude = (plain._topology.plastic_weights[0].detach() - before)[edges].abs().sum()
        variant_magnitude = (
            (variant._topology.plastic_weights[0].detach() - before)[edges].abs().sum()
        )
        assert float(variant_magnitude) == pytest.approx(float(plain_magnitude), rel=1e-6)

    def test_nothing_is_written_off_the_edge_set(self) -> None:
        _, variant = self._pair()
        weight = variant._topology.plastic_weights[0]
        edges = _edges(variant._topology)
        before = weight.detach().clone()
        _step(variant, 2.0)
        assert torch.equal(weight.detach()[~edges], before[~edges])

    def test_it_composes_with_dales_law(self) -> None:
        topo = _topology(synapse_signs="atlas")
        rule = _rule(
            topo,
            synapse_signs=_signs_of(topo),
            enforce_signs=True,
            decorrelation=DecorrelationOptions(mechanism="anti_hebbian_inhibitory"),
        )
        for reward in (3.0, -3.0, 3.0):
            _step(rule, reward)
        signs = cast("torch.Tensor", topo.chem_sign)
        weight = rule._topology.plastic_weights[0].detach()
        grounded = signs != 0
        assert not bool(((weight * signs.to(weight.dtype)) < 0)[grounded].any())

    def test_the_share_counts_the_redirected_subset(self) -> None:
        _, variant = self._pair()
        share = _step(variant, 2.0).extra[DECORRELATION_SHARE_KEY]
        assert 0.0 < share < 1.0


class TestOja:
    def _oja(self, coefficient: float = 1.0, **overrides: object) -> ThreeFactorRule:
        return _rule(
            _topology(),
            decorrelation=DecorrelationOptions(mechanism="oja", oja_coefficient=coefficient),
            **overrides,
        )

    def test_the_term_opposes_growth_in_proportion_to_activity_and_weight(self) -> None:
        coefficient = 2.0
        plain = _rule(_topology())
        oja = self._oja(coefficient)
        before = plain._topology.plastic_weights[0].detach().clone()
        _step(plain, 1.0)
        _step(oja, 1.0)
        edges = _edges(oja._topology)
        plain_delta = plain._topology.plastic_weights[0].detach() - before
        oja_delta = oja._topology.plastic_weights[0].detach() - before
        post = oja._topology.plastic_post_activities[0].reshape(1, -1)
        expected = -(_ETA * coefficient) * post.square() * before
        assert torch.allclose(
            (oja_delta - plain_delta)[edges],
            expected[edges],
            atol=1e-6,
        )

    def test_an_inactive_unit_receives_no_term(self) -> None:
        oja = self._oja()
        with torch.no_grad():
            cast("ConnectomeTopology", oja._topology).prev_activity.zero_()
        plain = _rule(_topology())
        with torch.no_grad():
            cast("ConnectomeTopology", plain._topology).prev_activity.zero_()
        before = plain._topology.plastic_weights[0].detach().clone()
        _step(plain, 1.0)
        _step(oja, 1.0)
        assert torch.allclose(
            oja._topology.plastic_weights[0].detach() - before,
            plain._topology.plastic_weights[0].detach() - before,
            atol=1e-9,
        )

    def test_a_zero_weight_receives_no_term(self) -> None:
        oja = self._oja()
        weight = oja._topology.plastic_weights[0]
        with torch.no_grad():
            weight.zero_()
            oja._topology.eligibility_traces[0].zero_()  # no Hebbian term either
        _step(oja, 1.0)
        assert float(weight.detach().abs().max().item()) == 0.0

    def test_the_share_is_positive(self) -> None:
        assert _step(self._oja(), 1.0).extra[DECORRELATION_SHARE_KEY] > 0.0

    def test_the_dense_substrate_takes_the_same_path(self) -> None:
        torch.manual_seed(_SEED + 2)
        actor = nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 2))
        topo = MLPTopology(actor, enable_activity_traces=True, trace_decay=0.9)
        topo(torch.randn(6))  # accumulate a trace and a post activity
        rule = _rule(
            topo,
            decorrelation=DecorrelationOptions(mechanism="oja", oja_coefficient=1.0),
        )
        before = [w.detach().clone() for w in topo.plastic_weights]
        _step(rule, 1.0)
        assert any(
            not torch.equal(w.detach(), b)
            for w, b in zip(topo.plastic_weights, before, strict=True)
        )


class TestTheSeam:
    def test_the_connectome_exposes_its_post_activity_as_a_view(self) -> None:
        topo = _topology()
        assert topo.plastic_post_activities[0] is topo.prev_activity

    def test_each_vector_matches_the_post_axis(self) -> None:
        torch.manual_seed(_SEED + 3)
        actor = nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 2))
        mlp = MLPTopology(actor, enable_activity_traces=True, trace_decay=0.9)
        for topo in (_topology(), mlp):
            for weight, axis, post in zip(
                topo.plastic_weights,
                topo.plastic_fan_in_axes,
                topo.plastic_post_activities,
                strict=True,
            ):
                assert post.shape[0] == weight.shape[1 - axis]

    def test_the_activity_is_the_trace_s_own_post_factor(self) -> None:
        torch.manual_seed(_SEED + 4)
        actor = nn.Sequential(nn.Linear(4, 5), nn.Tanh())
        topo = MLPTopology(actor, enable_activity_traces=True, trace_decay=0.0)
        features = torch.randn(4)
        out = topo(features)
        assert torch.allclose(topo.plastic_post_activities[0], out.detach(), atol=1e-7)


class TestTelemetryIsRecorded:
    def test_the_shared_recorder_carries_the_share(self) -> None:
        history = BrainHistoryData()
        rule = _rule(
            _topology(),
            decorrelation=DecorrelationOptions(mechanism="oja", oja_coefficient=1.0),
        )
        record_plasticity_report(history, _step(rule, 1.0), reward=1.0)
        assert len(history.plasticity_decorrelation_share) == 1
