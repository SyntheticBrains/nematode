"""Three mechanisms that keep the plastic arms honest: noise, homeostasis, activation.

Each is default-off or default-unchanged, so every existing build is byte-identical (the
frozen-reference tests prove that separately). Pinned here: the configured noise is the initial
parameter on both brains and is rejected where it could not act; homeostatic incoming-norm
scaling holds every unit's norm at its target on both substrates, over the edge set only, under
the bound, and never under a freeze; the MLP builds bounded units on request with the matching
initialisation gain.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import pytest
import torch
from pydantic import ValidationError
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch._mlp_topology import MLPTopology
from quantumnematode.brain.arch._topology import PlasticTopology
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
    ConnectomeTopology,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.mlpppo import MLPPPOBrain, MLPPPOBrainConfig
from quantumnematode.brain.modules import ModuleName
from quantumnematode.learning_rules import ScalingOptions, ThreeFactorRule
from quantumnematode.learning_rules.three_factor import (
    NORM_DRIFT_KEY,
    ThreeFactorBatch,
    record_plasticity_report,
)
from torch import nn

if TYPE_CHECKING:
    from quantumnematode.brain.arch._rule import RuleStepReport
    from quantumnematode.brain.arch._topology import BrainTopology

_SEED = 9001
_MODULES = [ModuleName.FOOD_CHEMOTAXIS]
_ETA = 0.5  # large on purpose: without homeostasis the norms would move visibly in one step
_FLOOR = 1e-6


def _connectome(**overrides: object) -> ConnectomePPOBrain:
    cfg = ConnectomePPOBrainConfig(
        seed=_SEED,
        action_mode="continuous",
        learning_rule="three_factor",
        enable_activity_traces=True,
        **overrides,  # type: ignore[arg-type]
    )
    return ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)


def _mlp(**overrides: object) -> MLPPPOBrain:
    cfg = MLPPPOBrainConfig(
        seed=_SEED,
        action_mode="continuous",
        sensory_modules=_MODULES,
        **overrides,  # type: ignore[arg-type]
    )
    return MLPPPOBrain(config=cfg, device=DeviceType.CPU)


def _seeded_connectome_topology(chemical_mask_mode: str = "strict") -> ConnectomeTopology:
    topo = _connectome(chemical_mask_mode=chemical_mask_mode).topology
    torch.manual_seed(_SEED + 1)
    with torch.no_grad():
        topo.activity_traces.copy_(topo.apply_weight_mask(torch.randn_like(topo.activity_traces)))
    return topo


def _mlp_topology() -> MLPTopology:
    torch.manual_seed(_SEED + 2)
    actor = nn.Sequential(
        nn.Linear(6, 16),
        nn.Tanh(),
        nn.Linear(16, 16),
        nn.Tanh(),
        nn.Linear(16, 2),
    )
    topo = MLPTopology(actor, enable_activity_traces=True, trace_decay=0.9)
    with torch.no_grad():
        for trace in topo.eligibility_traces:
            trace.copy_(torch.randn_like(trace))
    return topo


def _rule(topology: PlasticTopology, **overrides: object) -> ThreeFactorRule:
    kwargs: dict[str, object] = {
        "plasticity_rate": _ETA,
        "weight_decay": 0.0,
        "weight_bound": 100.0,
        "baseline_rate": 0.0,
        "freeze_updates": False,
        "modulated": True,
        "scaling": ScalingOptions(scale_floor=_FLOOR),
        "homeostasis": True,
    }
    kwargs.update(overrides)
    return ThreeFactorRule(topology, device=torch.device("cpu"), **kwargs)  # type: ignore[arg-type]


def _step(rule: ThreeFactorRule, reward: float) -> RuleStepReport:
    return rule.step(cast("BrainTopology", rule._topology), ThreeFactorBatch(reward=reward))


def _incoming_norms(weight: torch.Tensor, mask: torch.Tensor, axis: int) -> torch.Tensor:
    return (weight.detach() * mask.to(weight.dtype)).norm(dim=axis)


# --- initial action noise ---------------------------------------------------------------


class TestInitialLogStd:
    def test_default_is_zero_on_both_brains(self) -> None:
        assert torch.equal(_connectome().topology.log_std, torch.zeros(2))
        assert torch.equal(_mlp().log_std, torch.zeros(2))

    def test_configured_value_is_the_initial_parameter_on_both_brains(self) -> None:
        assert torch.allclose(
            _connectome(initial_log_std=-1.0).topology.log_std,
            torch.full((2,), -1.0),
        )
        assert torch.allclose(_mlp(initial_log_std=-1.0).log_std, torch.full((2,), -1.0))

    def test_value_does_not_disturb_the_rest_of_the_initialisation(self) -> None:
        """log_std consumes no RNG, so every other parameter is unchanged by it."""
        base, shifted = _connectome(), _connectome(initial_log_std=-1.5)
        assert torch.equal(base.topology.w_chem, shifted.topology.w_chem)
        assert torch.equal(base.topology.readout, shifted.topology.readout)

    def test_rejected_under_the_state_dependent_head(self) -> None:
        with pytest.raises(ValidationError, match="state-dependent"):
            ConnectomePPOBrainConfig(
                action_mode="continuous",
                continuous_std_mode="state_dependent",
                initial_log_std=-1.0,
            )
        with pytest.raises(ValidationError, match="state-dependent"):
            MLPPPOBrainConfig(
                action_mode="continuous",
                continuous_std_mode="state_dependent",
                sensory_modules=_MODULES,
                initial_log_std=-1.0,
            )

    def test_zero_is_accepted_under_the_state_dependent_head(self) -> None:
        ConnectomePPOBrainConfig(action_mode="continuous", continuous_std_mode="state_dependent")

    def test_paired_wiring_arms_share_the_noise(self) -> None:
        wild = _connectome(initial_log_std=-0.7)
        rewired = _connectome(initial_log_std=-0.7, wiring="rewired_degree_preserving")
        assert torch.equal(wild.topology.log_std, rewired.topology.log_std)


# --- the seam ---------------------------------------------------------------------------


class TestFanInAxes:
    def test_connectome_axis_is_the_column(self) -> None:
        topo = _seeded_connectome_topology()
        assert topo.plastic_fan_in_axes == [0]
        # Reducing over that axis counts each POST neuron's incoming synapses.
        in_degree = topo.m_chem.sum(dim=0)
        assert int(in_degree.sum()) == int(topo.m_chem.sum())
        assert in_degree.shape == (topo.w_chem.shape[1],)

    def test_mlp_axis_is_the_row_for_every_layer(self) -> None:
        topo = _mlp_topology()
        assert topo.plastic_fan_in_axes == [1] * len(topo.plastic_weights)
        assert isinstance(topo, PlasticTopology)


# --- homeostasis --------------------------------------------------------------------------


class TestHomeostasis:
    def test_connectome_targets_are_near_one(self) -> None:
        """Inputs at scale 1/sqrt(k) give an incoming norm near 1: a chi_k / sqrt(k) draw.

        Tight for well-connected neurons; a one-input neuron's norm is a single half-normal
        draw and can sit anywhere, so the spread is asserted only where k is large.
        """
        topo = _seeded_connectome_topology()
        rule = _rule(topo)
        target = rule.norm_targets[0]
        in_degree = topo.m_chem.sum(dim=0)
        with_inputs = target > 0
        assert torch.equal(with_inputs, in_degree > 0)
        assert target[with_inputs].mean().item() == pytest.approx(1.0, abs=0.05)
        well_connected = in_degree >= 16
        assert int(well_connected.sum()) > 50
        assert target[well_connected].min().item() > 0.5
        assert target[well_connected].max().item() < 1.5

    def test_norms_return_to_target_after_an_update_on_the_connectome(self) -> None:
        topo = _seeded_connectome_topology()
        rule = _rule(topo)
        before = _incoming_norms(topo.w_chem, topo.m_chem, 0)
        report = _step(rule, reward=3.0)
        after = _incoming_norms(topo.w_chem, topo.m_chem, 0)
        target = rule.norm_targets[0]
        has = target > 0
        assert torch.allclose(after[has], target[has], atol=1e-5)
        assert (
            report.extra[NORM_DRIFT_KEY] > 0.01
        )  # the update DID move the norms before the rescale
        assert torch.equal(before, target)

    def test_norms_return_to_target_on_every_mlp_layer(self) -> None:
        topo = _mlp_topology()
        rule = _rule(topo)
        _step(rule, reward=-2.0)
        for weight, mask, target in zip(
            topo.plastic_weights,
            topo.plastic_masks,
            rule.norm_targets,
            strict=True,
        ):
            after = _incoming_norms(weight, mask, 1)
            assert torch.allclose(after, target, atol=1e-5)

    def test_weights_still_change_direction_under_homeostasis(self) -> None:
        """Homeostasis redistributes a unit's budget; it does not freeze the weights."""
        topo = _seeded_connectome_topology()
        before = topo.w_chem.detach().clone()
        _step(_rule(topo), reward=3.0)
        assert not torch.equal(topo.w_chem, before)

    def test_off_edge_entries_are_never_written_under_the_soft_mask(self) -> None:
        topo = _seeded_connectome_topology(chemical_mask_mode="soft_prior")
        off_edge = ~topo.m_chem
        with torch.no_grad():
            topo.w_chem[off_edge] = 0.25  # values off the edge set, as the soft prior allows
        rule = _rule(topo, weight_decay=0.1)  # decay would otherwise write off the edge set too
        _step(rule, reward=3.0)
        assert torch.all(topo.w_chem[off_edge] == 0.25)

    def test_units_without_inputs_are_untouched(self) -> None:
        topo = _seeded_connectome_topology()
        no_inputs = topo.m_chem.sum(dim=0) == 0
        assert int(no_inputs.sum()) > 0
        rule = _rule(topo)
        _step(rule, reward=3.0)
        assert torch.all(topo.w_chem[:, no_inputs] == 0)
        assert torch.all(rule.norm_targets[0][no_inputs] == 0)

    def test_bound_holds_after_the_rescale(self) -> None:
        topo = _seeded_connectome_topology()
        rule = _rule(topo, weight_bound=0.05)
        _step(rule, reward=10.0)
        assert float(topo.w_chem.abs().max()) <= 0.05 + 1e-7

    def test_decay_is_undone_by_the_rescale(self) -> None:
        """A uniform shrink of a unit's inputs is restored exactly: decay is inert here."""
        topo = _seeded_connectome_topology()
        with torch.no_grad():
            topo.activity_traces.zero_()  # no Hebbian term: only the decay acts
        rule = _rule(topo, weight_decay=0.5)
        before = topo.w_chem.detach().clone()
        _step(rule, reward=3.0)
        assert torch.allclose(topo.w_chem, before, atol=1e-6)

    def test_freeze_writes_nothing_and_still_reports(self) -> None:
        topo = _seeded_connectome_topology()
        before = topo.w_chem.detach().clone()
        rule = _rule(topo, freeze_updates=True)
        report = _step(rule, reward=3.0)
        assert torch.equal(topo.w_chem, before)
        assert report.extra[NORM_DRIFT_KEY] == 0.0

    def test_off_reports_nan_and_holds_no_targets(self) -> None:
        topo = _seeded_connectome_topology()
        rule = _rule(topo, homeostasis=False)
        report = _step(rule, reward=3.0)
        assert math.isnan(report.extra[NORM_DRIFT_KEY])
        assert rule.norm_targets == []

    def test_drift_is_recorded(self) -> None:
        rule = _rule(_seeded_connectome_topology())
        report = _step(rule, reward=1.0)
        history = BrainHistoryData()
        record_plasticity_report(history, report, reward=1.0)
        assert history.plasticity_norm_drift == [report.extra[NORM_DRIFT_KEY]]

    def test_brain_passes_the_switch_to_the_rule(self) -> None:
        brain = _connectome(plasticity_homeostasis=True)
        assert isinstance(brain._rule, ThreeFactorRule)
        assert brain._rule.homeostasis is True
        assert len(brain._rule.norm_targets) == 1
        default = _connectome()
        assert isinstance(default._rule, ThreeFactorRule)
        assert default._rule.homeostasis is False


# --- MLP activation -----------------------------------------------------------------------


def _hidden_modules(actor: nn.Sequential) -> list[nn.Module]:
    return [m for m in actor if not isinstance(m, nn.Linear)]


class TestMLPActivation:
    def test_default_builds_relu_with_the_historical_gain(self) -> None:
        brain = _mlp()
        assert all(isinstance(m, nn.ReLU) for m in _hidden_modules(brain.actor))
        first = [m for m in brain.actor if isinstance(m, nn.Linear)][1]  # a square hidden layer
        singular = torch.linalg.svdvals(first.weight.detach())
        assert torch.allclose(singular, torch.full_like(singular, math.sqrt(2.0)), atol=1e-5)

    def test_tanh_builds_bounded_units_with_gain_five_thirds(self) -> None:
        brain = _mlp(activation="tanh")
        hidden = _hidden_modules(brain.actor)
        assert hidden
        assert all(isinstance(m, nn.Tanh) for m in hidden)
        assert all(isinstance(m, nn.Tanh) for m in _hidden_modules(brain.critic))
        first = [m for m in brain.actor if isinstance(m, nn.Linear)][1]
        singular = torch.linalg.svdvals(first.weight.detach())
        assert torch.allclose(singular, torch.full_like(singular, 5.0 / 3.0), atol=1e-5)

    def test_default_weights_are_unchanged_by_the_option(self) -> None:
        """The relu build consumes the same RNG draws as before the option existed."""
        a, b = _mlp(), _mlp(activation="relu")
        for pa, pb in zip(a.actor.parameters(), b.actor.parameters(), strict=True):
            assert torch.equal(pa, pb)

    def test_tanh_actor_output_is_bounded_before_the_head(self) -> None:
        brain = _mlp(activation="tanh")
        hidden = nn.Sequential(*list(brain.actor)[:-1])  # everything up to the output layer
        x = 50.0 * torch.randn(8, brain.input_dim)
        assert float(hidden(x).abs().max()) <= 1.0
