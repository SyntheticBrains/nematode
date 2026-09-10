"""Routing the third factor through the instructive pathway.

Every panel so far broadcast one reward-prediction error to every plastic synapse. Routed, the
modulator reaches only synapses whose post-synaptic neuron the aminergic wiring instructs, and
the rest take the unmodulated Hebbian term — so a routed arm is an interpolation between two
arms the panels already measured, and that is exactly what these assert, entry for entry.
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
from quantumnematode.brain.arch.mlpppo import MLPPPOBrainConfig
from quantumnematode.brain.modules import ModuleName
from quantumnematode.learning_rules import ThreeFactorRule
from quantumnematode.learning_rules.three_factor import (
    INSTRUCTED_FRACTION_KEY,
    INSTRUCTED_SHARE_KEY,
    ThreeFactorBatch,
    record_plasticity_report,
)

if TYPE_CHECKING:
    from pathlib import Path

    from quantumnematode.brain.arch._rule import RuleStepReport
    from quantumnematode.brain.arch._topology import BrainTopology

_SEED = 5150
_ETA = 0.05


def _topology(**overrides: object) -> ConnectomeTopology:
    config: dict[str, object] = {
        "seed": _SEED,
        "action_mode": "continuous",
        "learning_rule": "three_factor",
        "enable_activity_traces": True,
    }
    config.update(overrides)
    brain = ConnectomePPOBrain(
        config=ConnectomePPOBrainConfig(**config),  # type: ignore[arg-type]
        device=DeviceType.CPU,
    )
    topo = brain.topology
    torch.manual_seed(_SEED + 1)
    with torch.no_grad():
        topo.activity_traces.copy_(topo.apply_weight_mask(torch.randn_like(topo.activity_traces)))
    return topo


def _rule(topology: ConnectomeTopology, **overrides: object) -> ThreeFactorRule:
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


def _pathway(topology: ConnectomeTopology) -> list[torch.Tensor]:
    return [cast("torch.Tensor", topology.chem_pathway)]


class TestTheDerivedPathway:
    def test_the_wild_type_reach_is_the_registered_one(self) -> None:
        topo = _topology()
        mask = cast("torch.Tensor", topo.chem_pathway)
        assert int(mask.sum()) == 2636
        assert topo.instructed_fraction == pytest.approx(0.711, abs=0.001)

    def test_the_mask_keys_on_the_post_synaptic_neuron(self) -> None:
        # Every column is uniform over the edge set: a synapse is instructed by whose column it
        # is in, never by whose row.
        topo = _topology()
        mask = cast("torch.Tensor", topo.chem_pathway)
        edges = cast("torch.Tensor", topo.m_chem).to(torch.bool)
        for column in range(mask.shape[1]):
            on_edges = mask[:, column][edges[:, column]]
            if on_edges.numel():
                assert bool(on_edges.all()) or not bool(on_edges.any())

    def test_the_mask_never_leaves_the_edge_set(self) -> None:
        topo = _topology()
        mask = cast("torch.Tensor", topo.chem_pathway)
        edges = cast("torch.Tensor", topo.m_chem).to(torch.bool)
        assert not bool((mask & ~edges).any())

    def test_a_rewired_substrate_derives_its_own_pathway(self) -> None:
        wild = _topology()
        rewired = _topology(wiring="rewired_degree_preserving")
        assert rewired.instructed_fraction != wild.instructed_fraction
        assert int(cast("torch.Tensor", rewired.chem_pathway).sum()) != int(
            cast("torch.Tensor", wild.chem_pathway).sum(),
        )


class TestRefusals:
    def test_routing_the_unmodulated_rule_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="unmodulated Hebbian floor"):
            ConnectomePPOBrainConfig(
                learning_rule="hebbian",
                enable_activity_traces=True,
                third_factor="pathway",
            )

    def test_the_dense_substrate_refuses_routing(self) -> None:
        with pytest.raises(ValidationError, match="not available on this substrate"):
            MLPPPOBrainConfig(
                sensory_modules=[next(iter(ModuleName))],
                third_factor="pathway",
            )

    def test_the_construction_guard_catches_a_copied_config(self) -> None:
        # `model_copy` skips validators, which is how the campaign runner derives its arms.
        config = ConnectomePPOBrainConfig(
            seed=_SEED,
            action_mode="continuous",
            learning_rule="three_factor",
            enable_activity_traces=True,
            third_factor="pathway",
        ).model_copy(update={"learning_rule": "hebbian"})
        with pytest.raises(ValueError, match="unmodulated Hebbian floor"):
            ConnectomePPOBrain(config=config, device=DeviceType.CPU)

    def test_the_dense_substrate_guard_catches_a_copied_config(self) -> None:
        from quantumnematode.brain.arch.mlpppo import MLPPPOBrain

        config = MLPPPOBrainConfig(
            sensory_modules=[next(iter(ModuleName))],
            learning_rule="three_factor",
            enable_activity_traces=True,
        ).model_copy(update={"third_factor": "pathway"})
        with pytest.raises(ValueError, match="not available on this substrate"):
            MLPPPOBrain(config=config, device=DeviceType.CPU)

    def test_the_modulated_rule_accepts_routing(self) -> None:
        config = ConnectomePPOBrainConfig(
            learning_rule="three_factor",
            enable_activity_traces=True,
            third_factor="pathway",
        )
        assert config.third_factor == "pathway"


class TestCreditReachesOnlyTheInstructedSet:
    def test_instructed_entries_match_the_global_rule_and_the_rest_the_unmodulated_one(
        self,
    ) -> None:
        topo_routed, topo_global, topo_plain = _topology(), _topology(), _topology()
        routed = _rule(topo_routed, pathway_masks=_pathway(topo_routed))
        glob = _rule(topo_global)
        plain = _rule(topo_plain, modulated=False)
        before = topo_global.plastic_weights[0].detach().clone()
        for rule in (routed, glob, plain):
            _step(rule, 3.0)
        mask = cast("torch.Tensor", topo_routed.chem_pathway)
        edges = cast("torch.Tensor", topo_routed.m_chem).to(torch.bool)
        routed_delta = topo_routed.plastic_weights[0].detach() - before
        global_delta = topo_global.plastic_weights[0].detach() - before
        plain_delta = topo_plain.plastic_weights[0].detach() - before
        assert torch.allclose(routed_delta[mask & edges], global_delta[mask & edges], atol=1e-7)
        assert torch.allclose(routed_delta[~mask & edges], plain_delta[~mask & edges], atol=1e-7)
        # The two halves must actually differ, or the assertion above is vacuous.
        assert not torch.allclose(
            global_delta[~mask & edges],
            plain_delta[~mask & edges],
            atol=1e-7,
        )

    def test_the_default_path_is_bit_identical(self) -> None:
        routed_off = _rule(_topology(), pathway_masks=None)
        plain = _rule(_topology())
        for reward in (1.0, -2.0, 0.5):
            _step(routed_off, reward)
            _step(plain, reward)
        assert torch.equal(
            routed_off._topology.plastic_weights[0],
            plain._topology.plastic_weights[0],
        )

    def test_nothing_is_written_off_the_edge_set(self) -> None:
        topo = _topology()
        rule = _rule(topo, pathway_masks=_pathway(topo), weight_decay=0.01)
        edges = cast("torch.Tensor", topo.m_chem).to(torch.bool)
        before = topo.plastic_weights[0].detach().clone()
        for reward in (2.0, -2.0):
            _step(rule, reward)
        assert torch.equal(topo.plastic_weights[0].detach()[~edges], before[~edges])


class TestTelemetry:
    def test_a_broadcast_third_factor_reports_the_whole(self) -> None:
        report = _step(_rule(_topology()), 2.0)
        assert math.isnan(report.extra[INSTRUCTED_FRACTION_KEY])
        assert report.extra[INSTRUCTED_SHARE_KEY] == 1.0

    def test_a_routed_third_factor_reports_its_fraction_and_split(self) -> None:
        topo = _topology()
        report = _step(_rule(topo, pathway_masks=_pathway(topo)), 2.0)
        assert report.extra[INSTRUCTED_FRACTION_KEY] == pytest.approx(0.711, abs=0.001)
        assert 0.0 < report.extra[INSTRUCTED_SHARE_KEY] < 1.0

    def test_the_shared_recorder_carries_both_keys(self) -> None:
        history = BrainHistoryData()
        topo = _topology()
        record_plasticity_report(
            history,
            _step(_rule(topo, pathway_masks=_pathway(topo)), 1.0),
            reward=1.0,
        )
        assert len(history.plasticity_instructed_fraction) == 1
        assert len(history.plasticity_instructed_share) == 1


class TestThePathwayIsNotPersisted:
    def test_it_is_absent_from_the_persisted_topology(self) -> None:
        # Derived from the connectome at construction, not learned: persisting it would refuse
        # every checkpoint written before routing existed, the clone assay's start points
        # included.
        brain = ConnectomePPOBrain(
            config=ConnectomePPOBrainConfig(
                seed=_SEED,
                action_mode="continuous",
                learning_rule="three_factor",
                enable_activity_traces=True,
                third_factor="pathway",
            ),
            device=DeviceType.CPU,
        )
        assert "chem_pathway" in ConnectomePPOBrain._TRANSIENT_BUFFERS
        components = brain.get_weight_components(components={"topology"})
        assert "chem_pathway" not in components["topology"].state

    def test_a_checkpoint_without_it_still_loads(self, tmp_path: Path) -> None:
        from quantumnematode.brain.weights import load_weights, save_weights

        def build(**over: object) -> ConnectomePPOBrain:
            config: dict[str, object] = {
                "seed": _SEED,
                "action_mode": "continuous",
                "learning_rule": "three_factor",
                "enable_activity_traces": True,
            }
            config.update(over)
            return ConnectomePPOBrain(
                config=ConnectomePPOBrainConfig(**config),  # type: ignore[arg-type]
                device=DeviceType.CPU,
            )

        file = tmp_path / "w.pt"
        save_weights(build(), file)
        blob = torch.load(file, weights_only=False)
        blob["topology"].pop("chem_pathway", None)  # a file written before routing existed
        torch.save(blob, file)
        target = build(third_factor="pathway")
        load_weights(target, file)
        assert int(cast("torch.Tensor", target.topology.chem_pathway).sum()) == 2636


class TestWeightFilesCrossRoutingModes:
    def test_a_global_file_loads_under_pathway(self, tmp_path: Path) -> None:
        from quantumnematode.brain.weights import load_weights, save_weights

        source = ConnectomePPOBrain(
            config=ConnectomePPOBrainConfig(
                seed=_SEED,
                action_mode="continuous",
                learning_rule="three_factor",
                enable_activity_traces=True,
            ),
            device=DeviceType.CPU,
        )
        target = ConnectomePPOBrain(
            config=ConnectomePPOBrainConfig(
                seed=_SEED,
                action_mode="continuous",
                learning_rule="three_factor",
                enable_activity_traces=True,
                third_factor="pathway",
            ),
            device=DeviceType.CPU,
        )
        file = tmp_path / "w.pt"
        save_weights(source, file)
        # Routing changes how learning is applied, not what the weights are, so this must load —
        # the clone assay starts every routed arm from a file saved under the broadcast rule.
        load_weights(target, file)
        assert torch.equal(
            target.topology.state_dict()["w_chem"],
            source.topology.state_dict()["w_chem"],
        )
