"""Which units the connectome draws a perturbation for, and what restricting it costs.

Four failures these pin, each of which would look like a result about the perturbation dimension:

* a declared set that silently does nothing, making its arm a duplicate of the unrestricted one
  reported under another name;
* a restricted set whose excluded synapses move anyway, because the rule's weight decay is
  unconditional and only the homeostatic rescale cancels it;
* a random stream that depends on which set is declared, so two sets at one seed would not be
  comparable;
* the mask drifting away from the readout it is derived from, or from the settling depth.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrainConfig,
    ConnectomeTopology,
)
from quantumnematode.connectome.loader import load_cook_2019_hermaphrodite

if TYPE_CHECKING:
    from quantumnematode.connectome.model import Connectome

# Measured from the Cook 2019 hermaphrodite graph at forward_pass_depth 4, and registered in the
# change's design table. A drift in the loader or the readout constants should fail here.
_EXPECTED = {
    "full": {"units": 302, "adaptable_synapses": 3709, "draws_per_decision": 1208},
    "causal": {"units": 277, "adaptable_synapses": 3538, "draws_per_decision": 672},
    "hop1": {"units": 109, "adaptable_synapses": 1476, "draws_per_decision": 436},
    "motor": {"units": 39, "adaptable_synapses": 323, "draws_per_decision": 156},
    "motor_last": {"units": 39, "adaptable_synapses": 323, "draws_per_decision": 39},
}
_CUMULATIVE_BY_HOP = (39, 109, 247, 277)
_NEVER_REACHING = 25


@pytest.fixture(scope="module")
def connectome() -> Connectome:
    """Load the real graph: every count below is a fact about it, not about a fixture."""
    return load_cook_2019_hermaphrodite()


def _topology(connectome: Connectome, perturbation_set: str, depth: int = 4) -> ConnectomeTopology:
    return ConnectomeTopology(
        connectome,
        enable_gap_junctions=True,
        forward_pass_depth=depth,
        node_noise=0.2,
        perturbation_set=perturbation_set,  # pyright: ignore[reportArgumentType]
        n_food_features=3,
        enforce_strict_mask=True,
        enable_predator_projection=False,
        enable_thermotaxis_projection=False,
        device=torch.device("cpu"),
        rng=np.random.default_rng(0),
        continuous=True,
        enable_activity_traces=True,
        trace_decay=0.9,
    )


class TestTheDeclaredSetMatchesTheGraph:
    @pytest.mark.parametrize("name", list(_EXPECTED))
    def test_the_dimension_is_what_the_design_registered(
        self,
        connectome: Connectome,
        name: str,
    ) -> None:
        reported = _topology(connectome, name).perturbation_dimension()
        for key, value in _EXPECTED[name].items():
            assert reported[key] == value, f"{name}.{key}"

    def test_the_hop_distribution_is_the_registered_one(self, connectome: Connectome) -> None:
        topology = _topology(connectome, "causal")
        distance = topology._readout_hop_distances(topology._motor_flat_indices.cpu().tolist())
        assert tuple(int((distance <= h).sum()) for h in range(4)) == _CUMULATIVE_BY_HOP
        # The largest budget is depth - 1 = 3, so these can never contribute at any step.
        assert int((distance >= 4).sum()) == _NEVER_REACHING

    def test_the_readout_pool_is_the_one_the_readout_uses(self, connectome: Connectome) -> None:
        # Derived from the same motor indices the action is pooled from, so the two cannot drift.
        topology = _topology(connectome, "motor")
        pool = set(topology._motor_flat_indices.cpu().tolist())
        perturbed = set(torch.nonzero(topology._perturbation_mask.any(dim=0)).flatten().tolist())
        assert perturbed == pool

    def test_the_mask_follows_the_settling_depth(self, connectome: Connectome) -> None:
        # Not hard-coded to four: the reach budget is depth - 1 - step.
        deep = _topology(connectome, "causal", depth=4).perturbation_dimension()
        shallow = _topology(connectome, "causal", depth=2).perturbation_dimension()
        assert shallow["settling_depth"] == 2
        assert shallow["draws_per_decision"] < deep["draws_per_decision"]

    def test_a_restricted_set_reports_only_causally_connected_draws(
        self,
        connectome: Connectome,
    ) -> None:
        for name in ("causal", "motor", "motor_last"):
            reported = _topology(connectome, name).perturbation_dimension()
            assert reported["causally_connected_draws"] == reported["draws_per_decision"], name

    def test_the_unrestricted_set_reports_the_disconnected_share(
        self,
        connectome: Connectome,
    ) -> None:
        # The finding this change exists for: 536 of 1208 draws cannot reach the action.
        reported = _topology(connectome, "full").perturbation_dimension()
        assert reported["causally_connected_draws"] == 672
        assert reported["draws_per_decision"] - reported["causally_connected_draws"] == 536

    def test_hop1_keeps_draws_it_cannot_use(self, connectome: Connectome) -> None:
        # A hop-1 unit has no budget at the last step, so 70 of its 436 draws are disconnected.
        # Registered as it is rather than silently intersected with the causal mask.
        reported = _topology(connectome, "hop1").perturbation_dimension()
        assert reported["causally_connected_draws"] == 366

    def test_full_is_built_without_consulting_the_graph(self, connectome: Connectome) -> None:
        # Every recorded plastic result ran the unrestricted set; it must not acquire a dependency
        # on a hop computation that did not exist when those results were produced.
        mask = _topology(connectome, "full")._perturbation_mask
        assert bool(mask.all())


class TestTheMaskActuallyMasks:
    def test_a_masked_unit_draws_nothing(self, connectome: Connectome) -> None:
        topology = _topology(connectome, "motor")
        pool = set(topology._motor_flat_indices.cpu().tolist())
        outside = [i for i in range(topology.n_neurons) if i not in pool]
        mask = topology._perturbation_mask
        assert not bool(mask[:, outside].any())

    def test_the_random_stream_does_not_depend_on_the_set(self, connectome: Connectome) -> None:
        # Drawn then masked, so two sets at one seed differ only in which draws are USED. If the
        # stream depended on the set, arms at the same seed would not be comparable at all.
        draws = {}
        for name in ("full", "motor"):
            topology = _topology(connectome, name)
            topology._perturbation_generator.manual_seed(7)
            draws[name] = torch.randn(
                (topology.n_neurons,),
                generator=topology._perturbation_generator,
            )
        assert torch.equal(draws["full"], draws["motor"])


class TestTheHomeostasisDependencyIsMeasured:
    """The rule's weight decay is unconditional; only the homeostatic rescale cancels it."""

    def _config(self, *, homeostasis: bool, perturbation_set: str) -> ConnectomePPOBrainConfig:
        return ConnectomePPOBrainConfig(
            learning_rule="three_factor",
            enable_activity_traces=True,
            plasticity_eligibility="node_perturbation",
            plasticity_node_noise=0.2,
            plasticity_perturbation_set=perturbation_set,  # pyright: ignore[reportArgumentType]
            plasticity_homeostasis=homeostasis,
        )

    def test_a_restricted_set_without_homeostasis_is_refused(self) -> None:
        with pytest.raises(ValueError, match="requires plasticity_homeostasis=True"):
            self._config(homeostasis=False, perturbation_set="motor")

    def test_a_restricted_set_with_homeostasis_is_accepted(self) -> None:
        assert self._config(homeostasis=True, perturbation_set="motor")

    def test_the_unrestricted_set_is_unaffected_by_the_rule(self) -> None:
        # `full` credits every unit, so there are no excluded synapses to protect and the
        # validator must not constrain the configuration every recorded result used.
        assert self._config(homeostasis=False, perturbation_set="full")

    @pytest.mark.parametrize("homeostasis", [True, False])
    def test_an_uncredited_units_norm_survives_only_with_homeostasis(
        self,
        *,
        homeostasis: bool,
    ) -> None:
        """Both directions, so the cancellation is measured rather than assumed.

        A weight with no eligibility still receives ``-rate * weight_decay * weight``. The
        homeostatic rescale returns each unit's incoming norm to its construction target, and decay
        is a purely radial shrink, so the two cancel radially. Without it the norm is lost, which
        over a run's ~1.05M updates is a collapse rather than a rounding difference.

        The assertion is on the **norm**, not on individual weights: the cancellation is radial and
        not bit-exact, so float32 round-off leaves a small single-weight excursion at a conserved
        norm. That residual is sized in the change's design and is measured per arm in the campaign
        rather than asserted away here.
        """
        from quantumnematode.brain.arch._mlp_topology import MLPTopology
        from quantumnematode.learning_rules import ScalingOptions, ThreeFactorRule
        from quantumnematode.learning_rules.three_factor import ThreeFactorBatch
        from torch import nn

        # A dense topology stands in deliberately: the interaction under test is the RULE's, not
        # the connectome's, and no forward pass is run, so every trace is zero and the decay term
        # is the only thing that can write a weight.
        actor = nn.Sequential(nn.Linear(4, 3), nn.Tanh(), nn.Linear(3, 2))
        topology = MLPTopology(
            actor,
            enable_activity_traces=True,
            trace_decay=0.9,
            plastic_layers="hidden",
        )
        before = topology.plastic_weights[0].detach().clone()
        rule = ThreeFactorRule(
            topology,
            plasticity_rate=0.1,
            weight_decay=0.5,
            weight_bound=3.0,
            baseline_rate=0.01,
            freeze_updates=False,
            modulated=True,
            eligibility="hebbian",
            scaling=ScalingOptions(normalise_modulator=False, normalise_trace=False),
            homeostasis=homeostasis,
            device=torch.device("cpu"),
        )
        for _ in range(200):
            rule.step(topology, ThreeFactorBatch(reward=1.0))
        after = topology.plastic_weights[0].detach()
        norm_ratio = float(after.norm()) / float(before.norm())
        if homeostasis:
            assert norm_ratio == pytest.approx(1.0, abs=1e-6), (
                "the homeostatic rescale must hold the incoming norm under decay alone"
            )
        else:
            assert norm_ratio < 1.0 - 1e-3, (
                "without homeostasis the uncredited weights must lose norm, which is the confound "
                "the validator refuses"
            )


# ───────────────────────── the R.1c arms as configured ──────────────────────

_CONFIG_DIR = Path(__file__).resolve().parents[6] / "configs" / "scenarios" / "foraging"
_BASE = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350"
_STEM = f"{_BASE}_nodepert"
# The recipe every arm adds to the committed base, and nothing else.
_RECIPE = {
    "brain.config.learning_rule": "three_factor",
    "brain.config.enable_activity_traces": True,
    "brain.config.plasticity_normalise_modulator": True,
    "brain.config.plasticity_normalise_trace": True,
    "brain.config.plasticity_homeostasis": True,
    "brain.config.initial_log_std": -1.0,
    "brain.config.plasticity_rate": 0.001,
    "brain.config.plasticity_eligibility": "node_perturbation",
    # Calibrated on this substrate: 0.2 was carried from the one-step control and costs the
    # learning arm 46% of its level here. Pinned in the test so a config drifting back to the
    # uncalibrated value cannot pass as registered.
    "brain.config.plasticity_node_noise": 0.1,
    "brain.config.trace_decay": 0.9,
}
_ARMS = ("full", "causal", "hop1", "motor", "motor_last")


def _flat(mapping: dict, prefix: str = "") -> dict:
    out: dict = {}
    for key, value in mapping.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flat(value, path + "."))
        else:
            out[path] = value
    return out


def _load_yaml(name: str) -> dict:
    import yaml

    return _flat(yaml.safe_load((_CONFIG_DIR / f"{name}.yml").read_text()))


class TestTheArmsDifferByTheRegisteredKeysOnly:
    """A stray key would be a second manipulation nobody registered."""

    @pytest.mark.parametrize("arm", _ARMS)
    @pytest.mark.parametrize("frozen", [False, True])
    def test_only_the_recipe_and_the_set_move(self, arm: str, *, frozen: bool) -> None:
        base = _load_yaml(_BASE)
        variant = _load_yaml(f"{_STEM}_{arm}{'_frozen' if frozen else ''}")
        allowed = dict(_RECIPE)
        allowed["brain.config.plasticity_perturbation_set"] = arm
        if frozen:
            allowed["brain.config.freeze_updates"] = True
        assert set(base) - set(variant) == set(), "no base key may be dropped"
        for key, value in variant.items():
            if key in base and base[key] == value:
                continue
            assert key in allowed, f"{key} moved but is not a registered key"
            assert value == allowed[key], f"{key} is {value!r}, registered as {allowed[key]!r}"

    @pytest.mark.parametrize("arm", _ARMS)
    def test_the_frozen_control_keeps_the_same_mask_and_sigma(self, arm: str) -> None:
        # The perturbation's cost to the policy must be present in BOTH arms of the pair, so the
        # contrast measures the update and not the net effect of switching perturbation on.
        learning = _load_yaml(f"{_STEM}_{arm}")
        frozen = _load_yaml(f"{_STEM}_{arm}_frozen")
        assert frozen["brain.config.plasticity_perturbation_set"] == arm
        assert (
            frozen["brain.config.plasticity_node_noise"]
            == learning["brain.config.plasticity_node_noise"]
        )
        assert frozen["brain.config.freeze_updates"] is True
        assert learning["brain.config.freeze_updates"] is False

    @pytest.mark.parametrize("arm", _ARMS)
    @pytest.mark.parametrize("frozen", [False, True])
    def test_every_arm_pins_homeostasis(self, arm: str, *, frozen: bool) -> None:
        # The validator refuses a restricted set without it; pinning it in the file means no arm
        # depends on a default to stay interpretable.
        variant = _load_yaml(f"{_STEM}_{arm}{'_frozen' if frozen else ''}")
        assert variant["brain.config.plasticity_homeostasis"] is True

    def test_the_cell_is_the_committed_hard_food_one(self) -> None:
        # Block V's cell, so a result here is comparable to the PPO reference already recorded.
        base = _load_yaml(_BASE)
        assert base["max_steps"] == 350
        assert base["environment.foraging.target_foods_to_collect"] == 20
        assert base["brain.config.forward_pass_depth"] == 4


class TestTheDerivedSetIsCheckedAtRuntime:
    """A declared set is only worth reporting if the wiring actually yields it."""

    def test_a_drifted_readout_pool_is_refused(self, connectome: Connectome) -> None:
        # If the hop walk and the readout disagree about which units are at distance zero, one of
        # them has changed and every mask built from the walk is wrong. Invariant, not a count, so
        # it holds for any connectome source.
        topology = _topology(connectome, "motor")
        distance = topology._readout_hop_distances(topology._motor_flat_indices.cpu().tolist())
        wrong_pool = [0, 1, 2]
        with pytest.raises(ValueError, match="are not the readout pool"):
            topology._validate_derived_set("motor", distance, wrong_pool)

    def test_a_set_selecting_nothing_is_refused(self, connectome: Connectome) -> None:
        # A mask that selects no unit would silence the eligibility entirely, and the arm would read
        # as a rule that learns nothing rather than one given nothing to learn from.
        topology = _topology(connectome, "motor")
        unreachable = torch.full((topology.n_neurons,), topology.n_neurons + 1, dtype=torch.long)
        with pytest.raises(ValueError, match="selects no unit"):
            topology._validate_derived_set("motor", unreachable, [])

    def test_the_real_graph_passes_every_set(self, connectome: Connectome) -> None:
        # The check runs on every restricted build, so a false positive would break all of them.
        for name in ("causal", "hop1", "motor", "motor_last"):
            assert _topology(connectome, name).perturbation_dimension()["units"] > 0
