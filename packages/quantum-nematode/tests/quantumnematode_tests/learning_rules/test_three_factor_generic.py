"""The generic rule is bit-identical to the pre-generalisation rule on the connectome.

Making the rule read a seam instead of naming the connectome's attributes
must not move a single bit of what it does to the connectome. The frozen
reference is the pre-rewrite update as a free function; both are driven
from deep-copied identical state across every mode the rule has.
"""

from __future__ import annotations

import copy

import pytest
import torch
from quantumnematode.brain.arch._topology import BrainTopology, PlasticTopology
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
    ConnectomeTopology,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.learning_rules import ConnectomeThreeFactorRule, ThreeFactorRule
from quantumnematode.learning_rules.three_factor import (
    BASELINE_KEY,
    MEAN_ABS_DELTA_KEY,
    PREDICTION_ERROR_KEY,
    SATURATED_FRACTION_KEY,
    ThreeFactorBatch,
)

from ._legacy_three_factor_reference import legacy_three_factor_step

_SEED = 5150
_HYPER = {
    "plasticity_rate": 0.2,
    "weight_decay": 0.01,
    "weight_bound": 0.4,  # low enough that some entries saturate
    "baseline_rate": 0.3,
}
_REWARDS = [1.0, -2.0, 3.5, 0.0, 2.25, -0.75]


def _topology() -> ConnectomeTopology:
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
    # A structured, non-uniform trace so masking, decay and clamping all bite.
    torch.manual_seed(_SEED + 1)
    with torch.no_grad():
        topo.activity_traces.copy_(topo.apply_weight_mask(torch.randn_like(topo.activity_traces)))
    return topo


class TestSeamConformance:
    def test_connectome_satisfies_both_protocols(self) -> None:
        topo = _topology()
        assert isinstance(topo, PlasticTopology)
        assert isinstance(topo, BrainTopology)

    def test_alias_is_the_same_class(self) -> None:
        assert ConnectomeThreeFactorRule is ThreeFactorRule


@pytest.mark.parametrize(
    ("modulated", "freeze_updates"),
    [(True, False), (False, False), (True, True), (False, True)],
)
class TestGenericRuleMatchesTheFrozenReference:
    """Every mode, driven from identical state, lands on identical bits."""

    def test_weights_and_telemetry_are_bit_identical(
        self,
        *,
        modulated: bool,
        freeze_updates: bool,
    ) -> None:
        live_topo = _topology()
        ref_topo = copy.deepcopy(live_topo)
        assert torch.equal(live_topo.w_chem, ref_topo.w_chem)

        rule = ThreeFactorRule(
            live_topo,
            freeze_updates=freeze_updates,
            modulated=modulated,
            device=torch.device("cpu"),
            **_HYPER,
        )
        ref_baseline = 0.0
        final_saturated = 0.0
        for reward in _REWARDS:
            report = rule.step(live_topo, ThreeFactorBatch(reward=reward))
            final_saturated = report.extra[SATURATED_FRACTION_KEY]
            delta, ref_baseline, mean_abs, saturated = legacy_three_factor_step(
                ref_topo,
                reward=reward,
                baseline=ref_baseline,
                freeze_updates=freeze_updates,
                modulated=modulated,
                **_HYPER,
            )
            assert torch.equal(live_topo.w_chem, ref_topo.w_chem)
            assert report.extra[PREDICTION_ERROR_KEY] == delta
            assert report.extra[BASELINE_KEY] == ref_baseline
            assert report.extra[MEAN_ABS_DELTA_KEY] == mean_abs
            assert report.extra[SATURATED_FRACTION_KEY] == saturated

        if not freeze_updates:
            # Guard against a vacuous pass: the drive must have moved weights
            # and, at this bound, saturated some of them.
            assert final_saturated > 0.0
