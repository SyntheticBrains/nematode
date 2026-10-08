"""The gap-only null and plastic gap junctions on the connectome brain.

Covers the connectome-ppo-brain requirements "A gap-only rewired null" (the chemical graph is the
wild type's; existing nulls unchanged) and "Plastic gap junctions under PPO" (symmetric, positive,
on existing pairs; the multipliers receive a gradient; off is byte-identical).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain, ConnectomePPOBrainConfig
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.connectome.loader import load_cook_2019_hermaphrodite
from quantumnematode.connectome.rewiring import rewire_degree_preserving
from quantumnematode.utils.config_loader import load_simulation_config

_REPO = Path(__file__).resolve().parents[6]
_THERMAL = (
    _REPO / "configs/scenarios/thermal_foraging/"
    "connectomeppo_small_continuous2d_thermal_klinotaxis_t35.yml"
)


def _thermal(**overrides: object) -> ConnectomePPOBrain:
    container = load_simulation_config(str(_THERMAL)).brain
    assert container is not None
    assert isinstance(container.config, ConnectomePPOBrainConfig)
    config = container.config.model_copy(update={"seed": 11, **overrides})
    return ConnectomePPOBrain(config=config, device=DeviceType.CPU)


# ── The gap-only null ────────────────────────────────────────────────────────────────────────


class TestGapOnlyRewiring:
    @pytest.mark.parametrize("seed", [1, 17])
    def test_the_chemical_graph_is_held_and_the_gaps_move(self, seed: int) -> None:
        cook = load_cook_2019_hermaphrodite()
        null = rewire_degree_preserving(cook, np.random.default_rng(seed), rewire_chemical=False)
        assert null.chemical_synapses == cook.chemical_synapses
        pairs = {(g.neuron_a, g.neuron_b) for g in cook.gap_junctions}
        moved = {(g.neuron_a, g.neuron_b) for g in null.gap_junctions}
        assert pairs != moved

        def degree(c: object) -> Counter[str]:
            out: Counter[str] = Counter()
            for g in c.gap_junctions:  # type: ignore[attr-defined]
                out[g.neuron_a] += 1
                out[g.neuron_b] += 1
            return out

        assert degree(null) == degree(cook)


class TestGapOnlyBrain:
    def test_the_chemical_mask_and_weights_are_the_wild_types(self) -> None:
        wild = _thermal().topology
        null = _thermal(wiring="rewired_gap_junctions_only").topology
        assert torch.equal(null.m_chem, wild.m_chem)
        assert torch.equal(null.w_chem, wild.w_chem)
        assert not torch.equal(null.g_gap, wild.g_gap)
        assert torch.equal((null.g_gap != 0).sum(dim=0), (wild.g_gap != 0).sum(dim=0))


# ── Plastic gaps ─────────────────────────────────────────────────────────────────────────────


class TestPlasticGaps:
    def test_off_allocates_nothing(self) -> None:
        topo = _thermal().topology
        assert not hasattr(topo, "gap_log_multiplier")
        assert "gap_log_multiplier" not in topo.state_dict()
        assert topo.gap_matrix() is topo.g_gap

    def test_on_appends_one_parameter_last(self) -> None:
        off = _thermal().topology.learnable_parameters
        on_topo = _thermal(plastic_gaps=True).topology
        on = on_topo.learnable_parameters
        assert len(on) == len(off) + 1
        assert on[-1] is on_topo.gap_log_multiplier
        assert [p.shape for p in on[:-1]] == [p.shape for p in off]

    def test_the_coupling_is_symmetric_positive_and_on_existing_pairs(self) -> None:
        topo = _thermal(plastic_gaps=True).topology
        with torch.no_grad():
            topo.gap_log_multiplier.copy_(torch.randn(topo.gap_log_multiplier.shape))
        coupling = topo.gap_matrix().detach()
        existing = topo.g_gap != 0
        assert torch.allclose(coupling, coupling.T)
        assert bool((coupling[existing] > 0).all())
        assert bool((coupling[~existing] == 0).all())

    def test_an_extreme_multiplier_cannot_overflow(self) -> None:
        topo = _thermal(plastic_gaps=True).topology
        with torch.no_grad():
            topo.gap_log_multiplier.fill_(1000.0)
        coupling = topo.gap_matrix().detach()
        existing = topo.g_gap != 0
        assert bool(torch.isfinite(coupling).all())
        assert bool((coupling[~existing] == 0).all())

    def test_it_starts_at_the_wiring_s_own_strengths(self) -> None:
        topo = _thermal(plastic_gaps=True).topology
        assert torch.equal(topo.gap_matrix().detach(), topo.g_gap)

    def test_the_multipliers_receive_a_gradient_and_move(self) -> None:
        brain = _thermal(
            plastic_gaps=True,
            rollout_buffer_size=16,
            num_minibatches=2,
            num_epochs=2,
        )
        before = brain.topology.gap_log_multiplier.detach().clone()
        torch.manual_seed(0)
        for step in range(16):
            params = BrainParams(
                food_concentration=0.1 * (step % 5),
                food_lateral_gradient=0.05 * (step % 3) - 0.05,
                food_dconcentration_dt=0.01 * step,
                temperature=20.0 + step % 4,
                temperature_lateral_gradient=0.1,
                temperature_ddt=0.01,
                cultivation_temperature=20.0,
            )
            brain.run_brain(
                params,
                reward=None,
                input_data=None,
                top_only=False,
                top_randomize=False,
            )
            brain.learn(params, reward=0.1 * (step % 3), episode_done=False)
        after = brain.topology.gap_log_multiplier.detach()
        assert not torch.equal(before, after)

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"dynamics": "leaky"}, "dynamics='leaky'"),
            ({"enable_gap_junctions": False}, "enable_gap_junctions"),
            (
                {"learning_rule": "three_factor", "enable_activity_traces": True},
                "requires learning_rule='ppo'",
            ),
        ],
    )
    def test_unsupported_combinations_are_refused(
        self,
        overrides: dict[str, object],
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            ConnectomePPOBrainConfig(plastic_gaps=True, **overrides)  # type: ignore[arg-type]
