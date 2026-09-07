"""Count-scaled chemical weight initialisation on the connectome brain.

Under ``weight_init: count_scaled`` every chemical weight is a standard-normal
draw times ``n / sqrt(sum n^2)`` over the post-synaptic neuron's incoming
synapse counts, drawn in the same edge order from the same generator as the
degree-scaled draw. That shared draw makes the option testable exactly: at one
seed the count-scaled and degree-scaled brains share every ``z``, so their
element-wise weight ratio is ``n * sqrt(k) / sqrt(sum n^2)`` -- proportional to
the count within a neuron, with squared ratios summing to the in-degree ``k``,
which is the unit-sum-of-squares identity on the scale factors.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.connectome.loader import load_cook_2019_hermaphrodite
from quantumnematode.connectome.rewiring import rewire_degree_preserving
from quantumnematode.utils.config_loader import load_simulation_config

_REPO_ROOT = Path(__file__).resolve().parents[6]
_ARMS = _REPO_ROOT / "configs" / "scenarios" / "foraging_predator_thermal"
_STEM = "connectomeppo_small_continuous2d_combined_klinotaxis_plastic_"
_FROZEN = _ARMS / f"{_STEM}frozen.yml"
_FROZEN_COUNT = _ARMS / f"{_STEM}frozen_countinit.yml"
_FROZEN_REWIRED_COUNT = _ARMS / f"{_STEM}frozen_rewired_null_countinit.yml"

_SEED = 23
_GAINS = (
    "food_gains",
    "predator_distal_gains",
    "predator_anterior_gains",
    "predator_posterior_gains",
    "thermotaxis_gains",
)


def _config(path: Path) -> ConnectomePPOBrainConfig:
    config = load_simulation_config(str(path)).brain
    assert config is not None
    assert isinstance(config.config, ConnectomePPOBrainConfig)
    return config.config


def _brain(path: Path, seed: int = _SEED, **overrides: object) -> ConnectomePPOBrain:
    cfg = _config(path).model_copy(update={"seed": seed, **overrides})
    return ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)


@pytest.fixture(scope="module")
def pair() -> tuple[ConnectomePPOBrain, ConnectomePPOBrain]:
    """Degree-scaled and count-scaled wild-type brains at one seed."""
    return _brain(_FROZEN), _brain(_FROZEN_COUNT)


def _incoming_counts() -> dict[str, dict[str, int]]:
    """Post -> {pre: synapse count} from the loaded connectome."""
    out: dict[str, dict[str, int]] = {}
    for syn in load_cook_2019_hermaphrodite().chemical_synapses:
        out.setdefault(syn.post, {})[syn.pre] = syn.weight
    return out


class TestDefaultIsUnchanged:
    def test_default_field_value(self) -> None:
        assert ConnectomePPOBrainConfig.model_fields["weight_init"].default == "degree_scaled"
        assert _config(_FROZEN).weight_init == "degree_scaled"
        assert _config(_FROZEN_COUNT).weight_init == "count_scaled"

    def test_explicit_degree_scaled_is_bit_identical_to_the_default(self) -> None:
        default = _brain(_FROZEN)
        explicit = _brain(_FROZEN, weight_init="degree_scaled")
        assert torch.equal(default.topology.w_chem, explicit.topology.w_chem)
        assert torch.equal(default.topology.readout, explicit.topology.readout)

    def test_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="weight_init"):
            ConnectomePPOBrainConfig.model_validate(
                _config(_FROZEN).model_dump() | {"weight_init": "log_scaled"},
            )


class TestCountScaledMagnitudes:
    def test_same_edges_same_signs(
        self,
        pair: tuple[ConnectomePPOBrain, ConnectomePPOBrain],
    ) -> None:
        degree, count = pair
        assert torch.equal(degree.topology.m_chem, count.topology.m_chem)
        mask = degree.topology.m_chem
        assert torch.equal(
            torch.sign(degree.topology.w_chem[mask]),
            torch.sign(count.topology.w_chem[mask]),
        )
        assert not torch.equal(degree.topology.w_chem, count.topology.w_chem)

    def test_ratio_is_proportional_to_count_and_factors_have_unit_energy(
        self,
        pair: tuple[ConnectomePPOBrain, ConnectomePPOBrain],
    ) -> None:
        """Within every neuron: ratio / n constant, and sum(ratio^2) == in-degree k.

        Since ``ratio = (n / sqrt(sum n^2)) / (1 / sqrt(k))``, the second identity is
        ``sum (n / sqrt(sum n^2))^2 == 1`` on the count-scaled factors themselves.
        """
        degree, count = pair
        topo = degree.topology
        idx = topo._idx
        wd = degree.topology.w_chem.detach().numpy().astype(np.float64)
        wc = count.topology.w_chem.detach().numpy().astype(np.float64)
        checked = 0
        for post, incoming in _incoming_counts().items():
            j = idx[post]
            pres = sorted(incoming)
            ratios = np.array([wc[idx[p], j] / wd[idx[p], j] for p in pres])
            counts = np.array([incoming[p] for p in pres], dtype=np.float64)
            per_count = ratios / counts
            assert np.allclose(per_count, per_count[0], rtol=1e-4), post
            assert np.isclose(float(np.sum(ratios**2)), float(len(pres)), rtol=1e-4), post
            checked += 1
        assert checked > 200

    def test_gap_junctions_are_untouched(
        self,
        pair: tuple[ConnectomePPOBrain, ConnectomePPOBrain],
    ) -> None:
        degree, count = pair
        assert torch.equal(degree.topology.g_gap, count.topology.g_gap)

    def test_periphery_is_untouched(
        self,
        pair: tuple[ConnectomePPOBrain, ConnectomePPOBrain],
    ) -> None:
        degree, count = pair
        assert torch.equal(degree.topology.readout, count.topology.readout)
        assert torch.equal(degree.topology.log_std, count.topology.log_std)
        for gain in _GAINS:
            assert torch.equal(getattr(degree.topology, gain), getattr(count.topology, gain))


class TestRewiredArmCarriesCounts:
    @pytest.fixture(scope="class")
    def arms(self) -> tuple[ConnectomePPOBrain, ConnectomePPOBrain]:
        return _brain(_FROZEN_COUNT), _brain(_FROZEN_REWIRED_COUNT)

    def test_degrees_preserved_and_periphery_identical(
        self,
        arms: tuple[ConnectomePPOBrain, ConnectomePPOBrain],
    ) -> None:
        wild, rewired = arms
        mw, mr = wild.topology.m_chem, rewired.topology.m_chem
        assert not torch.equal(mw, mr)
        assert torch.equal(mw.sum(0), mr.sum(0))
        assert torch.equal(mw.sum(1), mr.sum(1))
        assert torch.equal(wild.topology.readout, rewired.topology.readout)
        assert torch.equal(wild.topology.log_std, rewired.topology.log_std)
        for gain in _GAINS:
            assert torch.equal(getattr(wild.topology, gain), getattr(rewired.topology, gain))

    def test_count_multisets_move_with_the_edges(self) -> None:
        """At the arm's rewire seed some neuron's incident counts differ; degrees do not."""
        wild = load_cook_2019_hermaphrodite()
        rewired = rewire_degree_preserving(wild, np.random.default_rng(_SEED))
        wild_counts = Counter(sorted((s.post, s.weight) for s in wild.chemical_synapses))
        rewired_counts = Counter(sorted((s.post, s.weight) for s in rewired.chemical_synapses))
        assert wild_counts != rewired_counts
        assert Counter(s.weight for s in wild.chemical_synapses) == Counter(
            s.weight for s in rewired.chemical_synapses
        )

    def test_rewired_factors_are_normalised_on_the_rewired_edge_set(self) -> None:
        """Each rewired neuron's count factors still have unit sum of squares."""
        rewired_degree = _brain(_FROZEN_REWIRED_COUNT, weight_init="degree_scaled")
        rewired_count = _brain(_FROZEN_REWIRED_COUNT)
        mask = rewired_degree.topology.m_chem
        assert torch.equal(mask, rewired_count.topology.m_chem)
        wd = rewired_degree.topology.w_chem.detach().to(torch.float64)
        wc = rewired_count.topology.w_chem.detach().to(torch.float64)
        ratio_sq = torch.where(mask, (wc / torch.where(mask, wd, 1.0)) ** 2, 0.0)
        in_degree = mask.sum(0).to(torch.float64)
        has_inputs = in_degree > 0
        assert torch.allclose(ratio_sq.sum(0)[has_inputs], in_degree[has_inputs], rtol=1e-4)
