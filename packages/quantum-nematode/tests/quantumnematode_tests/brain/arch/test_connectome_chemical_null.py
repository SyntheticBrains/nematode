"""The chemical-only rewired null on the brain: what it changes, and what it leaves alone.

Covers the connectome-ppo-brain requirement "Degree-preserving rewired-null wiring option": the
chemical-only null holds gap junctions and autapses at the wild type, so the built gap-junction
buffer is the wild type's bit for bit while the chemical mask differs; the value reaches the brain,
and the measured prior and the fan-in draw both build on it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.utils.config_loader import load_simulation_config

_REPO_ROOT = Path(__file__).resolve().parents[6]
_WILD = (
    _REPO_ROOT
    / "configs"
    / "scenarios"
    / "foraging"
    / "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350.yml"
)
_SEED = 11


def _brain(**overrides: object) -> ConnectomePPOBrain:
    container = load_simulation_config(str(_WILD)).brain
    assert container is not None
    assert isinstance(container.config, ConnectomePPOBrainConfig)
    cfg = container.config.model_copy(update={"seed": _SEED, **overrides})
    return ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)


@pytest.fixture(scope="module")
def arms() -> dict[str, ConnectomePPOBrain]:
    """Build the wild type and both nulls at one seed."""
    return {
        w: _brain(wiring=w)
        for w in ("wild_type", "rewired_degree_preserving", "rewired_chemical_only")
    }


def test_the_value_validates() -> None:
    """The new wiring value is accepted by the configuration model."""
    container = load_simulation_config(str(_WILD)).brain
    assert container is not None
    assert isinstance(container.config, ConnectomePPOBrainConfig)
    ConnectomePPOBrainConfig.model_validate(
        {**container.config.model_dump(), "wiring": "rewired_chemical_only"},
    )


def test_gap_junctions_are_bit_identical_to_the_wild_type(
    arms: dict[str, ConnectomePPOBrain],
) -> None:
    """The coupling the null holds is the wild type's; the full null's is not."""
    wild = arms["wild_type"].topology.g_gap
    assert torch.equal(arms["rewired_chemical_only"].topology.g_gap, wild)
    assert not torch.equal(arms["rewired_degree_preserving"].topology.g_gap, wild)


def test_the_chemical_mask_differs_and_keeps_the_autapses(
    arms: dict[str, ConnectomePPOBrain],
) -> None:
    """The value reaches the brain: a different mask, the same diagonal, the same edge count."""
    wild = arms["wild_type"].topology.m_chem
    chem = arms["rewired_chemical_only"].topology.m_chem
    assert not torch.equal(chem, wild)
    assert int(chem.sum()) == int(wild.sum())
    assert torch.equal(torch.diagonal(chem), torch.diagonal(wild))
    assert int(torch.diagonal(wild).sum()) == 38
    assert int(torch.diagonal(arms["rewired_degree_preserving"].topology.m_chem).sum()) < 38


def test_in_and_out_degree_match_the_wild_type(arms: dict[str, ConnectomePPOBrain]) -> None:
    """Per-neuron fan-in, and so every per-neuron scale, is the wild type's."""
    wild = arms["wild_type"].topology.m_chem
    chem = arms["rewired_chemical_only"].topology.m_chem
    assert torch.equal(chem.sum(dim=0), wild.sum(dim=0))
    assert torch.equal(chem.sum(dim=1), wild.sum(dim=1))


@pytest.mark.parametrize(
    "overrides",
    [
        {"weight_prior": "measured"},
        {"weight_prior": "measured_signs", "weight_draw": "per_neuron_fanin"},
        {"weight_prior": "measured_shuffled", "weight_draw": "per_neuron_fanin"},
    ],
    ids=["measured", "signs-fanin", "shuffled-fanin"],
)
def test_the_measured_prior_and_the_fan_in_draw_build_on_it(overrides: dict[str, object]) -> None:
    """Chemical in-degree is kept, so the per-neuron placements apply unchanged."""
    brain = _brain(wiring="rewired_chemical_only", **overrides)
    assert brain.topology.w_chem.shape[0] == brain.topology.n_neurons
