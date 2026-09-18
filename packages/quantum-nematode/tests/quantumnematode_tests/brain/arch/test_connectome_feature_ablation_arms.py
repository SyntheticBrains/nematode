"""The L.4 and L.5 ablation arms: one key each, and each ablation is only what it claims to be.

* **One key.** Each of the eight configs differs from its committed L.1 wide parent in exactly one
  resolved key, so the arms are L.1's arms with one feature removed and nothing else.
* **Atlas signs change signs.** Magnitudes, and every other parameter, bitwise identical; only the
  sign structure of `w_chem` moves. B.1 asserted this at the pooled width; this asserts it at 39.
* **Gap junctions off changes no parameter.** `g_gap` is a buffer built from the data, so the
  ablation is the forward pass and nothing else.
* **The pool numbers are facts, not prose.** 199 gap junctions touch the readout pool, 47 lie within
  it, all 39 pool neurons carry one; 323 chemical inputs, 311 grounded, 275 E / 36 I.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain, ConnectomePPOBrainConfig
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.connectome.loader import load_cook_2019_hermaphrodite
from quantumnematode.connectome.neurotransmitters import read_atlas_transmitters, sign_for
from quantumnematode.utils.config_loader import load_simulation_config

_REPO_ROOT = Path(__file__).resolve().parents[6]
_DIR = _REPO_ROOT / "configs" / "scenarios" / "foraging"
_STEM = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_eprop"
_SEED = 7
_POOL_PREFIXES = ("VB", "DB", "VA", "DA")

_PAIRS: list[tuple[str, str, str]] = [
    (f"{_STEM}{_lsuf}{_wsuf}", f"{_STEM}{_lsuf}_{_ab}{_wsuf}", _key)
    for _ab, _key in (("atlas", "synapse_signs"), ("nogap", "enable_gap_junctions"))
    for _lsuf in ("_readout_only_wide", "_frozen_wide")
    for _wsuf in ("", "_rewired_null")
]


def _cfg(name: str) -> ConnectomePPOBrainConfig:
    config = load_simulation_config(str(_DIR / f"{name}.yml")).brain
    assert config is not None
    assert isinstance(config.config, ConnectomePPOBrainConfig)
    return config.config


def _brain(name: str) -> ConnectomePPOBrain:
    torch.manual_seed(_SEED)
    return ConnectomePPOBrain(
        config=_cfg(name).model_copy(update={"seed": _SEED}),
        device=DeviceType.CPU,
    )


def _pool() -> set[str]:
    c = load_cook_2019_hermaphrodite()
    return {
        n
        for n, x in c.neurons.items()
        if x.cell_class == "motor"
        and any(n.startswith(p) and n[len(p) :].isdigit() for p in _POOL_PREFIXES)
    }


class TestOneKeyEach:
    @pytest.mark.parametrize(
        ("parent", "child", "key"),
        _PAIRS,
        ids=[c.split("hard350_eprop_")[1] for _, c, _ in _PAIRS],
    )
    def test_the_child_differs_from_its_parent_in_that_key_alone(
        self,
        parent: str,
        child: str,
        key: str,
    ) -> None:
        a, b = _cfg(parent).model_dump(), _cfg(child).model_dump()
        assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {key}

    def test_the_ablation_values(self) -> None:
        assert _cfg(f"{_STEM}_readout_only_wide_atlas").synapse_signs == "atlas"
        assert _cfg(f"{_STEM}_readout_only_wide_nogap").enable_gap_junctions is False
        assert _cfg(f"{_STEM}_readout_only_wide").synapse_signs == "random"
        assert _cfg(f"{_STEM}_readout_only_wide").enable_gap_junctions is True


class TestEachAblationIsOnlyWhatItClaims:
    @pytest.fixture(scope="class")
    def base(self) -> ConnectomePPOBrain:
        return _brain(f"{_STEM}_readout_only_wide")

    def test_atlas_moves_signs_and_nothing_else(self, base: ConnectomePPOBrain) -> None:
        atlas = _brain(f"{_STEM}_readout_only_wide_atlas")
        pa, pb = dict(base.topology.named_parameters()), dict(atlas.topology.named_parameters())
        assert set(pa) == set(pb)
        for name, param in pa.items():
            other = pb[name].detach()
            if name == "w_chem":
                assert torch.equal(param.detach().abs(), other.abs()), "a magnitude moved"
                flipped = int((param.detach().sign() != other.sign()).sum().item())
                assert flipped > 0, "atlas grounding changed no sign at all"
            else:
                assert torch.equal(param.detach(), other), f"{name} moved under atlas signs"

    def test_no_gap_changes_no_parameter(self, base: ConnectomePPOBrain) -> None:
        nogap = _brain(f"{_STEM}_readout_only_wide_nogap")
        pa, pb = dict(base.topology.named_parameters()), dict(nogap.topology.named_parameters())
        assert set(pa) == set(pb)
        for name, param in pa.items():
            assert torch.equal(param.detach(), pb[name].detach()), (
                f"{name} moved with gap junctions off"
            )
        assert nogap.topology.enable_gap_junctions is False
        assert base.topology.enable_gap_junctions is True

    def test_both_keep_the_per_neuron_width(self) -> None:
        for name in (f"{_STEM}_readout_only_wide_atlas", f"{_STEM}_readout_only_wide_nogap"):
            readout = _brain(name).topology.readout
            assert isinstance(readout, torch.Tensor)
            assert tuple(readout.shape) == (2, 39)


class TestThePoolNumbersAreFacts:
    def test_gap_junctions_on_the_readout_pool(self) -> None:
        c, pool = load_cook_2019_hermaphrodite(), _pool()
        assert len(pool) == 39
        gj = c.gap_junctions
        touch = [g for g in gj if g.neuron_a in pool or g.neuron_b in pool]
        within = [g for g in gj if g.neuron_a in pool and g.neuron_b in pool]
        with_one = ({g.neuron_a for g in touch} | {g.neuron_b for g in touch}) & pool
        assert len(gj) == 1093
        assert len(touch) == 199
        assert len(within) == 47
        assert len(with_one) == 39

    def test_chemical_inputs_to_the_pool_and_their_atlas_signs(self) -> None:
        c, pool = load_cook_2019_hermaphrodite(), _pool()
        atlas = read_atlas_transmitters()
        into = [s for s in c.chemical_synapses if s.post in pool]
        signs = [sign_for(atlas[s.pre]) for s in into if s.pre in atlas]
        grounded = [x for x in signs if x is not None]
        assert len(c.chemical_synapses) == 3709
        assert len(into) == 323
        assert len(grounded) == 311
        assert sum(1 for x in grounded if x > 0) == 275
        assert sum(1 for x in grounded if x < 0) == 36
