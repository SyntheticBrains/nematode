"""Strength-normalised gap junctions.

Covers the connectome-ppo-brain requirement "Strength-normalised gap junctions": the scenarios
"Strength-normalised couplings are bounded", "Unit weights make the two normalisations equal",
"The default is unchanged" and "A combination with nothing to normalise is refused".
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain, ConnectomePPOBrainConfig
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.connectome.loader import load_emmons_2024_hermaphrodite
from quantumnematode.utils.config_loader import load_simulation_config

_REPO = Path(__file__).resolve().parents[6]
_T35 = (
    _REPO
    / "configs/scenarios/thermal_foraging"
    / "connectomeppo_small_continuous2d_thermal_klinotaxis_t35.yml"
)
_SEED = 11


def _config(**overrides: object) -> ConnectomePPOBrainConfig:
    container = load_simulation_config(str(_T35)).brain
    assert container is not None
    assert isinstance(container.config, ConnectomePPOBrainConfig)
    return container.config.model_copy(
        update={"seed": _SEED, "connectome_source": "emmons_2024_hermaphrodite", **overrides},
    )


def _brain(**overrides: object) -> ConnectomePPOBrain:
    torch.manual_seed(0)
    return ConnectomePPOBrain(config=_config(**overrides), device=DeviceType.CPU)


def _gap(brain: ConnectomePPOBrain) -> np.ndarray:
    return brain.topology.g_gap.double().numpy()


@pytest.fixture(scope="module")
def strength() -> ConnectomePPOBrain:
    """Build the strength-normalised wild type once for the module."""
    return _brain(gap_normalisation="strength")


class TestBound:
    @pytest.mark.parametrize("wiring", ["wild_type", "rewired_gap_junctions_only"])
    def test_every_coupling_is_at_most_one_and_the_radius_is_one(self, wiring: str) -> None:
        """Scenario "Strength-normalised couplings are bounded", on the wild type and the null."""
        gap = _gap(_brain(gap_normalisation="strength", wiring=wiring))
        assert gap.max() <= 1.0 + 1e-6
        assert np.allclose(gap, gap.T)
        assert np.abs(np.linalg.eigvalsh(gap)).max() == pytest.approx(1.0, abs=1e-5)

    def test_counts_couple_far_above_one_under_degree(self) -> None:
        assert _gap(_brain()).max() > 50.0

    def test_a_neuron_without_partners_keeps_zero_couplings(
        self,
        strength: ConnectomePPOBrain,
    ) -> None:
        gap = _gap(strength)
        isolated = np.flatnonzero((gap != 0).sum(axis=0) == 0)
        assert isolated.size > 0
        assert not gap[isolated].any()
        assert not gap[:, isolated].any()

    def test_the_scale_multiplies_after_the_normalisation(
        self,
        strength: ConnectomePPOBrain,
    ) -> None:
        scaled = _gap(_brain(gap_normalisation="strength", gap_weight_scale=0.5))
        assert np.allclose(scaled, 0.5 * _gap(strength), atol=1e-7)


def test_unit_weights_make_the_two_normalisations_equal() -> None:
    """Scenario "Unit weights make the two normalisations equal"."""
    by_degree = _brain(gap_weight_mode="unit").topology.g_gap
    by_strength = _brain(gap_weight_mode="unit", gap_normalisation="strength").topology.g_gap
    assert torch.equal(by_degree, by_strength)


class TestDefault:
    def test_the_default_is_the_degree_normalisation(self) -> None:
        """Scenario "The default is unchanged": every entry is the count over sqrt(d_i d_j)."""
        brain = _brain()
        gap = _gap(brain)
        idx = brain.topology._idx
        raw = np.zeros(gap.shape, dtype=np.float32)
        for junction in load_emmons_2024_hermaphrodite().gap_junctions:
            a, b = idx[junction.neuron_a], idx[junction.neuron_b]
            raw[a, b] = raw[b, a] = float(junction.weight)
        degree = (raw != 0).sum(axis=0).astype(np.float64)
        scale = np.where(degree > 0, 1.0 / np.sqrt(np.where(degree > 0, degree, 1.0)), 0.0)
        scale = scale.astype(np.float32)
        expected = raw * scale[:, np.newaxis] * scale[np.newaxis, :]
        assert np.array_equal(brain.topology.g_gap.numpy(), expected)

    def test_explicit_degree_is_the_implicit_build(self) -> None:
        implicit, explicit = _brain(), _brain(gap_normalisation="degree")
        for name, value in implicit.topology.state_dict().items():
            assert torch.equal(value, explicit.topology.state_dict()[name]), name
        components = implicit.get_weight_components()
        assert "gap_normalisation" not in components["training_state"].state

    def test_strength_is_recorded_with_the_weights(self, strength: ConnectomePPOBrain) -> None:
        state = strength.get_weight_components()["training_state"].state
        assert state["gap_normalisation"] == "strength"


def test_weights_from_the_other_normalisation_are_refused(strength: ConnectomePPOBrain) -> None:
    """The gap matrix differs, so the load's wiring check refuses the warm start."""
    with pytest.raises(ValueError, match="g_gap"):
        _brain().load_weight_components(strength.get_weight_components())


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"enable_gap_junctions": False}, "no gap matrix to normalise"),
        ({"plastic_gaps": True}, "plastic_gaps"),
        ({"gap_normalisation": "partner"}, "Unsupported gap_normalisation"),
    ],
)
def test_unsupported_combinations_are_refused(overrides: dict[str, Any], match: str) -> None:
    """Scenario "A combination with nothing to normalise is refused", at build on a copy."""
    with pytest.raises(ValueError, match=match):
        _brain(**{"gap_normalisation": "strength", **overrides})


def test_the_validator_refuses_too() -> None:
    """A config written with strength and no gap junctions never validates."""
    with pytest.raises(ValueError, match="no gap matrix to normalise"):
        ConnectomePPOBrainConfig(gap_normalisation="strength", enable_gap_junctions=False)
