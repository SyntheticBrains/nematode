"""Gap-junction weight options and the gap-outlier wirings on the connectome brain.

Covers the connectome-ppo-brain requirements "Gap-junction weight options" (unit weights remove the
counts and keep the normalisers; the scale multiplies the normalised matrix; defaults are
byte-identical) and "Gap-outlier wirings" (held outliers keep their place; scattered outliers land
elsewhere; existing nulls are unchanged).
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import hashlib
from functools import cache
from pathlib import Path

import numpy as np
import pytest
import torch
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain, ConnectomePPOBrainConfig
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.connectome import (
    load_cook_2019_hermaphrodite,
    load_emmons_2024_hermaphrodite,
)
from quantumnematode.connectome.rewiring import rewire_degree_preserving
from quantumnematode.utils.config_loader import load_simulation_config

_REPO = Path(__file__).resolve().parents[6]
_T35 = (
    _REPO
    / "configs/scenarios/thermal_foraging"
    / "connectomeppo_small_continuous2d_thermal_klinotaxis_t35.yml"
)
_SEED = 11
_EMMONS = "emmons_2024_hermaphrodite"
_OUTLIER_MIN = 20

# Digests of the gap matrix and the chemical weights, and the shared generator's next draw, built
# at seed 11 before the gap options existed. Every quantity here is computed in numpy, so the
# digests do not depend on the platform's linear-algebra backend.
_PINNED = {
    ("cook_2019_hermaphrodite", "wild_type"): ("8d06d2ff604d992f", "9dd960d54a54a3b3"),
    ("cook_2019_hermaphrodite", "rewired_gap_junctions_only"): (
        "5a75e5a0d1cb75a0",
        "9dd960d54a54a3b3",
    ),
    ("cook_2019_hermaphrodite", "rewired_degree_preserving"): (
        "90ccfc302be5e18f",
        "8334483b16066445",
    ),
    (_EMMONS, "wild_type"): ("a95dc3532c457b73", "9dd960d54a54a3b3"),
    (_EMMONS, "rewired_gap_junctions_only"): ("077e48d561dd7329", "9dd960d54a54a3b3"),
    (_EMMONS, "rewired_degree_preserving"): ("53261e1ebaec4e41", "8334483b16066445"),
}
_PINNED_NEXT_DRAW = 0.23179450181263905


def _config(**overrides: object) -> ConnectomePPOBrainConfig:
    container = load_simulation_config(str(_T35)).brain
    assert container is not None
    assert isinstance(container.config, ConnectomePPOBrainConfig)
    return container.config.model_copy(update={"seed": _SEED, **overrides})


def _brain(**overrides: object) -> ConnectomePPOBrain:
    return ConnectomePPOBrain(config=_config(**overrides), device=DeviceType.CPU)


@cache
def _gap(**overrides: object) -> np.ndarray:
    """Return the built gap matrix, cached by its overrides."""
    return _brain(**overrides).topology.g_gap.detach().numpy().copy()


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


@cache
def _names() -> list[str]:
    return list(_brain().topology.neuron_names)


def _hand_gap(connectome: object, weight_of: object) -> np.ndarray:
    """Build the normalised gap matrix by hand: ``weight(count) / sqrt(d_i d_j)``."""
    names = _names()
    idx = {name: k for k, name in enumerate(names)}
    degree = np.zeros(len(names))
    for gj in connectome.gap_junctions:  # type: ignore[attr-defined]
        degree[idx[gj.neuron_a]] += 1
        degree[idx[gj.neuron_b]] += 1
    expected = np.zeros((len(names), len(names)))
    for gj in connectome.gap_junctions:  # type: ignore[attr-defined]
        a, b = idx[gj.neuron_a], idx[gj.neuron_b]
        value = weight_of(gj.weight) / np.sqrt(degree[a] * degree[b])  # type: ignore[operator]
        expected[a, b] = expected[b, a] = value
    return expected


def _outliers(min_count: int = _OUTLIER_MIN) -> set[tuple[str, str]]:
    """Return the Emmons gap pairs whose count is at least ``min_count``."""
    gaps = load_emmons_2024_hermaphrodite().gap_junctions
    return {(g.neuron_a, g.neuron_b) for g in gaps if g.weight >= min_count}


def _pair_values(gap: np.ndarray, pairs: set[tuple[str, str]]) -> dict[tuple[str, str], float]:
    idx = {name: k for k, name in enumerate(_names())}
    return {(a, b): float(gap[idx[a], idx[b]]) for a, b in pairs}


# ── Weight options ───────────────────────────────────────────────────────────────────────────


class TestGapWeightOptions:
    def test_unit_weights_remove_the_counts_and_keep_the_normalisers(self) -> None:
        """Scenario "Unit weights remove the counts and keep the normalisers"."""
        expected = _hand_gap(load_cook_2019_hermaphrodite(), lambda _count: 1.0)
        np.testing.assert_allclose(_gap(gap_weight_mode="unit"), expected, rtol=1e-6, atol=0)

    def test_log_count_compresses_each_count_before_the_normalisers(self) -> None:
        expected = _hand_gap(load_cook_2019_hermaphrodite(), np.log1p)
        np.testing.assert_allclose(_gap(gap_weight_mode="log_count"), expected, rtol=1e-6, atol=0)

    def test_count_is_the_hand_computation_too(self) -> None:
        expected = _hand_gap(load_cook_2019_hermaphrodite(), float)
        np.testing.assert_allclose(_gap(), expected, rtol=1e-6, atol=0)

    def test_every_mode_keeps_the_binary_pattern(self) -> None:
        pattern = _gap() != 0
        for mode in ("unit", "log_count"):
            assert np.array_equal(_gap(gap_weight_mode=mode) != 0, pattern)

    @pytest.mark.parametrize("mode", ["count", "unit", "log_count"])
    @pytest.mark.parametrize("scale", [0.25, 0.5, 2.0])
    def test_the_scale_multiplies_the_normalised_matrix(self, mode: str, scale: float) -> None:
        """Scenario "The scale multiplies the normalised matrix"."""
        scaled = _gap(gap_weight_mode=mode, gap_weight_scale=scale)
        np.testing.assert_allclose(scaled, _gap(gap_weight_mode=mode) * scale, rtol=1e-6, atol=0)

    def test_under_a_null_the_counts_travel_before_the_mode(self) -> None:
        """A unit-weight gap-only null differs from the unit wild type in placement alone."""
        cook = load_cook_2019_hermaphrodite()
        null = rewire_degree_preserving(cook, np.random.default_rng(_SEED), rewire_chemical=False)
        expected = _hand_gap(null, lambda _count: 1.0)
        built = _gap(gap_weight_mode="unit", wiring="rewired_gap_junctions_only")
        np.testing.assert_allclose(built, expected, rtol=1e-6, atol=0)

    def test_plastic_gaps_multiply_the_option_s_matrix(self) -> None:
        topo = _brain(plastic_gaps=True, gap_weight_mode="unit", gap_weight_scale=0.5).topology
        assert torch.equal(topo.gap_matrix().detach(), topo.g_gap)
        np.testing.assert_allclose(
            topo.g_gap.numpy(),
            _gap(gap_weight_mode="unit", gap_weight_scale=0.5),
        )

    def test_a_checkpoint_from_another_gap_weighting_is_refused(self) -> None:
        unit = _brain(gap_weight_mode="unit")
        with pytest.raises(ValueError, match="a different wiring"):
            unit.load_weight_components(_brain().get_weight_components())


class TestDefaultsAreByteIdentical:
    """Scenario "Defaults are byte-identical", and "Existing nulls are unchanged"."""

    @pytest.mark.parametrize(("source", "wiring"), sorted(_PINNED))
    def test_the_gap_matrix_weights_and_next_draw_are_pinned(
        self,
        source: str,
        wiring: str,
    ) -> None:
        brain = _brain(connectome_source=source, wiring=wiring)
        gap_digest, chem_digest = _PINNED[source, wiring]
        assert _digest(brain.topology.g_gap.detach().numpy()) == gap_digest
        assert _digest(brain.topology.w_chem.detach().numpy()) == chem_digest
        assert float(brain.rng.random()) == _PINNED_NEXT_DRAW

    def test_explicit_defaults_are_the_implicit_build(self) -> None:
        torch.manual_seed(0)
        implicit = _brain()
        torch.manual_seed(0)
        explicit = _brain(gap_weight_mode="count", gap_weight_scale=1.0)
        for mine, theirs in zip(
            implicit.topology.learnable_parameters,
            explicit.topology.learnable_parameters,
            strict=True,
        ):
            assert torch.equal(mine, theirs)
        torch.manual_seed(1)
        for step in range(4):
            params = BrainParams(
                temperature=20.0 + step,
                temperature_lateral_gradient=0.1,
                temperature_ddt=0.01,
                cultivation_temperature=20.0,
            )
            state = torch.get_rng_state()
            first = implicit.run_brain(
                params,
                reward=None,
                input_data=None,
                top_only=False,
                top_randomize=False,
            )
            torch.set_rng_state(state)
            second = explicit.run_brain(
                params,
                reward=None,
                input_data=None,
                top_only=False,
                top_randomize=False,
            )
            assert repr(first) == repr(second)
        assert float(implicit.rng.random()) == float(explicit.rng.random())


# ── Gap-outlier wirings ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def emmons_wild() -> ConnectomePPOBrain:
    """Build the Emmons wild type once for the outlier tests."""
    return _brain(connectome_source=_EMMONS)


class TestGapOutlierWirings:
    @pytest.mark.parametrize(
        "wiring",
        ["rewired_gap_outliers_held", "rewired_gap_outliers_scattered"],
    )
    def test_the_chemical_graph_and_gap_degrees_are_held(
        self,
        emmons_wild: ConnectomePPOBrain,
        wiring: str,
    ) -> None:
        null = _brain(connectome_source=_EMMONS, wiring=wiring).topology
        wild = emmons_wild.topology
        assert torch.equal(null.m_chem, wild.m_chem)
        assert torch.equal(null.w_chem, wild.w_chem)
        assert not torch.equal(null.g_gap, wild.g_gap)
        assert torch.equal((null.g_gap != 0).sum(dim=0), (wild.g_gap != 0).sum(dim=0))

    def test_held_outliers_keep_their_place(self, emmons_wild: ConnectomePPOBrain) -> None:
        """Scenario "Held outliers keep their place"."""
        held = _brain(connectome_source=_EMMONS, wiring="rewired_gap_outliers_held")
        outliers = _outliers()
        report = held.gap_swap_report
        assert report["accepted"] == report["target"]
        assert report["moved"] > len(outliers)
        # Degrees are held, so an outlier on its own pair with its own count has its wild-type
        # coupling exactly.
        values = _pair_values(held.topology.g_gap.numpy(), outliers)
        wild_values = _pair_values(emmons_wild.topology.g_gap.numpy(), outliers)
        for pair in outliers:
            assert values[pair] == pytest.approx(wild_values[pair], rel=1e-6)

    def test_scattered_outliers_land_elsewhere(self, emmons_wild: ConnectomePPOBrain) -> None:
        """Scenario "Scattered outliers land elsewhere"."""
        scattered = _brain(connectome_source=_EMMONS, wiring="rewired_gap_outliers_scattered")
        outliers = _outliers()
        report = scattered.gap_swap_report
        assert report["target"] == len(outliers)
        assert report["accepted"] == len(outliers)
        assert report["moved"] <= 2 * len(outliers)
        values = _pair_values(scattered.topology.g_gap.numpy(), outliers)
        wild_values = _pair_values(emmons_wild.topology.g_gap.numpy(), outliers)
        assert all(values[pair] != wild_values[pair] for pair in outliers)
        # The largest couplings now sit on new pairs.
        assert float(scattered.topology.g_gap.max()) > 1.0

    def test_the_threshold_narrows_the_outliers(self) -> None:
        report = _brain(
            connectome_source=_EMMONS,
            wiring="rewired_gap_outliers_scattered",
            gap_outlier_min_count=100,
        ).gap_swap_report
        assert report["target"] == report["accepted"] == len(_outliers(100))
        assert len(_outliers(100)) < len(_outliers())

    def test_other_wirings_report_nothing(self) -> None:
        assert _brain(wiring="rewired_gap_junctions_only").gap_swap_report == {}


# ── Refusals ─────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"enable_gap_junctions": False, "gap_weight_mode": "unit"}, "never read"),
        ({"enable_gap_junctions": False, "gap_weight_scale": 0.5}, "never read"),
        ({"gap_outlier_min_count": 10}, "never read"),
        (
            {"enable_gap_junctions": False, "wiring": "rewired_gap_outliers_held"},
            "requires enable_gap_junctions",
        ),
    ],
)
def test_unread_or_meaningless_settings_are_refused(
    overrides: dict[str, object],
    match: str,
) -> None:
    """A setting the build would accept and ignore is refused, by the validator and the build."""
    with pytest.raises(ValueError, match=match):
        ConnectomePPOBrainConfig(**overrides)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=match):
        _brain(**overrides)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"gap_weight_mode": "squared"}, "Unsupported gap_weight_mode"),
        ({"gap_weight_scale": 0.0}, "must be positive"),
        ({"wiring": "rewired_gap_outliers_held", "gap_outlier_min_count": 0}, "at least 1"),
        (
            {"wiring": "rewired_gap_outliers_scattered", "gap_outlier_min_count": 100_000},
            "no gap pair",
        ),
    ],
)
def test_a_copied_config_is_still_held_to_the_bounds(
    overrides: dict[str, object],
    match: str,
) -> None:
    """``model_copy`` skips the field bounds; the build repeats them."""
    with pytest.raises(ValueError, match=match):
        _brain(**overrides)
