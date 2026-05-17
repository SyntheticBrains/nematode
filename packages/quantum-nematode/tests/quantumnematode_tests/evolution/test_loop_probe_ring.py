"""Unit tests for the M6.11 env-derived F0 probe-ring helpers.

Covers the spec scenarios under "Env-Derived F0 Probe Ring" in
``openspec/changes/add-transgenerational-memory-redesign/specs/evolution-framework/spec.md``:
- probe ring uses env predator positions
- configurable count and radius_offset
- pure helper for gradient computation (``_compute_probe_gradient``)

Plus task-3.5 angular-distribution, food-gradient-variants doubling,
no-env-mutation invariant, and empty-predators fallback.

The ring builder is exercised at the ``_build_f0_probe_params`` level
via mock envs + mock predators (avoiding the full env construction
overhead). The helper ``_compute_probe_gradient`` is exercised directly
as a pure function.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from quantumnematode.env.env import PredatorType
from quantumnematode.evolution.loop import _compute_probe_gradient

if TYPE_CHECKING:
    from quantumnematode.evolution.loop import EvolutionLoop as _EvolutionLoop

# ---------------------------------------------------------------------------
# Pure helper: _compute_probe_gradient
# ---------------------------------------------------------------------------


def test_compute_probe_gradient_strength_at_zero_distance() -> None:
    """Strength at the predator's own cell SHALL equal 1.0."""
    strength, _ = _compute_probe_gradient((5, 5), (5, 5))
    assert strength == pytest.approx(1.0)


def test_compute_probe_gradient_falls_off_with_manhattan_distance() -> None:
    """Strength SHALL equal ``1 / (1 + manhattan_distance)`` per the spec helper formula."""
    # Manhattan distance from (5, 5) to (10, 10) = 10.
    strength, _ = _compute_probe_gradient((5, 5), (10, 10))
    assert strength == pytest.approx(1.0 / 11.0)


def test_compute_probe_gradient_direction_via_atan2() -> None:
    """Direction SHALL equal ``atan2(predator_y - probe_y, predator_x - probe_x)``."""
    _, direction = _compute_probe_gradient((0, 0), (3, 4))
    assert direction == pytest.approx(math.atan2(4, 3))


def test_compute_probe_gradient_is_pure_function() -> None:
    """Repeated calls SHALL return bit-identical results (no hidden state)."""
    r1 = _compute_probe_gradient((1, 2), (7, 9))
    r2 = _compute_probe_gradient((1, 2), (7, 9))
    assert r1 == r2


# ---------------------------------------------------------------------------
# Manhattan-ring offset helpers
# ---------------------------------------------------------------------------


def test_manhattan_ring_offsets_perimeter_is_4r() -> None:
    """The L1 ring at radius R SHALL have exactly 4*R perimeter cells."""
    from quantumnematode.evolution.loop import _manhattan_ring_offsets

    for radius in [1, 2, 3, 5, 7]:
        offsets = _manhattan_ring_offsets(radius)
        assert len(offsets) == 4 * radius


def test_manhattan_ring_offsets_all_at_exact_l1_distance() -> None:
    """Every offset on the ring SHALL satisfy |dx| + |dy| == radius."""
    from quantumnematode.evolution.loop import _manhattan_ring_offsets

    for radius in [1, 2, 5, 10]:
        for dx, dy in _manhattan_ring_offsets(radius):
            assert abs(dx) + abs(dy) == radius, (
                f"offset ({dx}, {dy}) not on L1 ring at radius {radius}"
            )


def test_manhattan_ring_offsets_radius_zero() -> None:
    """Radius 0 SHALL return a single (0, 0) offset."""
    from quantumnematode.evolution.loop import _manhattan_ring_offsets

    assert _manhattan_ring_offsets(0) == [(0, 0)]


def test_sample_ring_offsets_returns_full_ring_when_count_exceeds_perimeter() -> None:
    """When count >= perimeter the full ring SHALL be returned."""
    from quantumnematode.evolution.loop import _manhattan_ring_offsets, _sample_ring_offsets

    radius = 3
    full = _manhattan_ring_offsets(radius)  # 12 cells
    sampled = _sample_ring_offsets(radius, count=20)
    assert sampled == full


def test_sample_ring_offsets_downsamples_evenly() -> None:
    """Down-sampling to count < perimeter SHALL produce ``count`` evenly-spaced offsets."""
    from quantumnematode.evolution.loop import _sample_ring_offsets

    sampled = _sample_ring_offsets(radius=5, count=4)
    # All sampled offsets SHALL still satisfy |dx| + |dy| == 5.
    for dx, dy in sampled:
        assert abs(dx) + abs(dy) == 5
    assert len(sampled) == 4


# ---------------------------------------------------------------------------
# Ring builder: _build_f0_probe_params under the M6.11 env-derived path
# ---------------------------------------------------------------------------


def _make_mock_predator(position: tuple[int, int], damage_radius: int = 3) -> Mock:
    """Construct a mock Predator with stationary type + a position + damage radius."""
    pred = Mock()
    pred.position = position
    pred.damage_radius = damage_radius
    pred.predator_type = PredatorType.STATIONARY
    return pred


def _make_mock_env(predators: list[Mock]) -> Mock:
    """Construct a mock env exposing only ``predators`` (the ring builder reads no other attr)."""
    env = Mock()
    env.predators = predators
    return env


def _make_loop_with_probe_ring(probe_ring: object | None) -> _EvolutionLoop:
    """Construct a minimal stub EvolutionLoop with just enough fields for the probe builder."""
    from quantumnematode.evolution.loop import EvolutionLoop

    # Build a stub instance without going through __init__ (which requires
    # an optimizer + fitness + encoder + sim_config). The probe-ring
    # builder only reads ``self.sim_config.evolution.transgenerational.probe_ring``.
    loop = EvolutionLoop.__new__(EvolutionLoop)
    sim_config = Mock()
    transgenerational = Mock()
    transgenerational.probe_ring = probe_ring
    evolution_cfg = Mock()
    evolution_cfg.transgenerational = transgenerational
    sim_config.evolution = evolution_cfg
    loop.sim_config = sim_config
    return loop


def _probe_ring_config(
    count: int = 8,
    radius_offset: int = 1,
    *,
    include_food_gradient_variants: bool = False,
) -> object:
    """Construct a ProbeRingConfig matching the Pydantic schema."""
    from quantumnematode.utils.config_loader import ProbeRingConfig

    return ProbeRingConfig(
        count=count,
        radius_offset=radius_offset,
        include_food_gradient_variants=include_food_gradient_variants,
    )


def test_probe_ring_count_equals_num_predators_times_count() -> None:
    """Five stationary predators x count=8 SHALL produce exactly 40 probes."""
    loop = _make_loop_with_probe_ring(_probe_ring_config(count=8))
    predators = [_make_mock_predator((i * 5, i * 5)) for i in range(5)]
    env = _make_mock_env(predators)

    probe_params = loop._build_f0_probe_params(brain=None, env=env)

    assert len(probe_params) == 5 * 8


def test_probe_ring_strengths_are_nonzero() -> None:
    """Every probe in the env-derived ring SHALL have ``predator_gradient_strength > 0``."""
    loop = _make_loop_with_probe_ring(_probe_ring_config(count=8))
    env = _make_mock_env([_make_mock_predator((10, 10), damage_radius=3)])

    probe_params = loop._build_f0_probe_params(brain=None, env=env)

    for p in probe_params:
        assert p.predator_gradient_strength is not None
        assert p.predator_gradient_strength > 0


def test_probe_ring_configurable_count_and_radius_offset() -> None:
    """3 predators x count=4 x offset=2, damage_radius=3 SHALL produce 12 probes at distance 5.

    Spec scenario "configurable count and radius_offset": the probe ring's
    distance from each source predator equals ``damage_radius + radius_offset``.
    The Manhattan-ring builder enforces this exactly (no Euclidean drift) —
    every probe SHALL sit at L1 distance 5.
    """
    loop = _make_loop_with_probe_ring(_probe_ring_config(count=4, radius_offset=2))
    predators = [_make_mock_predator((20, 20), damage_radius=3) for _ in range(3)]
    env = _make_mock_env(predators)

    probe_params = loop._build_f0_probe_params(brain=None, env=env)

    assert len(probe_params) == 3 * 4
    radius = 3 + 2
    for p in probe_params:
        # strength = 1 / (1 + manhattan), so manhattan = 1/strength - 1.
        assert p.predator_gradient_strength is not None
        manhattan = round(1.0 / p.predator_gradient_strength - 1.0)
        assert manhattan == radius, (
            f"probe Manhattan distance {manhattan} != expected radius {radius}"
        )


def test_probe_ring_exact_manhattan_distance_at_default_count_8() -> None:
    """At the default count=8 every probe SHALL sit at exact L1 distance == damage_radius + offset.

    Regression test for the M6.11 Manhattan-ring rewrite (replaced an
    earlier Euclidean cos/sin projection that produced variable Manhattan
    distances 5-8 at radius=5 for the 8-position ring). The L1 ring at
    radius=5 has 4*5=20 perimeter cells; count=8 down-samples evenly.
    """
    loop = _make_loop_with_probe_ring(_probe_ring_config(count=8, radius_offset=1))
    env = _make_mock_env([_make_mock_predator((10, 10), damage_radius=4)])

    probe_params = loop._build_f0_probe_params(brain=None, env=env)

    assert len(probe_params) == 8
    radius = 4 + 1  # damage_radius + radius_offset
    for p in probe_params:
        assert p.predator_gradient_strength is not None
        manhattan = round(1.0 / p.predator_gradient_strength - 1.0)
        assert manhattan == radius, (
            f"probe Manhattan distance {manhattan} != expected radius {radius} at count=8"
        )


def test_probe_ring_directions_angularly_distributed() -> None:
    """Ring positions SHALL be approximately evenly distributed around the predator center."""
    loop = _make_loop_with_probe_ring(_probe_ring_config(count=8))
    env = _make_mock_env([_make_mock_predator((10, 10), damage_radius=3)])

    probe_params = loop._build_f0_probe_params(brain=None, env=env)

    # 8 probes — directions SHALL collectively span more than half the
    # circle (sanity check that we're not collapsed to one angle).
    directions: list[float] = []
    for p in probe_params:
        assert p.predator_gradient_direction is not None
        directions.append(p.predator_gradient_direction)
    spread = max(directions) - min(directions)
    assert spread > math.pi, f"directions not spread over more than half-circle (spread={spread})"


def test_probe_ring_include_food_gradient_variants_doubles_count() -> None:
    """``include_food_gradient_variants: true`` SHALL emit 2 x count probes per predator."""
    loop = _make_loop_with_probe_ring(
        _probe_ring_config(count=8, include_food_gradient_variants=True),
    )
    env = _make_mock_env([_make_mock_predator((10, 10), damage_radius=3)])

    probe_params = loop._build_f0_probe_params(brain=None, env=env)

    assert len(probe_params) == 16
    # Half SHALL have food_gradient_strength = 0, half > 0.
    zero_food = [p for p in probe_params if p.food_gradient_strength == 0.0]
    nonzero_food = [
        p
        for p in probe_params
        if p.food_gradient_strength is not None and p.food_gradient_strength > 0.0
    ]
    assert len(zero_food) == 8
    assert len(nonzero_food) == 8


def test_probe_ring_falls_back_to_legacy_when_no_probe_ring_config() -> None:
    """When ``probe_ring is None`` the builder SHALL emit the M6 legacy 3-probe path."""
    loop = _make_loop_with_probe_ring(probe_ring=None)
    env = _make_mock_env([_make_mock_predator((10, 10), damage_radius=3)])

    probe_params = loop._build_f0_probe_params(brain=None, env=env)

    assert len(probe_params) == 3
    # Legacy probes all have zero predator gradient.
    assert all(p.predator_gradient_strength == 0.0 for p in probe_params)


def test_probe_ring_falls_back_to_legacy_when_env_has_no_predators() -> None:
    """When the env has no stationary predators the builder SHALL emit the legacy 3-probe path."""
    loop = _make_loop_with_probe_ring(_probe_ring_config(count=8))
    env = _make_mock_env([])  # No predators

    probe_params = loop._build_f0_probe_params(brain=None, env=env)

    assert len(probe_params) == 3
    assert all(p.predator_gradient_strength == 0.0 for p in probe_params)


def test_probe_ring_falls_back_when_env_is_none() -> None:
    """When ``env is None`` (no env supplied) the builder SHALL emit the legacy 3-probe path."""
    loop = _make_loop_with_probe_ring(_probe_ring_config(count=8))

    probe_params = loop._build_f0_probe_params(brain=None, env=None)

    assert len(probe_params) == 3
    assert all(p.predator_gradient_strength == 0.0 for p in probe_params)


def test_probe_ring_does_not_mutate_env() -> None:
    """The builder SHALL NOT mutate the env's predators list."""
    loop = _make_loop_with_probe_ring(_probe_ring_config(count=4))
    pred = _make_mock_predator((10, 10), damage_radius=3)
    env = _make_mock_env([pred])
    original_position = pred.position
    original_damage_radius = pred.damage_radius

    _ = loop._build_f0_probe_params(brain=None, env=env)

    assert pred.position == original_position
    assert pred.damage_radius == original_damage_radius
    assert len(env.predators) == 1
