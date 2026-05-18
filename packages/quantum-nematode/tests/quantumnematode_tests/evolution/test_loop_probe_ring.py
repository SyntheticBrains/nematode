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


def _make_mock_env(predators: list[Mock], grid_size: int = 15) -> Mock:
    """Mock env exposing ``predators`` + ``grid_size`` (needed by safe-probe builder)."""
    env = Mock()
    env.predators = predators
    env.grid_size = grid_size
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


# ---------------------------------------------------------------------------
# safe_probes path (M6.9+ pilot-2 fix — conditional bias-network response)
# ---------------------------------------------------------------------------


def _probe_ring_with_safe_probes(
    *,
    count: int = 8,
    safe_count: int = 16,
    min_predator_distance: int = 6,
) -> object:
    """Build a ProbeRingConfig with the safe_probes sub-block enabled."""
    from quantumnematode.utils.config_loader import ProbeRingConfig, SafeProbesConfig

    return ProbeRingConfig(
        count=count,
        radius_offset=1,
        include_food_gradient_variants=False,
        safe_probes=SafeProbesConfig(
            count=safe_count,
            min_predator_distance=min_predator_distance,
        ),
    )


def test_safe_probes_default_none_omits_safe_set() -> None:
    """``probe_ring.safe_probes is None`` (default) SHALL NOT emit extra probes."""
    loop = _make_loop_with_probe_ring(_probe_ring_config(count=4))
    env = _make_mock_env([_make_mock_predator((5, 5))])
    probes = loop._build_f0_probe_params(brain=None, env=env)
    # 1 predator x 4 ring positions, no safe set.
    assert len(probes) == 4
    assert all(p.food_gradient_strength == 0.0 for p in probes)


def test_safe_probes_emits_additional_probes_when_configured() -> None:
    """``safe_probes`` set SHALL append ``safe_probes.count`` probes after the ring set."""
    loop = _make_loop_with_probe_ring(
        _probe_ring_with_safe_probes(count=4, safe_count=8, min_predator_distance=4),
    )
    env = _make_mock_env([_make_mock_predator((5, 5))])
    probes = loop._build_f0_probe_params(brain=None, env=env)
    # 1 predator x 4 ring + 8 safe = 12 probes.
    assert len(probes) == 12


def test_safe_probes_varying_food_gradient_strength() -> None:
    """Safe probes SHALL sweep ``food_gradient_strength`` across [0.1, 1.0] evenly."""
    loop = _make_loop_with_probe_ring(
        _probe_ring_with_safe_probes(count=4, safe_count=4, min_predator_distance=4),
    )
    env = _make_mock_env([_make_mock_predator((5, 5))])
    probes = loop._build_f0_probe_params(brain=None, env=env)
    safe_part = probes[4:]  # skip ring probes
    food_strengths = [p.food_gradient_strength or 0.0 for p in safe_part]
    # First and last should bracket [0.1, 1.0] inclusive (per the
    # _build_safe_probes formula: 0.1 + 0.9 * (i / (count-1))).
    assert food_strengths[0] == pytest.approx(0.1)
    assert food_strengths[-1] == pytest.approx(1.0)
    # All values monotonically increasing.
    assert all(food_strengths[i] <= food_strengths[i + 1] for i in range(len(food_strengths) - 1))


def test_safe_probes_pred_gradient_strength_lower_than_ring_probes() -> None:
    """Safe probes SHALL have weaker ``predator_gradient_strength`` than ring probes."""
    loop = _make_loop_with_probe_ring(
        _probe_ring_with_safe_probes(count=4, safe_count=4, min_predator_distance=6),
    )
    env = _make_mock_env([_make_mock_predator((7, 7))], grid_size=15)
    probes = loop._build_f0_probe_params(brain=None, env=env)
    ring = probes[:4]
    safe = probes[4:]
    ring_max = max(p.predator_gradient_strength or 0.0 for p in ring)
    safe_max = max(p.predator_gradient_strength or 0.0 for p in safe)
    # Ring is at distance damage_radius+1 = 4 from predator -> strong gradient
    # Safe is at distance >= 6 from predator -> weaker gradient
    assert safe_max < ring_max, (
        f"Expected safe probes to have weaker predator gradient than ring "
        f"(safe_max={safe_max:.4f}, ring_max={ring_max:.4f})"
    )


def test_safe_probes_skipped_with_warning_when_no_valid_positions() -> None:
    """If ``min_predator_distance`` is unsatisfiable on the grid, safe_probes SHALL skip (empty)."""
    loop = _make_loop_with_probe_ring(
        _probe_ring_with_safe_probes(count=4, safe_count=8, min_predator_distance=20),
    )
    # min_predator_distance=20 on a 15x15 grid: impossible.
    env = _make_mock_env([_make_mock_predator((7, 7))], grid_size=15)
    probes = loop._build_f0_probe_params(brain=None, env=env)
    # Only the ring probes (1 predator x 4 positions); safe set is empty.
    assert len(probes) == 4
