"""Tests for source-depletion dynamics (area-restricted search).

Covers the config-gated per-source depleting amplitude: the amount-scaled food field, the
once-per-step feeding decrement (never a field-read side effect), in-place flattening + removal at
exhaustion, per-source-amount integrity + copy, index-matching, and the byte-identical-when-off
contract — on both the continuous-2D and grid substrates.
"""

from __future__ import annotations

import pytest
from quantumnematode.env.continuous_2d import Continuous2DEnvironment, Continuous2DParams
from quantumnematode.env.env import DEFAULT_AGENT_ID, DynamicForagingEnvironment, ForagingParams


def _foraging(*, deplete: bool, foods: int = 3) -> ForagingParams:
    return ForagingParams(
        foods_on_grid=foods,
        min_food_distance=2,
        gradient_field_mode="fick",
        gradient_decay_constant=10.0,
        gradient_strength=1.0,
        no_respawn=True,
        source_depletion_enabled=deplete,
        source_initial_amount=1.0,
        depletion_per_feed=0.25,
        source_removal_eps=1e-3,
    )


def _cont(*, deplete: bool, foods: int = 3) -> Continuous2DEnvironment:
    return Continuous2DEnvironment(
        grid_size=60,
        seed=1,
        continuous=Continuous2DParams(world_size_mm=60.0, capture_radius_mm=2.0),
        foraging=_foraging(deplete=deplete, foods=foods),
    )


def _grid(*, deplete: bool) -> DynamicForagingEnvironment:
    return DynamicForagingEnvironment(grid_size=20, seed=1, foraging=_foraging(deplete=deplete))


# ── Disabled = byte-identical ─────────────────────────────────────────────────────────


def test_disabled_field_ignores_amounts():
    """With depletion off, neither concentration nor gradient consults food_amounts."""
    e = _cont(deplete=False)
    probe = e.foods[1]
    conc = e.get_food_concentration(position=probe)
    grad = e.get_separated_gradients(probe)
    e.food_amounts[0] = 0.0  # would zero source 0 IF consulted
    assert e.get_food_concentration(position=probe) == conc
    assert e.get_separated_gradients(probe) == grad


def test_disabled_consume_removes_outright():
    """Off: a consume removes the source outright (today's behaviour), keeping lists aligned."""
    e = _cont(deplete=False)
    n = len(e.foods)
    e._deplete_or_remove(0)
    assert len(e.foods) == n - 1
    assert len(e.food_amounts) == len(e.foods)


# ── Amount-scaled field ───────────────────────────────────────────────────────────────


def test_amount_scales_field_and_distance_zero():
    """A depleted source contributes a smaller bump; the distance==0 case reads the amount."""
    e = _cont(deplete=True)
    src = e.foods[0]
    full = e.get_food_concentration(position=src)
    e.food_amounts[0] = 0.5
    half = e.get_food_concentration(position=src)
    e.food_amounts[0] = 0.0
    zeroed = e.get_food_concentration(position=src)
    assert zeroed < half < full


# ── Once-per-step purity ──────────────────────────────────────────────────────────────


def test_field_reads_are_pure():
    """Sampling the field (repeatedly, incl. the gradient) never mutates food_amounts."""
    e = _cont(deplete=True)
    snapshot = list(e.food_amounts)
    for _ in range(5):
        e.get_food_concentration(position=e.foods[0])
        e.get_separated_gradients(e.foods[1])
    assert e.food_amounts == snapshot


def test_one_consume_decrements_once():
    """A single feeding event decrements the matched source exactly once."""
    e = _cont(deplete=True)
    a0 = e.food_amounts[0]
    e._deplete_or_remove(0)
    assert e.food_amounts[0] == pytest.approx(a0 - 0.25)


# ── In-place flattening + removal at exhaustion ───────────────────────────────────────


def test_persist_in_place_then_remove_at_exhaustion():
    """A source flattens in place across feeds, then is removed at exhaustion (no_respawn)."""
    e = _cont(deplete=True)
    pos = e.foods[0]
    n = len(e.foods)
    for expected in (0.75, 0.5, 0.25):
        e._deplete_or_remove(0)
        assert e.foods[0] == pos  # position unchanged — flattens in place
        assert e.food_amounts[0] == pytest.approx(expected)
        assert len(e.foods) == n
    e._deplete_or_remove(0)  # crosses removal_eps -> removed, no respawn
    assert len(e.foods) == n - 1
    assert len(e.food_amounts) == n - 1


def test_below_threshold_not_consumable():
    """A source at/below removal_eps does not count as reachable food."""
    e = _cont(deplete=True, foods=1)
    e.agents[DEFAULT_AGENT_ID].pos_continuous = e.foods[0]
    assert e.reached_goal()  # full source on which the agent sits -> reachable
    e.food_amounts[0] = 0.0
    assert not e.reached_goal()  # exhausted -> not food


# ── Integrity, copy, both substrates, index-matching ──────────────────────────────────


def test_copy_preserves_amounts():
    """A copied continuous-2D environment carries the per-source amounts."""
    e = _cont(deplete=True)
    e.food_amounts[0] = 0.5
    assert e.copy().food_amounts == e.food_amounts


def test_grid_substrate_depletes():
    """The grid substrate depletes via the same shared helper."""
    e = _grid(deplete=True)
    pos = e.foods[0]
    e.agents[DEFAULT_AGENT_ID].position = pos
    consumed = e.consume_food_for(DEFAULT_AGENT_ID)
    assert consumed == pos
    assert e.foods[0] == pos  # still present, flattened in place
    assert e.food_amounts[0] == pytest.approx(0.75)


def test_consume_matches_source_by_index():
    """Two coincident sources: the consume drains the MATCHED source, not another."""
    e = _cont(deplete=True, foods=2)
    e.foods[1] = e.foods[0]  # make source 1 coincide with source 0
    e.food_amounts[0] = 1.0
    e.food_amounts[1] = 1.0
    e.agents[DEFAULT_AGENT_ID].pos_continuous = e.foods[0]
    e.consume_food_for(DEFAULT_AGENT_ID)
    # Exactly one unit-quantum total was removed across the two coincident sources.
    assert sum(e.food_amounts) == pytest.approx(2.0 - 0.25)
    assert {round(a, 2) for a in e.food_amounts} == {1.0, 0.75}


def test_signal_absent_after_exhaustion():
    """A source consumed to exhaustion is removed and absent from every food signal."""
    e = _cont(deplete=True, foods=1)
    src = e.foods[0]
    for _ in range(4):  # 1.0 -> 0.0 over 4 feeds of 0.25
        e._deplete_or_remove(0)
    assert e.foods == []  # removed (no_respawn)
    assert e.get_food_concentration(position=src) == pytest.approx(0.0)
    assert e.get_nearest_food_distance_for(DEFAULT_AGENT_ID) is None
