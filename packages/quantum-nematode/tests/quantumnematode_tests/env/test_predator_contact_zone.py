"""Tests for ContactZone enum + get_agent_predator_contact_zone_for.

Covers the 32 cardinal-direction x surrounding-cell parameterisations plus
the NONE outside-radius case plus the agent-overlap edge case. The zone
discrimination is the env-side foundation of the corrected predator
mechanosensation channel — it maps approach direction to one of the four
biologically-meaningful receptive-field regions described in the spec.
"""

from __future__ import annotations

import pytest
from quantumnematode.brain.actions import Action
from quantumnematode.env import (
    Direction,
    DynamicForagingEnvironment,
    ForagingParams,
    PredatorParams,
)
from quantumnematode.env.env import DEFAULT_AGENT_ID, ContactZone
from quantumnematode.env.theme import Theme


def _make_env() -> DynamicForagingEnvironment:
    """Build a predator-enabled env with the agent at (10, 10)."""
    return DynamicForagingEnvironment(
        grid_size=20,
        start_pos=(10, 10),
        foraging=ForagingParams(foods_on_grid=0, target_foods_to_collect=10),
        theme=Theme.ASCII,
        action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        predator=PredatorParams(
            enabled=True,
            count=1,
            speed=1.0,
            detection_radius=8,
        ),
    )


# Surrounding-cell offsets (dx, dy) and their cardinal/diagonal labels.
# Cardinal cells are 1 step away (Manhattan distance 1).
# Diagonal cells are 2 steps away in Manhattan distance but immediately
# adjacent in Chebyshev distance (which is what touches at the corners).
_CARDINAL_OFFSETS = {
    "E": (1, 0),
    "W": (-1, 0),
    "N": (0, 1),
    "S": (0, -1),
}
_DIAGONAL_OFFSETS = {
    "NE": (1, 1),
    "NW": (-1, 1),
    "SE": (1, -1),
    "SW": (-1, -1),
}


def _expected_zone(heading: Direction, offset: tuple[int, int]) -> ContactZone:
    """Compute the expected ContactZone for a given heading + predator offset.

    Mirrors the production logic so the test asserts behaviour, not
    implementation: cardinal-forward → ANTERIOR; cardinal-rear → POSTERIOR;
    cardinal-side → LATERAL; diagonal-forward → ANTERIOR (cone boundary,
    forward bias); diagonal-rear → POSTERIOR.
    """
    forward_map = {
        Direction.UP: (0, 1),
        Direction.DOWN: (0, -1),
        Direction.RIGHT: (1, 0),
        Direction.LEFT: (-1, 0),
    }
    fwd = forward_map[heading]
    # Normalise the offset vector and dot with the forward unit vector.
    length = (offset[0] ** 2 + offset[1] ** 2) ** 0.5
    dot = (offset[0] * fwd[0] + offset[1] * fwd[1]) / length
    cos45 = 0.7071067811865476
    if dot >= cos45:
        return ContactZone.ANTERIOR
    if dot <= -cos45:
        return ContactZone.POSTERIOR
    return ContactZone.LATERAL


class TestContactZoneCardinalCells:
    """The 4 cardinal cells x 4 cardinal headings → 16 cases."""

    @pytest.mark.parametrize(("offset_label", "offset"), list(_CARDINAL_OFFSETS.items()))
    @pytest.mark.parametrize(
        "heading",
        [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT],
    )
    def test_cardinal_predator_relative_to_heading(
        self,
        heading: Direction,
        offset_label: str,
        offset: tuple[int, int],
    ) -> None:
        env = _make_env()
        # Place predator at agent_pos + offset; bump damage_radius so the
        # diagonal cells (Manhattan distance 2) also count as in-contact for
        # the diagonal test class.
        env.predators[0].position = (
            env.agent_pos[0] + offset[0],
            env.agent_pos[1] + offset[1],
        )
        env.predators[0].damage_radius = 2
        env.agents[DEFAULT_AGENT_ID].direction = heading

        observed = env.get_agent_predator_contact_zone_for(DEFAULT_AGENT_ID)
        expected = _expected_zone(heading, offset)
        assert observed == expected, (
            f"heading={heading.value} offset={offset_label}: "
            f"expected {expected.value}, got {observed.value}"
        )


class TestContactZoneDiagonalCells:
    """The 4 diagonal cells x 4 cardinal headings → 16 cases (cone boundary)."""

    @pytest.mark.parametrize(("offset_label", "offset"), list(_DIAGONAL_OFFSETS.items()))
    @pytest.mark.parametrize(
        "heading",
        [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT],
    )
    def test_diagonal_predator_relative_to_heading(
        self,
        heading: Direction,
        offset_label: str,
        offset: tuple[int, int],
    ) -> None:
        env = _make_env()
        env.predators[0].position = (
            env.agent_pos[0] + offset[0],
            env.agent_pos[1] + offset[1],
        )
        env.predators[0].damage_radius = 2
        env.agents[DEFAULT_AGENT_ID].direction = heading

        observed = env.get_agent_predator_contact_zone_for(DEFAULT_AGENT_ID)
        expected = _expected_zone(heading, offset)
        assert observed == expected, (
            f"heading={heading.value} offset={offset_label}: "
            f"expected {expected.value}, got {observed.value}"
        )


class TestContactZoneEdgeCases:
    """NONE outside radius + ANTERIOR on overlap + LATERAL for STAY heading."""

    def test_predator_outside_damage_radius_returns_none(self) -> None:
        env = _make_env()
        env.predators[0].position = (10 + 5, 10)
        env.predators[0].damage_radius = 1
        env.agents[DEFAULT_AGENT_ID].direction = Direction.RIGHT
        assert env.get_agent_predator_contact_zone_for(DEFAULT_AGENT_ID) == ContactZone.NONE

    def test_predator_exactly_on_agent_returns_anterior(self) -> None:
        env = _make_env()
        env.predators[0].position = env.agent_pos
        env.predators[0].damage_radius = 1
        env.agents[DEFAULT_AGENT_ID].direction = Direction.RIGHT
        assert env.get_agent_predator_contact_zone_for(DEFAULT_AGENT_ID) == ContactZone.ANTERIOR

    def test_stay_heading_classifies_any_contact_as_lateral(self) -> None:
        env = _make_env()
        env.predators[0].position = (env.agent_pos[0] + 1, env.agent_pos[1])
        env.predators[0].damage_radius = 1
        env.agents[DEFAULT_AGENT_ID].direction = Direction.STAY
        assert env.get_agent_predator_contact_zone_for(DEFAULT_AGENT_ID) == ContactZone.LATERAL

    def test_predator_disabled_returns_none(self) -> None:
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(foods_on_grid=0, target_foods_to_collect=10),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )
        # No predator config → env.predator.enabled is False
        assert env.get_agent_predator_contact_zone_for(DEFAULT_AGENT_ID) == ContactZone.NONE


class TestContactZoneBackCompat:
    """is_agent_in_predator_contact_for keeps its existing bool semantics."""

    def test_bool_method_returns_true_for_any_nonzero_zone(self) -> None:
        env = _make_env()
        env.predators[0].position = (env.agent_pos[0] + 1, env.agent_pos[1])
        env.predators[0].damage_radius = 1
        env.agents[DEFAULT_AGENT_ID].direction = Direction.RIGHT
        assert env.is_agent_in_predator_contact_for(DEFAULT_AGENT_ID) is True
        assert env.get_agent_predator_contact_zone_for(DEFAULT_AGENT_ID) != ContactZone.NONE

    def test_bool_method_returns_false_outside_radius(self) -> None:
        env = _make_env()
        env.predators[0].position = (env.agent_pos[0] + 5, env.agent_pos[1])
        env.predators[0].damage_radius = 1
        assert env.is_agent_in_predator_contact_for(DEFAULT_AGENT_ID) is False
        assert env.get_agent_predator_contact_zone_for(DEFAULT_AGENT_ID) == ContactZone.NONE
