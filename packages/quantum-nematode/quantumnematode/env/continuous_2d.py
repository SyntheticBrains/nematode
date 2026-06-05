"""Continuous-2D foraging environment (Phase 6 Tranche 5).

`Continuous2DEnvironment` subclasses `DynamicForagingEnvironment` and overrides
only the grid-coupled behaviours — kinematic ``(speed, turn)`` movement here;
capture-radius food consumption, continuous source placement, and Euclidean
distances follow — so the already-continuous sensing / source / state machinery
is reused (see the change design.md D1 and GitHub issue #206 for the eventual
god-class decomposition).

Positions: the float truth lives in ``AgentState.pos_continuous``; the integer
``AgentState.position`` is kept as a rounded discretized view for any inherited
grid-coupled reader, so the grid env's integer type contract is untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from quantumnematode.env.env import (
    DEFAULT_AGENT_ID,
    AgentState,
    DynamicForagingEnvironment,
)
from quantumnematode.logging_config import logger


@dataclass
class Continuous2DParams:
    """Continuous-2D substrate parameters (env-side mirror of `Continuous2DConfig`).

    Physical scale: ~1 mm worm body on a cm-scale plate. All values are in mm =
    internal coordinate units. The factory (`create_env_from_config`) maps the
    `Continuous2DConfig` pydantic model onto this dataclass, keeping the env
    package free of a `config_loader` import (which would be circular).
    """

    world_size_mm: float = 50.0
    body_length_mm: float = 1.0
    max_step_mm: float = 1.0
    capture_radius_mm: float = 1.0
    sweep_amplitude_mm: float = 0.5


def _wrap_to_pi(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi]``."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class Continuous2DEnvironment(DynamicForagingEnvironment):
    """Foraging environment with continuous-2D coordinates and kinematic movement.

    The worm is a point on a square ``world_size_mm`` arena. An action is a
    continuous ``(speed, turn)``: the heading rotates by ``turn`` (radians) and the
    worm advances ``speed`` (clamped to ``max_step_mm``) along the new heading,
    clamped to the world bounds. The integer coordinate extent passed to the parent
    is ``round(world_size_mm)`` so the existing field / sensing machinery builds for
    free (at ~1 mm per former grid cell).
    """

    def __init__(
        self,
        *,
        continuous: Continuous2DParams | None = None,
        **kwargs: object,
    ) -> None:
        self.continuous = continuous or Continuous2DParams()
        # The parent's integer coordinate extent = the continuous world size; the
        # caller does not set grid_size for the continuous substrate.
        kwargs.pop("grid_size", None)
        super().__init__(
            grid_size=round(self.continuous.world_size_mm),
            **kwargs,  # type: ignore[arg-type]
        )
        self._init_continuous_positions()

    def _init_continuous_positions(self) -> None:
        """Seed every agent's float position at the world centre, heading +x."""
        centre = self.continuous.world_size_mm / 2.0
        for agent_state in self.agents.values():
            agent_state.pos_continuous = (centre, centre)
            agent_state.heading_rad = 0.0
            agent_state.position = self._discretise((centre, centre))
            agent_state.body = [agent_state.position]  # point-worm

    def _discretise(self, pos: tuple[float, float]) -> tuple[int, int]:
        """Return a rounded, in-bounds integer view of a float position.

        Used to keep ``AgentState.position`` valid for inherited grid-coupled readers.
        """
        upper = self.grid_size - 1
        ix = min(upper, max(0, round(pos[0])))
        iy = min(upper, max(0, round(pos[1])))
        return (ix, iy)

    def _kinematic_move(self, agent_state: AgentState, speed: float, turn: float) -> None:
        """Apply a continuous ``(speed, turn)`` action to one agent.

        Rotate the heading by ``turn`` (wrapped to ``[-pi, pi]``), advance by
        ``speed`` (clamped to ``[0, max_step_mm]``) along the new heading, clamp the
        position to ``[0, world_size_mm]``, and keep the body a single point. The
        integer ``position`` view is re-synced for inherited grid-coupled readers.
        """
        speed = max(0.0, min(float(speed), self.continuous.max_step_mm))
        heading = _wrap_to_pi(agent_state.heading_rad + float(turn))
        agent_state.heading_rad = heading

        origin = agent_state.pos_continuous or (
            self.continuous.world_size_mm / 2.0,
            self.continuous.world_size_mm / 2.0,
        )
        world = self.continuous.world_size_mm
        new_x = min(world, max(0.0, origin[0] + speed * math.cos(heading)))
        new_y = min(world, max(0.0, origin[1] + speed * math.sin(heading)))

        agent_state.pos_continuous = (new_x, new_y)
        agent_state.position = self._discretise((new_x, new_y))
        agent_state.body = [agent_state.position]  # point-worm

    def move_agent_continuous(
        self,
        speed: float,
        turn: float,
        agent_id: str = DEFAULT_AGENT_ID,
    ) -> None:
        """Apply a continuous ``(speed, turn)`` action to the given agent.

        The continuous-substrate analogue of `move_agent`; the runner calls this
        (not the discrete `move_agent`) when the environment is continuous-2D.
        """
        self._kinematic_move(self.agents[agent_id], speed, turn)

    # ----- §3.4: capture-radius food consumption + Euclidean distances -----
    # Food sources remain on the integer lattice within the continuous arena
    # (inherited placement); the worm and capture are fully continuous. Float
    # food placement is deferred to avoid the `self.foods: list[tuple[int,int]]`
    # type ripple (same class of change as the position-type refinement).

    def _agent_xy(self, agent_id: str) -> tuple[float, float]:
        """Return the agent's continuous position (float truth, falling back to the int view)."""
        state = self.agents[agent_id]
        if state.pos_continuous is not None:
            return state.pos_continuous
        return (float(state.position[0]), float(state.position[1]))

    def reached_goal_for(self, agent_id: str) -> bool:
        """Return True if the agent is within the capture radius (Euclidean) of any food."""
        agent_x, agent_y = self._agent_xy(agent_id)
        radius = self.continuous.capture_radius_mm
        return any(
            math.hypot(agent_x - food_x, agent_y - food_y) <= radius
            for food_x, food_y in self.foods
        )

    def consume_food_for(self, agent_id: str) -> tuple[int, int] | None:
        """Consume the nearest food within the capture radius, respawn, and return it."""
        agent_x, agent_y = self._agent_xy(agent_id)
        nearest: tuple[int, int] | None = None
        nearest_distance = self.continuous.capture_radius_mm
        for food in self.foods:
            distance = math.hypot(agent_x - food[0], agent_y - food[1])
            if distance <= nearest_distance:
                nearest, nearest_distance = food, distance
        if nearest is not None:
            self.foods.remove(nearest)
            logger.info(f"Food consumed near {nearest} by {agent_id} (continuous-2D)")
            self.spawn_food()
            return nearest
        return None

    def get_nearest_food_distance_for(self, agent_id: str) -> int | None:
        """Return the Euclidean distance (rounded) to the nearest food, or None."""
        if not self.foods:
            return None
        agent_x, agent_y = self._agent_xy(agent_id)
        return min(
            round(math.hypot(agent_x - food_x, agent_y - food_y)) for food_x, food_y in self.foods
        )

    def get_nearest_food_distance(self) -> int | None:
        """Return the Euclidean distance (rounded) to the nearest food for the default agent."""
        return self.get_nearest_food_distance_for(DEFAULT_AGENT_ID)
