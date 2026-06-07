"""Continuous-2D foraging environment.

`Continuous2DEnvironment` subclasses `DynamicForagingEnvironment` and overrides
only the grid-coupled behaviours — kinematic ``(speed, turn)`` movement,
capture-radius food consumption, and Euclidean distances — so the
already-continuous sensing / source / state machinery is reused. (Decomposing the
shared machinery into composable strategies is tracked in GitHub issue #206.)

Positions: the float truth lives in ``AgentState.pos_continuous``; the integer
``AgentState.position`` is kept as a rounded discretized view for any inherited
grid-coupled reader, so the grid env's integer type contract is untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from quantumnematode.env.env import (
    DEFAULT_AGENT_ID,
    AgentState,
    DynamicForagingEnvironment,
)
from quantumnematode.logging_config import logger

if TYPE_CHECKING:
    from quantumnematode.brain.actions import Action


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

    def _new_like(self) -> Continuous2DEnvironment:
        """Construct a fresh `Continuous2DEnvironment` with this env's configuration.

        Overrides the base constructor hook so `copy()` clones as the continuous-2D
        subclass (preserving the `Continuous2DParams` substrate), not the grid base.
        Runtime state (foods, RNG, agents, predators, …) is transferred by the
        inherited `copy()`.
        """
        return Continuous2DEnvironment(
            continuous=replace(self.continuous),
            start_pos=(self.agent_pos[0], self.agent_pos[1]),
            viewport_size=self.viewport_size,
            max_body_length=len(self.body),
            theme=self.theme,
            action_set=self.action_set,
            rich_style_config=self.rich_style_config,
            seed=self.seed,
            foraging=self.foraging,
            predator=self.predator,
            health=self.health,
            thermotaxis=self.thermotaxis,
            aerotaxis=self.aerotaxis,
            pheromones=self.pheromones,
            social_feeding=self.social_feeding,
        )

    def copy(self) -> Continuous2DEnvironment:
        """Deep-copy as a `Continuous2DEnvironment` (preserves continuous params + state).

        Delegates to the inherited `copy()`, which builds the new instance via the
        overridden `_new_like()` (so the substrate type and parameters survive) and
        transfers all runtime state, including each agent's `pos_continuous` /
        `heading_rad`.
        """
        new_env = super().copy()
        # `_new_like()` guarantees the concrete type; narrow it for the type checker.
        if not isinstance(new_env, Continuous2DEnvironment):  # pragma: no cover - defensive
            msg = f"Expected Continuous2DEnvironment from copy(), got {type(new_env).__name__}"
            raise TypeError(msg)
        return new_env

    def _apply_movement(self, agent_state: AgentState, action: Action) -> None:
        """Apply a discrete action, then re-sync the continuous float position.

        This is the discrete-action fallback path (used when a discrete-action brain
        runs on the continuous substrate — continuous-action heads are not yet
        implemented). It
        runs the inherited grid move, then mirrors the resulting integer ``position``
        into ``pos_continuous`` so sensing and capture (which read the float truth)
        stay coherent with the worm's actual cell. Continuous-action brains use
        ``move_agent_continuous`` / ``_kinematic_move`` instead and never reach here.
        """
        super()._apply_movement(agent_state, action)
        agent_state.pos_continuous = (
            float(agent_state.position[0]),
            float(agent_state.position[1]),
        )

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

        # Float truth is the origin; if it's somehow unset, fall back to the
        # integer grid view (which is always synced) — NOT the world centre, which
        # would teleport the worm to the middle of the arena on its first move.
        origin = agent_state.pos_continuous or (
            float(agent_state.position[0]),
            float(agent_state.position[1]),
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
        """Apply a continuous ``(speed, turn)`` action (physical units) to one agent.

        Parameters
        ----------
        speed : float
            Forward displacement in mm (clamped to ``[0, max_step_mm]``).
        turn : float
            Heading change in radians (wrapped to ``[-π, π]``).
        agent_id : str
            The agent to move (defaults to the single/default agent).

        See Also
        --------
        move_agent_normalized : the entry point continuous-action brains use,
            which rescales a normalized action to physical units before calling
            ``_kinematic_move``.
        """
        self._kinematic_move(self.agents[agent_id], speed, turn)

    def move_agent_normalized(
        self,
        speed_norm: float,
        turn_norm: float,
        agent_id: str = DEFAULT_AGENT_ID,
    ) -> None:
        """Apply a normalized continuous action, rescaling to physical units.

        Parameters
        ----------
        speed_norm : float
            Normalized forward speed in ``[0, 1]`` (mapped to ``[0, max_step_mm]``).
        turn_norm : float
            Normalized heading change in ``[-1, 1]`` (mapped to ``[-π, π]``).
        agent_id : str
            The agent to move (defaults to the single/default agent).

        Notes
        -----
        Continuous-action brains emit a substrate-independent normalized action;
        rescaling here keeps the physical scale in the environment (the brain
        stays env-agnostic). ``_kinematic_move`` re-clamps speed and wraps turn as
        a safety net.
        """
        speed = speed_norm * self.continuous.max_step_mm
        turn = turn_norm * math.pi
        self._kinematic_move(self.agents[agent_id], speed, turn)

    # ----- float source placement + capture-radius consumption + Euclidean -----
    # Food sources are placed at real-valued coordinates within the continuous
    # arena (Rung-2 env fidelity); the worm, capture, and distances are fully
    # continuous. The float source type is confined to this subclass (it overrides
    # only candidate generation), so the grid base keeps its integer source type.

    def _generate_food_candidate(self) -> tuple[float, float]:  # type: ignore[override]
        """Generate a real-valued food candidate within the continuous arena.

        Overrides the grid base's integer-lattice candidate generation so food
        sources are placed at float coordinates. The hotspot-bias path is reused
        (hotspots stay integer-cell — rarely used on the continuous substrate);
        otherwise the candidate is uniform float in ``[0, grid_size - 1]``, the
        same coordinate range the inherited validity / Poisson-disk machinery
        expects. All validity checks (`_is_valid_food_position`) are Euclidean and
        float-safe.
        """
        foraging = self.foraging
        if (
            foraging.food_hotspots
            and foraging.food_hotspot_bias > 0
            and self.rng.random() < foraging.food_hotspot_bias
        ):
            return self._sample_hotspot_candidate()
        upper = float(self.grid_size - 1)
        return (float(self.rng.uniform(0.0, upper)), float(self.rng.uniform(0.0, upper)))

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

    def get_nearest_food_distance_for(self, agent_id: str) -> float | None:  # type: ignore[override]
        """Return the true (un-rounded) Euclidean distance to the nearest food, or None."""
        if not self.foods:
            return None
        agent_x, agent_y = self._agent_xy(agent_id)
        return min(math.hypot(agent_x - food_x, agent_y - food_y) for food_x, food_y in self.foods)

    def get_nearest_food_distance(self) -> float | None:  # type: ignore[override]
        """Return the true Euclidean distance to the nearest food for the default agent."""
        return self.get_nearest_food_distance_for(DEFAULT_AGENT_ID)
