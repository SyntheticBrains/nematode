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
    _HEADING_OFFSET,
    DEFAULT_AGENT_ID,
    AgentState,
    ContactZone,
    DynamicForagingEnvironment,
    PredatorType,
)
from quantumnematode.logging_config import logger

if TYPE_CHECKING:
    from quantumnematode.brain.actions import Action
    from quantumnematode.env.env import Predator

# Per-step heading perturbation for wandering (non-pursuing) continuous predators
# (±45°), the continuous analogue of the grid wanderer's random cardinal turn.
_PREDATOR_WANDER_RAD = math.pi / 4.0


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
    # Body/contact-scale Euclidean damage radius (mm) used when a predator's configured
    # ``damage_radius`` is <= 0 — the integer grid "same-cell" default is unreachable as a
    # Euclidean distance, so without this fallback continuous predators can never deal damage.
    predator_damage_radius_mm: float = 1.0


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
        # Mirror the discrete facing into the continuous heading so heading-aware
        # readers (e.g. the Euclidean contact-zone classifier) use the correct facing
        # after a discrete-action move. Derived from the same `_HEADING_OFFSET` the grid
        # contact-zone uses, so the two conventions stay consistent. STAY (zero offset)
        # leaves the heading unchanged.
        offset = _HEADING_OFFSET.get(agent_state.direction, (0, 0))
        if offset != (0, 0):
            agent_state.heading_rad = math.atan2(offset[1], offset[0])

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
    # arena; the worm, capture, and distances are fully
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
            hx, hy = self._sample_hotspot_candidate()
            return (float(hx), float(hy))
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

    # ----- continuous predator kinematics + Euclidean detection/damage ----------
    # On the continuous substrate predators move with continuous ``(speed, heading)``
    # kinematics and their detection / damage / contact geometry is Euclidean against
    # the worm's float ``pos_continuous`` — replacing the inherited integer-Manhattan
    # model. The grid base keeps its byte-stable integer model (these overrides are
    # subclass-only; the additive ``Predator.pos_continuous`` / ``heading_rad`` fields
    # are unread on the grid).

    def _predator_xy(self, pred: Predator) -> tuple[float, float]:
        """Return a predator's continuous position (float truth, falling back to the int view).

        Overrides the base integer-view helper so predator sensing fields, detection,
        damage, and contact all read the same continuous coordinates as movement and
        rendering.
        """
        if pred.pos_continuous is not None:
            return pred.pos_continuous
        return (float(pred.position[0]), float(pred.position[1]))

    def _initialize_predators(self) -> None:
        """Initialise predators at float coordinates within the continuous arena.

        Mirrors the grid base's spawn loop (Euclidean min-separation from the agent,
        ``MAX_POISSON_ATTEMPTS`` retries) but samples real-valued coordinates and
        seeds each predator's continuous position + a random initial heading; the
        integer ``position`` is the rounded, clamped view. Predator placement is float
        on the continuous substrate (the grid base keeps integer-lattice placement).
        """
        from quantumnematode.env.env import MAX_POISSON_ATTEMPTS

        self.predators = []
        min_spawn_distance = max(
            self.predator.detection_radius,
            self.predator.damage_radius,
        )
        upper = float(self.grid_size - 1)
        ax, ay = self._agent_xy(DEFAULT_AGENT_ID)
        for i in range(self.predator.count):
            predator_id = f"predator_{i}"
            candidate = (0.0, 0.0)
            for _ in range(MAX_POISSON_ATTEMPTS):
                candidate = (
                    float(self.rng.uniform(0.0, upper)),
                    float(self.rng.uniform(0.0, upper)),
                )
                if math.hypot(candidate[0] - ax, candidate[1] - ay) > min_spawn_distance:
                    break
            else:
                logger.warning(
                    f"Could not find safe spawn position for predator {predator_id} "
                    f"after {MAX_POISSON_ATTEMPTS} attempts (continuous-2D). "
                    f"Spawning at {candidate} anyway.",
                )
            self.predators.append(self._spawn_continuous_predator(predator_id, candidate))

    def _spawn_continuous_predator(
        self,
        predator_id: str,
        pos: tuple[float, float],
    ) -> Predator:
        """Build one predator at a float position with a random initial heading."""
        pred = self._make_predator(
            predator_id=predator_id,
            position=self._discretise(pos),
        )
        pred.pos_continuous = pos
        pred.heading_rad = float(self.rng.uniform(-math.pi, math.pi))
        return pred

    def update_predators(self, step_index: int = 0) -> None:  # noqa: ARG002
        """Move every predator with continuous ``(speed, heading)`` kinematics.

        Pursuit predators with an agent inside their (Euclidean) detection radius steer
        toward the nearest agent's float position and advance; predators with no target
        wander; stationary predators do not move. Bypasses the cardinal ``PredatorBrain``
        (the analytic rule is sufficient for pursue/wander/stationary). ``step_index`` is
        unused on the continuous path (no time-aware predator brain).
        """
        if not self.predator.enabled:
            return
        alive_xy = [self._agent_xy(aid) for aid, a in self.agents.items() if a.alive]
        for pred in self.predators:
            self._move_predator_continuous(pred, alive_xy)

    def _move_predator_continuous(
        self,
        pred: Predator,
        agent_positions: list[tuple[float, float]],
    ) -> None:
        """Apply one continuous kinematic step to a single predator."""
        if pred.predator_type == PredatorType.STATIONARY:
            return

        origin = self._predator_xy(pred)

        target: tuple[float, float] | None = None
        if agent_positions:
            target = min(
                agent_positions,
                key=lambda p: math.hypot(origin[0] - p[0], origin[1] - p[1]),
            )
        is_pursuing = (
            pred.predator_type == PredatorType.PURSUIT
            and target is not None
            and math.hypot(origin[0] - target[0], origin[1] - target[1]) <= pred.detection_radius
        )

        if is_pursuing and target is not None:
            heading = math.atan2(target[1] - origin[1], target[0] - origin[0])
        else:
            heading = _wrap_to_pi(
                pred.heading_rad
                + float(self.rng.uniform(-_PREDATOR_WANDER_RAD, _PREDATOR_WANDER_RAD)),
            )
        pred.heading_rad = heading

        step = pred.speed * self.continuous.max_step_mm
        world = self.continuous.world_size_mm
        new_x = min(world, max(0.0, origin[0] + step * math.cos(heading)))
        new_y = min(world, max(0.0, origin[1] + step * math.sin(heading)))
        pred.pos_continuous = (new_x, new_y)
        pred.position = self._discretise((new_x, new_y))
        if math.hypot(new_x - origin[0], new_y - origin[1]) > 1e-9:  # noqa: PLR2004
            pred.distance_traveled += 1

    def is_agent_in_danger_for(self, agent_id: str) -> bool:
        """Return True if the agent is within (Euclidean) detection radius of any predator."""
        if not self.predator.enabled:
            return False
        ax, ay = self._agent_xy(agent_id)
        # Env-level detection radius (matches the grid base's danger check, which uses
        # the shared `self.predator.detection_radius` rather than the per-predator value
        # used by pursuit steering).
        radius = self.predator.detection_radius
        return any(
            math.hypot(ax - px, ay - py) <= radius
            for px, py in (self._predator_xy(pred) for pred in self.predators)
        )

    def _effective_damage_radius(self, pred: Predator) -> float:
        """Euclidean damage radius (mm) for a predator on the continuous substrate.

        Falls back to the body/contact-scale ``predator_damage_radius_mm`` when the
        configured ``damage_radius`` is <= 0 — the integer grid "same-cell" default is
        unreachable as a Euclidean distance, so without this fallback continuous predators
        could never deal damage. An explicit positive ``damage_radius`` takes precedence.
        """
        if pred.damage_radius > 0:
            return float(pred.damage_radius)
        return self.continuous.predator_damage_radius_mm

    def is_agent_in_damage_radius_for(self, agent_id: str) -> bool:
        """Return True if the agent is within any predator's (Euclidean) damage radius."""
        if not self.predator.enabled:
            return False
        ax, ay = self._agent_xy(agent_id)
        for pred in self.predators:
            px, py = self._predator_xy(pred)
            if math.hypot(ax - px, ay - py) <= self._effective_damage_radius(pred):
                logger.debug(
                    f"Agent {agent_id} in damage radius of {pred.predator_type.value} "
                    f"predator at {(px, py)} (continuous-2D)",
                )
                return True
        return False

    def get_agent_predator_contact_zone_for(self, agent_id: str) -> ContactZone:
        """Classify the nearest in-damage-radius predator contact by continuous heading.

        Euclidean analogue of the grid contact-zone method: selects the nearest predator
        within its own damage radius by Euclidean distance, then classifies the approach
        cone via the dot product of the predator→agent unit vector and the worm's
        continuous forward unit vector ``(cos heading_rad, sin heading_rad)``, retaining
        the ±45° anterior/lateral/posterior cones (diagonal boundary → ANTERIOR).
        """
        if not self.predator.enabled or not self.predators:
            return ContactZone.NONE

        agent_state = self.agents[agent_id]
        ax, ay = self._agent_xy(agent_id)

        nearest_pred: Predator | None = None
        nearest_dist: float | None = None
        for pred in self.predators:
            px, py = self._predator_xy(pred)
            distance = math.hypot(ax - px, ay - py)
            if distance > self._effective_damage_radius(pred):
                continue
            if nearest_dist is None or distance < nearest_dist:
                nearest_pred, nearest_dist = pred, distance

        if nearest_pred is None:
            return ContactZone.NONE

        px, py = self._predator_xy(nearest_pred)
        rel_dx, rel_dy = px - ax, py - ay
        rel_len = math.hypot(rel_dx, rel_dy)
        if rel_len == 0.0:
            # Overlap → ANTERIOR by convention (nose-touch dominance).
            return ContactZone.ANTERIOR

        # Worm forward unit vector (world y-up), the same `heading_rad` convention as
        # `_kinematic_move`. The forward vector is already unit-length, so the dot
        # product is the cosine of the approach angle.
        forward_x = math.cos(agent_state.heading_rad)
        forward_y = math.sin(agent_state.heading_rad)
        dot = (rel_dx * forward_x + rel_dy * forward_y) / rel_len
        cos45 = 0.7071067811865476
        if dot >= cos45:
            return ContactZone.ANTERIOR
        if dot <= -cos45:
            return ContactZone.POSTERIOR
        return ContactZone.LATERAL
