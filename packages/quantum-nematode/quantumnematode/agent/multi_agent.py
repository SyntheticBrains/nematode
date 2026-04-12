"""Multi-agent simulation orchestrator for multi-agent infrastructure.

Coordinates multiple independent QuantumNematodeAgent instances in a shared
DynamicForagingEnvironment with synchronous simultaneous stepping, food
competition resolution, and per-agent + aggregate metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

import numpy as np  # noqa: TC002 - needed at runtime for np.random.Generator

from quantumnematode.brain.arch import ClassicalBrain
from quantumnematode.env.env import DEFAULT_AGENT_ID
from quantumnematode.report.dtypes import TerminationReason

if TYPE_CHECKING:
    from quantumnematode.agent.agent import QuantumNematodeAgent, RewardConfig
    from quantumnematode.agent.runners import EpisodeResult
    from quantumnematode.brain.actions import ActionData
    from quantumnematode.dtypes import GridPosition
    from quantumnematode.env import DynamicForagingEnvironment
    from quantumnematode.env.pygame_renderer import PygameRenderer


# ── Constants ────────────────────────────────────────────────────────────────

MIN_GRID_SIZE_BASE = 5

# Minimum alarm pheromone concentration to count as "in alarm zone"
ALARM_EVASION_THRESHOLD = 0.1

# Number of steps to look back for food-sharing event detection
FOOD_SHARING_LOOKBACK_STEPS = 20

# Number of steps after alarm emission to check for direction change response
ALARM_RESPONSE_WINDOW = 5


# ── Food Competition ─────────────────────────────────────────────────────────


class FoodCompetitionPolicy(StrEnum):
    """Policy for resolving food competition when multiple agents reach the same food."""

    FIRST_ARRIVAL = "first_arrival"  # Deterministic: sorted by agent_id
    RANDOM = "random"  # Random winner among contestants


def resolve_food_competition(
    contested: dict[GridPosition, list[str]],
    policy: FoodCompetitionPolicy,
    rng: np.random.Generator,
) -> dict[str, GridPosition | None]:
    """Resolve food competition when multiple agents occupy the same food cell.

    Parameters
    ----------
    contested : dict[GridPosition, list[str]]
        Map of food positions to list of agent_ids present at that position.
    policy : FoodCompetitionPolicy
        How to select the winner.
    rng : np.random.Generator
        RNG for RANDOM policy.

    Returns
    -------
    dict[str, GridPosition | None]
        Map of agent_id to the food position they win (or None if they lose).
    """
    results: dict[str, GridPosition | None] = {}

    for food_pos, agent_ids in contested.items():
        if len(agent_ids) == 1:
            results[agent_ids[0]] = food_pos
        elif len(agent_ids) > 1:
            if policy == FoodCompetitionPolicy.FIRST_ARRIVAL:
                winner = sorted(agent_ids)[0]
            else:  # RANDOM
                winner = str(rng.choice(agent_ids))

            for aid in agent_ids:
                results[aid] = food_pos if aid == winner else None

    return results


# ── Spawn Placement ──────────────────────────────────────────────────────────


def validate_multi_agent_grid(grid_size: int, num_agents: int) -> None:
    """Validate that the grid is large enough for the number of agents.

    Parameters
    ----------
    grid_size : int
        Grid dimension.
    num_agents : int
        Number of agents.

    Raises
    ------
    ValueError
        If grid is too small.
    """
    min_size = max(MIN_GRID_SIZE_BASE, math.ceil(5 * math.sqrt(num_agents)))
    if grid_size < min_size:
        msg = f"Grid size {grid_size} too small for {num_agents} agents. Minimum: {min_size}."
        raise ValueError(msg)


# ── Multi-Agent Episode Result ───────────────────────────────────────────────


@dataclass
class MultiAgentEpisodeResult:
    """Result of a multi-agent episode.

    Attributes
    ----------
    agent_results : dict[str, EpisodeResult]
        Per-agent episode results with termination reason and path.
    total_food_collected : int
        Sum of food collected across all agents.
    per_agent_food : dict[str, int]
        Food collected per agent.
    food_competition_events : int
        Number of times 2+ agents were at the same food cell.
    proximity_events : int
        Number of step-agent pairs where 2+ agents were within detection radius.
    agents_alive_at_end : int
        Number of agents still alive when episode ended.
    mean_agent_success : float
        Fraction of agents that completed their food target.
    food_gini_coefficient : float
        Gini coefficient of food distribution (0=equal, 1=monopoly).
    social_feeding_events : int
        Step-agent pairs where social feeding decay reduction was applied
        (both social and solitary phenotypes counted when near conspecifics).
    aggregation_index : float
        Mean normalized inverse pairwise distance averaged over all steps
        (0=maximally dispersed, 1=all agents co-located).
    alarm_evasion_events : int
        Zone exits: agent concentration dropped from above ALARM_EVASION_THRESHOLD
        to at or below it between consecutive steps.
    food_sharing_events : int
        Non-emitter agent approached a food-marking pheromone source within
        FOOD_SHARING_LOOKBACK_STEPS of emission.
    territorial_index : float
        Spatial Gini coefficient of per-agent foraging spreads. Each agent's
        spread is the mean Manhattan distance of their food collection positions
        from their centroid. 0 = equal foraging patterns, 1 = maximal specialization.
    alarm_response_rate : float
        Fraction of alarm response opportunities where a nearby agent changed
        direction within ALARM_RESPONSE_WINDOW steps of an alarm emission.
    per_agent_reward : dict[str, float]
        Total accumulated reward per agent over the episode.
    per_agent_satiety : dict[str, float]
        Satiety remaining per agent at episode end.
    """

    agent_results: dict[str, EpisodeResult]
    total_food_collected: int
    per_agent_food: dict[str, int]
    food_competition_events: int
    proximity_events: int
    agents_alive_at_end: int
    mean_agent_success: float
    food_gini_coefficient: float
    social_feeding_events: int = 0
    aggregation_index: float = 0.0
    alarm_evasion_events: int = 0
    food_sharing_events: int = 0
    territorial_index: float = 0.0
    alarm_response_rate: float = 0.0
    per_agent_reward: dict[str, float] = field(default_factory=dict)
    per_agent_satiety: dict[str, float] = field(default_factory=dict)


def _compute_gini_values(values: list[float]) -> float:
    """Compute Gini coefficient for a list of non-negative values.

    Parameters
    ----------
    values : list[float]
        Non-negative values (int or float).

    Returns
    -------
    float
        Gini coefficient in [0, 1]. 0 = perfect equality, 1 = maximum inequality.
    """
    if not values or all(v == 0 for v in values):
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    weighted_sum = 0.0
    for i, v in enumerate(sorted_vals):
        weighted_sum += (2 * (i + 1) - n - 1) * v
    return weighted_sum / (n * total)


def _compute_gini(values: list[int]) -> float:
    """Compute Gini coefficient for a list of non-negative integers."""
    return _compute_gini_values([float(v) for v in values])


def _compute_aggregation_index(
    positions: list[tuple[int, int]],
    grid_size: int,
) -> float:
    """Compute aggregation index from agent positions.

    Mean normalized inverse pairwise distance. 0 = maximally dispersed,
    1 = all agents at same position.

    Parameters
    ----------
    positions : list[tuple[int, int]]
        Positions of alive agents.
    grid_size : int
        Grid size for normalization.

    Returns
    -------
    float
        Aggregation index in [0, 1].
    """
    if len(positions) < 2:  # noqa: PLR2004
        return 0.0
    max_dist = 2 * (grid_size - 1)
    if max_dist == 0:
        return 1.0
    total_proximity = 0.0
    n_pairs = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = abs(positions[i][0] - positions[j][0]) + abs(
                positions[i][1] - positions[j][1],
            )
            total_proximity += 1.0 - (dist / max_dist)
            n_pairs += 1
    return total_proximity / n_pairs


def _compute_territorial_index(
    per_agent_food_positions: dict[str, list[tuple[int, int]]],
) -> float:
    """Compute territorial index from per-agent food collection positions.

    Spatial Gini coefficient of per-agent foraging spreads. Each agent's
    spread is the mean Manhattan distance of food positions from the centroid.
    High Gini = some agents forage tightly while others range widely.

    Parameters
    ----------
    per_agent_food_positions : dict[str, list[tuple[int, int]]]
        Food collection positions per agent.

    Returns
    -------
    float
        Territorial index in [0, 1]. 0 = equal, 1 = maximal specialization.
    """
    # Only consider agents that collected food
    spreads: list[float] = []
    for positions in per_agent_food_positions.values():
        if not positions:
            continue
        if len(positions) == 1:
            spreads.append(0.0)
            continue
        # Compute centroid
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        # Mean Manhattan distance from centroid
        mean_dist = sum(abs(p[0] - cx) + abs(p[1] - cy) for p in positions) / len(positions)
        spreads.append(mean_dist)

    if len(spreads) < 2:  # noqa: PLR2004
        return 0.0

    return _compute_gini_values(spreads)


# ── Multi-Agent Simulation Orchestrator ──────────────────────────────────────


@dataclass
class MultiAgentSimulation:
    """Orchestrates multiple agents in a shared environment.

    Sits above individual QuantumNematodeAgent instances and coordinates
    the synchronous step loop: perception → movement → food competition →
    predators → effects → metrics.

    Attributes
    ----------
    env : DynamicForagingEnvironment
        Shared environment instance.
    agents : list[QuantumNematodeAgent]
        List of agent instances, each with a unique agent_id.
    food_policy : FoodCompetitionPolicy
        How to resolve food competition.
    social_detection_radius : int
        Manhattan distance radius for nearby agent counting.
    termination_policy : str
        What happens when an agent terminates: "freeze", "remove", or "end_all".
    """

    env: DynamicForagingEnvironment
    agents: list[QuantumNematodeAgent]
    food_policy: FoodCompetitionPolicy = FoodCompetitionPolicy.FIRST_ARRIVAL
    social_detection_radius: int = 5
    termination_policy: str = "freeze"
    agent_phenotypes: dict[str, str] = field(default_factory=dict)
    renderer: PygameRenderer | None = None

    # Runtime tracking (not init params)
    _followed_agent_id: str = field(default="", init=False)
    _renderer_closed: bool = field(default=False, init=False)
    _food_competition_events: int = field(default=0, init=False)
    _proximity_events: int = field(default=0, init=False)
    _social_feeding_events: int = field(default=0, init=False)
    _aggregation_index_sum: float = field(default=0.0, init=False)
    _aggregation_index_steps: int = field(default=0, init=False)
    _alarm_evasion_events: int = field(default=0, init=False)
    _food_sharing_events: int = field(default=0, init=False)
    _prev_alarm_concentration: dict[str, float] = field(default_factory=dict, init=False)
    _food_marking_buffer: list[tuple[tuple[int, int], int, str]] = field(
        default_factory=list,
        init=False,
    )
    _per_agent_food: dict[str, int] = field(default_factory=dict, init=False)
    _per_agent_total_reward: dict[str, float] = field(default_factory=dict, init=False)
    _per_agent_food_positions: dict[str, list[tuple[int, int]]] = field(
        default_factory=dict,
        init=False,
    )
    _alarm_response_opportunities: int = field(default=0, init=False)
    _alarm_response_successes: int = field(default=0, init=False)
    _alarm_response_buffer: list[tuple[int, str, dict[str, str]]] = field(
        default_factory=list,
        init=False,
    )
    _agent_terminations: dict[str, TerminationReason] = field(
        default_factory=dict,
        init=False,
    )

    _VALID_TERMINATION_POLICIES: ClassVar[frozenset[str]] = frozenset(
        {"freeze", "remove", "end_all"},
    )

    def __post_init__(self) -> None:
        """Validate agents and initialize tracking."""
        # Validate unique agent_ids
        ids = [a.agent_id for a in self.agents]
        if len(ids) != len(set(ids)):
            msg = f"Duplicate agent_ids: {ids}"
            raise ValueError(msg)

        # Validate termination policy
        if self.termination_policy not in self._VALID_TERMINATION_POLICIES:
            msg = (
                f"Invalid termination_policy '{self.termination_policy}'. "
                f"Must be one of: {sorted(self._VALID_TERMINATION_POLICIES)}"
            )
            raise ValueError(msg)

        # Validate all agents reference the shared env
        for agent in self.agents:
            if agent.env is not self.env:
                msg = f"Agent '{agent.agent_id}' does not share the environment instance."
                raise ValueError(msg)

        # Validate all agent_ids are registered in the environment
        missing_ids = [a.agent_id for a in self.agents if a.agent_id not in self.env.agents]
        if missing_ids:
            msg = (
                f"Agent IDs {missing_ids} not registered in environment. "
                f"Available: {list(self.env.agents.keys())}"
            )
            raise ValueError(msg)

        # Validate agent phenotypes
        _valid_phenotypes = {"social", "solitary"}
        for aid, phenotype in self.agent_phenotypes.items():
            if phenotype not in _valid_phenotypes:
                msg = (
                    f"Agent '{aid}': invalid social_phenotype '{phenotype}'. "
                    f"Must be one of: {sorted(_valid_phenotypes)}"
                )
                raise ValueError(msg)

        # Initialize per-agent food tracking
        self._per_agent_food = {a.agent_id: 0 for a in self.agents}

    @property
    def _alive_agents(self) -> list[QuantumNematodeAgent]:
        """Return list of agents that are still alive (handles 'remove' policy safely)."""
        return [
            a
            for a in self.agents
            if a.agent_id in self.env.agents and self.env.agents[a.agent_id].alive
        ]

    @property
    def renderer_closed(self) -> bool:
        """Whether the renderer window has been closed by the user."""
        return self._renderer_closed

    def _render_step(self, current_step: int, step: int, max_steps: int) -> None:
        """Render one frame of the multi-agent simulation.

        Builds AgentRenderState snapshots from current env state and
        per-agent trackers, then delegates to the renderer.
        """
        if self.renderer is None or self._renderer_closed:
            return

        from quantumnematode.env.pygame_renderer import AgentRenderState

        agents_state: list[AgentRenderState] = []
        for i, agent in enumerate(self.agents):
            aid = agent.agent_id
            if aid not in self.env.agents:
                continue
            env_state = self.env.agents[aid]
            agents_state.append(
                AgentRenderState(
                    agent_id=aid,
                    position=env_state.position,
                    body=list(env_state.body),
                    direction=env_state.direction.value,
                    alive=env_state.alive,
                    hp=env_state.hp,
                    max_hp=self.env.health.max_hp,
                    foods_collected=self._per_agent_food.get(aid, 0),
                    satiety=agent.current_satiety,
                    max_satiety=agent.max_satiety,
                    color_index=i % 8,
                ),
            )

        # Default followed agent to first if not set
        if not self._followed_agent_id and agents_state:
            self._followed_agent_id = agents_state[0].agent_id

        self._followed_agent_id = self.renderer.render_multi_agent_frame(
            env=self.env,
            agents=agents_state,
            followed_agent_id=self._followed_agent_id,
            step=step,
            max_steps=max_steps,
            current_step=current_step,
        )

        if self.renderer.closed:
            self._renderer_closed = True

    def _compute_nearby_agents_count(self, agent_id: str) -> int:
        """Count other agents (alive and frozen) within social detection radius.

        Frozen agents are physically present and detectable. Removed agents
        (via "remove" policy) are not counted. Self is excluded.

        Parameters
        ----------
        agent_id : str
            The agent to compute count for.

        Returns
        -------
        int
            Number of other agents within radius.
        """
        pos = self.env.agents[agent_id].position
        count = 0
        for aid, state in self.env.agents.items():
            # Skip self and the backward-compat "default" placeholder agent
            if aid in (agent_id, DEFAULT_AGENT_ID):
                continue
            distance = abs(pos[0] - state.position[0]) + abs(pos[1] - state.position[1])
            if distance <= self.social_detection_radius:
                count += 1
        return count

    def run_episode(  # noqa: C901, PLR0912, PLR0915
        self,
        reward_config: RewardConfig,
        max_steps: int,
    ) -> MultiAgentEpisodeResult:
        """Run a complete multi-agent episode.

        Parameters
        ----------
        reward_config : RewardConfig
            Reward configuration (shared across agents).
        max_steps : int
            Maximum steps before episode ends.

        Returns
        -------
        MultiAgentEpisodeResult
            Per-agent and aggregate results.
        """
        # Reset tracking
        self._food_competition_events = 0
        self._proximity_events = 0
        self._social_feeding_events = 0
        self._aggregation_index_sum = 0.0
        self._aggregation_index_steps = 0
        self._alarm_evasion_events = 0
        self._food_sharing_events = 0
        self._prev_alarm_concentration = {}
        self._food_marking_buffer = []
        self._per_agent_food = {a.agent_id: 0 for a in self.agents}
        self._per_agent_total_reward = {a.agent_id: 0.0 for a in self.agents}
        self._per_agent_food_positions = {a.agent_id: [] for a in self.agents}
        self._alarm_response_opportunities = 0
        self._alarm_response_successes = 0
        self._alarm_response_buffer = []
        self._agent_terminations = {}

        # Episode preparation for each agent
        for agent in self.agents:
            agent.brain.prepare_episode()
            if agent._stam is not None:
                agent._stam.reset()
            agent._previous_position = None
            agent._food_handler.reset()
            agent._satiety_manager.reset()
            agent._episode_tracker.reset()

        reward_per_agent: dict[str, float] = {a.agent_id: 0.0 for a in self.agents}
        action_per_agent: dict[str, ActionData | None] = {a.agent_id: None for a in self.agents}
        nearby_per_agent: dict[str, int] = {a.agent_id: 0 for a in self.agents}

        pheromones_enabled = self.env.pheromones.enabled

        for current_step in range(max_steps):
            alive = self._alive_agents
            if not alive:
                break

            # Reset per-step reward accumulator (effects + calculator rewards)
            for aid in reward_per_agent:
                reward_per_agent[aid] = 0.0

            # ── 1. PERCEPTION + DECISION ─────────────────────────
            actions: dict[str, ActionData] = {}
            for agent in alive:
                aid = agent.agent_id
                nearby = self._compute_nearby_agents_count(aid)
                nearby_per_agent[aid] = nearby

                gradient_strength, _ = agent.env.get_state(
                    agent.env.agents[aid].position,
                    disable_log=True,
                )
                input_data = agent._prepare_input_data(gradient_strength)
                params = agent._create_brain_params(
                    action=action_per_agent.get(aid),
                    nearby_agents_count=nearby,
                )
                brain_output = agent.brain.run_brain(
                    params=params,
                    reward=reward_per_agent[aid],
                    input_data=input_data,
                    top_only=True,
                    top_randomize=True,
                )
                if len(brain_output) != 1:
                    msg = f"Agent {aid}: invalid action length {len(brain_output)}"
                    raise ValueError(msg)
                actions[aid] = brain_output[0]

            # ── 2. MOVEMENT ──────────────────────────────────────
            for agent in alive:
                aid = agent.agent_id
                self.env.move_agent_for(aid, actions[aid].action)
                agent._episode_tracker.track_step()

                # Update path and food history
                pos = self.env.agents[aid].position
                agent.path.append((pos[0], pos[1]))
                agent.food_history.append(list(self.env.foods))
                agent._food_handler.track_step()

            # ── 2b. AGGREGATION PHEROMONE EMISSION ───────────────
            if pheromones_enabled and self.env.pheromone_field_aggregation is not None:
                for agent in alive:
                    aid = agent.agent_id
                    agent_pos = self.env.agents[aid].position
                    self.env.emit_aggregation_pheromone(agent_pos, current_step, aid)

            # ── 3. FOOD COMPETITION ──────────────────────────────
            self._resolve_food_step(alive, current_step)

            # ── 4. PREDATORS ─────────────────────────────────────
            self.env.update_predators()

            # Per-agent predator damage (copy list — terminations modify alive set)
            for agent in list(alive):
                aid = agent.agent_id
                if aid not in self.env.agents:
                    continue  # Removed by policy
                if self.env.is_agent_in_damage_radius_for(aid):
                    actual_damage = self.env.apply_predator_damage_for(aid)
                    # Emit alarm pheromone when agent takes damage
                    if actual_damage > 0 and pheromones_enabled:
                        agent_pos = self.env.agents[aid].position
                        self.env.emit_alarm_pheromone(agent_pos, current_step, aid)
                        # Record nearby agents' directions for alarm response tracking
                        nearby_dirs: dict[str, str] = {}
                        for other in alive:
                            oid = other.agent_id
                            if oid == aid or oid not in self.env.agents:
                                continue
                            if not self.env.agents[oid].alive:
                                continue
                            opos = self.env.agents[oid].position
                            dist = abs(opos[0] - agent_pos[0]) + abs(
                                opos[1] - agent_pos[1],
                            )
                            if dist <= self.social_detection_radius:
                                nearby_dirs[oid] = str(self.env.agents[oid].direction)
                        if nearby_dirs:
                            self._alarm_response_buffer.append(
                                (current_step, aid, nearby_dirs),
                            )
                    if self.env.agents[aid].hp <= 0:
                        self._handle_termination(
                            agent,
                            TerminationReason.HEALTH_DEPLETED,
                            nearby_agents_count=nearby_per_agent.get(aid, 0),
                        )

            # ── 5. EFFECTS ───────────────────────────────────────
            # Refresh alive list after predator phase
            still_alive = self._alive_agents
            for agent in list(still_alive):
                aid = agent.agent_id
                cached_nearby = nearby_per_agent.get(aid, 0)

                # Satiety decay (with social feeding reduction if applicable)
                decay_mult = 1.0
                if self.env.social_feeding.enabled and cached_nearby > 0:
                    phenotype = self.agent_phenotypes.get(aid, "social")
                    if phenotype == "social":
                        decay_mult = self.env.social_feeding.decay_reduction
                    else:
                        decay_mult = self.env.social_feeding.solitary_decay
                    self._social_feeding_events += 1
                agent._satiety_manager.decay_satiety(multiplier=decay_mult)
                agent._episode_tracker.track_satiety(agent.current_satiety)
                agent._episode_tracker.track_health(self.env.agents[aid].hp)

                # Temperature effects
                if self.env.thermotaxis.enabled:
                    temp_reward, _temp_damage = self.env.apply_temperature_effects_for(aid)
                    reward_per_agent[aid] += temp_reward
                    if self.env.agents[aid].hp <= 0:
                        self._handle_termination(
                            agent,
                            TerminationReason.HEALTH_DEPLETED,
                            nearby_agents_count=cached_nearby,
                        )
                        continue

                # Oxygen effects
                if self.env.aerotaxis.enabled:
                    o2_reward, _o2_damage = self.env.apply_oxygen_effects_for(aid)
                    reward_per_agent[aid] += o2_reward
                    if self.env.agents[aid].hp <= 0:
                        self._handle_termination(
                            agent,
                            TerminationReason.HEALTH_DEPLETED,
                            nearby_agents_count=cached_nearby,
                        )
                        continue

                # Starvation check
                if agent._satiety_manager.is_starved():
                    self._handle_termination(
                        agent,
                        TerminationReason.STARVED,
                        nearby_agents_count=cached_nearby,
                    )
                    continue

                # Check if agent collected all target foods
                if (
                    agent._episode_tracker.foods_collected
                    >= self.env.foraging.target_foods_to_collect
                ):
                    self._handle_termination(
                        agent,
                        TerminationReason.COMPLETED_ALL_FOOD,
                        nearby_agents_count=cached_nearby,
                    )
                    continue

            # ── 6. LEARNING ──────────────────────────────────────
            for agent in self._alive_agents:
                aid = agent.agent_id
                # Compute reward for this step (adds to effects-phase rewards)
                reward_per_agent[aid] += agent.calculate_reward(
                    reward_config,
                    agent.env,
                    agent.path,
                    max_steps=max_steps,
                    stuck_position_count=0,
                )
                self._per_agent_total_reward[aid] += reward_per_agent[aid]
                action_per_agent[aid] = actions.get(aid)

                if isinstance(agent.brain, ClassicalBrain):
                    params = agent._create_brain_params(
                        action=action_per_agent.get(aid),
                        nearby_agents_count=nearby_per_agent.get(aid, 0),
                    )
                    agent.brain.learn(
                        params=params,
                        reward=reward_per_agent[aid],
                        episode_done=False,
                    )
                agent.brain.update_memory(reward_per_agent[aid])

            # ── 7. RENDER ───────────────────────────────────────────
            self._render_step(current_step, current_step, max_steps)
            if self._renderer_closed:
                break

            # ── PHEROMONE FIELD UPDATE ────────────────────────────
            if pheromones_enabled:
                self.env.update_pheromone_fields(current_step)

            # ── PROXIMITY TRACKING ───────────────────────────────
            for agent in self._alive_agents:
                if self._compute_nearby_agents_count(agent.agent_id) > 0:
                    self._proximity_events += 1

            # ── COLLECTIVE METRICS ──────────────────────────────
            alive_now = self._alive_agents
            # Aggregation index
            if len(alive_now) >= 2:  # noqa: PLR2004
                positions = [self.env.agents[a.agent_id].position for a in alive_now]
                self._aggregation_index_sum += _compute_aggregation_index(
                    positions,
                    self.env.grid_size,
                )
                self._aggregation_index_steps += 1

            # Alarm evasion: did agents move away from alarm pheromone?
            if pheromones_enabled:
                for agent in alive_now:
                    aid = agent.agent_id
                    alarm_conc = self.env.get_pheromone_alarm_concentration(
                        position=self.env.agents[aid].position,
                        current_step=current_step,
                    )
                    prev_conc = self._prev_alarm_concentration.get(aid, 0.0)
                    # Evasion: was in alarm zone last step, now exited it
                    if (
                        prev_conc > ALARM_EVASION_THRESHOLD
                        and alarm_conc <= ALARM_EVASION_THRESHOLD
                    ):
                        self._alarm_evasion_events += 1
                    self._prev_alarm_concentration[aid] = alarm_conc

            # Food sharing: did non-emitter approach food-marking source?
            if pheromones_enabled:
                # Prune old entries from buffer
                self._food_marking_buffer = [
                    (pos, step, eid)
                    for pos, step, eid in self._food_marking_buffer
                    if current_step - step <= FOOD_SHARING_LOOKBACK_STEPS
                ]
                # Check if any alive agent (not the emitter) is near a food-marking site
                for fpos, _fstep, emitter_id in list(self._food_marking_buffer):
                    for agent in alive_now:
                        aid = agent.agent_id
                        if aid == emitter_id:
                            continue
                        apos = self.env.agents[aid].position
                        dist = abs(apos[0] - fpos[0]) + abs(apos[1] - fpos[1])
                        if dist <= self.social_detection_radius:
                            self._food_sharing_events += 1
                            # Remove from buffer to avoid double-counting
                            self._food_marking_buffer.remove((fpos, _fstep, emitter_id))
                            break

            # Alarm response: check if agents changed direction after alarm emissions
            expired: list[int] = []
            for idx, (emit_step, _emitter_id, nearby_dirs) in enumerate(
                self._alarm_response_buffer,
            ):
                if current_step - emit_step > ALARM_RESPONSE_WINDOW:
                    # Count all remaining tracked agents as non-responses
                    self._alarm_response_opportunities += len(nearby_dirs)
                    expired.append(idx)
                    continue
                if current_step - emit_step < 1:
                    continue  # Need at least 1 step to observe change
                for oid, original_dir in list(nearby_dirs.items()):
                    if oid not in self.env.agents:
                        # Removed agent — count as non-response
                        self._alarm_response_opportunities += 1
                        del nearby_dirs[oid]
                        continue
                    current_dir = str(self.env.agents[oid].direction)
                    if current_dir != original_dir:
                        self._alarm_response_successes += 1
                        self._alarm_response_opportunities += 1
                        # Remove this agent from tracking (counted once)
                        del nearby_dirs[oid]
                    elif current_step - emit_step == ALARM_RESPONSE_WINDOW:
                        # Window expired without change — count as non-response
                        self._alarm_response_opportunities += 1
                        del nearby_dirs[oid]
            # Remove fully expired entries
            for idx in reversed(expired):
                self._alarm_response_buffer.pop(idx)

        # ── EPISODE END ──────────────────────────────────────────
        # Terminate remaining alive agents (max_steps reached)
        for agent in list(self._alive_agents):
            self._handle_termination(
                agent,
                TerminationReason.MAX_STEPS,
                nearby_agents_count=nearby_per_agent.get(agent.agent_id, 0),
            )

        return self._build_result()

    def _resolve_food_step(
        self,
        alive_agents: list[QuantumNematodeAgent],
        current_step: int,
    ) -> None:
        """Resolve food collection with competition for this step."""
        # Build map: food_position -> list of agent_ids at that position
        contested: dict[tuple[int, int], list[str]] = {}
        for agent in alive_agents:
            aid = agent.agent_id
            pos = self.env.agents[aid].position
            if self.env.reached_goal_for(aid):
                contested.setdefault(pos, []).append(aid)

        if not contested:
            return

        # Count competition events
        for agents_at_food in contested.values():
            if len(agents_at_food) > 1:
                self._food_competition_events += 1

        # Resolve competition
        winners = resolve_food_competition(contested, self.food_policy, self.env.rng)

        # Process winners
        for aid, food_pos in winners.items():
            if food_pos is not None:
                agent = self._get_agent(aid)
                consumed = self.env.consume_food_for(aid)
                if consumed is not None:
                    self._per_agent_food[aid] += 1
                    self._per_agent_food_positions[aid].append(consumed)
                    agent._episode_tracker.track_food_collection()
                    # Restore satiety (same logic as FoodConsumptionHandler)
                    satiety_gain = (
                        agent._satiety_manager.max_satiety
                        * agent._food_handler.satiety_gain_fraction
                    )
                    agent._satiety_manager.restore_satiety(satiety_gain)
                    # Apply food healing
                    agent_state = self.env.agents[aid]
                    agent_state.hp = min(
                        self.env.health.max_hp,
                        agent_state.hp + self.env.health.food_healing,
                    )
                    # Emit food-marking pheromone at consumed food position
                    self.env.emit_food_pheromone(consumed, current_step, aid)
                    # Track for food-sharing metric (only when pheromones active)
                    if self.env.pheromones.enabled:
                        self._food_marking_buffer.append((consumed, current_step, aid))

    def _handle_termination(
        self,
        agent: QuantumNematodeAgent,
        reason: TerminationReason,
        nearby_agents_count: int = 0,
    ) -> None:
        """Handle agent termination according to policy."""
        aid = agent.agent_id
        if aid in self._agent_terminations:
            return  # Already terminated

        self._agent_terminations[aid] = reason
        agent_state = self.env.agents[aid]

        # Final learning step (include nearby_agents_count for consistent BrainParams)
        if isinstance(agent.brain, ClassicalBrain):
            params = agent._create_brain_params(nearby_agents_count=nearby_agents_count)
            agent.brain.learn(params=params, reward=0.0, episode_done=True)
        agent.brain.update_memory(0.0)
        agent.brain.post_process_episode(
            episode_success=(reason == TerminationReason.COMPLETED_ALL_FOOD),
        )

        # Track in metrics
        agent._metrics_tracker.track_episode_completion(
            success=(reason == TerminationReason.COMPLETED_ALL_FOOD),
            steps=agent._episode_tracker.steps,
            reward=agent._episode_tracker.rewards,
            foods_collected=agent._episode_tracker.foods_collected,
            distance_efficiencies=agent._episode_tracker.distance_efficiencies,
            predator_encounters=agent._episode_tracker.predator_encounters,
            successful_evasions=agent._episode_tracker.successful_evasions,
            termination_reason=reason,
        )

        # Apply termination policy
        if self.termination_policy == "freeze":
            agent_state.alive = False
        elif self.termination_policy == "remove":
            agent_state.alive = False
            del self.env.agents[aid]
        elif self.termination_policy == "end_all":
            agent_state.alive = False
            # Iteratively terminate all remaining agents (avoids recursion)
            remaining = [
                other
                for other in self.agents
                if other.agent_id != aid and other.agent_id not in self._agent_terminations
            ]
            for other in remaining:
                self._terminate_single(other, TerminationReason.MAX_STEPS)

    def _terminate_single(
        self,
        agent: QuantumNematodeAgent,
        reason: TerminationReason,
        nearby_agents_count: int = 0,
    ) -> None:
        """Terminate a single agent without policy propagation (used by end_all)."""
        aid = agent.agent_id
        if aid in self._agent_terminations:
            return
        self._agent_terminations[aid] = reason
        if aid in self.env.agents:
            self.env.agents[aid].alive = False

        if isinstance(agent.brain, ClassicalBrain):
            params = agent._create_brain_params(nearby_agents_count=nearby_agents_count)
            agent.brain.learn(params=params, reward=0.0, episode_done=True)
        agent.brain.update_memory(0.0)
        agent.brain.post_process_episode(
            episode_success=(reason == TerminationReason.COMPLETED_ALL_FOOD),
        )
        agent._metrics_tracker.track_episode_completion(
            success=(reason == TerminationReason.COMPLETED_ALL_FOOD),
            steps=agent._episode_tracker.steps,
            reward=agent._episode_tracker.rewards,
            foods_collected=agent._episode_tracker.foods_collected,
            distance_efficiencies=agent._episode_tracker.distance_efficiencies,
            predator_encounters=agent._episode_tracker.predator_encounters,
            successful_evasions=agent._episode_tracker.successful_evasions,
            termination_reason=reason,
        )

    def _get_agent(self, agent_id: str) -> QuantumNematodeAgent:
        """Get agent instance by ID."""
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        msg = f"Agent '{agent_id}' not found"
        raise KeyError(msg)

    def _build_result(self) -> MultiAgentEpisodeResult:
        """Build the multi-agent episode result from tracked data."""
        from quantumnematode.agent.runners import EpisodeResult

        agent_results: dict[str, EpisodeResult] = {}
        for agent in self.agents:
            aid = agent.agent_id
            reason = self._agent_terminations.get(aid, TerminationReason.MAX_STEPS)
            agent_results[aid] = EpisodeResult(
                agent_path=list(agent.path),
                termination_reason=reason,
                food_history=agent.food_history or None,
            )

        total_food = sum(self._per_agent_food.values())
        food_values = list(self._per_agent_food.values())
        successes = sum(
            1
            for r in self._agent_terminations.values()
            if r == TerminationReason.COMPLETED_ALL_FOOD
        )
        # Count alive agents excluding the backward-compat "default" placeholder
        alive_count = sum(
            1 for a in self.env.agents.values() if a.alive and a.agent_id != DEFAULT_AGENT_ID
        )

        agg_index = (
            self._aggregation_index_sum / self._aggregation_index_steps
            if self._aggregation_index_steps > 0
            else 0.0
        )

        return MultiAgentEpisodeResult(
            agent_results=agent_results,
            total_food_collected=total_food,
            per_agent_food=dict(self._per_agent_food),
            food_competition_events=self._food_competition_events,
            proximity_events=self._proximity_events,
            agents_alive_at_end=alive_count,
            mean_agent_success=successes / len(self.agents) if self.agents else 0.0,
            food_gini_coefficient=_compute_gini(food_values),
            social_feeding_events=self._social_feeding_events,
            aggregation_index=agg_index,
            alarm_evasion_events=self._alarm_evasion_events,
            food_sharing_events=self._food_sharing_events,
            territorial_index=_compute_territorial_index(self._per_agent_food_positions),
            alarm_response_rate=(
                self._alarm_response_successes / self._alarm_response_opportunities
                if self._alarm_response_opportunities > 0
                else 0.0
            ),
            per_agent_reward=dict(self._per_agent_total_reward),
            per_agent_satiety={a.agent_id: a._satiety_manager.current_satiety for a in self.agents},
        )
