"""Multi-agent simulation orchestrator for Phase 4 multi-agent infrastructure.

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


# ── Constants ────────────────────────────────────────────────────────────────

MIN_GRID_SIZE_BASE = 5


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
    """

    agent_results: dict[str, EpisodeResult]
    total_food_collected: int
    per_agent_food: dict[str, int]
    food_competition_events: int
    proximity_events: int
    agents_alive_at_end: int
    mean_agent_success: float
    food_gini_coefficient: float


def _compute_gini(values: list[int]) -> float:
    """Compute Gini coefficient for a list of non-negative integers."""
    if not values or all(v == 0 for v in values):
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    cumulative = 0.0
    weighted_sum = 0.0
    for i, v in enumerate(sorted_vals):
        cumulative += v
        weighted_sum += (2 * (i + 1) - n - 1) * v
    return weighted_sum / (n * total) if total > 0 else 0.0


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

    # Runtime tracking (not init params)
    _food_competition_events: int = field(default=0, init=False)
    _proximity_events: int = field(default=0, init=False)
    _per_agent_food: dict[str, int] = field(default_factory=dict, init=False)
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
        self._per_agent_food = {a.agent_id: 0 for a in self.agents}
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

        for _step in range(max_steps):
            alive = self._alive_agents
            if not alive:
                break

            # ── 1. PERCEPTION + DECISION ─────────────────────────
            actions: dict[str, ActionData] = {}
            for agent in alive:
                aid = agent.agent_id
                nearby = self._compute_nearby_agents_count(aid)

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

                # Update visited cells
                self.env.agents[aid].visited_cells.add(pos)

            # ── 3. FOOD COMPETITION ──────────────────────────────
            self._resolve_food_step(alive)

            # ── 4. PREDATORS ─────────────────────────────────────
            self.env.update_predators()

            # Per-agent predator damage (copy list — terminations modify alive set)
            for agent in list(alive):
                aid = agent.agent_id
                if aid not in self.env.agents:
                    continue  # Removed by policy
                if self.env.is_agent_in_damage_radius_for(aid):
                    self.env.apply_predator_damage_for(aid)
                    if self.env.agents[aid].hp <= 0:
                        self._handle_termination(agent, TerminationReason.HEALTH_DEPLETED)

            # ── 5. EFFECTS ───────────────────────────────────────
            # Refresh alive list after predator phase
            still_alive = self._alive_agents
            for agent in list(still_alive):
                aid = agent.agent_id

                # Satiety decay
                agent._satiety_manager.decay_satiety()
                agent._episode_tracker.track_satiety(agent.current_satiety)
                agent._episode_tracker.track_health(self.env.agents[aid].hp)

                # Temperature effects
                if self.env.thermotaxis.enabled:
                    temp_reward, _temp_damage = self.env.apply_temperature_effects_for(aid)
                    reward_per_agent[aid] += temp_reward
                    if self.env.agents[aid].hp <= 0:
                        self._handle_termination(agent, TerminationReason.HEALTH_DEPLETED)
                        continue

                # Oxygen effects
                if self.env.aerotaxis.enabled:
                    o2_reward, _o2_damage = self.env.apply_oxygen_effects_for(aid)
                    reward_per_agent[aid] += o2_reward
                    if self.env.agents[aid].hp <= 0:
                        self._handle_termination(agent, TerminationReason.HEALTH_DEPLETED)
                        continue

                # Starvation check
                if agent._satiety_manager.is_starved():
                    self._handle_termination(agent, TerminationReason.STARVED)
                    continue

                # Check if agent collected all target foods
                if (
                    agent._episode_tracker.foods_collected
                    >= self.env.foraging.target_foods_to_collect
                ):
                    self._handle_termination(agent, TerminationReason.COMPLETED_ALL_FOOD)
                    continue

            # ── 6. LEARNING ──────────────────────────────────────
            for agent in self._alive_agents:
                aid = agent.agent_id
                # Compute reward for this step
                reward_per_agent[aid] = agent.calculate_reward(
                    reward_config,
                    agent.env,
                    agent.path,
                    max_steps=max_steps,
                    stuck_position_count=0,
                )
                action_per_agent[aid] = actions.get(aid)

                if isinstance(agent.brain, ClassicalBrain):
                    params = agent._create_brain_params(action=action_per_agent.get(aid))
                    agent.brain.learn(
                        params=params,
                        reward=reward_per_agent[aid],
                        episode_done=False,
                    )
                agent.brain.update_memory(reward_per_agent[aid])

            # ── PROXIMITY TRACKING ───────────────────────────────
            for agent in self._alive_agents:
                if self._compute_nearby_agents_count(agent.agent_id) > 0:
                    self._proximity_events += 1

        # ── EPISODE END ──────────────────────────────────────────
        # Terminate remaining alive agents (max_steps reached)
        for agent in list(self._alive_agents):
            self._handle_termination(agent, TerminationReason.MAX_STEPS)

        return self._build_result()

    def _resolve_food_step(
        self,
        alive_agents: list[QuantumNematodeAgent],
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

    def _handle_termination(
        self,
        agent: QuantumNematodeAgent,
        reason: TerminationReason,
    ) -> None:
        """Handle agent termination according to policy."""
        aid = agent.agent_id
        if aid in self._agent_terminations:
            return  # Already terminated

        self._agent_terminations[aid] = reason
        agent_state = self.env.agents[aid]

        # Final learning step
        if isinstance(agent.brain, ClassicalBrain):
            params = agent._create_brain_params()
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
    ) -> None:
        """Terminate a single agent without policy propagation (used by end_all)."""
        aid = agent.agent_id
        if aid in self._agent_terminations:
            return
        self._agent_terminations[aid] = reason
        if aid in self.env.agents:
            self.env.agents[aid].alive = False

        if isinstance(agent.brain, ClassicalBrain):
            params = agent._create_brain_params()
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
        alive_count = sum(1 for a in self.env.agents.values() if a.alive)

        return MultiAgentEpisodeResult(
            agent_results=agent_results,
            total_food_collected=total_food,
            per_agent_food=dict(self._per_agent_food),
            food_competition_events=self._food_competition_events,
            proximity_events=self._proximity_events,
            agents_alive_at_end=alive_count,
            mean_agent_success=successes / len(self.agents) if self.agents else 0.0,
            food_gini_coefficient=_compute_gini(food_values),
        )
