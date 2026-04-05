"""Tests for multi-agent simulation orchestrator (Phase 4 Deliverable 1)."""

from __future__ import annotations

import pytest
from quantumnematode.agent.agent import QuantumNematodeAgent, RewardConfig, SatietyConfig
from quantumnematode.agent.multi_agent import (
    FoodCompetitionPolicy,
    MultiAgentEpisodeResult,
    MultiAgentSimulation,
    _compute_gini,
    resolve_food_competition,
    validate_multi_agent_grid,
)
from quantumnematode.brain.arch.mlpppo import MLPPPOBrain, MLPPPOBrainConfig
from quantumnematode.env import (
    DynamicForagingEnvironment,
    ForagingParams,
    HealthParams,
    PredatorParams,
    PredatorType,
)
from quantumnematode.report.dtypes import TerminationReason
from quantumnematode.utils.seeding import get_rng


def _make_env(
    grid_size: int = 20,
    seed: int = 42,
    foods: int = 5,
    target: int = 10,
    predators: bool = False,
) -> DynamicForagingEnvironment:
    """Create a test environment."""
    return DynamicForagingEnvironment(
        grid_size=grid_size,
        start_pos=(grid_size // 2, grid_size // 2),
        seed=seed,
        foraging=ForagingParams(
            foods_on_grid=foods,
            target_foods_to_collect=target,
            min_food_distance=2,
            agent_exclusion_radius=2,
            gradient_decay_constant=8.0,
        ),
        health=HealthParams(max_hp=100.0),
        predator=PredatorParams(
            enabled=predators,
            count=1 if predators else 0,
            predator_type=PredatorType.PURSUIT,
            speed=0.5,
            detection_radius=5,
            damage_radius=1,
        ),
    )


def _make_agent(
    env: DynamicForagingEnvironment,
    agent_id: str,
    position: tuple[int, int] | None = None,
) -> QuantumNematodeAgent:
    """Create a test agent with MLP PPO brain."""
    config = MLPPPOBrainConfig(
        sensory_modules=["food_chemotaxis"],
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        num_hidden_layers=1,
    )
    brain = MLPPPOBrain(config=config, num_actions=4)

    if agent_id != "default" and agent_id not in env.agents:
        env.add_agent(agent_id, position=position)

    return QuantumNematodeAgent(
        brain=brain,
        env=env,
        agent_id=agent_id,
        satiety_config=SatietyConfig(initial_satiety=200.0),
    )


class TestFoodCompetitionPolicy:
    """Tests for food competition resolution."""

    def test_first_arrival_deterministic(self) -> None:
        """FIRST_ARRIVAL: lexicographically first agent wins."""
        rng = get_rng(42)
        contested = {(5, 5): ["agent_1", "agent_0"]}
        result = resolve_food_competition(contested, FoodCompetitionPolicy.FIRST_ARRIVAL, rng)
        assert result["agent_0"] == (5, 5)
        assert result["agent_1"] is None

    def test_random_uses_rng(self) -> None:
        """RANDOM: one winner selected via RNG."""
        rng = get_rng(42)
        contested = {(5, 5): ["agent_0", "agent_1"]}
        result = resolve_food_competition(contested, FoodCompetitionPolicy.RANDOM, rng)
        winners = [aid for aid, pos in result.items() if pos is not None]
        assert len(winners) == 1

    def test_three_agents_one_food(self) -> None:
        """Three agents at same food: only one wins."""
        rng = get_rng(42)
        contested = {(5, 5): ["agent_0", "agent_1", "agent_2"]}
        result = resolve_food_competition(contested, FoodCompetitionPolicy.FIRST_ARRIVAL, rng)
        assert result["agent_0"] == (5, 5)
        assert result["agent_1"] is None
        assert result["agent_2"] is None

    def test_different_foods_no_competition(self) -> None:
        """Agents at different foods both win."""
        rng = get_rng(42)
        contested = {(5, 5): ["agent_0"], (10, 10): ["agent_1"]}
        result = resolve_food_competition(contested, FoodCompetitionPolicy.FIRST_ARRIVAL, rng)
        assert result["agent_0"] == (5, 5)
        assert result["agent_1"] == (10, 10)

    def test_single_agent_at_food(self) -> None:
        """Single agent at food: no competition."""
        rng = get_rng(42)
        contested = {(5, 5): ["agent_0"]}
        result = resolve_food_competition(contested, FoodCompetitionPolicy.FIRST_ARRIVAL, rng)
        assert result["agent_0"] == (5, 5)


class TestGridSizeValidation:
    """Tests for grid size validation."""

    def test_valid_grid(self) -> None:
        """Grid large enough for agents."""
        validate_multi_agent_grid(20, 5)  # Should not raise

    def test_too_small_grid(self) -> None:
        """Grid too small raises ValueError."""
        with pytest.raises(ValueError, match="too small"):
            validate_multi_agent_grid(5, 10)

    def test_minimum_formula(self) -> None:
        """Check minimum size formula."""
        # 10 agents: min = max(5, ceil(5*sqrt(10))) = ceil(15.81) = 16
        validate_multi_agent_grid(16, 10)
        with pytest.raises(ValueError):
            validate_multi_agent_grid(15, 10)


class TestGiniCoefficient:
    """Tests for Gini coefficient computation."""

    def test_equal_distribution(self) -> None:
        """All agents get same food: Gini = 0."""
        assert _compute_gini([5, 5, 5, 5]) == pytest.approx(0.0)

    def test_total_inequality(self) -> None:
        """One agent gets all food: Gini approaches 1."""
        gini = _compute_gini([0, 0, 0, 100])
        assert gini > 0.7  # High inequality

    def test_empty(self) -> None:
        """Empty or all-zero: Gini = 0."""
        assert _compute_gini([]) == 0.0
        assert _compute_gini([0, 0, 0]) == 0.0


class TestMultiAgentSimulation:
    """Tests for the MultiAgentSimulation orchestrator."""

    def test_two_agent_episode_completes(self) -> None:
        """Two agents run an episode to completion."""
        env = _make_env(grid_size=20, seed=42)
        agent_0 = _make_agent(env, "agent_0", position=(5, 5))
        agent_1 = _make_agent(env, "agent_1", position=(15, 15))
        # Remove the default agent — we only want our two
        # Note: "default" agent remains in env but is not included in the
        # MultiAgentSimulation.agents list, so it doesn't participate.

        sim = MultiAgentSimulation(
            env=env,
            agents=[agent_0, agent_1],
            food_policy=FoodCompetitionPolicy.FIRST_ARRIVAL,
            social_detection_radius=5,
        )
        reward_config = RewardConfig()
        result = sim.run_episode(reward_config, max_steps=50)

        assert isinstance(result, MultiAgentEpisodeResult)
        assert "agent_0" in result.agent_results
        assert "agent_1" in result.agent_results
        assert result.total_food_collected >= 0

    def test_five_agent_episode_completes(self) -> None:
        """Five agents run an episode to completion."""
        env = _make_env(grid_size=30, seed=42, foods=10, target=20)
        agents = []
        # Note: "default" agent remains in env but is not included in the
        # MultiAgentSimulation.agents list, so it doesn't participate.
        for i in range(5):
            pos = (5 + i * 5, 15)
            agents.append(_make_agent(env, f"agent_{i}", position=pos))

        sim = MultiAgentSimulation(
            env=env,
            agents=agents,
            food_policy=FoodCompetitionPolicy.FIRST_ARRIVAL,
        )
        result = sim.run_episode(RewardConfig(), max_steps=30)

        assert len(result.agent_results) == 5
        assert result.agents_alive_at_end >= 0
        assert 0.0 <= result.mean_agent_success <= 1.0
        assert 0.0 <= result.food_gini_coefficient <= 1.0

    def test_terminated_agents_freeze(self) -> None:
        """Terminated agent freezes in place (default policy)."""
        env = _make_env(grid_size=20, seed=42)
        agent_0 = _make_agent(env, "agent_0", position=(5, 5))
        # Note: "default" agent remains in env but is not included in the
        # MultiAgentSimulation.agents list, so it doesn't participate.

        sim = MultiAgentSimulation(
            env=env,
            agents=[agent_0],
            termination_policy="freeze",
        )
        # Run episode — agent should eventually terminate (starved or max_steps)
        result = sim.run_episode(RewardConfig(), max_steps=10)
        assert not env.agents["agent_0"].alive

    def test_duplicate_agent_ids_raises(self) -> None:
        """Duplicate agent_ids raises ValueError."""
        env = _make_env()
        agent_0a = _make_agent(env, "agent_0", position=(5, 5))
        # Create second agent with same ID — need to work around add_agent check
        config = MLPPPOBrainConfig(
            sensory_modules=["food_chemotaxis"],
            actor_hidden_dim=16,
            critic_hidden_dim=16,
            num_hidden_layers=1,
        )
        brain = MLPPPOBrain(config=config, num_actions=4)
        agent_0b = QuantumNematodeAgent(brain=brain, env=env, agent_id="agent_0")

        with pytest.raises(ValueError, match="Duplicate"):
            MultiAgentSimulation(env=env, agents=[agent_0a, agent_0b])

    def test_per_agent_food_tracking(self) -> None:
        """Per-agent food collection is tracked correctly."""
        env = _make_env(grid_size=20, seed=42)
        agent_0 = _make_agent(env, "agent_0", position=(5, 5))
        # Note: "default" agent remains in env but is not included in the
        # MultiAgentSimulation.agents list, so it doesn't participate.

        sim = MultiAgentSimulation(env=env, agents=[agent_0])
        result = sim.run_episode(RewardConfig(), max_steps=50)

        assert "agent_0" in result.per_agent_food
        assert result.total_food_collected == result.per_agent_food["agent_0"]

    def test_all_termination_policies(self) -> None:
        """All three termination policies are accepted."""
        for policy in ("freeze", "remove", "end_all"):
            env = _make_env(grid_size=20, seed=42)
            agent = _make_agent(env, "agent_0", position=(5, 5))
            # Note: "default" agent remains in env but doesn't participate.

            sim = MultiAgentSimulation(
                env=env,
                agents=[agent],
                termination_policy=policy,
            )
            result = sim.run_episode(RewardConfig(), max_steps=10)
            assert isinstance(result, MultiAgentEpisodeResult)
