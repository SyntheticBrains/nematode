"""Tests for multi-agent simulation orchestrator."""

from __future__ import annotations

from typing import Any

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
from quantumnematode.brain.modules import ModuleName
from quantumnematode.env import (
    DynamicForagingEnvironment,
    ForagingParams,
    HealthParams,
    PredatorParams,
    PredatorType,
)
from quantumnematode.utils.seeding import get_rng


def _make_env(
    grid_size: int = 20,
    seed: int = 42,
    foods: int = 5,
    target: int = 10,
    *,
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
        sensory_modules=[ModuleName.FOOD_CHEMOTAXIS],
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
        with pytest.raises(ValueError, match="too small"):
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
            pos = (5 + i * 5, 10)  # y=10 avoids default agent at grid center (15,15)
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
        sim.run_episode(RewardConfig(), max_steps=10)
        assert not env.agents["agent_0"].alive

    def test_duplicate_agent_ids_raises(self) -> None:
        """Duplicate agent_ids raises ValueError."""
        env = _make_env()
        agent_0a = _make_agent(env, "agent_0", position=(5, 5))
        # Create second agent with same ID — need to work around add_agent check
        config = MLPPPOBrainConfig(
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS],
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

    def test_invalid_termination_policy_raises(self) -> None:
        """Invalid termination policy raises ValueError."""
        env = _make_env(grid_size=20, seed=42)
        agent = _make_agent(env, "agent_0", position=(5, 5))
        with pytest.raises(ValueError, match="Invalid termination_policy"):
            MultiAgentSimulation(
                env=env,
                agents=[agent],
                termination_policy="invalid",
            )

    def test_remove_policy_no_keyerror(self) -> None:
        """Remove policy doesn't crash with KeyError on subsequent steps."""
        env = _make_env(grid_size=20, seed=42)
        a0 = _make_agent(env, "agent_0", position=(5, 5))
        a1 = _make_agent(env, "agent_1", position=(15, 15))

        sim = MultiAgentSimulation(
            env=env,
            agents=[a0, a1],
            termination_policy="remove",
        )
        # Should complete without KeyError even when agents are removed
        result = sim.run_episode(RewardConfig(), max_steps=20)
        assert isinstance(result, MultiAgentEpisodeResult)


class TestPerPredatorMetrics:
    """Per-predator metrics in MultiAgentEpisodeResult.

    Covers:

    - per_predator_distance_traveled records movements
    - per_predator_prey_proximity_steps increments when agent in range
    - per_predator_kills attribution under distinct distances (closest wins)
    - per_predator_kills attribution under equal distances (lex tie-break)
    - defensive global-closest fallback when no predator covers the agent
    """

    def test_per_predator_distance_traveled_records_movements(self) -> None:
        """After an episode, distance_traveled is a sum of cardinal moves."""
        env = _make_env(grid_size=20, seed=42, predators=True)
        a0 = _make_agent(env, "agent_0", position=(5, 5))
        sim = MultiAgentSimulation(env=env, agents=[a0])
        result = sim.run_episode(RewardConfig(), max_steps=30)
        # _make_env spawns predators with speed=0.5, so over 30 sim steps
        # a single predator advances its accumulator 30 x 0.5 = 15 times,
        # firing at most 15 cardinal steps. distance_traveled is the count
        # of post-clamp position changes, bounded above by the firing count.
        # Tightening the upper bound to 15 (vs a loose <= 30) catches
        # double-counting bugs where a single accumulator-step might
        # increment distance more than once.
        max_distance = 15
        n_predators = len(env.predators)
        assert n_predators >= 1
        for pid, dist in result.per_predator_distance_traveled.items():
            assert 0 <= dist <= max_distance, (
                f"{pid}: distance {dist} exceeds physical max {max_distance} (speed=0.5 x 30 steps)"
            )

    def test_per_predator_prey_proximity_steps_increments_when_in_range(self) -> None:
        """With predator pinned next to the agent, proximity counter > 0."""
        env = _make_env(grid_size=20, seed=42, predators=True)
        a0 = _make_agent(env, "agent_0", position=(5, 5))
        # Pin every predator within detection_radius (5) of the agent at (5, 5)
        # so that the proximity counter is guaranteed to fire on every step.
        # Without this pinning, _make_env's default Poisson-disk spawn could
        # place predators outside detection range, leaving the assertion
        # trivially satisfied at total == 0.
        for pred in env.predators:
            pred.position = (6, 5)  # Manhattan distance 1 from agent
        sim = MultiAgentSimulation(env=env, agents=[a0])
        max_steps = 20
        result = sim.run_episode(RewardConfig(), max_steps=max_steps)
        total_proximity = sum(result.per_predator_prey_proximity_steps.values())
        # Each predator should register proximity on every step where the
        # agent is alive. The agent may die mid-episode (default predator
        # damage), so the lower bound is "at least one step in range".
        assert total_proximity >= 1, (
            f"expected ≥1 proximity-step, got {total_proximity} "
            f"(predator detection_radius=5; agent pinned at (5,5); predators at (6,5))"
        )

    def test_per_predator_metric_keys_match_predator_ids(self) -> None:
        """All three per-predator dicts use the synthesised predator_id keys."""
        env = _make_env(grid_size=20, seed=42, predators=True)
        a0 = _make_agent(env, "agent_0", position=(5, 5))
        sim = MultiAgentSimulation(env=env, agents=[a0])
        result = sim.run_episode(RewardConfig(), max_steps=10)
        expected_ids = {p.predator_id for p in env.predators}
        assert set(result.per_predator_kills.keys()) == expected_ids
        assert set(result.per_predator_prey_proximity_steps.keys()) == expected_ids
        assert set(result.per_predator_distance_traveled.keys()) == expected_ids

    def test_kill_attribution_distinct_distances(self) -> None:
        """When 2 predators cover an agent at different distances, closest gets the kill."""
        # Build a synthetic env with 2 predators at known positions.
        env = _make_env(grid_size=20, seed=42, predators=False)
        # Override predator config + manually spawn two predators with
        # known IDs and positions for deterministic attribution test.
        from quantumnematode.env import HeuristicPredatorBrain
        from quantumnematode.env.env import Predator

        env.predator.enabled = True
        env.predator.count = 2
        env.predator.damage_radius = 2
        env.predators = [
            Predator(
                position=(10, 10),
                predator_id="predator_0",
                damage_radius=2,
                brain=HeuristicPredatorBrain(),
            ),
            Predator(
                position=(12, 10),
                predator_id="predator_1",
                damage_radius=2,
                brain=HeuristicPredatorBrain(),
            ),
        ]
        sim = MultiAgentSimulation(env=env, agents=[_make_agent(env, "agent_0", position=(5, 5))])
        # Manually init the per-predator metric counters (normally done
        # in run_episode).
        sim._kills_by_predator = {p.predator_id: 0 for p in env.predators}
        # Agent at (10, 11): predator_0 distance 1, predator_1 distance 3.
        sim._attribute_kill_to_predator((10, 11))
        assert sim._kills_by_predator["predator_0"] == 1
        assert sim._kills_by_predator["predator_1"] == 0

    def test_kill_attribution_lex_tiebreak(self) -> None:
        """When predators are equidistant, predator_id lex order wins."""
        env = _make_env(grid_size=20, seed=42, predators=False)
        from quantumnematode.env import HeuristicPredatorBrain
        from quantumnematode.env.env import Predator

        env.predator.enabled = True
        env.predator.count = 2
        env.predator.damage_radius = 2
        # Both predators 1 cell from agent at (10, 11).
        env.predators = [
            Predator(
                position=(10, 10),
                predator_id="predator_0",
                damage_radius=2,
                brain=HeuristicPredatorBrain(),
            ),
            Predator(
                position=(10, 12),
                predator_id="predator_1",
                damage_radius=2,
                brain=HeuristicPredatorBrain(),
            ),
        ]
        sim = MultiAgentSimulation(env=env, agents=[_make_agent(env, "agent_0", position=(5, 5))])
        sim._kills_by_predator = {p.predator_id: 0 for p in env.predators}
        # Both are dist 1; lex tie-break gives predator_0.
        sim._attribute_kill_to_predator((10, 11))
        assert sim._kills_by_predator["predator_0"] == 1
        assert sim._kills_by_predator["predator_1"] == 0

    def test_kill_attribution_fallback_no_covering_predator(self) -> None:
        """Defensive: if no predator covers, credit global-closest (lex tie-break)."""
        env = _make_env(grid_size=20, seed=42, predators=False)
        from quantumnematode.env import HeuristicPredatorBrain
        from quantumnematode.env.env import Predator

        env.predator.enabled = True
        env.predator.count = 2
        env.predator.damage_radius = 1
        # Both predators OUT of damage radius (Manhattan > 1) for the
        # agent at (10, 10). predator_0 is closer (3) than predator_1 (5).
        env.predators = [
            Predator(
                position=(10, 13),
                predator_id="predator_0",
                damage_radius=1,
                brain=HeuristicPredatorBrain(),
            ),
            Predator(
                position=(10, 15),
                predator_id="predator_1",
                damage_radius=1,
                brain=HeuristicPredatorBrain(),
            ),
        ]
        sim = MultiAgentSimulation(env=env, agents=[_make_agent(env, "agent_0", position=(5, 5))])
        sim._kills_by_predator = {p.predator_id: 0 for p in env.predators}
        sim._attribute_kill_to_predator((10, 10))
        # Fallback: global-closest — predator_0 at dist 3.
        assert sim._kills_by_predator["predator_0"] == 1
        assert sim._kills_by_predator["predator_1"] == 0


class TestLearningPredatorPerStepHook:
    """Per-step `predator.brain.learn(reward, done)` integration into MultiAgentSimulation.

    Wired in step 6b of `run_episode`; only fires for predator brains
    exposing a `learn` attribute (currently `MLPPPOPredatorBrain` with
    `enable_learning=True`). Heuristic predator brains are untouched.
    """

    def test_learning_predator_buffer_grows_during_episode(self) -> None:
        """Multi-agent runner SHALL call predator.brain.learn() per step when available."""
        from quantumnematode.env.mlpppo_predator_brain import MLPPPOPredatorBrain

        env = _make_env(grid_size=20, seed=42, predators=True)
        # Replace the heuristic predator with a learning MLPPPOPredator.
        learning_brain = MLPPPOPredatorBrain(
            seed=42,
            enable_learning=True,
            rollout_buffer_size=4096,
        )
        env.predators[0].brain = learning_brain

        # Spy on `learn()` so we can verify it fired even though the
        # terminal `learn(episode_done=True)` call drains the buffer
        # before the assertion runs (per the brain's documented
        # episode-end flush contract).
        learn_calls: list[dict[str, Any]] = []
        original_learn = learning_brain.learn

        def spy_learn(*, reward: float, episode_done: bool) -> None:
            learn_calls.append({"reward": reward, "episode_done": episode_done})
            original_learn(reward=reward, episode_done=episode_done)

        learning_brain.learn = spy_learn  # type: ignore[method-assign]

        a0 = _make_agent(env, "agent_0", position=(5, 5))
        sim = MultiAgentSimulation(env=env, agents=[a0])
        max_steps = 20
        result = sim.run_episode(RewardConfig(), max_steps=max_steps)

        # The runner fires one `learn()` per sim step (mid-episode) plus
        # one terminal `learn(reward=0.0, episode_done=True)` flush.
        # At least one mid-episode call AND the terminal call must
        # appear.
        mid_episode_calls = [c for c in learn_calls if not c["episode_done"]]
        terminal_calls = [c for c in learn_calls if c["episode_done"]]
        assert len(mid_episode_calls) >= 1, (
            f"expected ≥1 mid-episode learn() call; got {learn_calls}"
        )
        assert len(terminal_calls) == 1, (
            f"expected exactly one terminal learn(episode_done=True) call; got {terminal_calls}"
        )
        # Episode result still has the standard kill-rate counters.
        assert isinstance(result.per_predator_kills, dict)

    def test_heuristic_predator_skipped_no_attribute_error(self) -> None:
        """Heuristic predator brains (no `learn` method) SHALL NOT raise."""
        # Existing test verifies _make_env+_make_agent+sim.run_episode
        # works with heuristic predators; here we verify the per-step
        # `hasattr(brain, "learn")` gate keeps that path clean.
        env = _make_env(grid_size=20, seed=42, predators=True)
        # env.predators[0] is a HeuristicPredatorBrain by default - no
        # `learn` method.
        assert not hasattr(env.predators[0].brain, "learn"), (
            "test precondition: default HeuristicPredatorBrain has no learn method"
        )
        a0 = _make_agent(env, "agent_0", position=(5, 5))
        sim = MultiAgentSimulation(env=env, agents=[a0])
        # Should NOT raise.
        sim.run_episode(RewardConfig(), max_steps=10)

    def test_reward_coefficients_propagate(self) -> None:
        """Custom predator reward coefficients SHALL change rewards passed to learn()."""
        from quantumnematode.env.mlpppo_predator_brain import MLPPPOPredatorBrain

        env = _make_env(grid_size=20, seed=42, predators=True)
        # Pin predator next to agent so proximity_reward fires every step.
        env.predators[0].position = (6, 5)  # Manhattan 1 from agent at (5, 5)
        learning_brain = MLPPPOPredatorBrain(
            seed=42,
            enable_learning=True,
            rollout_buffer_size=4096,
        )
        env.predators[0].brain = learning_brain

        # Spy on `learn(reward=...)` to capture rewards as they're
        # passed in by the runner. Inspecting the buffer post-episode
        # is unreliable because the terminal `learn(episode_done=True)`
        # call drains the buffer when it has ≥ num_minibatches entries.
        rewards_observed: list[float] = []
        original_learn = learning_brain.learn

        def spy_learn(*, reward: float, episode_done: bool) -> None:
            rewards_observed.append(reward)
            original_learn(reward=reward, episode_done=episode_done)

        learning_brain.learn = spy_learn  # type: ignore[method-assign]

        a0 = _make_agent(env, "agent_0", position=(5, 5))
        # Use big proximity_reward, zero kill/step components, to isolate
        # the proximity-shaping signal.
        sim = MultiAgentSimulation(
            env=env,
            agents=[a0],
            predator_kill_reward=0.0,
            predator_proximity_reward=10.0,
            predator_step_penalty=0.0,
        )
        sim.run_episode(RewardConfig(), max_steps=5)

        # At least one of the rewards passed to learn() during the
        # episode SHALL be ≥9.0 (the proximity coefficient was 10.0
        # and the predator was Manhattan-1 from the agent).
        assert any(r >= 9.0 for r in rewards_observed), (
            f"expected at least one large proximity reward (~10.0); got rewards={rewards_observed}"
        )
