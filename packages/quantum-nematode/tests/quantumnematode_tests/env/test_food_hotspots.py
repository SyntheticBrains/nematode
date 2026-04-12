"""Tests for food hotspot spawning, no-respawn mode, and satiety-gated consumption."""

from __future__ import annotations

import numpy as np
from quantumnematode.agent import QuantumNematodeAgent, RewardConfig, SatietyConfig
from quantumnematode.brain.arch.qvarcircuit import QVarCircuitBrain, QVarCircuitBrainConfig
from quantumnematode.brain.modules import ModuleName
from quantumnematode.env import DynamicForagingEnvironment, ForagingParams


class TestBackwardCompatibility:
    """Verify defaults produce identical behavior to pre-hotspot code."""

    def test_default_foraging_params_unchanged(self) -> None:
        """Default ForagingParams has no hotspots, no satiety gate, respawn enabled."""
        params = ForagingParams()
        assert params.food_hotspots is None
        assert params.food_hotspot_bias == 0.0
        assert params.food_hotspot_decay == 8.0
        assert params.no_respawn is False
        assert params.satiety_food_threshold is None

    def test_default_env_spawns_food_normally(self) -> None:
        """Environment with default params spawns food as before."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
            seed=42,
        )
        assert len(env.foods) == 5


class TestHotspotSpawning:
    """Tests for food hotspot-biased spawning."""

    def test_hotspot_bias_1_spawns_near_center(self) -> None:
        """With bias=1.0 and one hotspot, food spawns near the hotspot center."""
        hotspot = (25, 25, 1.0)
        env = DynamicForagingEnvironment(
            grid_size=50,
            foraging=ForagingParams(
                foods_on_grid=20,
                target_foods_to_collect=30,
                min_food_distance=1,
                agent_exclusion_radius=1,
                food_hotspots=[hotspot],
                food_hotspot_bias=1.0,
                food_hotspot_decay=5.0,
            ),
            seed=42,
        )
        # Measure mean distance from foods to hotspot center
        distances = [np.sqrt((f[0] - 25) ** 2 + (f[1] - 25) ** 2) for f in env.foods]
        mean_dist = np.mean(distances)
        # With decay=5.0, most food should be within ~15 cells of center
        assert mean_dist < 15.0, f"Mean distance {mean_dist} too large for hotspot bias=1.0"

    def test_no_hotspot_bias_spawns_uniformly(self) -> None:
        """With bias=0.0, food spawns uniformly (no clustering)."""
        env = DynamicForagingEnvironment(
            grid_size=50,
            foraging=ForagingParams(
                foods_on_grid=30,
                target_foods_to_collect=50,
                min_food_distance=1,
                agent_exclusion_radius=1,
                food_hotspots=[(25, 25, 1.0)],
                food_hotspot_bias=0.0,  # disabled
            ),
            seed=42,
        )
        # With uniform spawning on 50x50, mean distance from center should be larger
        distances = [np.sqrt((f[0] - 25) ** 2 + (f[1] - 25) ** 2) for f in env.foods]
        mean_dist = np.mean(distances)
        # Uniform on 50x50 has expected distance ~18 from center
        assert mean_dist > 10.0, f"Mean distance {mean_dist} suspiciously small for uniform"

    def test_multiple_hotspots_distribute_proportionally(self) -> None:
        """Food distributes across hotspots roughly proportional to weights."""
        env = DynamicForagingEnvironment(
            grid_size=50,
            foraging=ForagingParams(
                foods_on_grid=40,
                target_foods_to_collect=60,
                min_food_distance=1,
                agent_exclusion_radius=1,
                food_hotspots=[
                    (10, 10, 3.0),  # 3x weight
                    (40, 40, 1.0),  # 1x weight
                ],
                food_hotspot_bias=1.0,
                food_hotspot_decay=5.0,
            ),
            seed=42,
        )
        # Count food closer to each hotspot
        near_h1 = sum(1 for f in env.foods if np.sqrt((f[0] - 10) ** 2 + (f[1] - 10) ** 2) < 15)
        near_h2 = sum(1 for f in env.foods if np.sqrt((f[0] - 40) ** 2 + (f[1] - 40) ** 2) < 15)
        # With 3:1 weight ratio, h1 should have more food
        assert near_h1 > near_h2, f"Expected more food near h1 ({near_h1}) than h2 ({near_h2})"

    def test_hotspot_respawn_near_center(self) -> None:
        """After consumption, respawned food also appears near hotspots."""
        env = DynamicForagingEnvironment(
            grid_size=50,
            start_pos=(25, 25),
            foraging=ForagingParams(
                foods_on_grid=5,
                target_foods_to_collect=10,
                min_food_distance=1,
                agent_exclusion_radius=1,
                food_hotspots=[(25, 25, 1.0)],
                food_hotspot_bias=1.0,
                food_hotspot_decay=5.0,
            ),
            seed=42,
        )
        # Place agent on a food item and consume it
        if env.foods:
            food_pos = env.foods[0]
            env.agents["default"].position = food_pos
            env.consume_food()
        # After respawn, check new food is still near hotspot
        distances = [np.sqrt((f[0] - 25) ** 2 + (f[1] - 25) ** 2) for f in env.foods]
        assert all(d < 25 for d in distances), "Respawned food too far from hotspot"

    def test_partial_bias_mixed_spawning(self) -> None:
        """With bias=0.5, some food spawns near hotspot, some doesn't."""
        near_counts = []
        for seed in range(10):
            env = DynamicForagingEnvironment(
                grid_size=50,
                foraging=ForagingParams(
                    foods_on_grid=10,
                    target_foods_to_collect=20,
                    min_food_distance=1,
                    agent_exclusion_radius=1,
                    food_hotspots=[(10, 10, 1.0)],
                    food_hotspot_bias=0.5,
                    food_hotspot_decay=5.0,
                ),
                seed=seed,
            )
            near = sum(1 for f in env.foods if np.sqrt((f[0] - 10) ** 2 + (f[1] - 10) ** 2) < 12)
            near_counts.append(near)
        # With 50% bias, expect some runs with clustering and some without
        mean_near = np.mean(near_counts)
        assert 2 < mean_near < 9, f"Partial bias mean near count {mean_near} outside expected range"

    def test_hotspot_initializes_with_safe_zone_bias(self) -> None:
        """Environment initializes without error when both biases are configured.

        Note: safe_zone_food_bias only takes effect when thermotaxis is enabled.
        This test verifies the parameters don't conflict, not the combined behavior.
        """
        env = DynamicForagingEnvironment(
            grid_size=50,
            foraging=ForagingParams(
                foods_on_grid=10,
                target_foods_to_collect=20,
                min_food_distance=1,
                agent_exclusion_radius=1,
                food_hotspots=[(25, 25, 1.0)],
                food_hotspot_bias=0.8,
                safe_zone_food_bias=0.5,
            ),
            seed=42,
        )
        assert len(env.foods) > 0


class TestNoRespawnMode:
    """Tests for static food (no respawn) mode."""

    def test_no_respawn_food_decreases(self) -> None:
        """Food count decreases after consumption and never increases."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(
                foods_on_grid=5,
                target_foods_to_collect=10,
                min_food_distance=1,
                agent_exclusion_radius=1,
                no_respawn=True,
            ),
            seed=42,
        )
        initial_count = len(env.foods)
        assert initial_count == 5

        # Consume a food item
        if env.foods:
            food_pos = env.foods[0]
            env.agents["default"].position = food_pos
            env.consume_food()

        assert len(env.foods) == initial_count - 1

    def test_no_respawn_spawn_food_returns_false(self) -> None:
        """spawn_food() returns False immediately when no_respawn is True."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            foraging=ForagingParams(
                foods_on_grid=5,
                target_foods_to_collect=10,
                no_respawn=True,
            ),
            seed=42,
        )
        # Remove a food to make room
        env.foods.pop()
        result = env.spawn_food()
        assert result is False


class TestSatietyGate:
    """Tests for satiety-gated food collection."""

    def _create_env_and_agent(
        self,
        threshold: float | None = 0.8,
        initial_satiety: float = 200.0,
    ) -> tuple[DynamicForagingEnvironment, QuantumNematodeAgent]:
        """Create an environment and agent for satiety gate testing."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(
                foods_on_grid=5,
                target_foods_to_collect=10,
                min_food_distance=1,
                agent_exclusion_radius=1,
                satiety_food_threshold=threshold,
            ),
            seed=42,
        )
        brain_config = QVarCircuitBrainConfig(
            modules={ModuleName.FOOD_CHEMOTAXIS: [0, 1]},
            num_layers=1,
        )
        brain = QVarCircuitBrain(config=brain_config, shots=10)
        agent = QuantumNematodeAgent(
            brain=brain,
            env=env,
            satiety_config=SatietyConfig(initial_satiety=initial_satiety),
        )
        return env, agent

    def test_sated_agent_cannot_eat(self) -> None:
        """Agent at high satiety cannot consume food."""
        env, agent = self._create_env_and_agent(threshold=0.8, initial_satiety=200.0)
        # Satiety is 200, threshold is 0.8 * 200 = 160 → agent is sated
        assert not agent.can_eat

        # Place agent on food
        if env.foods:
            env.agents["default"].position = env.foods[0]

        result = agent._food_handler.check_and_consume_food()
        assert result.food_consumed is False

    def test_hungry_agent_can_eat(self) -> None:
        """Agent below threshold can consume normally."""
        env, agent = self._create_env_and_agent(threshold=0.8, initial_satiety=200.0)
        # Decay satiety below threshold: 0.8 * 200 = 160
        for _ in range(50):
            agent._satiety_manager.decay_satiety()
        # Satiety is now 150, threshold is 160 → agent can eat
        assert agent.can_eat

        # Place agent on food
        if env.foods:
            env.agents["default"].position = env.foods[0]

        result = agent._food_handler.check_and_consume_food()
        assert result.food_consumed is True

    def test_satiety_gate_disabled(self) -> None:
        """With threshold=None, no restriction on consumption."""
        env, agent = self._create_env_and_agent(threshold=None, initial_satiety=200.0)
        assert agent.can_eat

        if env.foods:
            env.agents["default"].position = env.foods[0]

        result = agent._food_handler.check_and_consume_food()
        assert result.food_consumed is True

    def test_food_remains_on_grid_when_sated(self) -> None:
        """Food stays on grid when sated agent tries to eat."""
        env, agent = self._create_env_and_agent(threshold=0.8, initial_satiety=200.0)
        food_count_before = len(env.foods)

        if env.foods:
            env.agents["default"].position = env.foods[0]
            agent._food_handler.check_and_consume_food()

        assert len(env.foods) == food_count_before


class TestMultiAgentSatietyGate:
    """Tests for satiety gate in multi-agent food competition."""

    def test_sated_agents_excluded_from_competition(self) -> None:
        """Sated agents don't compete for food; hungry agent gets it."""
        from quantumnematode.agent.multi_agent import MultiAgentSimulation

        env = DynamicForagingEnvironment(
            grid_size=20,
            foraging=ForagingParams(
                foods_on_grid=5,
                target_foods_to_collect=10,
                min_food_distance=1,
                agent_exclusion_radius=1,
                satiety_food_threshold=0.5,
            ),
            seed=42,
        )
        brain_config = QVarCircuitBrainConfig(
            modules={ModuleName.FOOD_CHEMOTAXIS: [0, 1]},
            num_layers=1,
        )

        # Agent 0: sated (high satiety)
        env.add_agent("agent_0", position=(5, 5), max_body_length=3)
        brain_0 = QVarCircuitBrain(config=brain_config, shots=10)
        agent_0 = QuantumNematodeAgent(
            brain=brain_0,
            env=env,
            agent_id="agent_0",
            satiety_config=SatietyConfig(initial_satiety=200.0),
        )

        # Agent 1: hungry — placed adjacent, will be moved to same position
        env.add_agent("agent_1", position=(6, 5), max_body_length=3)
        brain_1 = QVarCircuitBrain(config=brain_config, shots=10)
        agent_1 = QuantumNematodeAgent(
            brain=brain_1,
            env=env,
            agent_id="agent_1",
            satiety_config=SatietyConfig(initial_satiety=200.0),
        )
        # Decay agent_1's satiety to make it hungry
        for _ in range(120):
            agent_1._satiety_manager.decay_satiety()

        sim = MultiAgentSimulation(env=env, agents=[agent_0, agent_1])

        # Agent 0 is sated (200 > 0.5 * 200 = 100)
        # Agent 1 is hungry after decay (~80 <= 100)
        assert sim._is_agent_sated(agent_0)
        assert not sim._is_agent_sated(agent_1)

        # Exercise actual food competition: place both on a food tile
        food_pos = (5, 5)
        env.foods.append(food_pos)
        env.agents["agent_1"].position = food_pos  # Move hungry agent to food

        # Ensure per-agent tracking dicts are initialized
        for aid in ("agent_0", "agent_1"):
            sim._per_agent_food.setdefault(aid, 0)
            sim._per_agent_food_positions.setdefault(aid, [])

        food_before = sim._per_agent_food.copy()
        sim._resolve_food_step([agent_0, agent_1], current_step=0)

        # Hungry agent_1 should have eaten, sated agent_0 should not
        assert sim._per_agent_food["agent_1"] == food_before.get("agent_1", 0) + 1
        assert sim._per_agent_food["agent_0"] == food_before.get("agent_0", 0)


class TestRewardSuppression:
    """Tests for goal bonus suppression when agent can't eat."""

    def test_goal_bonus_suppressed_when_cant_eat(self) -> None:
        """No goal bonus when can_eat=False."""
        from quantumnematode.agent.reward_calculator import RewardCalculator

        config = RewardConfig(reward_goal=5.0)
        calc = RewardCalculator(config)
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
            seed=42,
        )
        # Place agent on food
        if env.foods:
            env.agents["default"].position = env.foods[0]

        reward_can_eat = calc.calculate_reward(
            env=env,
            path=[(10, 10)],
            can_eat=True,
        )
        reward_cant_eat = calc.calculate_reward(
            env=env,
            path=[(10, 10)],
            can_eat=False,
        )
        # The can_eat=True version should have higher reward (goal bonus)
        assert reward_can_eat > reward_cant_eat


class TestYAMLConfigLoading:
    """Tests for YAML config loading with new fields."""

    def test_foraging_config_with_hotspots(self) -> None:
        """ForagingConfig correctly parses and converts hotspot fields."""
        from quantumnematode.utils.config_loader import ForagingConfig

        config = ForagingConfig(
            foods_on_grid=10,
            food_hotspots=[[12.0, 12.0, 1.0], [38.0, 38.0, 0.5]],
            food_hotspot_bias=0.8,
            food_hotspot_decay=6.0,
            satiety_food_threshold=0.8,
            no_respawn=True,
        )
        params = config.to_params()

        assert params.food_hotspots is not None
        assert len(params.food_hotspots) == 2
        assert params.food_hotspots[0] == (12, 12, 1.0)
        assert params.food_hotspots[1] == (38, 38, 0.5)
        assert params.food_hotspot_bias == 0.8
        assert params.food_hotspot_decay == 6.0
        assert params.satiety_food_threshold == 0.8
        assert params.no_respawn is True

    def test_foraging_config_default_hotspots(self) -> None:
        """ForagingConfig with no hotspot fields produces None."""
        from quantumnematode.utils.config_loader import ForagingConfig

        config = ForagingConfig()
        params = config.to_params()
        assert params.food_hotspots is None

    def test_foraging_config_rejects_invalid_bias(self) -> None:
        """ForagingConfig rejects food_hotspot_bias outside [0.0, 1.0]."""
        import pytest
        from pydantic import ValidationError
        from quantumnematode.utils.config_loader import ForagingConfig

        with pytest.raises(ValidationError):
            ForagingConfig(food_hotspot_bias=1.5)
        with pytest.raises(ValidationError):
            ForagingConfig(food_hotspot_bias=-0.1)

    def test_foraging_config_rejects_invalid_decay(self) -> None:
        """ForagingConfig rejects non-positive food_hotspot_decay."""
        import pytest
        from pydantic import ValidationError
        from quantumnematode.utils.config_loader import ForagingConfig

        with pytest.raises(ValidationError):
            ForagingConfig(food_hotspot_decay=0.0)
        with pytest.raises(ValidationError):
            ForagingConfig(food_hotspot_decay=-1.0)

    def test_foraging_config_rejects_invalid_threshold(self) -> None:
        """ForagingConfig rejects satiety_food_threshold outside (0.0, 1.0]."""
        import pytest
        from pydantic import ValidationError
        from quantumnematode.utils.config_loader import ForagingConfig

        with pytest.raises(ValidationError):
            ForagingConfig(satiety_food_threshold=0.0)
        with pytest.raises(ValidationError):
            ForagingConfig(satiety_food_threshold=1.5)
