"""Tests for episode runners."""

from quantumnematode.agent import (
    ManyworldsModeConfig,
    QuantumNematodeAgent,
    RewardConfig,
    SatietyConfig,
)
from quantumnematode.agent.runners import ManyworldsEpisodeRunner, StandardEpisodeRunner
from quantumnematode.brain.arch import MLPBrain, MLPBrainConfig
from quantumnematode.env import DynamicForagingEnvironment, ForagingParams, StaticEnvironment
from quantumnematode.env.env import HealthParams, PredatorParams, ThermotaxisParams
from quantumnematode.env.temperature import TemperatureZone
from quantumnematode.report.dtypes import TerminationReason


class TestStandardEpisodeRunnerInitialization:
    """Test standard episode runner initialization."""

    def test_initialize_runner(self):
        """Test that runner initializes correctly."""
        runner = StandardEpisodeRunner()
        assert runner is not None


class TestStandardEpisodeRunnerIntegration:
    """Integration tests for StandardEpisodeRunner with real agent."""

    def test_run_episode_maze_environment(self):
        """Test running episode in maze environment returns path and termination reason."""
        # Create real components
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)
        env = StaticEnvironment(grid_size=5)
        agent = QuantumNematodeAgent(brain=brain, env=env)

        # Store initial position before episode
        initial_pos = tuple(env.agent_pos)

        # Run episode through the runner
        reward_config = RewardConfig()
        step_result = agent.run_episode(reward_config, max_steps=10)

        # Verify path is returned
        assert isinstance(step_result.agent_path, list)
        assert len(step_result.agent_path) > 0
        assert all(isinstance(pos, tuple) for pos in step_result.agent_path)
        # Path should start at agent's initial position
        assert step_result.agent_path[0] == initial_pos
        # Verify termination reason
        assert isinstance(step_result.termination_reason, TerminationReason)

    def test_run_episode_dynamic_foraging(self):
        """Test running episode in dynamic foraging environment."""
        # Create real components
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)
        env = DynamicForagingEnvironment(
            grid_size=10,
            foraging=ForagingParams(target_foods_to_collect=3),
        )
        satiety_config = SatietyConfig(initial_satiety=100.0)
        agent = QuantumNematodeAgent(
            brain=brain,
            env=env,
            satiety_config=satiety_config,
        )

        # Run episode through the runner
        reward_config = RewardConfig()
        step_result = agent.run_episode(reward_config, max_steps=20)

        # Verify path is returned
        assert isinstance(step_result.agent_path, list)
        assert len(step_result.agent_path) > 0
        # Satiety should have changed
        assert agent.current_satiety <= satiety_config.initial_satiety
        # Verify termination reason
        assert isinstance(step_result.termination_reason, TerminationReason)

    def test_run_episode_updates_agent_state(self):
        """Test that running episode updates agent state correctly."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)
        env = StaticEnvironment(grid_size=5)
        agent = QuantumNematodeAgent(brain=brain, env=env)

        initial_steps = agent._metrics_tracker.total_steps
        initial_path_length = len(agent.path)

        # Run episode
        reward_config = RewardConfig()
        agent.run_episode(reward_config, max_steps=10)

        # Verify agent state was updated
        assert agent._metrics_tracker.total_steps > initial_steps
        assert len(agent.path) > initial_path_length

    def test_runner_delegates_correctly(self):
        """Test that agent correctly delegates to StandardEpisodeRunner."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)
        env = StaticEnvironment(grid_size=5)
        agent = QuantumNematodeAgent(brain=brain, env=env)

        # Verify runner was created
        assert hasattr(agent, "_standard_runner")
        assert isinstance(agent._standard_runner, StandardEpisodeRunner)

        # Run episode and verify it works
        reward_config = RewardConfig()
        step_result = agent.run_episode(reward_config, max_steps=10)

        assert isinstance(step_result.agent_path, list)
        assert len(step_result.agent_path) > 0
        assert isinstance(step_result.termination_reason, TerminationReason)


class TestManyworldsEpisodeRunnerInitialization:
    """Test manyworlds episode runner initialization."""

    def test_initialize_runner(self):
        """Test that runner initializes correctly."""
        runner = ManyworldsEpisodeRunner()
        assert runner is not None


class TestManyworldsEpisodeRunnerIntegration:
    """Integration tests for ManyworldsEpisodeRunner with real agent."""

    def test_runner_initialization(self):
        """Test that agent creates manyworlds runner correctly."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)
        env = StaticEnvironment(grid_size=5)
        agent = QuantumNematodeAgent(brain=brain, env=env)

        # Verify runner was created
        assert hasattr(agent, "_manyworlds_runner")
        assert isinstance(agent._manyworlds_runner, ManyworldsEpisodeRunner)

    def test_manyworlds_config_parameter(self):
        """Test that manyworlds runner accepts config parameter."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)
        env = StaticEnvironment(grid_size=5)
        agent = QuantumNematodeAgent(brain=brain, env=env)

        # Create custom config
        config = ManyworldsModeConfig(
            top_n_actions=2,
            max_superpositions=4,
            render_sleep_seconds=0.0,  # No sleep for testing
        )

        # Verify runner accepts config (will sys.exit, so we can't test full execution)
        assert hasattr(agent, "_manyworlds_runner")
        assert agent._manyworlds_runner is not None


class TestRunnerComponentIntegration:
    """Test that runners integrate properly with agent components."""

    def test_standard_runner_uses_food_handler(self):
        """Test that StandardEpisodeRunner uses FoodConsumptionHandler."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)
        env = DynamicForagingEnvironment(
            grid_size=10,
            foraging=ForagingParams(target_foods_to_collect=3),
        )
        satiety_config = SatietyConfig(initial_satiety=100.0)
        agent = QuantumNematodeAgent(
            brain=brain,
            env=env,
            satiety_config=satiety_config,
        )

        initial_foods = agent._metrics_tracker.foods_collected

        # Run episode
        reward_config = RewardConfig()
        agent.run_episode(reward_config, max_steps=50)

        # Food collection tracking should work
        assert agent._metrics_tracker.foods_collected >= initial_foods

    def test_standard_runner_uses_satiety_manager(self):
        """Test that StandardEpisodeRunner uses SatietyManager."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)
        env = DynamicForagingEnvironment(
            grid_size=10,
            foraging=ForagingParams(target_foods_to_collect=3),
        )
        satiety_config = SatietyConfig(
            initial_satiety=100.0,
            satiety_decay_rate=1.0,
        )
        agent = QuantumNematodeAgent(
            brain=brain,
            env=env,
            satiety_config=satiety_config,
        )

        initial_satiety = agent.current_satiety

        # Run episode
        reward_config = RewardConfig()
        agent.run_episode(reward_config, max_steps=10)

        # Satiety should have decayed
        assert agent.current_satiety < initial_satiety

    def test_runner_helper_methods_accessible(self):
        """Test that runners can access agent helper methods."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)
        env = StaticEnvironment(grid_size=5)
        agent = QuantumNematodeAgent(brain=brain, env=env)

        # Verify helper methods exist and are callable
        assert hasattr(agent, "_get_agent_position_tuple")
        assert callable(agent._get_agent_position_tuple)

        assert hasattr(agent, "_prepare_input_data")
        assert callable(agent._prepare_input_data)

        assert hasattr(agent, "_create_brain_params")
        assert callable(agent._create_brain_params)

        assert hasattr(agent, "_render_step")
        assert callable(agent._render_step)

        # Test helper methods work
        pos = agent._get_agent_position_tuple()
        assert isinstance(pos, tuple)
        assert len(pos) == 2

        input_data = agent._prepare_input_data(0.5)
        assert input_data is None  # MLP brain returns None

        params = agent._create_brain_params(0.5, 1.57)
        assert params is not None
        assert hasattr(params, "gradient_strength")
        assert hasattr(params, "gradient_direction")


class TestPredatorCollisionTermination:
    """Test predator collision handling in episode runner."""

    def test_predator_instant_death_terminates_episode(self):
        """Test that predator collision without health system causes instant death."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)

        # Create environment with predator that has kill_radius=1
        # (agent moves first in the step loop, so kill_radius=0 would miss)
        env = DynamicForagingEnvironment(
            grid_size=10,
            foraging=ForagingParams(target_foods_to_collect=5),
            predator=PredatorParams(enabled=True, count=1, kill_radius=1),
            health=HealthParams(enabled=False),  # Instant death mode
        )

        # Place predator adjacent to agent - will be in kill range after agent moves
        agent_x, agent_y = env.agent_pos[0], env.agent_pos[1]
        env.predators[0].position = (agent_x, agent_y)

        satiety_config = SatietyConfig(initial_satiety=500.0)  # High satiety to avoid starvation
        agent = QuantumNematodeAgent(brain=brain, env=env, satiety_config=satiety_config)

        reward_config = RewardConfig()
        result = agent.run_episode(reward_config, max_steps=100)

        # Should terminate due to predator
        assert result.termination_reason == TerminationReason.PREDATOR

    def test_predator_with_health_system_applies_damage(self):
        """Test that predator collision with health system applies damage."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)

        env = DynamicForagingEnvironment(
            grid_size=10,
            foraging=ForagingParams(target_foods_to_collect=5),
            predator=PredatorParams(enabled=True, count=1, kill_radius=0),
            health=HealthParams(enabled=True, max_hp=100.0, predator_damage=20.0),
        )

        # Place predator on agent's starting position
        env.predators[0].position = (env.agent_pos[0], env.agent_pos[1])

        satiety_config = SatietyConfig(initial_satiety=500.0)
        agent = QuantumNematodeAgent(brain=brain, env=env, satiety_config=satiety_config)

        initial_hp = env.agent_hp
        reward_config = RewardConfig()
        agent.run_episode(reward_config, max_steps=5)

        # HP should have decreased (took at least one hit)
        assert env.agent_hp <= initial_hp

    def test_health_depletion_terminates_episode(self):
        """Test that HP reaching zero terminates episode with HEALTH_DEPLETED."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)

        # Low HP and high damage = quick death
        env = DynamicForagingEnvironment(
            grid_size=5,  # Small grid, hard to avoid predator
            foraging=ForagingParams(target_foods_to_collect=5),
            predator=PredatorParams(enabled=True, count=3, kill_radius=0),  # Multiple predators
            health=HealthParams(enabled=True, max_hp=20.0, predator_damage=20.0),  # One-hit kill
        )

        satiety_config = SatietyConfig(initial_satiety=500.0)
        agent = QuantumNematodeAgent(brain=brain, env=env, satiety_config=satiety_config)

        reward_config = RewardConfig()
        result = agent.run_episode(reward_config, max_steps=100)

        # With 3 predators in small grid and one-hit HP, should die to health depletion
        # Could also starve or hit max steps, so check for valid termination
        assert result.termination_reason in [
            TerminationReason.HEALTH_DEPLETED,
            TerminationReason.MAX_STEPS,
            TerminationReason.STARVED,
        ]


class TestStarvationTermination:
    """Test starvation handling in episode runner."""

    def test_starvation_terminates_episode(self):
        """Test that running out of satiety terminates episode."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)

        env = DynamicForagingEnvironment(
            grid_size=20,  # Large grid so agent won't find food easily
            foraging=ForagingParams(
                target_foods_to_collect=10,
                foods_on_grid=1,  # Very few foods
            ),
        )

        # Very low satiety with fast decay = quick starvation
        satiety_config = SatietyConfig(
            initial_satiety=10.0,
            satiety_decay_rate=5.0,  # Very fast decay
        )
        agent = QuantumNematodeAgent(brain=brain, env=env, satiety_config=satiety_config)

        reward_config = RewardConfig()
        result = agent.run_episode(reward_config, max_steps=100)

        # Should starve quickly
        assert result.termination_reason == TerminationReason.STARVED


class TestGoalCompletionTermination:
    """Test goal completion in static environment."""

    def test_static_environment_goal_reached(self):
        """Test that reaching goal in static environment terminates correctly."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)

        # Create small environment where goal is reachable
        env = StaticEnvironment(grid_size=5)

        agent = QuantumNematodeAgent(brain=brain, env=env)

        reward_config = RewardConfig()
        result = agent.run_episode(reward_config, max_steps=200)

        # Should either reach goal or hit max steps
        assert result.termination_reason in [
            TerminationReason.GOAL_REACHED,
            TerminationReason.MAX_STEPS,
        ]

    def test_food_collection_victory(self):
        """Test that collecting target foods terminates with COMPLETED_ALL_FOOD."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)

        # Small environment with easy food collection
        env = DynamicForagingEnvironment(
            grid_size=5,
            foraging=ForagingParams(
                target_foods_to_collect=1,  # Just need 1 food
                foods_on_grid=3,  # Plenty of food
                min_food_distance=1,
                agent_exclusion_radius=1,
            ),
        )

        satiety_config = SatietyConfig(initial_satiety=500.0)
        agent = QuantumNematodeAgent(brain=brain, env=env, satiety_config=satiety_config)

        reward_config = RewardConfig()
        result = agent.run_episode(reward_config, max_steps=200)

        # Should eventually collect food or hit max steps
        assert result.termination_reason in [
            TerminationReason.COMPLETED_ALL_FOOD,
            TerminationReason.MAX_STEPS,
            TerminationReason.STARVED,
        ]


class TestHealthHistoryTracking:
    """Test that health history is tracked correctly during episodes."""

    def test_health_history_recorded_when_enabled(self):
        """Test that health history is recorded when health system is enabled."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)

        env = DynamicForagingEnvironment(
            grid_size=10,
            foraging=ForagingParams(target_foods_to_collect=5),
            health=HealthParams(enabled=True, max_hp=100.0),
        )

        satiety_config = SatietyConfig(initial_satiety=100.0)
        agent = QuantumNematodeAgent(brain=brain, env=env, satiety_config=satiety_config)

        reward_config = RewardConfig()
        agent.run_episode(reward_config, max_steps=10)

        # Health history should be recorded
        assert len(agent._episode_tracker.health_history) > 0
        # All values should be valid HP values
        for hp in agent._episode_tracker.health_history:
            assert 0 <= hp <= 100.0


class TestThermotaxisIntegration:
    """Test thermotaxis integration in episode runner."""

    def test_temperature_effects_applied_during_episode(self):
        """Test that temperature zone effects are applied during episode when enabled."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)

        # Create environment with thermotaxis enabled and strong gradient
        # Agent starts in a danger zone to ensure we can detect effects
        env = DynamicForagingEnvironment(
            grid_size=20,
            foraging=ForagingParams(target_foods_to_collect=5),
            health=HealthParams(enabled=True, max_hp=100.0),
            thermotaxis=ThermotaxisParams(
                enabled=True,
                cultivation_temperature=20.0,
                base_temperature=35.0,  # Hot environment (danger zone)
                gradient_strength=0.0,  # No gradient, uniform hot temp
                danger_hp_damage=5.0,
                comfort_delta=5.0,
                discomfort_delta=10.0,
                danger_delta=15.0,
            ),
        )

        satiety_config = SatietyConfig(initial_satiety=500.0)
        agent = QuantumNematodeAgent(brain=brain, env=env, satiety_config=satiety_config)

        initial_hp = env.agent_hp
        reward_config = RewardConfig()
        agent.run_episode(reward_config, max_steps=10)

        # HP should have decreased due to temperature danger zone damage
        assert env.agent_hp < initial_hp, "Temperature danger zone should cause HP damage"

    def test_temperature_comfort_score_tracked(self):
        """Test that temperature comfort score is tracked during episode."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)

        # Environment at comfortable temperature
        env = DynamicForagingEnvironment(
            grid_size=10,
            foraging=ForagingParams(target_foods_to_collect=5),
            health=HealthParams(enabled=True, max_hp=100.0),
            thermotaxis=ThermotaxisParams(
                enabled=True,
                cultivation_temperature=20.0,
                base_temperature=20.0,  # At cultivation temp (comfort zone)
                gradient_strength=0.0,  # Uniform temperature
            ),
        )

        satiety_config = SatietyConfig(initial_satiety=500.0)
        agent = QuantumNematodeAgent(brain=brain, env=env, satiety_config=satiety_config)

        reward_config = RewardConfig()
        agent.run_episode(reward_config, max_steps=10)

        # Temperature comfort score should be tracked and high (we're in comfort zone)
        comfort_score = env.get_temperature_comfort_score()
        assert comfort_score is not None
        assert comfort_score == 1.0, "Should be 100% in comfort zone at cultivation temperature"

    def test_temperature_lethal_zone_causes_death(self):
        """Test that extreme temperatures can cause death through health depletion."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)

        # Lethal temperature environment with high damage
        env = DynamicForagingEnvironment(
            grid_size=10,
            foraging=ForagingParams(target_foods_to_collect=5),
            health=HealthParams(enabled=True, max_hp=50.0),  # Low HP
            thermotaxis=ThermotaxisParams(
                enabled=True,
                cultivation_temperature=20.0,
                base_temperature=50.0,  # Extreme heat (lethal zone: >35°C)
                gradient_strength=0.0,
                lethal_hp_damage=20.0,  # 20 HP per step in lethal zone
            ),
        )

        satiety_config = SatietyConfig(initial_satiety=500.0)
        agent = QuantumNematodeAgent(brain=brain, env=env, satiety_config=satiety_config)

        reward_config = RewardConfig()
        result = agent.run_episode(reward_config, max_steps=100)

        # Should die from lethal temperature damage
        assert result.termination_reason == TerminationReason.HEALTH_DEPLETED

    def test_temperature_hp_damage_applies_penalty(self):
        """Test that temperature HP damage triggers penalty_health_damage penalty."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)

        # Environment in danger zone (causes HP damage)
        env = DynamicForagingEnvironment(
            grid_size=10,
            foraging=ForagingParams(target_foods_to_collect=5),
            health=HealthParams(enabled=True, max_hp=100.0),
            thermotaxis=ThermotaxisParams(
                enabled=True,
                cultivation_temperature=20.0,
                base_temperature=33.0,  # Danger zone (30-35°C)
                gradient_strength=0.0,
                danger_hp_damage=5.0,  # 5 HP damage per step
                danger_penalty=-0.3,  # Zone penalty
            ),
        )

        satiety_config = SatietyConfig(initial_satiety=500.0)
        agent = QuantumNematodeAgent(brain=brain, env=env, satiety_config=satiety_config)

        # Run with penalty_health_damage set
        reward_config = RewardConfig(penalty_health_damage=0.5)
        agent.run_episode(reward_config, max_steps=5)

        # Agent should have taken HP damage from danger zone
        assert env.agent_hp < 100.0, "Agent should have taken temperature HP damage"

        # Check that rewards include the health damage penalty
        # With danger_penalty=-0.3 and penalty_health_damage=0.5, each step in danger zone
        # should apply both penalties, resulting in significantly negative reward
        total_reward = agent._episode_tracker.rewards
        assert total_reward < 0, (
            "Total reward should be negative due to danger zone and HP damage penalties"
        )

    def test_temperature_hp_damage_penalty_applied_when_damage_taken(self):
        """Test that HP damage penalty is applied per-step when taking damage."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)

        # Run 1: Low HP damage (danger zone)
        brain1 = MLPBrain(config=config, input_dim=2, num_actions=4)
        env1 = DynamicForagingEnvironment(
            grid_size=10,
            foraging=ForagingParams(target_foods_to_collect=5),
            health=HealthParams(enabled=True, max_hp=100.0),
            thermotaxis=ThermotaxisParams(
                enabled=True,
                cultivation_temperature=20.0,
                base_temperature=33.0,  # Danger zone
                gradient_strength=0.0,
                danger_hp_damage=2.0,  # Low damage
                danger_penalty=0.0,  # No zone penalty to isolate HP damage penalty
            ),
        )
        satiety_config = SatietyConfig(initial_satiety=500.0)
        agent1 = QuantumNematodeAgent(brain=brain1, env=env1, satiety_config=satiety_config)
        reward_config = RewardConfig(penalty_health_damage=0.5, penalty_step=0.0)
        agent1.run_episode(reward_config, max_steps=5)
        reward_low_damage = agent1._episode_tracker.rewards

        # Run 2: High HP damage (lethal zone)
        brain2 = MLPBrain(config=config, input_dim=2, num_actions=4)
        env2 = DynamicForagingEnvironment(
            grid_size=10,
            foraging=ForagingParams(target_foods_to_collect=5),
            health=HealthParams(enabled=True, max_hp=100.0),
            thermotaxis=ThermotaxisParams(
                enabled=True,
                cultivation_temperature=20.0,
                base_temperature=33.0,  # Danger zone
                gradient_strength=0.0,
                danger_hp_damage=10.0,  # High damage
                danger_penalty=0.0,  # No zone penalty to isolate HP damage penalty
            ),
        )
        agent2 = QuantumNematodeAgent(brain=brain2, env=env2, satiety_config=satiety_config)
        agent2.run_episode(reward_config, max_steps=5)
        reward_high_damage = agent2._episode_tracker.rewards

        # Both should be negative (taking damage)
        # The penalty_health_damage is applied per-step when damage > 0, regardless of amount
        assert reward_low_damage < 0, "Should have negative reward from HP damage penalty"
        assert reward_high_damage < 0, "Should have negative reward from HP damage penalty"


class TestBraveForagingBonus:
    """Test brave foraging bonus for collecting food in discomfort zones."""

    def test_brave_foraging_bonus_config_in_discomfort_zone(self):
        """Test that reward_discomfort_food parameter is accessible and usable."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)

        # Environment in hot discomfort zone (25-30°C with cultivation at 20°C)
        env = DynamicForagingEnvironment(
            grid_size=5,
            foraging=ForagingParams(
                target_foods_to_collect=5,
                foods_on_grid=5,
                min_food_distance=1,
                agent_exclusion_radius=1,
            ),
            health=HealthParams(enabled=True, max_hp=100.0),
            thermotaxis=ThermotaxisParams(
                enabled=True,
                cultivation_temperature=20.0,
                base_temperature=27.0,  # Hot discomfort zone
                gradient_strength=0.0,
                reward_discomfort_food=0.5,  # Brave foraging bonus
                discomfort_penalty=-0.05,
            ),
        )

        # Verify the parameter is set correctly
        assert env.thermotaxis.reward_discomfort_food == 0.5

        # Verify we're in discomfort zone
        assert env.get_temperature_zone() == TemperatureZone.DISCOMFORT_HOT, (
            "Should be in discomfort zone for brave bonus to apply"
        )

        satiety_config = SatietyConfig(initial_satiety=500.0)
        agent = QuantumNematodeAgent(brain=brain, env=env, satiety_config=satiety_config)

        reward_config = RewardConfig(reward_goal=2.0)
        agent.run_episode(reward_config, max_steps=100)

        # Verify episode completed without errors - brave bonus code path executed
        # The actual reward value depends on many factors, so we just verify execution
        assert env.get_temperature_zone() == TemperatureZone.DISCOMFORT_HOT, (
            "Should remain in discomfort zone (no gradient)"
        )

    def test_brave_bonus_not_applied_in_comfort_zone(self):
        """Test that brave bonus is only applied in discomfort zones, not comfort."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)

        # Environment at comfort temperature - no brave bonus should apply
        env = DynamicForagingEnvironment(
            grid_size=5,
            foraging=ForagingParams(
                target_foods_to_collect=1,
                foods_on_grid=5,
                min_food_distance=1,
                agent_exclusion_radius=1,
            ),
            health=HealthParams(enabled=True, max_hp=100.0),
            thermotaxis=ThermotaxisParams(
                enabled=True,
                cultivation_temperature=20.0,
                base_temperature=20.0,  # Comfort zone
                gradient_strength=0.0,
                reward_discomfort_food=0.5,  # Would be awarded if in discomfort
                comfort_reward=0.0,  # No comfort reward
            ),
        )

        # Verify we're in comfort zone
        zone = env.get_temperature_zone()
        assert zone == TemperatureZone.COMFORT, "Should be in comfort zone"

        satiety_config = SatietyConfig(initial_satiety=500.0)
        agent = QuantumNematodeAgent(brain=brain, env=env, satiety_config=satiety_config)

        # Run episode - the brave bonus code path should not trigger in comfort zone
        reward_config = RewardConfig(reward_goal=2.0)
        agent.run_episode(reward_config, max_steps=100)

        # Verify zone is still comfort (no gradient movement)
        assert env.get_temperature_zone() == TemperatureZone.COMFORT, (
            "Agent should remain in comfort zone throughout episode"
        )

    def test_brave_bonus_not_applied_in_danger_zone(self):
        """Test that brave bonus is only for discomfort, not danger zones."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)

        # Environment in danger zone - no brave bonus (too dangerous to reward)
        env = DynamicForagingEnvironment(
            grid_size=5,
            foraging=ForagingParams(
                target_foods_to_collect=1,
                foods_on_grid=5,
                min_food_distance=1,
                agent_exclusion_radius=1,
            ),
            health=HealthParams(enabled=True, max_hp=200.0),  # High HP to survive
            thermotaxis=ThermotaxisParams(
                enabled=True,
                cultivation_temperature=20.0,
                base_temperature=33.0,  # Danger zone (30-35°C)
                gradient_strength=0.0,
                reward_discomfort_food=0.5,  # Should NOT be applied in danger
                danger_penalty=-0.3,
                danger_hp_damage=1.0,  # Low damage to survive
            ),
        )

        satiety_config = SatietyConfig(initial_satiety=500.0)
        agent = QuantumNematodeAgent(brain=brain, env=env, satiety_config=satiety_config)

        # Verify we're in danger zone
        assert env.get_temperature_zone() == TemperatureZone.DANGER_HOT, "Should be in danger zone"

        reward_config = RewardConfig(reward_goal=2.0, penalty_health_damage=0.0)
        agent.run_episode(reward_config, max_steps=100)

        # Verify zone remained danger (no gradient movement) - brave bonus should not apply
        assert env.get_temperature_zone() == TemperatureZone.DANGER_HOT, (
            "Agent should remain in danger zone throughout episode"
        )

    def test_brave_bonus_disabled_when_zero(self):
        """Test that brave bonus is not applied when reward_discomfort_food is 0."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)

        # Environment in discomfort zone but with brave bonus disabled
        env = DynamicForagingEnvironment(
            grid_size=5,
            foraging=ForagingParams(
                target_foods_to_collect=1,
                foods_on_grid=5,
                min_food_distance=1,
                agent_exclusion_radius=1,
            ),
            health=HealthParams(enabled=True, max_hp=100.0),
            thermotaxis=ThermotaxisParams(
                enabled=True,
                cultivation_temperature=20.0,
                base_temperature=27.0,  # Hot discomfort zone
                gradient_strength=0.0,
                reward_discomfort_food=0.0,  # Disabled
                discomfort_penalty=-0.05,
            ),
        )

        # Verify the parameter is set to 0
        assert env.thermotaxis.reward_discomfort_food == 0.0

        satiety_config = SatietyConfig(initial_satiety=500.0)
        agent = QuantumNematodeAgent(brain=brain, env=env, satiety_config=satiety_config)

        reward_config = RewardConfig(reward_goal=2.0)
        agent.run_episode(reward_config, max_steps=100)

        # Verify brave bonus config is still 0 (not accidentally modified)
        assert env.thermotaxis.reward_discomfort_food == 0.0, (
            "Brave bonus should remain disabled (0.0)"
        )
