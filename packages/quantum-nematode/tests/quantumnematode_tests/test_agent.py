"""Tests for QuantumNematodeAgent core functionality."""

import pytest
from quantumnematode.agent import (
    ManyworldsModeConfig,
    QuantumNematodeAgent,
    RewardConfig,
    SatietyConfig,
)
from quantumnematode.brain.arch.modular import ModularBrain, ModularBrainConfig
from quantumnematode.brain.modules import ModuleName
from quantumnematode.env import DynamicForagingEnvironment, MazeEnvironment


class TestSatietyConfig:
    """Test SatietyConfig data model."""

    def test_default_satiety_config(self):
        """Test creating SatietyConfig with default values."""
        config = SatietyConfig()

        assert config.initial_satiety > 0
        assert config.satiety_decay_rate > 0
        assert 0 < config.satiety_gain_per_food < 1

    def test_custom_satiety_config(self):
        """Test creating SatietyConfig with custom values."""
        config = SatietyConfig(
            initial_satiety=150.0,
            satiety_decay_rate=2.0,
            satiety_gain_per_food=0.5,
        )

        assert config.initial_satiety == 150.0
        assert config.satiety_decay_rate == 2.0
        assert config.satiety_gain_per_food == 0.5


class TestRewardConfig:
    """Test RewardConfig data model."""

    def test_default_reward_config(self):
        """Test creating RewardConfig with default values."""
        config = RewardConfig()

        assert config.reward_goal > 0
        assert config.reward_distance_scale > 0
        assert config.penalty_step > 0
        assert config.penalty_anti_dithering > 0
        assert config.penalty_stuck_position > 0
        assert config.stuck_position_threshold > 0

    def test_custom_reward_config(self):
        """Test creating RewardConfig with custom values."""
        config = RewardConfig(
            reward_goal=5.0,
            reward_distance_scale=1.0,
            penalty_step=0.01,
            penalty_anti_dithering=0.05,
            penalty_stuck_position=0.2,
            stuck_position_threshold=5,
            reward_exploration=0.1,
        )

        assert config.reward_goal == 5.0
        assert config.reward_distance_scale == 1.0
        assert config.reward_exploration == 0.1


class TestManyworldsModeConfig:
    """Test ManyworldsModeConfig data model."""

    def test_default_manyworlds_config(self):
        """Test creating ManyworldsModeConfig with default values."""
        config = ManyworldsModeConfig()

        assert config.max_superpositions > 0
        assert config.max_columns > 0
        assert config.render_sleep_seconds >= 0
        assert config.top_n_actions > 0

    def test_custom_manyworlds_config(self):
        """Test creating ManyworldsModeConfig with custom values."""
        config = ManyworldsModeConfig(
            max_superpositions=32,
            max_columns=8,
            render_sleep_seconds=1.0,
            top_n_actions=3,
            top_n_randomize=False,
        )

        assert config.max_superpositions == 32
        assert config.max_columns == 8
        assert config.render_sleep_seconds == 1.0
        assert config.top_n_actions == 3
        assert config.top_n_randomize is False


class TestQuantumNematodeAgentInitialization:
    """Test QuantumNematodeAgent initialization."""

    @pytest.fixture
    def modular_brain(self):
        """Create a simple modular brain for testing."""
        config = ModularBrainConfig(
            modules={ModuleName.CHEMOTAXIS: [0, 1]},
            num_layers=1,
        )
        return ModularBrain(config=config, shots=50)

    def test_agent_init_with_default_maze_env(self, modular_brain):
        """Test agent initialization creates default maze environment."""
        agent = QuantumNematodeAgent(brain=modular_brain)

        assert agent.brain is modular_brain
        assert isinstance(agent.env, MazeEnvironment)
        assert agent.steps == 0
        assert len(agent.path) == 1  # Initial position
        assert agent.success_count == 0

    def test_agent_init_with_custom_maze_env(self, modular_brain):
        """Test agent initialization with custom maze environment."""
        env = MazeEnvironment(grid_size=15, max_body_length=3)
        agent = QuantumNematodeAgent(brain=modular_brain, env=env)

        assert agent.env is env
        assert agent.env.grid_size == 15

    def test_agent_init_with_dynamic_env(self, modular_brain):
        """Test agent initialization with dynamic foraging environment."""
        env = DynamicForagingEnvironment(
            grid_size=30,
            num_initial_foods=5,
            max_active_foods=8,
        )
        agent = QuantumNematodeAgent(brain=modular_brain, env=env)

        assert isinstance(agent.env, DynamicForagingEnvironment)
        assert agent.env.grid_size == 30

    def test_agent_init_with_satiety_config(self, modular_brain):
        """Test agent initialization with custom satiety config."""
        satiety_config = SatietyConfig(
            initial_satiety=200.0,
            satiety_decay_rate=1.5,
            satiety_gain_per_food=0.4,
        )
        agent = QuantumNematodeAgent(
            brain=modular_brain,
            satiety_config=satiety_config,
        )

        assert agent.satiety == 200.0
        assert agent.max_satiety == 200.0
        assert agent.satiety_config.satiety_decay_rate == 1.5

    def test_agent_path_initialization(self, modular_brain):
        """Test that agent path is initialized with starting position."""
        agent = QuantumNematodeAgent(brain=modular_brain)

        assert len(agent.path) > 0
        assert agent.path[0] == tuple(agent.env.agent_pos)


class TestQuantumNematodeAgentGoalDistance:
    """Test agent goal distance calculation."""

    @pytest.fixture
    def modular_brain(self):
        """Create a simple modular brain for testing."""
        config = ModularBrainConfig(
            modules={ModuleName.CHEMOTAXIS: [0, 1]},
            num_layers=1,
        )
        return ModularBrain(config=config, shots=50)

    def test_calculate_goal_distance_maze(self, modular_brain):
        """Test goal distance calculation for maze environment."""
        env = MazeEnvironment(
            grid_size=10,
            start_pos=(0, 0),
            food_pos=(5, 5),
        )
        agent = QuantumNematodeAgent(brain=modular_brain, env=env)

        distance = agent.calculate_goal_distance()
        # Manhattan distance from (0,0) to (5,5) should be 10
        assert distance == 10

    def test_calculate_goal_distance_after_movement(self, modular_brain):
        """Test that goal distance changes after agent moves."""
        env = MazeEnvironment(
            grid_size=10,
            start_pos=(0, 0),
            food_pos=(5, 5),
        )
        agent = QuantumNematodeAgent(brain=modular_brain, env=env)

        initial_distance = agent.calculate_goal_distance()

        # Move agent closer to goal
        env.agent_pos = (1, 1)
        new_distance = agent.calculate_goal_distance()

        assert new_distance < initial_distance


class TestQuantumNematodeAgentReset:
    """Test agent reset functionality."""

    @pytest.fixture
    def modular_brain(self):
        """Create a simple modular brain for testing."""
        config = ModularBrainConfig(
            modules={ModuleName.CHEMOTAXIS: [0, 1]},
            num_layers=1,
        )
        return ModularBrain(config=config, shots=50)

    def test_reset_environment_maze(self, modular_brain):
        """Test resetting maze environment."""
        agent = QuantumNematodeAgent(brain=modular_brain)

        # Modify agent state
        agent.steps = 100
        agent.path = [(0, 0), (1, 1), (2, 2)]
        agent.success_count = 5

        # Reset environment
        agent.reset_environment()

        # Steps and path should be reset, but success_count should persist
        assert agent.steps == 0
        assert len(agent.path) == 1
        assert agent.success_count == 5  # Success count should not be reset

    def test_reset_environment_dynamic(self, modular_brain):
        """Test resetting dynamic foraging environment."""
        env = DynamicForagingEnvironment(
            grid_size=30,
            num_initial_foods=5,
            max_active_foods=8,
        )
        satiety_config = SatietyConfig(initial_satiety=100.0)
        agent = QuantumNematodeAgent(
            brain=modular_brain,
            env=env,
            satiety_config=satiety_config,
        )

        # Modify agent state
        agent.satiety = 50.0
        agent.foods_collected = 10

        # Reset environment
        agent.reset_environment()

        # Satiety should be restored to initial value
        assert agent.satiety == 100.0
        # Foods collected should be reset
        assert agent.foods_collected == 0

    def test_reset_brain(self, modular_brain):
        """Test resetting brain history data."""
        agent = QuantumNematodeAgent(brain=modular_brain)

        # Add some history data
        agent.brain.history_data.rewards.append(10.0)
        agent.brain.history_data.rewards.append(15.0)

        # Reset brain
        agent.reset_brain()

        # History should be cleared
        assert len(agent.brain.history_data.rewards) == 0


class TestQuantumNematodeAgentMetrics:
    """Test agent metrics calculation."""

    @pytest.fixture
    def modular_brain(self):
        """Create a simple modular brain for testing."""
        config = ModularBrainConfig(
            modules={ModuleName.CHEMOTAXIS: [0, 1]},
            num_layers=1,
        )
        return ModularBrain(config=config, shots=50)

    def test_calculate_metrics_static_env(self, modular_brain):
        """Test metrics calculation for static maze environment."""
        agent = QuantumNematodeAgent(brain=modular_brain)

        # Simulate some successful runs
        agent.success_count = 7
        agent.total_steps = 500
        agent.total_rewards = 100.0

        total_runs = 10
        metrics = agent.calculate_metrics(total_runs)

        assert metrics.success_rate == 0.7  # 7 out of 10
        assert metrics.average_steps == 50.0  # 500 / 10
        assert metrics.average_reward == 10.0  # 100 / 10
        # Static environment should not have foraging metrics
        assert metrics.foraging_efficiency is None
        assert metrics.average_distance_efficiency is None
        assert metrics.average_foods_collected is None

    def test_calculate_metrics_dynamic_env(self, modular_brain):
        """Test metrics calculation for dynamic foraging environment."""
        env = DynamicForagingEnvironment(
            grid_size=30,
            num_initial_foods=5,
            max_active_foods=8,
        )
        agent = QuantumNematodeAgent(brain=modular_brain, env=env)

        # Simulate some data
        agent.success_count = 3  # All runs successful
        agent.total_steps = 300
        agent.total_rewards = 150.0
        agent.foods_collected = 27  # Total foods collected across all runs
        agent.distance_efficiencies = [0.85, 0.90, 0.88]

        total_runs = 3
        metrics = agent.calculate_metrics(total_runs)

        assert metrics.success_rate == 1.0  # 3 / 3
        assert metrics.average_reward == 50.0  # 150 / 3
        assert metrics.average_steps == 100.0  # 300 / 3
        # Dynamic environment should have foraging metrics
        assert metrics.foraging_efficiency is not None
        assert metrics.foraging_efficiency == pytest.approx(0.09, rel=0.01)  # 27 / 300
        assert metrics.average_distance_efficiency == pytest.approx(0.877, rel=0.01)
        assert metrics.average_foods_collected == 9.0  # 27 / 3

    def test_calculate_metrics_no_data(self, modular_brain):
        """Test metrics calculation with minimal data."""
        agent = QuantumNematodeAgent(brain=modular_brain)

        # Initialize with some minimal data
        agent.total_steps = 0
        agent.total_rewards = 0.0
        agent.success_count = 0

        total_runs = 1
        metrics = agent.calculate_metrics(total_runs)

        # Should handle edge case gracefully
        assert metrics.success_rate == 0.0
        assert metrics.average_steps == 0.0
        assert metrics.average_reward == 0.0
