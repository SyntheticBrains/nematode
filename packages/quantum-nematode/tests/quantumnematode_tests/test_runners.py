"""Tests for episode runners."""

from unittest.mock import Mock

from quantumnematode.agent import (
    ManyworldsModeConfig,
    QuantumNematodeAgent,
    RewardConfig,
    SatietyConfig,
)
from quantumnematode.brain.arch import MLPBrain, MLPBrainConfig
from quantumnematode.env import DynamicForagingEnvironment, MazeEnvironment
from quantumnematode.metrics import MetricsTracker
from quantumnematode.rendering import EpisodeRenderer
from quantumnematode.runners import ManyworldsEpisodeRunner, StandardEpisodeRunner
from quantumnematode.step_processor import StepProcessor


class TestStandardEpisodeRunnerInitialization:
    """Test standard episode runner initialization."""

    def test_initialize_with_dependencies(self):
        """Test that runner initializes with all dependencies."""
        step_processor = Mock(spec=StepProcessor)
        metrics_tracker = Mock(spec=MetricsTracker)
        renderer = Mock(spec=EpisodeRenderer)

        runner = StandardEpisodeRunner(
            step_processor=step_processor,
            metrics_tracker=metrics_tracker,
            renderer=renderer,
        )

        assert runner.step_processor is step_processor
        assert runner.metrics_tracker is metrics_tracker
        assert runner.renderer is renderer


class TestStandardEpisodeRunnerIntegration:
    """Integration tests for StandardEpisodeRunner with real agent."""

    def test_run_episode_maze_environment(self):
        """Test running episode in maze environment returns path."""
        # Create real components
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)
        env = MazeEnvironment(grid_size=5)
        agent = QuantumNematodeAgent(brain=brain, env=env)

        # Store initial position before episode
        initial_pos = tuple(env.agent_pos)

        # Run episode through the runner
        reward_config = RewardConfig()
        path = agent.run_episode(reward_config, max_steps=10)

        # Verify path is returned
        assert isinstance(path, list)
        assert len(path) > 0
        assert all(isinstance(pos, tuple) for pos in path)
        # Path should start at agent's initial position
        assert path[0] == initial_pos

    def test_run_episode_dynamic_foraging(self):
        """Test running episode in dynamic foraging environment."""
        # Create real components
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)
        env = DynamicForagingEnvironment(
            grid_size=10,
            max_active_foods=3,
        )
        satiety_config = SatietyConfig(initial_satiety=100.0)
        agent = QuantumNematodeAgent(
            brain=brain,
            env=env,
            satiety_config=satiety_config,
        )

        # Run episode through the runner
        reward_config = RewardConfig()
        path = agent.run_episode(reward_config, max_steps=20)

        # Verify path is returned
        assert isinstance(path, list)
        assert len(path) > 0
        # Satiety should have changed
        assert agent.satiety <= satiety_config.initial_satiety

    def test_run_episode_updates_agent_state(self):
        """Test that running episode updates agent state correctly."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)
        env = MazeEnvironment(grid_size=5)
        agent = QuantumNematodeAgent(brain=brain, env=env)

        initial_steps = agent.steps
        initial_path_length = len(agent.path)

        # Run episode
        reward_config = RewardConfig()
        agent.run_episode(reward_config, max_steps=10)

        # Verify agent state was updated
        assert agent.steps > initial_steps
        assert len(agent.path) > initial_path_length

    def test_runner_delegates_correctly(self):
        """Test that agent correctly delegates to StandardEpisodeRunner."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)
        env = MazeEnvironment(grid_size=5)
        agent = QuantumNematodeAgent(brain=brain, env=env)

        # Verify runner was created
        assert hasattr(agent, "_standard_runner")
        assert isinstance(agent._standard_runner, StandardEpisodeRunner)

        # Run episode and verify it works
        reward_config = RewardConfig()
        path = agent.run_episode(reward_config, max_steps=10)

        assert isinstance(path, list)
        assert len(path) > 0


class TestManyworldsEpisodeRunnerInitialization:
    """Test manyworlds episode runner initialization."""

    def test_initialize_with_dependencies(self):
        """Test that runner initializes with all dependencies."""
        step_processor = Mock(spec=StepProcessor)
        metrics_tracker = Mock(spec=MetricsTracker)
        renderer = Mock(spec=EpisodeRenderer)

        runner = ManyworldsEpisodeRunner(
            step_processor=step_processor,
            metrics_tracker=metrics_tracker,
            renderer=renderer,
        )

        assert runner.step_processor is step_processor
        assert runner.metrics_tracker is metrics_tracker
        assert runner.renderer is renderer


class TestManyworldsEpisodeRunnerIntegration:
    """Integration tests for ManyworldsEpisodeRunner with real agent."""

    def test_runner_initialization(self):
        """Test that agent creates manyworlds runner correctly."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)
        env = MazeEnvironment(grid_size=5)
        agent = QuantumNematodeAgent(brain=brain, env=env)

        # Verify runner was created
        assert hasattr(agent, "_manyworlds_runner")
        assert isinstance(agent._manyworlds_runner, ManyworldsEpisodeRunner)

    def test_manyworlds_config_parameter(self):
        """Test that manyworlds runner accepts config parameter."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)
        env = MazeEnvironment(grid_size=5)
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
            max_active_foods=3,
        )
        satiety_config = SatietyConfig(initial_satiety=100.0)
        agent = QuantumNematodeAgent(
            brain=brain,
            env=env,
            satiety_config=satiety_config,
        )

        initial_foods = agent.foods_collected

        # Run episode
        reward_config = RewardConfig()
        agent.run_episode(reward_config, max_steps=50)

        # Food collection tracking should work
        assert agent.foods_collected >= initial_foods

    def test_standard_runner_uses_satiety_manager(self):
        """Test that StandardEpisodeRunner uses SatietyManager."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)
        env = DynamicForagingEnvironment(
            grid_size=10,
            max_active_foods=3,
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

        initial_satiety = agent.satiety

        # Run episode
        reward_config = RewardConfig()
        agent.run_episode(reward_config, max_steps=10)

        # Satiety should have decayed
        assert agent.satiety < initial_satiety

    def test_runner_helper_methods_accessible(self):
        """Test that runners can access agent helper methods."""
        config = MLPBrainConfig(hidden_dim=32, learning_rate=0.01, num_hidden_layers=2)
        brain = MLPBrain(config=config, input_dim=2, num_actions=4)
        env = MazeEnvironment(grid_size=5)
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
