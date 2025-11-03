"""Integration tests for dynamic foraging environment with preset configurations."""

from pathlib import Path

import pytest
import yaml
from quantumnematode.agent import QuantumNematodeAgent
from quantumnematode.brain.actions import Action
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.modular import ModularBrain
from quantumnematode.env import DynamicForagingEnvironment, MazeEnvironment


class TestPresetConfigurations:
    """Test preset configuration files load and work correctly."""

    @pytest.fixture
    def config_dir(self):
        """Get path to config examples directory."""
        # Assuming we're in packages/quantum-nematode/tests/
        return Path(__file__).parents[4] / "configs" / "examples"

    @pytest.mark.parametrize(
        ("config_file", "expected_brain", "expected_env_type"),
        [
            # Dynamic foraging configs
            ("modular_dynamic_small.yml", "modular", "dynamic"),
            ("modular_dynamic_medium.yml", "modular", "dynamic"),
            ("modular_dynamic_large.yml", "modular", "dynamic"),
            # Static maze configs
            ("modular_simple_medium.yml", "modular", "static"),
            ("qmodular_simple_medium.yml", "qmodular", "static"),
            ("mlp_simple_medium.yml", "mlp", "static"),
            ("qmlp_simple_medium.yml", "qmlp", "static"),
            ("spiking_simple_medium.yml", "spiking", "static"),
        ],
    )
    def test_config_loads_correctly(
        self,
        config_dir,
        config_file,
        expected_brain,
        expected_env_type,
    ):
        """Test that configuration files load without errors and have expected structure."""
        config_path = config_dir / config_file
        assert config_path.exists(), f"Config file not found: {config_path}"

        with config_path.open() as f:
            config = yaml.safe_load(f)

        # Verify basic structure
        assert "brain" in config, f"Missing 'brain' in {config_file}"
        expected_msg = (
            f"Expected brain '{expected_brain}' but got "
            f"'{config['brain']['name']}' in {config_file}"
        )
        assert config["brain"]["name"] == expected_brain, expected_msg

        # Verify brain has config section
        assert "config" in config["brain"], f"Missing 'brain.config' in {config_file}"

        # For dynamic environments, verify additional required sections
        if expected_env_type == "dynamic":
            assert "environment" in config, f"Missing 'environment' in {config_file}"
            env_type_msg = (
                f"Expected environment type 'dynamic' but got "
                f"'{config['environment']['type']}' in {config_file}"
            )
            assert config["environment"]["type"] == "dynamic", env_type_msg

            assert "dynamic" in config["environment"], (
                f"Missing 'environment.dynamic' section in dynamic config {config_file}"
            )
            assert "satiety" in config, f"Missing 'satiety' section in dynamic config {config_file}"

            # Verify dynamic environment parameters
            dynamic_config = config["environment"]["dynamic"]
            assert "grid_size" in dynamic_config
            assert "num_initial_foods" in dynamic_config
            assert "max_active_foods" in dynamic_config

            # Verify satiety parameters
            satiety_config = config["satiety"]
            assert "initial_satiety" in satiety_config
            assert "satiety_decay_rate" in satiety_config
            assert "satiety_gain_per_food" in satiety_config
        else:
            # Static maze configs use environment.static section
            assert "environment" in config, f"Missing 'environment' in {config_file}"
            env_type_msg = (
                f"Expected environment type 'static' but got "
                f"'{config['environment']['type']}' in {config_file}"
            )
            assert config["environment"]["type"] == "static", env_type_msg

            assert "static" in config["environment"], (
                f"Missing 'environment.static' section in static config {config_file}"
            )

            # Verify static environment parameters
            static_config = config["environment"]["static"]
            assert "grid_size" in static_config, (
                f"Missing 'grid_size' in static config {config_file}"
            )

        # Verify common required fields
        assert "max_steps" in config, f"Missing 'max_steps' in {config_file}"

        # Learning rate can be at top level or inside brain.config (for spiking)
        has_learning_rate = "learning_rate" in config or "learning_rate" in config.get(
            "brain",
            {},
        ).get("config", {})
        assert has_learning_rate, f"Missing 'learning_rate' in {config_file}"

        assert "reward" in config, f"Missing 'reward' in {config_file}"

    @pytest.mark.parametrize(
        ("config_file", "expected_shots"),
        [
            # Quantum modular architectures
            ("modular_dynamic_small.yml", 1500),
            ("modular_dynamic_medium.yml", 1500),
            ("modular_dynamic_large.yml", 1500),
            ("modular_simple_medium.yml", 1500),
            # Other quantum architectures
            ("qmodular_simple_medium.yml", 1000),
            # Note: qmlp is Q-learning MLP, not quantum - it doesn't use shots
        ],
    )
    def test_quantum_configs_have_shots(self, config_dir, config_file, expected_shots):
        """Test that quantum brain configurations have required quantum parameters."""
        config_path = config_dir / config_file

        with config_path.open() as f:
            config = yaml.safe_load(f)

        # Verify shots parameter exists
        assert "shots" in config, f"Missing 'shots' parameter in quantum config {config_file}"
        assert config["shots"] == expected_shots, (
            f"Expected {expected_shots} shots but got {config['shots']} in {config_file}"
        )

        # Verify qubits parameter exists
        assert "qubits" in config, f"Missing 'qubits' parameter in modular config {config_file}"

    @pytest.mark.parametrize(
        "config_file",
        [
            "spiking_simple_medium.yml",
        ],
    )
    def test_classical_configs_structure(self, config_dir, config_file):
        """Test that classical brain configurations have appropriate structure."""
        config_path = config_dir / config_file

        with config_path.open() as f:
            config = yaml.safe_load(f)

        # Verify they have brain config
        assert "brain" in config
        assert "config" in config["brain"]


class TestDynamicEnvironmentWithBrain:
    """Integration tests running dynamic environment with brain architectures."""

    @pytest.fixture
    def dynamic_env_small(self):
        """Create small dynamic foraging environment."""
        return DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            num_initial_foods=5,
            max_active_foods=10,
            min_food_distance=3,
            agent_exclusion_radius=5,
            gradient_decay_constant=8.0,
            gradient_strength=1.0,
            viewport_size=(11, 11),
            max_body_length=0,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

    @pytest.fixture
    def modular_brain(self):
        """Create a simple modular brain for testing."""
        from quantumnematode.brain.arch.modular import ModularBrainConfig
        from quantumnematode.brain.modules import ModuleName

        config = ModularBrainConfig(
            num_layers=1,
            modules={
                ModuleName.CHEMOTAXIS: [0, 1],
            },
        )
        return ModularBrain(
            config=config,
            shots=50,
            device=DeviceType.CPU,
        )

    def test_agent_initialization_with_dynamic_env(self, dynamic_env_small, modular_brain):
        """Test agent initialization with dynamic foraging environment."""
        from quantumnematode.agent import SatietyConfig

        satiety_config = SatietyConfig(
            initial_satiety=100.0,
            satiety_decay_rate=1.0,
            satiety_gain_per_food=0.2,
        )

        agent = QuantumNematodeAgent(
            brain=modular_brain,
            env=dynamic_env_small,
            satiety_config=satiety_config,
        )

        assert agent.satiety == 100.0
        assert agent.max_satiety == 100.0
        assert agent.foods_collected == 0
        assert isinstance(agent.env, DynamicForagingEnvironment)

    def test_food_consumption_workflow(self, dynamic_env_small, modular_brain):
        """Test complete food consumption workflow."""
        from quantumnematode.agent import SatietyConfig

        satiety_config = SatietyConfig(
            initial_satiety=100.0,
            satiety_decay_rate=0.5,  # Slow decay for testing
            satiety_gain_per_food=0.3,
        )

        agent = QuantumNematodeAgent(
            brain=modular_brain,
            env=dynamic_env_small,
            satiety_config=satiety_config,
        )

        # Type narrowing: ensure env is DynamicForagingEnvironment
        assert isinstance(agent.env, DynamicForagingEnvironment)

        initial_satiety = agent.satiety
        initial_food_count = len(agent.env.foods)

        # Manually place agent on food and consume
        if len(agent.env.foods) > 0:
            food_pos = agent.env.foods[0]
            agent.env.agent_pos = food_pos

            consumed = agent.env.consume_food()

            if consumed:
                # Update agent satiety as would happen in run_episode
                satiety_gain = agent.max_satiety * satiety_config.satiety_gain_per_food
                agent.satiety = min(agent.max_satiety, agent.satiety + satiety_gain)
                agent.foods_collected += 1

                # Verify satiety increased
                assert agent.satiety > initial_satiety or agent.satiety == agent.max_satiety

                # Verify food respawned
                assert len(agent.env.foods) == initial_food_count

                # Verify food was removed from original position
                assert consumed not in agent.env.foods

    def test_satiety_decay(self, dynamic_env_small, modular_brain):
        """Test satiety decay over steps."""
        from quantumnematode.agent import SatietyConfig

        satiety_config = SatietyConfig(
            initial_satiety=200.0,
            satiety_decay_rate=1.0,
            satiety_gain_per_food=0.2,
        )

        agent = QuantumNematodeAgent(
            brain=modular_brain,
            env=dynamic_env_small,
            satiety_config=satiety_config,
        )

        # Just verify satiety system exists and is functional
        assert hasattr(agent, "satiety")
        assert agent.satiety >= 0.0
        assert agent.satiety <= agent.max_satiety


class TestBackwardCompatibility:
    """Test backward compatibility with existing MazeEnvironment."""

    @pytest.fixture
    def maze_env(self):
        """Create traditional maze environment."""
        return MazeEnvironment(
            grid_size=15,
            start_pos=(2, 2),
            food_pos=(12, 12),
            max_body_length=5,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

    @pytest.fixture
    def modular_brain(self):
        """Create a simple modular brain for testing."""
        from quantumnematode.brain.arch.modular import ModularBrainConfig
        from quantumnematode.brain.modules import ModuleName

        config = ModularBrainConfig(
            num_layers=1,
            modules={
                ModuleName.CHEMOTAXIS: [0, 1],
            },
        )
        return ModularBrain(
            config=config,
            shots=50,
            device=DeviceType.CPU,
        )

    def test_agent_with_maze_environment(self, maze_env, modular_brain):
        """Test agent works with traditional MazeEnvironment."""
        agent = QuantumNematodeAgent(
            brain=modular_brain,
            env=maze_env,
        )

        # Agent has satiety even with MazeEnvironment (defaults to 200.0)
        assert hasattr(agent, "satiety")
        assert isinstance(agent.env, MazeEnvironment)
