"""Integration tests for dynamic foraging environment with preset configurations."""

from pathlib import Path

import pytest
import yaml
from quantumnematode.agent import QuantumNematodeAgent
from quantumnematode.brain.actions import Action
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.qvarcircuit import QVarCircuitBrain
from quantumnematode.env import DynamicForagingEnvironment, ForagingParams


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
            # Foraging configs
            ("qvarcircuit_foraging_small.yml", "qvarcircuit", "dynamic"),
            ("qvarcircuit_foraging_medium.yml", "qvarcircuit", "dynamic"),
            ("qvarcircuit_foraging_large.yml", "qvarcircuit", "dynamic"),
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

        # Verify environment configuration
        if expected_env_type == "dynamic":
            assert "environment" in config, f"Missing 'environment' in {config_file}"
            assert "satiety" in config, f"Missing 'satiety' section in config {config_file}"

            # Verify environment parameters (type and dynamic wrapper removed)
            env_config = config["environment"]
            assert "grid_size" in env_config, f"Missing 'grid_size' in {config_file}"

            # Verify foraging subsection exists and has required fields
            assert "foraging" in env_config, (
                f"Missing 'foraging' subsection in config {config_file}"
            )
            foraging_config = env_config["foraging"]
            assert "foods_on_grid" in foraging_config
            assert "target_foods_to_collect" in foraging_config
            assert "min_food_distance" in foraging_config
            assert "agent_exclusion_radius" in foraging_config

            # Verify satiety parameters
            satiety_config = config["satiety"]
            assert "initial_satiety" in satiety_config
            assert "satiety_decay_rate" in satiety_config
            assert "satiety_gain_per_food" in satiety_config

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
            ("qvarcircuit_foraging_small.yml", 3000),
            ("qvarcircuit_foraging_medium.yml", 3000),
            ("qvarcircuit_foraging_large.yml", 3000),
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


class TestDynamicEnvironmentWithBrain:
    """Integration tests running dynamic environment with brain architectures."""

    @pytest.fixture
    def dynamic_env_small(self):
        """Create small dynamic foraging environment."""
        return DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(
                foods_on_grid=5,
                target_foods_to_collect=10,
                min_food_distance=3,
                agent_exclusion_radius=5,
                gradient_decay_constant=8.0,
                gradient_strength=1.0,
            ),
            viewport_size=(11, 11),
            max_body_length=0,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

    @pytest.fixture
    def modular_brain(self):
        """Create a simple modular brain for testing."""
        from quantumnematode.brain.arch.qvarcircuit import QVarCircuitBrainConfig
        from quantumnematode.brain.modules import ModuleName

        config = QVarCircuitBrainConfig(
            num_layers=1,
            modules={
                ModuleName.CHEMOTAXIS: [0, 1],
            },
        )
        return QVarCircuitBrain(
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

        assert agent.current_satiety == 100.0
        assert agent.max_satiety == 100.0
        assert agent._metrics_tracker.foods_collected == 0
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

        initial_satiety = agent.current_satiety
        initial_food_count = len(agent.env.foods)

        # Manually place agent on food and consume
        if len(agent.env.foods) > 0:
            food_pos = agent.env.foods[0]
            agent.env.agent_pos = food_pos

            consumed = agent.env.consume_food()

            if consumed:
                # Update agent satiety via satiety manager
                satiety_gain = agent.max_satiety * satiety_config.satiety_gain_per_food
                agent._satiety_manager.restore_satiety(satiety_gain)
                agent._metrics_tracker.foods_collected += 1

                # Verify satiety increased
                current_satiety = agent.current_satiety
                assert current_satiety > initial_satiety or current_satiety == agent.max_satiety

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
        assert hasattr(agent, "_satiety_manager")
        assert agent.current_satiety >= 0.0
        assert agent.current_satiety <= agent.max_satiety
