"""Tests for experiment metadata models."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from quantumnematode.experiment.metadata import (
    BenchmarkMetadata,
    BrainMetadata,
    EnvironmentMetadata,
    ExperimentMetadata,
    GradientMetadata,
    LearningRateMetadata,
    ParameterInitializer,
    ResultsMetadata,
    RewardMetadata,
    SystemMetadata,
)


class TestEnvironmentMetadata:
    """Test EnvironmentMetadata model."""

    def test_create_dynamic_environment(self):
        """Test creating dynamic environment metadata."""
        env_meta = EnvironmentMetadata(
            type="dynamic",
            grid_size=50,
            num_foods=20,
            target_foods_to_collect=5,
            initial_satiety=100.0,
            satiety_decay_rate=0.1,
            viewport_size=[5, 5],
        )

        assert env_meta.type == "dynamic"
        assert env_meta.grid_size == 50
        assert env_meta.num_foods == 20
        assert env_meta.target_foods_to_collect == 5
        assert env_meta.initial_satiety == 100.0
        assert env_meta.satiety_decay_rate == 0.1
        assert env_meta.viewport_size == [5, 5]

    def test_create_predator_environment(self):
        """Test creating predator-enabled environment metadata."""
        env_meta = EnvironmentMetadata(
            type="dynamic",
            grid_size=20,
            num_foods=5,
            target_foods_to_collect=10,
            initial_satiety=200.0,
            satiety_decay_rate=1.0,
            viewport_size=[11, 11],
            predators_enabled=True,
            num_predators=2,
            predator_speed=1.0,
            predator_detection_radius=8,
            predator_kill_radius=0,
            predator_gradient_decay=12.0,
            predator_gradient_strength=1.0,
        )

        assert env_meta.type == "dynamic"
        assert env_meta.predators_enabled is True
        assert env_meta.num_predators == 2
        assert env_meta.predator_speed == 1.0
        assert env_meta.predator_detection_radius == 8
        assert env_meta.predator_kill_radius == 0
        assert env_meta.predator_gradient_decay == 12.0
        assert env_meta.predator_gradient_strength == 1.0

    def test_create_no_predator_environment(self):
        """Test creating environment without predators has None for predator fields."""
        env_meta = EnvironmentMetadata(
            type="dynamic",
            grid_size=20,
            num_foods=5,
            predators_enabled=False,
        )

        assert env_meta.predators_enabled is False
        assert env_meta.num_predators is None
        assert env_meta.predator_speed is None
        assert env_meta.predator_detection_radius is None
        assert env_meta.predator_kill_radius is None
        assert env_meta.predator_gradient_decay is None
        assert env_meta.predator_gradient_strength is None

    def test_model_dump(self):
        """Test Pydantic model_dump."""
        env_meta = EnvironmentMetadata(type="dynamic", grid_size=30, num_foods=10)

        data = env_meta.model_dump()
        assert isinstance(data, dict)
        assert data["type"] == "dynamic"
        assert data["grid_size"] == 30

    def test_model_dump_with_predators(self):
        """Test Pydantic model_dump with predator fields."""
        env_meta = EnvironmentMetadata(
            type="dynamic",
            grid_size=20,
            predators_enabled=True,
            num_predators=3,
            predator_gradient_decay=10.0,
            predator_gradient_strength=1.5,
        )

        data = env_meta.model_dump()
        assert data["predators_enabled"] is True
        assert data["num_predators"] == 3
        assert data["predator_gradient_decay"] == 10.0
        assert data["predator_gradient_strength"] == 1.5


class TestBrainMetadata:
    """Test BrainMetadata model."""

    def test_create_quantum_brain(self):
        """Test creating quantum brain metadata."""
        brain_meta = BrainMetadata(
            type="modular",
            qubits=4,
            shots=1000,
            learning_rate=0.01,
        )

        assert brain_meta.type == "modular"
        assert brain_meta.qubits == 4
        assert brain_meta.shots == 1000
        assert brain_meta.learning_rate == 0.01

    def test_create_classical_brain(self):
        """Test creating classical brain metadata."""
        brain_meta = BrainMetadata(
            type="mlp",
            hidden_dim=128,
            num_hidden_layers=2,
            learning_rate=0.001,
        )

        assert brain_meta.type == "mlp"
        assert brain_meta.qubits is None
        assert brain_meta.hidden_dim == 128
        assert brain_meta.num_hidden_layers == 2

    def test_create_brain_with_random_initializer(self):
        """Test creating brain metadata with random parameter initializer."""
        brain_meta = BrainMetadata(
            type="modular",
            qubits=4,
            shots=1000,
            learning_rate=0.01,
            parameter_initializer=ParameterInitializer(type="random_pi"),
        )

        assert brain_meta.type == "modular"
        assert brain_meta.parameter_initializer is not None
        assert brain_meta.parameter_initializer.type == "random_pi"
        assert brain_meta.parameter_initializer.manual_parameter_values is None

    def test_create_brain_with_manual_initializer(self):
        """Test creating brain metadata with manual parameter initialization."""
        manual_params = {
            "θ_rx1_0": -2.4,
            "θ_rx1_1": -2.7,
            "θ_rx2_0": -1.3,
            "θ_rx2_1": -2.4,
            "θ_ry1_0": 2.9,
            "θ_ry1_1": -2.4,
        }

        brain_meta = BrainMetadata(
            type="modular",
            qubits=2,
            num_layers=2,
            learning_rate=1.0,
            parameter_initializer=ParameterInitializer(
                type="manual",
                manual_parameter_values=manual_params,
            ),
        )

        assert brain_meta.parameter_initializer is not None
        assert brain_meta.parameter_initializer.type == "manual"
        assert brain_meta.parameter_initializer.manual_parameter_values == manual_params
        assert brain_meta.parameter_initializer.manual_parameter_values is not None
        assert brain_meta.parameter_initializer.manual_parameter_values["θ_rx1_0"] == -2.4
        assert brain_meta.parameter_initializer.manual_parameter_values["θ_ry1_1"] == -2.4

    def test_create_brain_no_initializer(self):
        """Test creating brain metadata without initializer info."""
        brain_meta = BrainMetadata(
            type="mlp",
            hidden_dim=64,
            learning_rate=0.001,
        )

        assert brain_meta.parameter_initializer is None

    def test_model_dump(self):
        """Test Pydantic model_dump."""
        brain_meta = BrainMetadata(type="modular", qubits=4, learning_rate=0.01)

        data = brain_meta.model_dump()
        assert isinstance(data, dict)
        assert data["type"] == "modular"
        assert data["qubits"] == 4
        assert data["learning_rate"] == 0.01

    def test_model_dump_with_initializer(self):
        """Test Pydantic model_dump with parameter initializer."""
        manual_params = {"θ_rx1_0": -2.4, "θ_ry1_0": 2.9}
        brain_meta = BrainMetadata(
            type="modular",
            qubits=2,
            learning_rate=1.0,
            parameter_initializer=ParameterInitializer(
                type="manual",
                manual_parameter_values=manual_params,
            ),
        )

        data = brain_meta.model_dump()
        assert data["parameter_initializer"] is not None
        assert data["parameter_initializer"]["type"] == "manual"
        assert data["parameter_initializer"]["manual_parameter_values"] == manual_params
        assert data["parameter_initializer"]["manual_parameter_values"]["θ_rx1_0"] == -2.4


class TestRewardMetadata:
    """Test RewardMetadata model."""

    def test_create_reward_metadata(self, reward_metadata: RewardMetadata):
        """Test creating reward metadata."""
        assert reward_metadata.reward_goal == 2.0
        assert reward_metadata.reward_distance_scale == 0.5
        assert reward_metadata.reward_exploration == 0.05
        assert reward_metadata.penalty_step == 0.005
        assert reward_metadata.penalty_anti_dithering == 0.02
        assert reward_metadata.penalty_stuck_position == 0.1
        assert reward_metadata.stuck_position_threshold == 3
        assert reward_metadata.penalty_starvation == 10.0
        assert reward_metadata.penalty_predator_death == 10.0
        assert reward_metadata.penalty_predator_proximity == 0.1

    def test_model_dump(self):
        """Test Pydantic model_dump."""
        reward_meta = RewardMetadata(
            reward_goal=2.0,
            reward_distance_scale=0.5,
            reward_exploration=0.05,
            penalty_step=0.005,
            penalty_anti_dithering=0.02,
            penalty_stuck_position=0.1,
            stuck_position_threshold=3,
            penalty_starvation=10.0,
            penalty_predator_death=10.0,
            penalty_predator_proximity=0.1,
        )

        data = reward_meta.model_dump()
        assert isinstance(data, dict)
        assert data["reward_goal"] == 2.0
        assert data["penalty_predator_proximity"] == 0.1


class TestLearningRateMetadata:
    """Test LearningRateMetadata model."""

    def test_create_static_learning_rate(self):
        """Test creating static learning rate metadata."""
        lr_meta = LearningRateMetadata(
            method="static",
            initial_learning_rate=0.01,
        )

        assert lr_meta.method == "static"
        assert lr_meta.initial_learning_rate == 0.01
        assert lr_meta.decay_type is None
        assert lr_meta.decay_rate is None
        assert lr_meta.min_lr is None

    def test_create_dynamic_learning_rate(self):
        """Test creating dynamic learning rate metadata."""
        lr_meta = LearningRateMetadata(
            method="dynamic",
            initial_learning_rate=0.001,
            decay_type="exponential",
            decay_rate=0.9995,
            min_lr=0.0003,
        )

        assert lr_meta.method == "dynamic"
        assert lr_meta.initial_learning_rate == 0.001
        assert lr_meta.decay_type == "exponential"
        assert lr_meta.decay_rate == 0.9995
        assert lr_meta.min_lr == 0.0003

    def test_create_step_decay_learning_rate(self):
        """Test creating learning rate metadata with step decay."""
        lr_meta = LearningRateMetadata(
            method="dynamic",
            initial_learning_rate=0.01,
            decay_type="step",
            decay_rate=0.5,
            decay_factor=0.5,
            min_lr=0.0001,
            step_size=100,
        )

        assert lr_meta.method == "dynamic"
        assert lr_meta.initial_learning_rate == 0.01
        assert lr_meta.decay_type == "step"
        assert lr_meta.decay_rate == 0.5
        assert lr_meta.decay_factor == 0.5
        assert lr_meta.min_lr == 0.0001
        assert lr_meta.step_size == 100

    def test_create_polynomial_decay_learning_rate(self):
        """Test creating learning rate metadata with polynomial decay."""
        lr_meta = LearningRateMetadata(
            method="dynamic",
            initial_learning_rate=0.01,
            decay_type="polynomial",
            decay_rate=0.1,
            power=2.0,
            min_lr=0.0001,
            max_steps=500,
        )

        assert lr_meta.method == "dynamic"
        assert lr_meta.initial_learning_rate == 0.01
        assert lr_meta.decay_type == "polynomial"
        assert lr_meta.decay_rate == 0.1
        assert lr_meta.power == 2.0
        assert lr_meta.min_lr == 0.0001
        assert lr_meta.max_steps == 500

    def test_model_dump(self):
        """Test Pydantic model_dump."""
        lr_meta = LearningRateMetadata(
            method="dynamic",
            initial_learning_rate=0.001,
            decay_type="exponential",
            decay_rate=0.9995,
        )

        data = lr_meta.model_dump()
        assert isinstance(data, dict)
        assert data["method"] == "dynamic"
        assert data["decay_type"] == "exponential"


class TestGradientMetadata:
    """Test GradientMetadata model."""

    def test_create_gradient_metadata(self):
        """Test creating gradient metadata."""
        grad_meta = GradientMetadata(method="clip")

        assert grad_meta.method == "clip"

    def test_create_with_different_methods(self):
        """Test creating gradient metadata with different methods."""
        for method in ["raw", "clip", "normalize"]:
            grad_meta = GradientMetadata(method=method)
            assert grad_meta.method == method

    def test_model_dump(self):
        """Test Pydantic model_dump."""
        grad_meta = GradientMetadata(method="clip")

        data = grad_meta.model_dump()
        assert isinstance(data, dict)
        assert data["method"] == "clip"


class TestResultsMetadata:
    """Test ResultsMetadata model."""

    def test_create_basic_results(self):
        """Test creating basic results metadata."""
        results = ResultsMetadata(
            total_runs=50,
            success_rate=0.85,
            avg_steps=45.5,
            avg_reward=100.5,
        )

        assert results.total_runs == 50
        assert results.success_rate == 0.85
        assert results.avg_steps == 45.5
        assert results.avg_reward == 100.5

    def test_create_foraging_results(self):
        """Test creating foraging results with optional fields."""
        results = ResultsMetadata(
            total_runs=50,
            success_rate=0.92,
            avg_steps=38.2,
            avg_reward=150.0,
            avg_foods_collected=18.5,
            avg_distance_efficiency=0.75,
            completed_all_food=46,
            starved=2,
            max_steps_reached=2,
        )

        assert results.avg_foods_collected == 18.5
        assert results.avg_distance_efficiency == 0.75
        assert results.completed_all_food == 46
        assert results.starved == 2

    def test_model_dump(self):
        """Test Pydantic model_dump."""
        results = ResultsMetadata(
            total_runs=10,
            success_rate=0.8,
            avg_steps=50.0,
            avg_reward=75.0,
        )

        data = results.model_dump()
        assert isinstance(data, dict)
        assert data["total_runs"] == 10
        assert data["success_rate"] == 0.8


class TestSystemMetadata:
    """Test SystemMetadata model."""

    def test_create_system_metadata(self):
        """Test creating system metadata."""
        system = SystemMetadata(
            python_version="3.12.0",
            qiskit_version="1.0.0",
            torch_version="2.1.0",
            device_type="cpu",
        )

        assert system.python_version == "3.12.0"
        assert system.qiskit_version == "1.0.0"
        assert system.torch_version == "2.1.0"
        assert system.device_type == "cpu"
        assert system.qpu_backend is None

    def test_create_with_qpu(self):
        """Test creating system metadata with QPU backend."""
        system = SystemMetadata(
            python_version="3.12.0",
            qiskit_version="1.0.0",
            device_type="qpu",
            qpu_backend="ibm_brisbane",
        )

        assert system.device_type == "qpu"
        assert system.qpu_backend == "ibm_brisbane"

    def test_model_dump(self):
        """Test Pydantic model_dump."""
        system = SystemMetadata(
            python_version="3.12.0",
            qiskit_version="1.0.0",
            device_type="cpu",
        )

        data = system.model_dump()
        assert isinstance(data, dict)
        assert data["python_version"] == "3.12.0"
        assert data["device_type"] == "cpu"


class TestBenchmarkMetadata:
    """Test BenchmarkMetadata model."""

    def test_create_benchmark_metadata(self):
        """Test creating benchmark metadata."""
        benchmark = BenchmarkMetadata(
            contributor="Jane Doe",
            github_username="janedoe",
            category="dynamic_medium_quantum",
            verified=False,
            notes="Optimized learning rate schedule",
        )

        assert benchmark.contributor == "Jane Doe"
        assert benchmark.github_username == "janedoe"
        assert benchmark.category == "dynamic_medium_quantum"
        assert benchmark.verified is False
        assert benchmark.notes == "Optimized learning rate schedule"

    def test_create_without_optional_fields(self):
        """Test creating benchmark with only required fields."""
        benchmark = BenchmarkMetadata(
            contributor="John Smith",
            category="dynamic_small_classical",
        )

        assert benchmark.contributor == "John Smith"
        assert benchmark.github_username is None
        assert benchmark.notes is None

    def test_model_dump(self):
        """Test Pydantic model_dump."""
        benchmark = BenchmarkMetadata(
            contributor="Alice",
            category="dynamic_small_quantum",
        )

        data = benchmark.model_dump()
        assert isinstance(data, dict)
        assert data["contributor"] == "Alice"
        assert data["category"] == "dynamic_small_quantum"


class TestExperimentMetadata:
    """Test complete ExperimentMetadata model."""

    def test_create_complete_experiment(self):
        """Test creating complete experiment metadata."""
        now = datetime.now(UTC)
        experiment = ExperimentMetadata(
            experiment_id="20250101_120000",
            timestamp=now,
            config_file="configs/test.yml",
            config_hash="abc123",
            git_commit="def456",
            git_branch="main",
            git_dirty=False,
            environment=EnvironmentMetadata(type="dynamic", grid_size=50),
            brain=BrainMetadata(type="modular", qubits=4, learning_rate=0.01),
            reward=RewardMetadata(
                reward_goal=2.0,
                reward_distance_scale=0.5,
                reward_exploration=0.05,
                penalty_step=0.005,
                penalty_anti_dithering=0.02,
                penalty_stuck_position=0.1,
                stuck_position_threshold=3,
                penalty_starvation=10.0,
                penalty_predator_death=10.0,
                penalty_predator_proximity=0.1,
            ),
            learning_rate=LearningRateMetadata(
                method="dynamic",
                initial_learning_rate=0.001,
                decay_type="exponential",
                decay_rate=0.9995,
                min_lr=0.0003,
            ),
            gradient=GradientMetadata(method="clip"),
            results=ResultsMetadata(
                total_runs=50,
                success_rate=0.9,
                avg_steps=40.0,
                avg_reward=120.0,
            ),
            system=SystemMetadata(
                python_version="3.12.0",
                qiskit_version="1.0.0",
                device_type="cpu",
            ),
        )

        assert experiment.experiment_id == "20250101_120000"
        assert experiment.config_file == "configs/test.yml"
        assert experiment.git_commit == "def456"
        assert experiment.git_branch == "main"
        assert experiment.git_dirty is False
        assert experiment.environment.grid_size == 50
        assert experiment.brain.qubits == 4
        assert experiment.reward.reward_goal == 2.0
        assert experiment.reward.penalty_predator_proximity == 0.1
        assert experiment.learning_rate is not None
        assert experiment.learning_rate.method == "dynamic"
        assert experiment.learning_rate.decay_type == "exponential"
        assert experiment.gradient.method == "clip"
        assert experiment.results.success_rate == 0.9
        assert experiment.system.python_version == "3.12.0"

    def test_create_with_benchmark(self):
        """Test creating experiment with benchmark metadata."""
        now = datetime.now(UTC)
        experiment = ExperimentMetadata(
            experiment_id="20250101_130000",
            timestamp=now,
            config_file="configs/test.yml",
            config_hash="xyz789",
            environment=EnvironmentMetadata(type="static", grid_size=10),
            brain=BrainMetadata(type="mlp", learning_rate=0.001),
            reward=RewardMetadata(
                reward_goal=0.2,
                reward_distance_scale=0.1,
                reward_exploration=0.05,
                penalty_step=0.01,
                penalty_anti_dithering=0.05,
                penalty_stuck_position=0.02,
                stuck_position_threshold=2,
                penalty_starvation=10.0,
                penalty_predator_death=10.0,
                penalty_predator_proximity=0.1,
            ),
            learning_rate=LearningRateMetadata(
                method="static",
                initial_learning_rate=0.001,
            ),
            gradient=GradientMetadata(method="raw"),
            results=ResultsMetadata(
                total_runs=10,
                success_rate=0.8,
                avg_steps=50.0,
                avg_reward=80.0,
            ),
            system=SystemMetadata(
                python_version="3.12.0",
                qiskit_version="1.0.0",
                device_type="cpu",
            ),
            benchmark=BenchmarkMetadata(
                contributor="Test User",
                category="dynamic_small_classical",
            ),
        )

        assert experiment.benchmark is not None
        assert experiment.benchmark.contributor == "Test User"
        assert experiment.benchmark.category == "dynamic_small_classical"

    def test_serialization(self):
        """Test complete serialization and deserialization."""
        now = datetime.now(UTC)
        experiment = ExperimentMetadata(
            experiment_id="20250101_140000",
            timestamp=now,
            config_file="configs/benchmark.yml",
            config_hash="hash123",
            git_commit="commit123",
            git_branch="feature",
            git_dirty=True,
            environment=EnvironmentMetadata(
                type="dynamic",
                grid_size=50,
                num_foods=20,
            ),
            brain=BrainMetadata(
                type="modular",
                qubits=6,
                shots=2000,
                learning_rate=0.02,
            ),
            reward=RewardMetadata(
                reward_goal=2.0,
                reward_distance_scale=0.5,
                reward_exploration=0.05,
                penalty_step=0.005,
                penalty_anti_dithering=0.02,
                penalty_stuck_position=0.1,
                stuck_position_threshold=3,
                penalty_starvation=10.0,
                penalty_predator_death=10.0,
                penalty_predator_proximity=0.1,
            ),
            learning_rate=LearningRateMetadata(
                method="dynamic",
                initial_learning_rate=0.001,
                decay_type="exponential",
                decay_rate=0.9995,
                min_lr=0.0003,
            ),
            gradient=GradientMetadata(method="clip"),
            results=ResultsMetadata(
                total_runs=30,
                success_rate=0.95,
                avg_steps=35.5,
                avg_reward=140.0,
                avg_foods_collected=19.8,
            ),
            system=SystemMetadata(
                python_version="3.12.1",
                qiskit_version="1.1.0",
                torch_version="2.2.0",
                device_type="gpu",
            ),
            exports_path="exports/20250101_140000",
        )

        # Test full serialization (with exclude_config_details=False for backward compat)
        data = experiment.to_dict(exclude_config_details=False)
        assert isinstance(data, dict)
        assert data["experiment_id"] == "20250101_140000"
        assert data["environment"]["grid_size"] == 50
        assert data["brain"]["qubits"] == 6
        assert data["reward"]["reward_goal"] == 2.0
        assert data["learning_rate"]["method"] == "dynamic"
        assert data["gradient"]["method"] == "clip"
        assert data["results"]["success_rate"] == 0.95

        # Test full deserialization roundtrip
        restored = ExperimentMetadata.from_dict(data)
        assert restored.experiment_id == experiment.experiment_id
        assert restored.environment.grid_size == experiment.environment.grid_size
        assert restored.brain.qubits == experiment.brain.qubits
        assert restored.reward.reward_goal == experiment.reward.reward_goal
        assert restored.learning_rate is not None
        assert experiment.learning_rate is not None
        assert restored.learning_rate.method == experiment.learning_rate.method
        assert restored.gradient.method == experiment.gradient.method
        assert restored.results.total_runs == experiment.results.total_runs
        assert restored.system.python_version == experiment.system.python_version
        assert restored.exports_path == experiment.exports_path

    def test_serialization_lean_format(self):
        """Test lean serialization format (excludes duplicate config data)."""
        now = datetime.now(UTC)
        experiment = ExperimentMetadata(
            experiment_id="20250101_140000",
            timestamp=now,
            config_file="configs/benchmark.yml",
            config_hash="hash123",
            environment=EnvironmentMetadata(
                type="dynamic",
                grid_size=50,
                predators_enabled=True,
            ),
            brain=BrainMetadata(type="ppo", hidden_dim=128),
            reward=RewardMetadata(
                reward_goal=2.0,
                reward_distance_scale=0.5,
                reward_exploration=0.05,
                penalty_step=0.005,
                penalty_anti_dithering=0.02,
                penalty_stuck_position=0.1,
                stuck_position_threshold=3,
                penalty_starvation=10.0,
                penalty_predator_death=10.0,
                penalty_predator_proximity=0.1,
            ),
            gradient=GradientMetadata(method="clip"),
            results=ResultsMetadata(
                total_runs=30,
                success_rate=0.95,
                avg_steps=35.5,
                avg_reward=140.0,
            ),
            system=SystemMetadata(
                python_version="3.12.1",
                qiskit_version="1.1.0",
                device_type="cpu",
            ),
        )

        # Test lean serialization (default behavior)
        data = experiment.to_dict()  # exclude_config_details=True by default
        assert isinstance(data, dict)
        assert data["experiment_id"] == "20250101_140000"

        # Config sections should be excluded
        assert "environment" not in data
        assert "brain" not in data
        assert "reward" not in data
        assert "gradient" not in data

        # config_summary should be present with essential fields
        assert "config_summary" in data
        assert data["config_summary"]["brain_type"] == "ppo"
        assert data["config_summary"]["environment_type"] == "dynamic"
        assert data["config_summary"]["grid_size"] == 50
        assert data["config_summary"]["predators_enabled"] is True

        # Results and system should still be present
        assert data["results"]["success_rate"] == 0.95
        assert data["system"]["python_version"] == "3.12.1"

        # Verify config_summary comes before results in the dict
        keys = list(data.keys())
        config_summary_idx = keys.index("config_summary")
        results_idx = keys.index("results")
        assert config_summary_idx < results_idx, "config_summary should appear before results"

        # Test deserialization from lean format
        restored = ExperimentMetadata.from_dict(data)
        assert restored.experiment_id == experiment.experiment_id
        # Essential fields are restored from config_summary
        assert restored.environment.type == "dynamic"
        assert restored.environment.grid_size == 50
        assert restored.environment.predators_enabled is True
        assert restored.brain.type == "ppo"
        # Results are fully restored
        assert restored.results.success_rate == experiment.results.success_rate
        assert restored.system.python_version == experiment.system.python_version

    def test_validation_required_fields(self):
        """Test that required fields are validated."""
        with pytest.raises(ValidationError):
            # Missing required fields
            ExperimentMetadata()  # pyright: ignore[reportCallIssue]
