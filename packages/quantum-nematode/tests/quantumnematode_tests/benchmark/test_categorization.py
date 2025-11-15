"""Tests for benchmark categorization."""

from datetime import UTC, datetime

import pytest
from quantumnematode.benchmark.categorization import (
    determine_benchmark_category,
    get_category_directory,
)
from quantumnematode.experiment.metadata import (
    BrainMetadata,
    EnvironmentMetadata,
    ExperimentMetadata,
    ResultsMetadata,
    SystemMetadata,
)


def create_test_experiment(env: EnvironmentMetadata, brain: BrainMetadata) -> ExperimentMetadata:
    """Create test experiment metadata."""
    return ExperimentMetadata(
        experiment_id="test_id",
        timestamp=datetime.now(UTC),
        config_file="test.yml",
        config_hash="abc123",
        environment=env,
        brain=brain,
        results=ResultsMetadata(
            total_runs=20,
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


class TestDetermineBenchmarkCategory:
    """Test benchmark category determination."""

    def test_static_maze_quantum(self):
        """Test categorizing static maze with quantum brain."""
        env = EnvironmentMetadata(type="static", grid_size=10)
        brain = BrainMetadata(type="modular", qubits=4, learning_rate=0.01)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "static_maze_quantum"

    def test_static_maze_classical(self):
        """Test categorizing static maze with classical brain."""
        env = EnvironmentMetadata(type="static", grid_size=10)
        brain = BrainMetadata(type="mlp", learning_rate=0.001)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "static_maze_classical"

    def test_dynamic_small_quantum(self):
        """Test categorizing small dynamic environment with quantum brain."""
        env = EnvironmentMetadata(type="dynamic", grid_size=15, num_foods=10)
        brain = BrainMetadata(type="qmodular", qubits=6, learning_rate=0.02)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "dynamic_small_quantum"

    def test_dynamic_small_classical(self):
        """Test categorizing small dynamic environment with classical brain."""
        env = EnvironmentMetadata(type="dynamic", grid_size=20, num_foods=15)
        brain = BrainMetadata(type="qmlp", learning_rate=0.001)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "dynamic_small_classical"

    def test_dynamic_medium_quantum(self):
        """Test categorizing medium dynamic environment with quantum brain."""
        env = EnvironmentMetadata(type="dynamic", grid_size=50, num_foods=20)
        brain = BrainMetadata(type="modular", qubits=4, learning_rate=0.01)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "dynamic_medium_quantum"

    def test_dynamic_medium_classical(self):
        """Test categorizing medium dynamic environment with classical brain."""
        env = EnvironmentMetadata(type="dynamic", grid_size=40, num_foods=18)
        brain = BrainMetadata(type="mlp", learning_rate=0.001)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "dynamic_medium_classical"

    def test_dynamic_large_quantum(self):
        """Test categorizing large dynamic environment with quantum brain."""
        env = EnvironmentMetadata(type="dynamic", grid_size=100, num_foods=50)
        brain = BrainMetadata(type="modular", qubits=8, learning_rate=0.005)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "dynamic_large_quantum"

    def test_dynamic_large_classical(self):
        """Test categorizing large dynamic environment with classical brain."""
        env = EnvironmentMetadata(type="dynamic", grid_size=75, num_foods=30)
        brain = BrainMetadata(type="qmlp", learning_rate=0.0005)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "dynamic_large_classical"

    def test_boundary_case_small_medium(self):
        """Test boundary between small and medium (grid_size=20)."""
        env = EnvironmentMetadata(type="dynamic", grid_size=20, num_foods=10)
        brain = BrainMetadata(type="modular", qubits=4, learning_rate=0.01)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "dynamic_small_quantum"

    def test_boundary_case_medium_large(self):
        """Test boundary between medium and large (grid_size=50)."""
        env = EnvironmentMetadata(type="dynamic", grid_size=50, num_foods=20)
        brain = BrainMetadata(type="mlp", learning_rate=0.001)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "dynamic_medium_classical"

    def test_quantum_brain_types(self):
        """Test all quantum brain types categorize correctly."""
        env = EnvironmentMetadata(type="dynamic", grid_size=30, num_foods=15)

        for brain_type in ["modular", "qmodular"]:
            brain = BrainMetadata(type=brain_type, qubits=4, learning_rate=0.01)
            experiment = create_test_experiment(env, brain)
            category = determine_benchmark_category(experiment)
            assert category.endswith("_quantum")

    def test_classical_brain_types(self):
        """Test all classical brain types categorize correctly."""
        env = EnvironmentMetadata(type="dynamic", grid_size=30, num_foods=15)

        for brain_type in ["mlp", "qmlp", "spiking"]:
            brain = BrainMetadata(type=brain_type, learning_rate=0.001)
            experiment = create_test_experiment(env, brain)
            category = determine_benchmark_category(experiment)
            assert category.endswith("_classical")

    def test_dynamic_predator_small_quantum(self):
        """Test categorizing small dynamic environment with predators and quantum brain."""
        env = EnvironmentMetadata(
            type="dynamic",
            grid_size=20,
            num_foods=10,
            predators_enabled=True,
            num_predators=2,
        )
        brain = BrainMetadata(type="modular", qubits=4, learning_rate=0.01)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "dynamic_predator_small_quantum"

    def test_dynamic_predator_small_classical(self):
        """Test categorizing small dynamic environment with predators and classical brain."""
        env = EnvironmentMetadata(
            type="dynamic",
            grid_size=15,
            num_foods=10,
            predators_enabled=True,
            num_predators=2,
        )
        brain = BrainMetadata(type="mlp", learning_rate=0.001)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "dynamic_predator_small_classical"

    def test_dynamic_predator_medium_quantum(self):
        """Test categorizing medium dynamic environment with predators and quantum brain."""
        env = EnvironmentMetadata(
            type="dynamic",
            grid_size=50,
            num_foods=20,
            predators_enabled=True,
            num_predators=3,
        )
        brain = BrainMetadata(type="qmodular", qubits=6, learning_rate=0.01)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "dynamic_predator_medium_quantum"

    def test_dynamic_predator_medium_classical(self):
        """Test categorizing medium dynamic environment with predators and classical brain."""
        env = EnvironmentMetadata(
            type="dynamic",
            grid_size=40,
            num_foods=18,
            predators_enabled=True,
            num_predators=3,
        )
        brain = BrainMetadata(type="qmlp", learning_rate=0.001)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "dynamic_predator_medium_classical"

    def test_dynamic_predator_large_quantum(self):
        """Test categorizing large dynamic environment with predators and quantum brain."""
        env = EnvironmentMetadata(
            type="dynamic",
            grid_size=100,
            num_foods=50,
            predators_enabled=True,
            num_predators=5,
        )
        brain = BrainMetadata(type="modular", qubits=8, learning_rate=0.005)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "dynamic_predator_large_quantum"

    def test_dynamic_predator_large_classical(self):
        """Test categorizing large dynamic environment with predators and classical brain."""
        env = EnvironmentMetadata(
            type="dynamic",
            grid_size=75,
            num_foods=30,
            predators_enabled=True,
            num_predators=5,
        )
        brain = BrainMetadata(type="mlp", learning_rate=0.0005)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "dynamic_predator_large_classical"

    def test_predators_disabled_uses_regular_category(self):
        """Test that predators_enabled=False uses regular (non-predator) categories."""
        env = EnvironmentMetadata(
            type="dynamic",
            grid_size=30,
            num_foods=15,
            predators_enabled=False,
        )
        brain = BrainMetadata(type="modular", qubits=4, learning_rate=0.01)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "dynamic_medium_quantum"
        assert "predator" not in category


class TestGetCategoryDirectory:
    """Test category directory path generation."""

    def test_static_maze_quantum_dir(self):
        """Test static maze quantum directory path."""
        path = get_category_directory("static_maze_quantum")
        assert path == "static_maze/quantum"

    def test_static_maze_classical_dir(self):
        """Test static maze classical directory path."""
        path = get_category_directory("static_maze_classical")
        assert path == "static_maze/classical"

    def test_dynamic_small_quantum_dir(self):
        """Test dynamic small quantum directory path."""
        path = get_category_directory("dynamic_small_quantum")
        assert path == "dynamic_small/quantum"

    def test_dynamic_medium_classical_dir(self):
        """Test dynamic medium classical directory path."""
        path = get_category_directory("dynamic_medium_classical")
        assert path == "dynamic_medium/classical"

    def test_dynamic_large_quantum_dir(self):
        """Test dynamic large quantum directory path."""
        path = get_category_directory("dynamic_large_quantum")
        assert path == "dynamic_large/quantum"

    def test_invalid_category_format(self):
        """Test handling invalid category format."""
        with pytest.raises(ValueError, match="Invalid category format"):
            get_category_directory("invalid")

    def test_all_valid_categories(self):
        """Test all 8 valid category combinations."""
        valid_categories = [
            "static_maze_quantum",
            "static_maze_classical",
            "dynamic_small_quantum",
            "dynamic_small_classical",
            "dynamic_medium_quantum",
            "dynamic_medium_classical",
            "dynamic_large_quantum",
            "dynamic_large_classical",
        ]

        for category in valid_categories:
            path = get_category_directory(category)
            assert "/" in path
            assert path.count("/") == 1
            parts = path.split("/")
            assert len(parts) == 2
            assert parts[1] in ["quantum", "classical"]

    def test_predator_category_directories(self):
        """Test directory paths for all 6 predator category combinations."""
        predator_categories = [
            ("dynamic_predator_small_quantum", "dynamic_predator_small/quantum"),
            ("dynamic_predator_small_classical", "dynamic_predator_small/classical"),
            ("dynamic_predator_medium_quantum", "dynamic_predator_medium/quantum"),
            ("dynamic_predator_medium_classical", "dynamic_predator_medium/classical"),
            ("dynamic_predator_large_quantum", "dynamic_predator_large/quantum"),
            ("dynamic_predator_large_classical", "dynamic_predator_large/classical"),
        ]

        for category, expected_path in predator_categories:
            path = get_category_directory(category)
            assert path == expected_path
