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
    GradientMetadata,
    LearningRateMetadata,
    ResultsMetadata,
    RewardMetadata,
    SystemMetadata,
)

# Centralized list of all valid benchmark categories
# Used by multiple tests to avoid duplication and drift
VALID_CATEGORIES = [
    # Foraging categories (without predators)
    "foraging_small_quantum",
    "foraging_small_classical",
    "foraging_medium_quantum",
    "foraging_medium_classical",
    "foraging_large_quantum",
    "foraging_large_classical",
    # Predator evasion categories (with predators)
    "predator_small_quantum",
    "predator_small_classical",
    "predator_medium_quantum",
    "predator_medium_classical",
    "predator_large_quantum",
    "predator_large_classical",
]

# Expected directory mappings for predator categories
PREDATOR_CATEGORY_DIRECTORIES = {
    "predator_small_quantum": "predator_small/quantum",
    "predator_small_classical": "predator_small/classical",
    "predator_medium_quantum": "predator_medium/quantum",
    "predator_medium_classical": "predator_medium/classical",
    "predator_large_quantum": "predator_large/quantum",
    "predator_large_classical": "predator_large/classical",
}


def create_test_experiment(env: EnvironmentMetadata, brain: BrainMetadata) -> ExperimentMetadata:
    """Create test experiment metadata."""
    return ExperimentMetadata(
        experiment_id="test_id",
        timestamp=datetime.now(UTC),
        config_file="test.yml",
        config_hash="abc123",
        environment=env,
        brain=brain,
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
            method="dynamic",
            initial_learning_rate=0.01,
        ),
        gradient=GradientMetadata(method="raw"),
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


class TestDetermineBenchmarkCategory:
    """Test benchmark category determination."""

    def test_foraging_small_quantum(self):
        """Test categorizing small foraging environment with quantum brain."""
        env = EnvironmentMetadata(type="dynamic", grid_size=15, num_foods=10)
        brain = BrainMetadata(type="qmodular", qubits=6, learning_rate=0.02)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "foraging_small_quantum"

    def test_foraging_small_classical(self):
        """Test categorizing small foraging environment with classical brain."""
        env = EnvironmentMetadata(type="dynamic", grid_size=20, num_foods=15)
        brain = BrainMetadata(type="qmlp", learning_rate=0.001)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "foraging_small_classical"

    def test_foraging_medium_quantum(self):
        """Test categorizing medium foraging environment with quantum brain."""
        env = EnvironmentMetadata(type="dynamic", grid_size=50, num_foods=20)
        brain = BrainMetadata(type="modular", qubits=4, learning_rate=0.01)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "foraging_medium_quantum"

    def test_foraging_medium_classical(self):
        """Test categorizing medium foraging environment with classical brain."""
        env = EnvironmentMetadata(type="dynamic", grid_size=40, num_foods=18)
        brain = BrainMetadata(type="mlp", learning_rate=0.001)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "foraging_medium_classical"

    def test_foraging_large_quantum(self):
        """Test categorizing large foraging environment with quantum brain."""
        env = EnvironmentMetadata(type="dynamic", grid_size=100, num_foods=50)
        brain = BrainMetadata(type="modular", qubits=8, learning_rate=0.005)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "foraging_large_quantum"

    def test_foraging_large_classical(self):
        """Test categorizing large foraging environment with classical brain."""
        env = EnvironmentMetadata(type="dynamic", grid_size=75, num_foods=30)
        brain = BrainMetadata(type="qmlp", learning_rate=0.0005)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "foraging_large_classical"

    def test_boundary_case_small_medium(self):
        """Test boundary between small and medium (grid_size=20)."""
        env = EnvironmentMetadata(type="dynamic", grid_size=20, num_foods=10)
        brain = BrainMetadata(type="modular", qubits=4, learning_rate=0.01)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "foraging_small_quantum"

    def test_boundary_case_medium_large(self):
        """Test boundary between medium and large (grid_size=50)."""
        env = EnvironmentMetadata(type="dynamic", grid_size=50, num_foods=20)
        brain = BrainMetadata(type="mlp", learning_rate=0.001)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "foraging_medium_classical"

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

    def test_predator_small_quantum(self):
        """Test categorizing small predator environment with quantum brain."""
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
        assert category == "predator_small_quantum"

    def test_predator_small_classical(self):
        """Test categorizing small predator environment with classical brain."""
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
        assert category == "predator_small_classical"

    def test_predator_medium_quantum(self):
        """Test categorizing medium predator environment with quantum brain."""
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
        assert category == "predator_medium_quantum"

    def test_predator_medium_classical(self):
        """Test categorizing medium predator environment with classical brain."""
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
        assert category == "predator_medium_classical"

    def test_predator_large_quantum(self):
        """Test categorizing large predator environment with quantum brain."""
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
        assert category == "predator_large_quantum"

    def test_predator_large_classical(self):
        """Test categorizing large predator environment with classical brain."""
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
        assert category == "predator_large_classical"

    def test_predators_disabled_uses_foraging_category(self):
        """Test that predators_enabled=False uses foraging (non-predator) categories."""
        env = EnvironmentMetadata(
            type="dynamic",
            grid_size=30,
            num_foods=15,
            predators_enabled=False,
        )
        brain = BrainMetadata(type="modular", qubits=4, learning_rate=0.01)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "foraging_medium_quantum"
        assert "predator" not in category


class TestGetCategoryDirectory:
    """Test category directory path generation."""

    def test_foraging_small_quantum_dir(self):
        """Test foraging small quantum directory path."""
        path = get_category_directory("foraging_small_quantum")
        assert path == "foraging_small/quantum"

    def test_foraging_medium_classical_dir(self):
        """Test foraging medium classical directory path."""
        path = get_category_directory("foraging_medium_classical")
        assert path == "foraging_medium/classical"

    def test_foraging_large_quantum_dir(self):
        """Test foraging large quantum directory path."""
        path = get_category_directory("foraging_large_quantum")
        assert path == "foraging_large/quantum"

    def test_invalid_category_format(self):
        """Test handling invalid category format."""
        with pytest.raises(ValueError, match="Invalid category format"):
            get_category_directory("invalid")

    def test_all_valid_categories(self):
        """Test all valid category combinations (12 total: 6 foraging + 6 predator)."""
        assert len(VALID_CATEGORIES) == 12
        for category in VALID_CATEGORIES:
            path = get_category_directory(category)
            env_category, brain_class = category.rsplit("_", 1)
            assert path == f"{env_category}/{brain_class}"
            assert brain_class in ["quantum", "classical"]

    def test_predator_category_directories(self):
        """Test directory paths for all 6 predator category combinations."""
        for category, expected_path in PREDATOR_CATEGORY_DIRECTORIES.items():
            path = get_category_directory(category)
            assert path == expected_path
