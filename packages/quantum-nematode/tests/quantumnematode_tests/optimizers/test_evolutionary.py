"""Tests for evolutionary optimization algorithms."""

import numpy as np
from quantumnematode.optimizers.evolutionary import (
    CMAESOptimizer,
    EvolutionResult,
    FitnessConfig,
    GeneticAlgorithmOptimizer,
)


class TestEvolutionResult:
    """Tests for EvolutionResult dataclass."""

    def test_evolution_result_creation(self):
        """Test creating an EvolutionResult."""
        result = EvolutionResult(
            best_params=[0.1, 0.2, 0.3],
            best_fitness=-0.5,
            generations=10,
        )
        assert result.best_params == [0.1, 0.2, 0.3]
        assert result.best_fitness == -0.5
        assert result.generations == 10
        assert result.history == []

    def test_evolution_result_with_history(self):
        """Test creating an EvolutionResult with history."""
        history = [{"generation": 1, "best_fitness": -0.3}]
        result = EvolutionResult(
            best_params=[0.1],
            best_fitness=-0.5,
            generations=1,
            history=history,
        )
        assert len(result.history) == 1


class TestCMAESOptimizer:
    """Tests for CMA-ES optimizer."""

    def test_cmaes_initialization(self):
        """Test CMA-ES initialization with defaults."""
        optimizer = CMAESOptimizer(num_params=10)
        assert optimizer.num_params == 10
        assert optimizer.population_size == 20
        assert optimizer.sigma0 == 0.5
        assert optimizer.generation == 0

    def test_cmaes_initialization_with_custom_params(self):
        """Test CMA-ES initialization with custom parameters."""
        x0 = [0.1, 0.2, 0.3]
        optimizer = CMAESOptimizer(
            num_params=3,
            x0=x0,
            population_size=10,
            sigma0=0.3,
        )
        assert optimizer.num_params == 3
        assert optimizer.population_size == 10
        assert optimizer.sigma0 == 0.3

    def test_cmaes_ask(self):
        """Test asking for candidate solutions."""
        optimizer = CMAESOptimizer(num_params=5, population_size=10)
        solutions = optimizer.ask()

        assert len(solutions) == 10
        assert all(len(sol) == 5 for sol in solutions)
        assert all(isinstance(sol, list) for sol in solutions)

    def test_cmaes_tell(self):
        """Test reporting fitness values."""
        optimizer = CMAESOptimizer(num_params=5, population_size=10)
        solutions = optimizer.ask()
        fitnesses = [np.sum(np.array(sol) ** 2) for sol in solutions]  # Sphere function

        optimizer.tell(solutions, fitnesses)

        assert optimizer.generation == 1
        assert len(optimizer._history) == 1
        assert "best_fitness" in optimizer._history[0]

    def test_cmaes_result(self):
        """Test getting result after evolution."""
        optimizer = CMAESOptimizer(num_params=3, population_size=10)

        # Run a few generations
        for _ in range(5):
            solutions = optimizer.ask()
            fitnesses = [np.sum(np.array(sol) ** 2) for sol in solutions]
            optimizer.tell(solutions, fitnesses)

        result = optimizer.result
        assert isinstance(result, EvolutionResult)
        assert len(result.best_params) == 3
        assert result.generations == 5
        assert len(result.history) == 5

    def test_cmaes_convergence_on_sphere(self):
        """Test CMA-ES converges on simple sphere function."""
        optimizer = CMAESOptimizer(
            num_params=2,
            x0=[1.0, 1.0],
            population_size=10,
            sigma0=0.5,
        )

        # Run until convergence or max generations
        for _ in range(50):
            if optimizer.stop():
                break
            solutions = optimizer.ask()
            fitnesses = [np.sum(np.array(sol) ** 2) for sol in solutions]
            optimizer.tell(solutions, fitnesses)

        result = optimizer.result
        # Should find near-zero solution
        assert result.best_fitness < 0.1


class TestGeneticAlgorithmOptimizer:
    """Tests for Genetic Algorithm optimizer."""

    def test_ga_initialization(self):
        """Test GA initialization with defaults."""
        optimizer = GeneticAlgorithmOptimizer(num_params=10, seed=42)
        assert optimizer.num_params == 10
        assert optimizer.population_size == 50
        assert optimizer.sigma0 == 0.5
        assert optimizer.generation == 0

    def test_ga_initialization_with_custom_params(self):
        """Test GA initialization with custom parameters."""
        optimizer = GeneticAlgorithmOptimizer(
            num_params=5,
            population_size=20,
            sigma0=0.3,
            elite_fraction=0.1,
            mutation_rate=0.2,
            crossover_rate=0.9,
            seed=42,
        )
        assert optimizer.num_params == 5
        assert optimizer.population_size == 20
        assert optimizer.elite_fraction == 0.1
        assert optimizer.mutation_rate == 0.2
        assert optimizer.crossover_rate == 0.9

    def test_ga_ask(self):
        """Test asking for candidate solutions."""
        optimizer = GeneticAlgorithmOptimizer(num_params=5, population_size=20, seed=42)
        solutions = optimizer.ask()

        assert len(solutions) == 20
        assert all(len(sol) == 5 for sol in solutions)
        # Parameters should be in [-pi, pi]
        for sol in solutions:
            for val in sol:
                assert -np.pi <= val <= np.pi

    def test_ga_tell(self):
        """Test reporting fitness values."""
        optimizer = GeneticAlgorithmOptimizer(num_params=5, population_size=20, seed=42)
        solutions = optimizer.ask()
        fitnesses = [np.sum(np.array(sol) ** 2) for sol in solutions]

        optimizer.tell(solutions, fitnesses)

        assert optimizer.generation == 1
        assert len(optimizer._history) == 1

    def test_ga_elite_preservation(self):
        """Test that elites are preserved across generations."""
        optimizer = GeneticAlgorithmOptimizer(
            num_params=3,
            population_size=10,
            elite_fraction=0.2,
            seed=42,
        )

        # First generation
        solutions1 = optimizer.ask()
        fitnesses1 = [float(i) for i in range(10)]  # Best fitness is 0 for first individual
        optimizer.tell(solutions1, fitnesses1)

        # Best individual should be preserved
        best_before = optimizer._best_individual.copy()

        # Second generation
        solutions2 = optimizer.ask()
        fitnesses2 = [f + 1.0 for f in fitnesses1]  # All worse
        optimizer.tell(solutions2, fitnesses2)

        # Best should still be the same
        assert optimizer._best_individual == best_before

    def test_ga_stop_always_false(self):
        """Test that GA stop() always returns False."""
        optimizer = GeneticAlgorithmOptimizer(num_params=5, seed=42)

        for _ in range(10):
            solutions = optimizer.ask()
            rng = np.random.default_rng(42)
            fitnesses = [float(rng.random()) for _ in solutions]
            optimizer.tell(solutions, fitnesses)
            assert optimizer.stop() is False

    def test_ga_result(self):
        """Test getting result after evolution."""
        optimizer = GeneticAlgorithmOptimizer(num_params=3, population_size=10, seed=42)

        for _ in range(5):
            solutions = optimizer.ask()
            fitnesses = [np.sum(np.array(sol) ** 2) for sol in solutions]
            optimizer.tell(solutions, fitnesses)

        result = optimizer.result
        assert isinstance(result, EvolutionResult)
        assert len(result.best_params) == 3
        assert result.generations == 5
        assert len(result.history) == 5

    def test_ga_deterministic_with_seed(self):
        """Test that GA is deterministic with same seed."""
        optimizer1 = GeneticAlgorithmOptimizer(num_params=3, population_size=10, seed=42)
        optimizer2 = GeneticAlgorithmOptimizer(num_params=3, population_size=10, seed=42)

        solutions1 = optimizer1.ask()
        solutions2 = optimizer2.ask()

        for sol1, sol2 in zip(solutions1, solutions2, strict=False):
            assert sol1 == sol2


class TestFitnessConfig:
    """Tests for FitnessConfig."""

    def test_fitness_config_defaults(self):
        """Test FitnessConfig default values."""
        config = FitnessConfig()
        assert config.episodes_per_evaluation == 15
        assert config.parallel_workers == 1
        assert config.episode_max_steps == 200

    def test_fitness_config_custom(self):
        """Test FitnessConfig with custom values."""
        config = FitnessConfig(
            episodes_per_evaluation=20,
            parallel_workers=4,
            episode_max_steps=100,
        )
        assert config.episodes_per_evaluation == 20
        assert config.parallel_workers == 4
        assert config.episode_max_steps == 100


class TestEvolutionaryOptimizerComparison:
    """Comparative tests between CMA-ES and GA."""

    def test_both_find_minimum_on_sphere(self):
        """Test that both algorithms can minimize sphere function."""
        optimizer_classes = [CMAESOptimizer, GeneticAlgorithmOptimizer]
        for optimizer_class in optimizer_classes:
            if optimizer_class == GeneticAlgorithmOptimizer:
                optimizer = optimizer_class(
                    num_params=2,
                    population_size=20,
                    seed=42,
                )
            else:
                optimizer = optimizer_class(
                    num_params=2,
                    x0=[2.0, 2.0],
                    population_size=20,
                )

            for _ in range(30):
                solutions = optimizer.ask()
                fitnesses = [np.sum(np.array(sol) ** 2) for sol in solutions]
                optimizer.tell(solutions, fitnesses)

            result = optimizer.result
            # Should find reasonably good solution
            assert result.best_fitness < 1.0, f"{optimizer_class.__name__} failed"

    def test_both_improve_over_generations(self):
        """Test that both algorithms improve fitness over generations."""
        optimizer_classes = [CMAESOptimizer, GeneticAlgorithmOptimizer]
        for optimizer_class in optimizer_classes:
            if optimizer_class == GeneticAlgorithmOptimizer:
                optimizer = optimizer_class(num_params=3, population_size=20, seed=42)
            else:
                optimizer = optimizer_class(num_params=3, population_size=20)

            first_best: float = float("inf")
            last_best: float = float("inf")

            for gen in range(20):
                solutions = optimizer.ask()
                fitnesses = [np.sum(np.array(sol) ** 2) for sol in solutions]
                optimizer.tell(solutions, fitnesses)

                current_best = min(fitnesses)
                if gen == 0:
                    first_best = current_best
                last_best = current_best

            # Should improve (lower fitness is better)
            assert last_best <= first_best, f"{optimizer_class.__name__} didn't improve"
