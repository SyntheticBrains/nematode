"""Evolutionary optimization for quantum brain parameters.

This module provides population-based optimization algorithms that sidestep
gradient-based learning entirely. Instead of computing noisy parameter-shift
gradients, these optimizers use fitness evaluation over multiple episodes.

Supported algorithms:
- CMA-ES (Covariance Matrix Adaptation Evolution Strategy) - recommended
- Genetic Algorithm (GA) - simpler, more interpretable

Key advantages over gradient-based learning:
- No gradient noise from parameter-shift rule
- Works with sparse rewards (aggregates over episodes)
- Maintains population diversity (avoids local minima)
- Fewer hyperparameters to tune
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import TYPE_CHECKING

import cma
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

# Crossover probability for uniform crossover
CROSSOVER_PROBABILITY = 0.5


@dataclass
class EvolutionResult:
    """Results from evolutionary optimization."""

    best_params: list[float]
    best_fitness: float
    generations: int
    history: list[dict] = field(default_factory=list)


class EvolutionaryOptimizer(ABC):
    """Base class for evolutionary optimization algorithms.

    Subclasses must implement ask/tell interface for population-based search.
    """

    def __init__(
        self,
        num_params: int,
        population_size: int = 20,
        sigma0: float = 0.5,
    ) -> None:
        """Initialize the evolutionary optimizer.

        Args:
            num_params: Number of parameters to optimize.
            population_size: Number of candidate solutions per generation.
            sigma0: Initial step size / search radius.
        """
        self.num_params = num_params
        self.population_size = population_size
        self.sigma0 = sigma0
        self.generation = 0

    @abstractmethod
    def ask(self) -> list[list[float]]:
        """Request candidate solutions for evaluation.

        Returns
        -------
            List of parameter arrays (population_size x num_params).
        """

    @abstractmethod
    def tell(self, solutions: list[list[float]], fitnesses: list[float]) -> None:
        """Report fitness values for candidate solutions.

        Args:
            solutions: Parameter arrays that were evaluated.
            fitnesses: Corresponding fitness values (lower is better for CMA-ES).
        """

    @abstractmethod
    def stop(self) -> bool:
        """Check if optimization should stop.

        Returns
        -------
            True if convergence or stopping criteria met.
        """

    @property
    @abstractmethod
    def result(self) -> EvolutionResult:
        """Get the current best result.

        Returns
        -------
            EvolutionResult with best parameters and metadata.
        """


class CMAESOptimizer(EvolutionaryOptimizer):
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.

    CMA-ES is recommended for quantum circuit optimization because:
    - Works well in 10-50 parameter range (typical quantum circuits)
    - Self-adapts search distribution without manual tuning
    - Handles noisy fitness evaluations gracefully
    - Standard in variational quantum circuit literature
    """

    def __init__(
        self,
        num_params: int,
        x0: list[float] | None = None,
        population_size: int = 20,
        sigma0: float = 0.5,
        seed: int | None = None,
    ) -> None:
        """Initialize CMA-ES optimizer.

        Args:
            num_params: Number of parameters to optimize.
            x0: Initial parameter values. Defaults to zeros.
            population_size: Population size (lambda). Defaults to 20.
            sigma0: Initial step size. Defaults to 0.5.
            seed: Random seed for reproducibility.
        """
        super().__init__(num_params, population_size, sigma0)

        if x0 is None:
            x0 = [0.0] * num_params

        options: dict = {"popsize": population_size, "verbose": -9}
        if seed is not None:
            options["seed"] = seed

        self._es: cma.CMAEvolutionStrategy = cma.CMAEvolutionStrategy(
            x0,
            sigma0,
            options,
        )
        self._history: list[dict] = []
        self._last_solutions: list[list[float]] = []

    def ask(self) -> list[list[float]]:
        """Request candidate solutions from CMA-ES.

        Returns
        -------
            List of parameter arrays to evaluate.
        """
        self._last_solutions = self._es.ask()
        return [list(sol) for sol in self._last_solutions]

    def tell(self, solutions: list[list[float]], fitnesses: list[float]) -> None:
        """Report fitness values to CMA-ES.

        Args:
            solutions: Parameter arrays that were evaluated.
            fitnesses: Corresponding fitness values (lower is better).
        """
        self._es.tell(solutions, fitnesses)
        self.generation += 1

        # Record history
        self._history.append(
            {
                "generation": self.generation,
                "best_fitness": min(fitnesses),
                "mean_fitness": np.mean(fitnesses),
                "std_fitness": np.std(fitnesses),
            },
        )

    def stop(self) -> bool:
        """Check if CMA-ES has converged.

        Returns
        -------
            True if CMA-ES internal stopping criteria met.
        """
        return bool(self._es.stop())

    @property
    def result(self) -> EvolutionResult:
        """Get the best result from CMA-ES.

        Returns
        -------
            EvolutionResult with best parameters found.
        """
        cma_result = self._es.result
        xbest = cma_result.xbest if cma_result.xbest is not None else [0.0] * self.num_params
        return EvolutionResult(
            best_params=list(xbest),
            best_fitness=float(cma_result.fbest) if cma_result.fbest is not None else float("inf"),
            generations=self.generation,
            history=self._history,
        )


class GeneticAlgorithmOptimizer(EvolutionaryOptimizer):
    """Simple genetic algorithm optimizer.

    Provides a more interpretable alternative to CMA-ES with explicit
    selection, crossover, and mutation operators.
    """

    def __init__(  # noqa: PLR0913
        self,
        num_params: int,
        population_size: int = 50,
        sigma0: float = 0.5,
        elite_fraction: float = 0.2,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        seed: int | None = None,
    ) -> None:
        """Initialize genetic algorithm optimizer.

        Args:
            num_params: Number of parameters to optimize.
            population_size: Number of individuals in population.
            sigma0: Initial parameter range and mutation std.
            elite_fraction: Fraction of population to keep as elite.
            mutation_rate: Probability of mutating each parameter.
            crossover_rate: Probability of crossover vs direct copy.
            seed: Random seed for reproducibility.

        Raises
        ------
            ValueError: If population_size < 1 or hyperparameters outside [0.0, 1.0].
        """
        # Validate population_size
        if population_size < 1:
            msg = f"population_size must be >= 1, got {population_size}"
            raise ValueError(msg)

        super().__init__(num_params, population_size, sigma0)

        # Validate hyperparameters are in [0.0, 1.0]
        if not 0.0 <= elite_fraction <= 1.0:
            msg = f"elite_fraction must be in [0.0, 1.0], got {elite_fraction}"
            raise ValueError(msg)
        if not 0.0 <= mutation_rate <= 1.0:
            msg = f"mutation_rate must be in [0.0, 1.0], got {mutation_rate}"
            raise ValueError(msg)
        if not 0.0 <= crossover_rate <= 1.0:
            msg = f"crossover_rate must be in [0.0, 1.0], got {crossover_rate}"
            raise ValueError(msg)

        self.elite_fraction = elite_fraction
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self._rng = np.random.default_rng(seed)

        # Initialize population uniformly in [-pi, pi]
        self._population = [
            list(self._rng.uniform(-np.pi, np.pi, num_params)) for _ in range(population_size)
        ]
        self._fitnesses: list[float] = [float("inf")] * population_size
        self._best_individual: list[float] = self._population[0].copy()
        self._best_fitness: float = float("inf")
        self._history: list[dict] = []

    def ask(self) -> list[list[float]]:
        """Return current population for evaluation.

        Returns
        -------
            List of parameter arrays (current population).
        """
        return [ind.copy() for ind in self._population]

    def tell(self, solutions: list[list[float]], fitnesses: list[float]) -> None:
        """Update population based on fitness values.

        Performs selection, crossover, and mutation to create next generation.

        Args:
            solutions: Parameter arrays that were evaluated.
            fitnesses: Corresponding fitness values (lower is better).
        """
        self._fitnesses = list(fitnesses)

        # Update best
        min_idx = int(np.argmin(fitnesses))
        if fitnesses[min_idx] < self._best_fitness:
            self._best_fitness = fitnesses[min_idx]
            self._best_individual = solutions[min_idx].copy()

        # Sort by fitness (ascending = best first)
        sorted_indices = np.argsort(fitnesses)
        sorted_population = [solutions[i] for i in sorted_indices]

        # Elite selection (ensure n_elite never exceeds population)
        n_elite = min(self.population_size, max(1, int(self.population_size * self.elite_fraction)))
        elite = [ind.copy() for ind in sorted_population[:n_elite]]

        # Generate children
        children: list[list[float]] = []
        while len(children) < self.population_size - n_elite:
            # Tournament selection for parents
            parent1 = self._tournament_select(solutions, fitnesses)
            parent2 = self._tournament_select(solutions, fitnesses)

            # Crossover
            if self._rng.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()

            # Mutation
            child = self._mutate(child)
            children.append(child)

        # New population = elite + children
        self._population = elite + children
        self.generation += 1

        # Record history
        self._history.append(
            {
                "generation": self.generation,
                "best_fitness": self._best_fitness,
                "mean_fitness": np.mean(fitnesses),
                "std_fitness": np.std(fitnesses),
            },
        )

    def _tournament_select(
        self,
        population: list[list[float]],
        fitnesses: list[float],
        tournament_size: int = 3,
    ) -> list[float]:
        """Select individual via tournament selection.

        Handles edge cases where tournament_size > population size by using
        sampling with replacement when necessary.
        """
        pop_size = len(population)
        effective_size = min(tournament_size, pop_size)
        # Use replace=True when tournament_size exceeds population to avoid errors
        replace = tournament_size > pop_size
        indices = self._rng.choice(pop_size, size=effective_size, replace=replace)
        best_idx = indices[np.argmin([fitnesses[i] for i in indices])]
        return population[best_idx].copy()

    def _crossover(self, parent1: list[float], parent2: list[float]) -> list[float]:
        """Uniform crossover between two parents."""
        mask = self._rng.random(self.num_params) < CROSSOVER_PROBABILITY
        return [p1 if m else p2 for p1, p2, m in zip(parent1, parent2, mask, strict=False)]

    def _mutate(self, individual: list[float]) -> list[float]:
        """Gaussian mutation with probability mutation_rate per parameter."""
        mutated = individual.copy()
        for i in range(self.num_params):
            if self._rng.random() < self.mutation_rate:
                mutated[i] += self._rng.normal(0, self.sigma0)
                # Wrap to [-pi, pi]
                mutated[i] = ((mutated[i] + np.pi) % (2 * np.pi)) - np.pi
        return mutated

    def stop(self) -> bool:
        """GA doesn't have automatic stopping - always returns False.

        Use generation count or fitness plateau detection externally.
        """
        return False

    @property
    def result(self) -> EvolutionResult:
        """Get the best result from GA.

        Returns
        -------
            EvolutionResult with best parameters found.
        """
        return EvolutionResult(
            best_params=self._best_individual.copy(),
            best_fitness=self._best_fitness,
            generations=self.generation,
            history=self._history,
        )


@dataclass
class FitnessConfig:
    """Configuration for fitness evaluation."""

    episodes_per_evaluation: int = 15
    parallel_workers: int = 1
    episode_max_steps: int = 200


class FitnessFunction:
    """Evaluates fitness of parameter sets by running episodes.

    Fitness is computed as negative success rate (for minimization):
    fitness = -successes / episodes_per_evaluation

    Lower fitness is better (CMA-ES minimizes).
    """

    def __init__(
        self,
        brain_factory: "Callable[[list[float]], object]",
        env_factory: "Callable[[], object]",
        config: FitnessConfig | None = None,
    ) -> None:
        """Initialize fitness function.

        Args:
            brain_factory: Callable that takes param_array and returns a brain instance.
            env_factory: Callable that returns a fresh environment instance.
            config: Fitness evaluation configuration.
        """
        self.brain_factory = brain_factory
        self.env_factory = env_factory
        self.config = config or FitnessConfig()

    def __call__(self, param_array: list[float]) -> float:
        """Evaluate fitness of a parameter set.

        Runs episodes_per_evaluation episodes and returns negative success rate.

        Args:
            param_array: Flat array of parameter values.

        Returns
        -------
            Negative success rate (lower is better).
        """
        brain = self.brain_factory(param_array)
        successes = 0

        for _ in range(self.config.episodes_per_evaluation):
            env = self.env_factory()
            success = self._run_episode(brain, env)
            if success:
                successes += 1

        success_rate = successes / self.config.episodes_per_evaluation
        return -success_rate  # Negate for minimization

    def _run_episode(self, brain: object, env: object) -> bool:
        """Run a single episode and return whether it was successful.

        Args:
            brain: Brain instance with run_brain method.
            env: Environment instance with step/reset methods.

        Returns
        -------
            True if episode was successful (collected target foods).
        """
        # Import here to avoid circular imports
        from quantumnematode.brain.arch import BrainParams

        state = env.reset()  # type: ignore[attr-defined]

        for _ in range(self.config.episode_max_steps):
            # Convert state to BrainParams
            params = BrainParams(
                gradient_strength=state.get("gradient_strength"),
                gradient_direction=state.get("gradient_direction"),
                agent_direction=state.get("agent_direction"),
                agent_position=state.get("agent_position"),
                food_gradient_strength=state.get("food_gradient_strength"),
                food_gradient_direction=state.get("food_gradient_direction"),
                predator_gradient_strength=state.get("predator_gradient_strength"),
                predator_gradient_direction=state.get("predator_gradient_direction"),
                satiety=state.get("satiety"),
            )

            # Get action from brain
            actions = brain.run_brain(params)  # type: ignore[attr-defined]
            if not actions:
                break

            action = actions[0].action

            # Step environment
            state, _reward, done, info = env.step(action)  # type: ignore[attr-defined]

            if done:
                # Check termination reason
                reason = info.get("termination_reason", "")
                return reason == "target_reached" or info.get("success", False)

        return False

    def evaluate_population(
        self,
        population: list[list[float]],
    ) -> list[float]:
        """Evaluate fitness for a population of parameter sets.

        Args:
            population: List of parameter arrays.

        Returns
        -------
            List of fitness values (same order as population).
        """
        if self.config.parallel_workers > 1:
            return self._evaluate_parallel(population)
        return [self(params) for params in population]

    def _evaluate_parallel(self, population: list[list[float]]) -> list[float]:
        """Evaluate population in parallel using multiprocessing.

        Args:
            population: List of parameter arrays.

        Returns
        -------
            List of fitness values.
        """
        with Pool(processes=self.config.parallel_workers) as pool:
            fitnesses = pool.map(self, population)
        return list(fitnesses)
