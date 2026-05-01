"""Evolutionary optimization for quantum brain parameters.

This module provides population-based optimization algorithms that sidestep
gradient-based learning entirely. Instead of computing noisy parameter-shift
gradients, these optimizers use fitness evaluation over multiple episodes.

Supported algorithms:
- CMA-ES (Covariance Matrix Adaptation Evolution Strategy) - recommended for
  unbounded continuous search at moderate-to-large genome dim
- Genetic Algorithm (GA) - simpler, more interpretable
- Optuna TPE (Tree-structured Parzen Estimator) - Bayesian-style sampler
  for small-genome bounded hyperparameter search; especially relevant for
  schemas with mixed-scale dimensions and narrow viable regions

Key advantages over gradient-based learning:
- No gradient noise from parameter-shift rule
- Works with sparse rewards (aggregates over episodes)
- Maintains population diversity (avoids local minima)
- Fewer hyperparameters to tune
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cma
import numpy as np

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

    def __init__(  # noqa: PLR0913, C901 — input-validation branches push complexity past the threshold
        self,
        num_params: int,
        x0: list[float] | None = None,
        population_size: int = 20,
        sigma0: float = 0.5,
        seed: int | None = None,
        *,
        diagonal: bool = False,
        stds: list[float] | None = None,
    ) -> None:
        """Initialize CMA-ES optimizer.

        Args:
            num_params: Number of parameters to optimize.
            x0: Initial parameter values. Defaults to zeros.
            population_size: Population size (lambda). Defaults to 20.
            sigma0: Initial step size. Defaults to 0.5.
            seed: Random seed for reproducibility.
            diagonal: If True, restrict the covariance matrix to its
                diagonal (sep-CMA-ES; sets ``CMA_diagonal=True`` in the
                underlying ``cma`` library).  Drops ``tell()`` cost from
                O(n²) to O(n) — a tractability requirement for large
                genomes (e.g. neuroevolution at n>~1000), where
                full-covariance ``tell()`` becomes minutes per generation.

                Trade-off: gives up off-diagonal covariance adaptation,
                so per-generation convergence is slower — typically 2-10x
                more generations are needed to reach the same fitness
                target on non-separable problems (Ros & Hansen 2008).  At
                large n, full-cov is NOT a competing option (it doesn't
                fit in memory or finish a generation in finite time), so
                net wall-clock to convergence is dramatically faster
                despite the slower per-generation convergence.

                Defaults to False (full covariance).  Use False for small
                genomes (n<~100); True for neural-network weight evolution.
            stds: Per-parameter standard deviations.  When provided,
                CMA-ES uses ``stds[i] * sigma0`` as the initial step
                size for parameter i.  Necessary when parameters live
                on different scales — e.g. mixed hyperparameter schemas
                with log-scale floats (range ~6 in log-units), tight
                linear floats like gamma (range ~0.1), and ints
                (range ~200).  Without per-parameter scaling, a single
                uniform sigma cannot be appropriate for all dimensions:
                too large for tight ranges (samples saturate at bounds)
                or too small for wide ranges (no exploration).  Length
                must equal num_params.  Defaults to None (uniform
                sigma across all dimensions).
        """
        super().__init__(num_params, population_size, sigma0)

        if x0 is None:
            x0 = [0.0] * num_params
        else:
            # Defensive: cma library would fail downstream with cryptic
            # errors if x0 is wrong-length or contains NaN/inf.  Catch
            # at construction time with a clear message.
            if len(x0) != num_params:
                msg = f"CMAESOptimizer: x0 length {len(x0)} does not match num_params {num_params}."
                raise ValueError(msg)
            for i, value in enumerate(x0):
                if not math.isfinite(value):
                    msg = (
                        f"CMAESOptimizer: x0[{i}] is not finite ({value!r}); "
                        "all entries must be finite real numbers."
                    )
                    raise ValueError(msg)

        options: dict = {"popsize": population_size, "verbose": -9}
        if seed is not None:
            options["seed"] = seed
        if diagonal:
            options["CMA_diagonal"] = True
        if stds is not None:
            if len(stds) != num_params:
                msg = (
                    f"CMAESOptimizer: stds length {len(stds)} does not match "
                    f"num_params {num_params}."
                )
                raise ValueError(msg)
            for i, value in enumerate(stds):
                if not math.isfinite(value) or value <= 0:
                    msg = (
                        f"CMAESOptimizer: stds[{i}] = {value!r}; per-parameter "
                        "stds must be finite and strictly positive."
                    )
                    raise ValueError(msg)
            options["CMA_stds"] = list(stds)

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
        x0: list[float] | None = None,
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
            x0: Initial parameter values. If provided, first individual uses these
                values and rest are initialized around them with noise.
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

        # Initialize population
        if x0 is not None:
            # First individual is x0 exactly, rest are x0 + noise
            self._population = [list(x0)]
            for _ in range(population_size - 1):
                noisy = [v + self._rng.normal(0, sigma0) for v in x0]
                # Wrap to [-pi, pi]
                noisy = [((v + np.pi) % (2 * np.pi)) - np.pi for v in noisy]
                self._population.append(noisy)
        else:
            # No x0: initialize uniformly in [-pi, pi]
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


class OptunaTPEOptimizer(EvolutionaryOptimizer):
    """Optuna TPE sampler wrapped in the population-based ask/tell interface.

    TPE = Tree-structured Parzen Estimator.

    Why TPE rather than CMA-ES for some hyperparameter-shaped problems:

    - **Bounded sampling**.  CMA-ES samples from an unbounded Gaussian and
      relies on the encoder to clip out-of-bounds values at decode.  When
      the schema has narrow viable regions (e.g. a learning rate with a
      lower bound right at a dead-zone boundary), CMA-ES's covariance
      adaptation can collapse the search distribution onto that boundary
      and clip every sample there.  TPE samples from a uniform prior over
      the configured bounds and never clips — a genome at the bound is
      just one point of many, not the destination the search converges
      toward.
    - **Per-parameter mixed scales handled natively**.  CMA-ES needs
      explicit ``CMA_stds`` to handle log-scale lr (range ~7) alongside
      tight-range gamma (range ~0.1).  TPE's KDE is fit independently
      per parameter, so no per-parameter scaling tuning is required.
    - **Sample-efficient at small genome dim**.  TPE's kernel density
      estimate over good vs bad trials directly biases sampling toward
      promising regions — at small evolved-field counts (n on the order
      of a handful, with low-hundreds of evaluations) this typically
      converges faster than CMA-ES.

    Why we still call it "evolutionary" and use ask/tell:

    - The framework's ``EvolutionLoop`` is a population-based ask/tell
      loop and we want optimisers to be drop-in.  Optuna's native
      ``study.optimize(func, n_trials=N)`` doesn't fit that surface, but
      Optuna also exposes ``study.ask()`` / ``study.tell(trial, value)``
      which does.  Per generation we ``ask()`` ``population_size`` trials,
      hand the suggested params back to the loop, then ``tell()`` each
      trial its fitness once the loop reports back.

    Sign convention:

    - ``EvolutionLoop`` minimises (lower fitness is better; it negates
      success rates upstream).  ``OptunaTPEOptimizer`` configures the
      Optuna study with ``direction="minimize"`` so the dispatch site
      doesn't need to know which optimiser is in use.

    Bounds requirement:

    - TPE genuinely needs per-parameter ``[low, high]`` bounds — unlike
      CMA-ES which can run unbounded.  Callers MUST pass ``bounds`` or
      construction fails clearly at init time.  The
      ``HyperparameterEncoder`` already knows these via the schema; the
      CLI threads them through ``_build_optimizer``.
    """

    def __init__(
        self,
        num_params: int,
        *,
        bounds: list[tuple[float, float]],
        population_size: int = 12,
        seed: int | None = None,
    ) -> None:
        """Initialise the Optuna TPE optimiser.

        Args:
            num_params: Number of parameters to optimise.
            bounds: Per-parameter ``(low, high)`` ranges.  Length MUST equal
                ``num_params``.  TPE samples from these bounds via
                ``trial.suggest_float``; out-of-bounds values are
                impossible by construction (vs CMA-ES which clips at
                decode time).
            population_size: Number of trials per generation.  TPE's
                ``ask()`` is normally serial, but we batch
                ``population_size`` trials per generation to match the
                ``EvolutionLoop`` interface.
            seed: Seed for the underlying ``TPESampler``.  When provided,
                reproduces the same trial-suggestion sequence given the
                same fitness values.
        """
        # Lazy import: optuna is an optional-but-installed dependency we
        # want to avoid at module-import time so test collection stays
        # fast even on environments without optuna available.  At
        # construction time we genuinely need it.
        import optuna

        super().__init__(num_params, population_size, sigma0=0.0)

        if len(bounds) != num_params:
            msg = (
                f"OptunaTPEOptimizer: bounds length {len(bounds)} does not "
                f"match num_params {num_params}."
            )
            raise ValueError(msg)
        for i, (low, high) in enumerate(bounds):
            if not (math.isfinite(low) and math.isfinite(high)):
                msg = (
                    f"OptunaTPEOptimizer: bounds[{i}] = ({low!r}, {high!r}); "
                    "all bounds must be finite real numbers."
                )
                raise ValueError(msg)
            if low >= high:
                msg = (
                    f"OptunaTPEOptimizer: bounds[{i}] = ({low}, {high}); "
                    "low must be strictly less than high."
                )
                raise ValueError(msg)

        self._bounds = list(bounds)
        # Silence Optuna's default per-trial INFO logs — at population_size *
        # generations trials per run, the volume drowns the loop's own logs.
        # Users who want them can re-enable via OPTUNA_LOG_LEVEL env or
        # ``optuna.logging.set_verbosity(...)`` directly.
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(seed=seed)
        self._study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
        )
        # Trial objects we've ``ask()``'d for the current generation but
        # haven't ``tell()``'d yet.  Keyed by the position in the
        # population so ``tell(solutions, fitnesses)`` can correlate the
        # caller's solutions list with the underlying Trial.
        self._pending_trials: list = []
        self._best_params: list[float] | None = None
        self._best_fitness: float = float("inf")
        self._history: list[dict] = []

    def ask(self) -> list[list[float]]:
        """Request ``population_size`` candidate solutions from TPE.

        Each call corresponds to one generation in the framework's
        loop.  The Optuna trials are stashed in ``self._pending_trials``
        so the next ``tell()`` can correlate fitness values back to
        their originating trial.
        """
        # If a previous ask() was made without a tell(), the user is
        # double-asking — that's a contract violation and we'd silently
        # leak Trial objects.  Surface it.
        if self._pending_trials:
            msg = (
                "OptunaTPEOptimizer.ask() called without a preceding tell(); "
                "the loop must call tell() with the previous generation's "
                "fitnesses before requesting a new generation."
            )
            raise RuntimeError(msg)

        solutions: list[list[float]] = []
        for _ in range(self.population_size):
            trial = self._study.ask()
            params = [
                trial.suggest_float(f"x{i}", low, high)
                for i, (low, high) in enumerate(self._bounds)
            ]
            self._pending_trials.append(trial)
            solutions.append(params)
        return solutions

    def tell(self, solutions: list[list[float]], fitnesses: list[float]) -> None:
        """Report fitness values for the most recent ``ask()`` solutions.

        Args:
            solutions: Parameter arrays from the most recent ``ask()``.
                Length must equal ``population_size``.  The arrays
                themselves are NOT used for the tell — Optuna correlates
                via the trial objects stashed in ``self._pending_trials``
                — but a length check on ``solutions`` catches a
                miscount on the caller's side.
            fitnesses: Corresponding fitness values (lower is better).
        """
        if len(self._pending_trials) != len(fitnesses):
            msg = (
                "OptunaTPEOptimizer.tell(): "
                f"have {len(self._pending_trials)} pending trials but "
                f"{len(fitnesses)} fitnesses; ask/tell pairing broken."
            )
            raise RuntimeError(msg)
        if len(solutions) != len(fitnesses):
            msg = (
                "OptunaTPEOptimizer.tell(): "
                f"solutions length {len(solutions)} does not match "
                f"fitnesses length {len(fitnesses)}."
            )
            raise ValueError(msg)

        for trial, params, fitness in zip(
            self._pending_trials,
            solutions,
            fitnesses,
            strict=True,
        ):
            self._study.tell(trial, fitness)
            if fitness < self._best_fitness:
                self._best_fitness = fitness
                self._best_params = list(params)

        # Mirror CMA-ES's history schema (generation/best_fitness/
        # mean_fitness/std_fitness) so the aggregator script and any
        # downstream tooling that reads history.csv works regardless of
        # which optimiser produced the run.  All three are in the
        # framework's minimisation-space (lower = better) — the loop's
        # ``_write_artefacts`` flips them to "positive = better" before
        # writing the CSV, so optimisers MUST store minimisation-space
        # values here.  ``best_fitness`` is the global running min;
        # mean/std are computed over THIS generation's fitnesses.
        self._history.append(
            {
                "generation": self.generation + 1,
                "best_fitness": self._best_fitness,
                "mean_fitness": float(np.mean(fitnesses)) if fitnesses else 0.0,
                "std_fitness": float(np.std(fitnesses)) if fitnesses else 0.0,
            },
        )
        self._pending_trials = []
        self.generation += 1

    def stop(self) -> bool:
        """TPE has no internal stopping criterion; the loop's gen budget governs."""
        return False

    @property
    def result(self) -> EvolutionResult:
        """Best params + fitness across all generations seen so far."""
        # Defensive: if ``result`` is queried before any ``tell()`` has
        # landed, ``_best_params`` is still None.  Return a sentinel
        # rather than raise so the caller's logging path doesn't crash
        # (CMA-ES exposes an analogous default through ``cma``).
        best = (
            self._best_params.copy() if self._best_params is not None else [0.0] * self.num_params
        )
        return EvolutionResult(
            best_params=best,
            best_fitness=self._best_fitness,
            generations=self.generation,
            history=self._history,
        )


@dataclass
class FitnessConfig:
    """Configuration for fitness evaluation.

    Attributes
    ----------
        episodes_per_evaluation: Number of episodes to run per fitness evaluation.
        parallel_workers: Number of parallel workers for population evaluation.
            When > 1, uses multiprocessing which requires picklable factories.
            Defaults to 1 (sequential evaluation).
        episode_max_steps: Maximum steps per episode before timeout.
    """

    episodes_per_evaluation: int = 15
    parallel_workers: int = 1
    episode_max_steps: int = 200
