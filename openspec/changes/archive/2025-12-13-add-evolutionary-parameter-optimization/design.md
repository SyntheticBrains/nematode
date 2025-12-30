# Design: Evolutionary Parameter Optimization

## Context

Quantum variational circuits face a fundamental challenge: gradient estimation via parameter-shift rule is inherently noisy. Combined with sparse rewards (success every 5-10 episodes), gradient-based learning adds noise rather than signal. Our experiments show learning actively degrades performance from 22.5% to 14.9%.

Evolutionary optimization sidesteps gradients entirely, using population-based search with episode-level fitness aggregation. This is a well-established approach for quantum circuit optimization (see Variational Quantum Eigensolver literature).

### Stakeholders

- Researchers seeking optimal quantum parameters
- Quantum advantage investigation (requires clean comparison)
- Future work on hybrid classical-quantum architectures

## Goals / Non-Goals

### Goals

- Find optimal or near-optimal quantum circuit parameters without gradient noise
- Establish true capacity ceiling of quantum circuits (currently unknown)
- Enable apples-to-apples comparison with classical baselines
- Provide stable, reproducible optimization results

### Non-Goals

- Replace gradient-based learning (it remains available)
- Prove quantum advantage (this enables investigation, not proof)
- Optimize for speed (research clarity over performance)
- Support distributed/cloud evolution (local multiprocessing only)

## Decisions

### Decision 1: CMA-ES as Primary Algorithm

**Choice**: Use CMA-ES (Covariance Matrix Adaptation Evolution Strategy) as the primary evolutionary algorithm.

**Rationale**:

- Works exceptionally well in 10-50 parameter range (we have 12-24)
- Self-adapts search distribution without manual tuning
- Handles noisy fitness evaluations gracefully
- Well-established Python library (`cma`)
- Standard in quantum variational circuit optimization literature

**Alternatives Considered**:

- Simple Genetic Algorithm: More interpretable, but requires tuning mutation rates
- Particle Swarm: Good for continuous spaces, but less robust to noise
- Random Search: Baseline comparison only, not for production use

### Decision 2: Fitness = Success Rate Over N Episodes

**Choice**: Fitness is negative of success rate averaged over N episodes (N=10-20).

**Rationale**:

- CMA-ES minimizes, so use negative success rate
- N=10-20 balances noise reduction vs evaluation cost
- Success rate is the metric we care about (not reward magnitude)
- Simple, interpretable, directly comparable to baseline

**Alternative Considered**:

- Weighted combination of success + foods collected: More complex, unclear benefit
- Single-episode fitness: Too noisy, would require more generations

### Decision 3: Parallel Fitness Evaluation

**Choice**: Use Python `multiprocessing` for parallel fitness evaluation.

**Rationale**:

- Each population member is independent
- Easy 4-8x speedup on modern machines
- No external dependencies (Ray, Dask)
- Brain instances are picklable

**Implementation**:

```python
from multiprocessing import Pool

with Pool(processes=8) as pool:
    fitnesses = pool.map(evaluate_fitness, population)
```

### Decision 4: Classical Baseline with Matched Parameters

**Choice**: Create `LinearClassicalBrain` with exactly 12 parameters.

**Rationale**:

- Fair comparison requires parameters to be matched in count
- Linear model: 4 actions × 3 inputs = 12 weights (no bias)
- Or small MLP: 3→4 with 12 weights distributed
- Same evolution process, only brain differs

**Comparison Protocol**:

1. Evolve quantum brain (12 params) → fitness_q
2. Evolve classical brain (12 params) → fitness_c
3. If fitness_q > fitness_c: Evidence of quantum advantage
4. Report with confidence intervals

### Decision 5: Configuration Schema Extension

**Choice**: Extend existing YAML schema with `evolution` section.

```yaml
evolution:
  enabled: true
  algorithm: cmaes  # or 'ga'
  population_size: 20
  generations: 50
  episodes_per_evaluation: 15
  parallel_workers: 8

  # CMA-ES specific
  sigma0: 0.5  # Initial step size

  # GA specific (if algorithm: ga)
  elite_fraction: 0.2
  mutation_rate: 0.1
  crossover_rate: 0.8
```

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    run_evolution.py                         │
├─────────────────────────────────────────────────────────────┤
│  1. Load config                                             │
│  2. Create EvolutionaryOptimizer (CMA-ES or GA)             │
│  3. For each generation:                                    │
│     - Ask optimizer for candidate solutions                 │
│     - Evaluate fitness in parallel                          │
│     - Tell optimizer the results                            │
│     - Log progress                                          │
│  4. Save best parameters                                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              EvolutionaryOptimizer (base)                   │
├─────────────────────────────────────────────────────────────┤
│  - ask() → list[param_array]                                │
│  - tell(solutions, fitnesses)                               │
│  - result → best_params                                     │
│  - stop() → bool                                            │
└─────────────────────────────────────────────────────────────┘
          │                                │
          ▼                                ▼
┌─────────────────────┐     ┌─────────────────────────────────┐
│   CMAESOptimizer    │     │    GeneticAlgorithmOptimizer    │
├─────────────────────┤     ├─────────────────────────────────┤
│ Wraps cma library   │     │ Selection, crossover, mutation  │
└─────────────────────┘     └─────────────────────────────────┘

                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    FitnessFunction                          │
├─────────────────────────────────────────────────────────────┤
│  - __call__(param_array) → fitness                          │
│  - Creates brain with params                                │
│  - Runs N episodes                                          │
│  - Returns -success_rate                                    │
└─────────────────────────────────────────────────────────────┘
```

## Risks / Trade-offs

### Risk 1: Computational Cost

- **Issue**: 20 population × 15 episodes × 50 generations = 15,000 episodes
- **Mitigation**: Parallel evaluation, resume from checkpoint
- **Trade-off**: Accept longer runtime for cleaner results

### Risk 2: Local Optima

- **Issue**: Evolution may converge to local optimum
- **Mitigation**: CMA-ES adapts search distribution; run multiple seeds
- **Trade-off**: More generations/larger population if needed

### Risk 3: Quantum Advantage May Not Exist

- **Issue**: Comparison may show classical wins at 12 parameters
- **Mitigation**: This is a valid research result, not a failure
- **Trade-off**: Understanding limits is valuable

## Migration Plan

1. **Phase 1**: Implement core evolution infrastructure (no breaking changes)
2. **Phase 2**: Add classical baseline for comparison
3. **Phase 3**: Run experiments and document results
4. **Rollback**: Simply don't use evolution; existing code unchanged

## Open Questions

1. **Optimal episodes per evaluation**: Start with 15, may need tuning
2. **Seed initialization**: Random vs informed (use best known params)?
3. **Multi-objective**: Should we optimize success + efficiency jointly?
4. **Longer episodes**: Current 200 steps enough for evolution?
