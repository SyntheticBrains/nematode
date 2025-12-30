# Change: Add Evolutionary Parameter Optimization for Quantum Brains

## Why

After 15+ experiments with gradient-based learning on quantum brains, we have established that:

1. **Gradient noise is fundamental**: The parameter-shift rule produces noisy gradient estimates (2 circuit evaluations per parameter per step), and sparse rewards amplify this noise
2. **Learning actively harms performance**: Informed initializations achieve 22.5% success, but learning degrades this to 14.9%
3. **Good parameters exist, but gradients can't find them**: The circuit has capacity, but gradient descent can't navigate the noisy, sparse-reward landscape

Evolutionary/genetic approaches (CMA-ES, genetic algorithms) sidestep gradient estimation entirely, using population-based search with episode-level fitness. This removes the fundamental bottleneck preventing quantum brain optimization.

Additionally, this serves as a **stepping stone to quantum advantage research**:

- First, establish the true capacity ceiling of quantum circuits (currently unknown due to learning noise)
- Then, compare against equivalent-parameter classical baselines (12 params each)
- If quantum outperforms classical at same parameter count, that's evidence of quantum advantage

## What Changes

- **New optimization mode**: Add evolutionary parameter optimization alongside gradient-based learning
- **Fitness evaluation**: Run N episodes per candidate, aggregate success rate as fitness
- **CMA-ES integration**: Use `cma` library for Covariance Matrix Adaptation Evolution Strategy
- **Simple GA alternative**: Provide simpler genetic algorithm option for interpretability
- **Classical baseline**: Add 12-parameter classical model for apples-to-apples comparison
- **New script**: `scripts/run_evolution.py` parallel to existing `run_simulation.py`
- **Config extension**: Add evolution parameters to YAML configuration schema

## Impact

- **Affected specs**: `brain-architecture` (new optimization paradigm)
- **Affected code**:
  - `quantumnematode/optimizers/` - New evolutionary optimizer module
  - `quantumnematode/brain/arch/` - Parameter get/set interfaces
  - `scripts/` - New evolution script
  - `configs/` - Evolution configuration examples
- **New dependencies**: `cma` (CMA-ES implementation)
- **Backward compatible**: Existing gradient-based learning unchanged; evolution is additive

## Success Criteria

1. Evolution script runs successfully with quantum brain
2. Find parameters achieving ≥30% success rate consistently
3. Variance between runs significantly lower than gradient-based
4. Classical baseline comparison completed for quantum advantage analysis
