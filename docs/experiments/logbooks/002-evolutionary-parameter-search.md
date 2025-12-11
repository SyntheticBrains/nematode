# 002: Evolutionary Parameter Search

**Status**: `active`

**Branch**: `feature/14-improve-quantum-predators-performance`

**Date Started**: 2025-12-09

**Date Completed**: -

## Objective

Use evolutionary algorithms (CMA-ES and Genetic Algorithm) to find optimal quantum circuit parameters, bypassing the noisy gradient-based learning that was shown to degrade performance in Experiment 001.

## Background

Experiment 001 revealed that gradient-based learning actively harms quantum circuit performance:
- Zero learning (22.5%) outperformed low-LR fine-tuning (18.75%)
- Parameter-shift gradients are noisy with sparse rewards
- Good initializations get destroyed by step-by-step updates

Evolution offers an alternative:
- No gradients needed - fitness is aggregate success rate
- Population maintains diversity - no catastrophic forgetting
- Elitism preserves good solutions

## Hypothesis

Evolutionary optimization will find parameters achieving higher success rates than gradient-based learning because:
1. Fitness evaluation over multiple episodes reduces noise
2. Population-based search explores globally, not just locally
3. No risk of destroying good parameters through learning

## Method

### Implementation
Created `scripts/run_evolution.py` and `quantumnematode/optimizers/evolutionary.py` with:
- **CMA-ES**: Covariance Matrix Adaptation Evolution Strategy (via `cma` library)
- **Genetic Algorithm**: Tournament selection, uniform crossover, Gaussian mutation

### Fitness Function
```python
def evaluate_fitness(params, config_path, episodes):
    brain = create_brain_with_params(params)
    successes = sum(run_episode(brain, env) for _ in range(episodes))
    return -successes / episodes  # Negative for minimization
```

### Configuration (Foraging Only)
```yaml
# configs/examples/evolution_foraging_only.yml
max_steps: 500
brain:
  name: modular
  config:
    modules:
      chemotaxis: [0, 1]
    num_layers: 2
qubits: 2
environment:
  type: dynamic
  dynamic:
    predators:
      enabled: false  # Simpler baseline first
```

### Run Parameters
| Parameter | CMA-ES | GA |
|-----------|--------|-----|
| Generations | 30 | 30 |
| Population | 15 | 30 |
| Episodes/eval | 10 | 10 |
| Parallel workers | 4 | 4 |
| Sigma (step size) | 0.5 | 0.3 |

## Results

### Phase 1: Foraging Only (No Predators)

| Algorithm | Best Success | Final Mean | Peak Generation | Stability |
|-----------|-------------|------------|-----------------|-----------|
| **CMA-ES** | **80%** | 17.3% | Gen 23 | Fluctuates |
| **GA** | 70% | 27.0% | Gen 20 | Stable |

### Phase 2: Small Environment with 2 Predators

| Algorithm | Best Success | Final Mean | Generations | Runtime |
|-----------|-------------|------------|-------------|---------|
| **GA** | **73.3%** | 25-30% | 150 (50+100) | ~21 hours |

#### GA Predator Run Details
Sessions: `20251210_072400` (gen 1-50), `20251210_210057` (gen 51-150)

| Phase | Generations | Best | Mean | Notes |
|-------|-------------|------|------|-------|
| Cold start | 1-37 | 0-20% | 0-3% | Slow exploration |
| Breakthrough | 38-50 | 53.3% | 15% | First viable solutions |
| Plateau | 51-62 | 60% | 25% | Ceiling hit |
| Peak | 63, 69 | **73.3%** | 26% | Best achieved |
| Stable | 70-150 | 46-67% | 25-33% | High variance persists |

Best parameters (73.3% success):
```json
{
  "θ_rx1_0": -0.346, "θ_rx1_1": -2.643,
  "θ_ry1_0": 0.693, "θ_ry1_1": -0.606,
  "θ_rz1_0": -3.026, "θ_rz1_1": 0.075,
  "θ_rx2_0": 2.474, "θ_rx2_1": 2.299,
  "θ_ry2_0": 2.369, "θ_ry2_1": 0.910,
  "θ_rz2_0": 1.512, "θ_rz2_1": 1.895
}
```

### CMA-ES Detailed Results
Session: `20251209_205950`

| Gen | Best | Mean | Notes |
|-----|------|------|-------|
| 1-4 | 0% | 0% | Cold start |
| 5 | 10% | 0.7% | First success |
| 18 | 70% | 16% | Rapid improvement |
| 23 | **80%** | 17% | Peak performance |
| 30 | 40% | 17% | Drifted from peak |

Best parameters found:
```json
{
  "theta_rx1_0": 0.200, "theta_rx1_1": -0.942,
  "theta_ry1_0": 1.129, "theta_ry1_1": -0.328,
  "theta_rz1_0": 0.145, "theta_rz1_1": -0.054,
  "theta_rx2_0": -0.617, "theta_rx2_1": 0.937,
  "theta_ry2_0": -0.559, "theta_ry2_1": 1.323,
  "theta_rz2_0": 0.861, "theta_rz2_1": -2.992
}
```

### GA Detailed Results
Session: `20251209_210000`

| Gen | Best | Mean | Notes |
|-----|------|------|-------|
| 1-5 | 10% | 0.3% | Slow start |
| 14 | 30% | 7.7% | Steady climb |
| 20 | **70%** | 21% | Peak, held stable |
| 30 | 70% | 27% | Mean still improving |

Best parameters found:
```json
{
  "theta_rx1_0": -0.538, "theta_rx1_1": -1.700,
  "theta_ry1_0": -2.396, "theta_ry1_1": 2.534,
  "theta_rz1_0": -2.976, "theta_rz1_1": 0.014,
  "theta_rx2_0": -1.517, "theta_rx2_1": -2.891,
  "theta_ry2_0": 2.383, "theta_ry2_1": 0.580,
  "theta_rz2_0": -0.473, "theta_rz2_1": 2.342
}
```

## Analysis

### Why CMA-ES Found Higher Peak
- Adapts search distribution based on successful directions
- Can make larger jumps in parameter space
- Found 80% solution but couldn't fully converge there

### Why GA is More Stable
- Elitism explicitly preserves best individual across generations
- Once 70% solution found, it's never lost
- Mean fitness continues improving even at gen 30

### Why Both Beat Gradient Learning
- Fitness over 10 episodes averages out environment randomness
- No per-step gradient noise from parameter-shift rule
- Population explores multiple solutions in parallel

### Predator Environment Analysis

**Key finding**: GA achieved **73.3%** on predators vs **22.5%** for gradient methods (3x improvement).

Observations from the 150-generation run:
1. **Long warmup required**: 37 generations of near-zero success before breakthrough
2. **High variance**: Best fitness fluctuated 40-73% even in later generations
3. **Mean lags best**: Population mean (~25-30%) well below best individual
4. **Plateau around 60-70%**: Multiple peaks at 73.3% but couldn't consolidate higher

Possible causes of variance:
- 15 episodes/eval may be insufficient to reliably distinguish solutions
- Predator movement randomness creates noisy fitness signal
- GA crossover may be disrupting good parameter combinations

### Comparison to Gradient-Based Results

| Environment | Evolution Best | Gradient Best | Improvement |
|-------------|---------------|---------------|-------------|
| Foraging-only | 80% (CMA-ES) | 83% | Similar |
| **With predators** | **73.3%** (GA) | **22.5%** | **3.3x** |

**Key insight**: Evolution matches gradient performance on foraging-only (~80%), but dramatically outperforms on predator environments where gradient noise causes catastrophic drift.

## Conclusions

1. **Evolution matches gradients on foraging-only**: Both achieve ~80% success rate
2. **Evolution dramatically outperforms on predators**: 73.3% vs 22.5% (3.3x improvement)
3. **GA more reliable**: Stable convergence, preserves best solutions
4. **CMA-ES finds higher peaks**: But can drift away from them
5. **Long warmup for predators**: ~37 generations before breakthrough, patience required

## Next Steps

### Immediate
- [x] ~~Longer GA run (50-100 generations) to see if 80%+ achievable~~ Done: 150 gen, peaked at 73.3%
- [ ] Validate best predator params over 100+ episodes
- [ ] Try CMA-ES on predators (may find higher peaks like it did for foraging)

### Reducing Variance
- [ ] Increase episodes per evaluation (20-25) to reduce fitness noise
- [ ] Larger population (40-50) for better coverage
- [ ] Seed from best known params instead of random init

### Algorithm Experiments
- [ ] CMA-ES with predators - compare peak finding vs GA
- [ ] Hybrid: CMA-ES for exploration, then GA for refinement
- [ ] Different sigma values for predator environment

## Data References

### Foraging Only
- **CMA-ES Session**: `20251209_205950`
  - Best: 80%, Config: `evolution_foraging_only.yml`
- **GA Session**: `20251209_210000`
  - Best: 70%, Config: `evolution_foraging_only.yml`

### With Predators (2 predators, small environment)
- **GA Session 1**: `20251210_072400` (gen 1-50)
  - Best: 53.3%, Config: `evolution_small_predators.yml`
- **GA Session 2**: `20251210_210057` (gen 51-150, resumed)
  - Best: 73.3%, Config: `evolution_small_predators.yml`
  - Best params: `best_params_20251210_210057.json`

### Scripts and Configs
- Evolution script: `scripts/run_evolution.py`
- Foraging config: `configs/examples/evolution_foraging_only.yml`
- Predator config: `configs/examples/evolution_small_predators.yml`
