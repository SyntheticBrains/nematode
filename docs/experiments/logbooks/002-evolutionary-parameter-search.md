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

### Summary Table (Foraging Only, No Predators)

| Algorithm | Best Success | Final Mean | Peak Generation | Stability |
|-----------|-------------|------------|-----------------|-----------|
| **CMA-ES** | **80%** | 17.3% | Gen 23 | Fluctuates |
| **GA** | 70% | 27.0% | Gen 20 | Stable |

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

### Comparison to Gradient-Based Results

**Important**: The poor gradient results from Experiment 001 (22.5% with zero LR) were for the **predator environment**. For foraging-only, gradient-based learning performs well:

| Approach | Best Success (Foraging-Only) | Notes |
|----------|------------------------------|-------|
| Gradient (benchmark 20251130_011145) | **83.0%** | Best overall |
| **CMA-ES Evolution** | 80% | Comparable |
| Gradient (benchmark 20251130_011154) | 80.5% | Similar to evolution |
| Gradient (benchmark 20251128_103121) | 72.0% | |
| **GA Evolution** | 70% | Stable convergence |
| Gradient (benchmark 20251127_104630) | 58.5% | Earlier config |

**Key insight**: Evolution matches gradient performance on foraging-only (~80%), but evolution's value is for predator environments where gradients fail.

## Conclusions

1. **Evolution matches gradients on foraging-only**: Both achieve ~80% success rate
2. **GA more reliable**: Stable convergence, preserves best solutions
3. **CMA-ES finds higher peaks**: But can drift away from them
4. **True value is for predators**: Evolution bypasses gradient noise where gradient methods fail (22.5% in Exp 001)
5. **Foraging-only is tractable**: Good baseline before adding predators

## Next Steps

### Immediate
- [ ] Longer GA run (50-100 generations) to see if 80%+ achievable
- [ ] Add `--init-params` flag to seed from best known parameters
- [ ] Run validation: test best params over 100+ episodes

### Predator Environment
- [ ] Create intermediate configs (1 slow predator, then 2, then faster)
- [ ] Expect much lower success rates initially (0-10%)
- [ ] May need 100+ generations with predators
- [ ] Consider curriculum: evolve on easy, transfer to hard

### Algorithm Improvements
- [ ] Add elitism to CMA-ES (track and preserve best ever)
- [ ] Implement fitness sharing for population diversity
- [ ] Try different episode counts per evaluation (15-20)

## Data References

- **CMA-ES Session**: `artifacts/evolutions/20251209_205950/`
  - Best params: `best_params_20251209_205950.json`
  - History: `history_20251209_205950.csv`
  - Configuration: `evolution_foraging_only.yml`
- **GA Session**: `artifacts/evolutions/20251209_210000/`
  - Best params: `best_params_20251209_210000.json`
  - History: `history_20251209_210000.csv`
  - Configuration: `evolution_foraging_only.yml`
- **Script**: `scripts/run_evolution.py`
