# 001: Quantum Predator Optimization

**Status**: `completed`

**Branch**: `feature/14-improve-quantum-predators-performance`

**Date Started**: 2025-12-06

**Date Completed**: 2025-12-08

## Objective

Close the performance gap between quantum circuits (~31% success) and classical MLP (~85% success) in predator avoidance environments through improved architectures and learning strategies.

## Background

The quantum nematode uses a variational quantum circuit for decision-making. Initial benchmarks showed:
- **Classical MLP**: 85-92% success rate
- **Quantum (2-qubit chemotaxis)**: 20-31% success rate

The 300x parameter gap (12 vs ~4,000) suggested limited capacity, but we explored whether better architectures or learning could help.

## Hypothesis

We tested multiple hypotheses:
1. Separate modules for appetitive/aversive behaviors would allow specialized learning
2. Dual quantum circuits with gating could handle conflicting signals
3. Better hyperparameters or learning strategies could improve convergence

## Method

### Architectures Tested

| Architecture | Qubits | Parameters | Description |
|--------------|--------|------------|-------------|
| Chemotaxis (baseline) | 2 | 12 | Single module, combined gradient |
| Appetitive/Aversive 4q | 4 | 24 | Entangled, separate modules |
| Appetitive/Aversive 2q | 2 | 12 | 1 qubit per module |
| Dual-Circuit | 4 | 24 | Two separate circuits with gating |

### Learning Strategies Tested

| Strategy | Description |
|----------|-------------|
| Standard gradient | Per-step parameter-shift updates |
| Informed initialization | Start from known good parameters |
| Zero learning | Use fixed parameters, no updates |
| Low LR fine-tuning | Small learning rate (0.001) |
| Success-only learning | Rollback parameters on failed episodes |
| Constant LR | No decay, fixed learning rate |

## Results

### Architecture Comparison

| Architecture | Best Success | Avg Foods | Predator Deaths | Verdict |
|--------------|-------------|-----------|-----------------|---------|
| Chemotaxis 2q | **31%** | 5.6 | 55% | Best quantum |
| App/Avr 4q entangled | 0-5% | 1.5 | 70% | Failed |
| App/Avr 2q | 0% | 1.2 | 80% | Failed |
| Dual-Circuit | 0.25% | 2.0 | 46% | Failed |

### Learning Strategy Comparison

| Strategy | Success Rate | Notes |
|----------|-------------|-------|
| Standard gradient | 6-20% | High variance between seeds |
| Informed init + learning | 14.9% | Learning destroys good params |
| **Zero learning** | **22.5%** | Best with informed params |
| Low LR fine-tuning | 18.75% | Still drifts |
| Success-only (short) | 28.75% | Best learning result |
| Success-only (long) | 14.9% | Degraded over time |

### Key Experiment Results

#### Attempt 10: 200-Run Baseline
Two sessions with identical config, wildly different outcomes:
- Session A: 20.5% success, 91% evasion rate
- Session B: 6% success, worse in every metric
- **Finding**: Random initialization variance is severe

#### Attempt 12: Zero Learning
- Started from parameters of best benchmark (29% success)
- Ran with learning_rate = 0
- Result: **22.5% average** across 4 sessions
- **Finding**: The informed parameters encode useful behavior

#### Attempt 14: Success-Only Learning
- Only keep parameter updates from successful episodes
- Short runs (20 episodes): **28.75%** average, one hit **45%**
- Long runs (200 episodes): Degraded to 14.9%
- **Finding**: Even sparse learning adds noise over time

#### Attempt 15: Dual-Circuit with Informed Init
- Two separate circuits for appetitive/aversive
- Gating mechanism to blend outputs
- Result: **0.25% success** (catastrophic)
- **Finding**: Added complexity without added capability

## Analysis

### Why Chemotaxis Works (Partially)
The combined gradient from `get_state()`:
1. Vectorially sums food attraction + predator repulsion
2. Produces single "optimal direction" signal
3. Circuit just learns to follow this pre-computed signal
4. **The environment does the hard work, not the brain**

### Why Separated Gradients Failed
- Appetitive and aversive give conflicting directions
- No natural mechanism to integrate in entangled circuit
- Action mapping (16 states → 4 via modulo) destroys semantics

### Why Dual-Circuit Failed
- Gating interfered with food-seeking
- Mirrored parameters have no principled basis
- 24 parameters still far below MLP's 4,000

### Why Learning Degrades Performance
- Parameter-shift gradients are noisy (statistical estimation)
- Sparse rewards (success every 5-10 episodes) = weak signal
- Step-by-step updates destroy good initializations
- No baseline subtraction or variance reduction

### The Fundamental Problem: Capacity
| Architecture | Parameters | Success Rate |
|--------------|-----------|--------------|
| MLP (classical) | ~4,000 | 85-92% |
| Quantum (2-qubit) | 12 | 20-31% |
| Quantum (4-qubit) | 24 | 0-5% |

The capacity gap is ~300x. More qubits didn't help because the learning problem became harder without proportional signal improvement.

## Conclusions

1. **2-qubit chemotaxis is the ceiling** for gradient-based learning (~31%)
2. **Learning actively harms performance** when starting from good parameters
3. **Separated gradients fail** - the combined gradient is essential
4. **Dual-circuit adds complexity without benefit** for this task
5. **The environment's signal does the heavy lifting**, not the quantum circuit

## Recommended Paths Forward

### Option A: Accept Limitations (Implemented in Exp 002)
- Use evolution instead of gradient learning
- Find optimal static parameters
- Accept 30-35% as quantum ceiling

### Option B: Hybrid Classical-Quantum
- Use MLP for decisions
- Quantum circuit for feature extraction only
- Classical learning (PyTorch autograd)

### Option C: More Qubits + Better Architecture (Research)
- 6-8 qubits for more capacity
- Better gradient estimation (SPSA, adjoint)
- Proper RL infrastructure (replay buffer, target networks)

## Data References

### Best Sessions
- Chemotaxis baseline: `20251207_035803` (20.5% success)
- Success-only short: `20251207_100041` (45% success in 20 runs)

### Config Files
- `modular_dynamic_small_predators.yml`
- `dual_circuit_optimized_50runs.yml`
- `modular_appetitive_aversive_predators.yml`

### Code Files Modified
- `quantumnematode/brain/modules.py`
- `quantumnematode/brain/arch/dual_circuit.py`
- `quantumnematode/brain/arch/modular.py`

## Appendix: Hyperparameters That Work

| Parameter | Working Value | Common Mistake |
|-----------|---------------|----------------|
| shots | 1500 | 3000 (too many) |
| initial_lr | 0.015 | 0.5 (too high) |
| decay_rate | 0.9995 (exp) | 0.001 (too fast) |
| min_lr | 0.003 | 0 or 0.001 |
| gradient_method | norm_clip | clip |
