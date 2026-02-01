# Optimization Methods for Brain Architectures

This document provides guidance on which optimization methods work best for each brain architecture in the Quantum Nematode project.

## Summary Table

| Architecture | Primary Method | Secondary Method | Success Rate | Notes |
|-------------|----------------|------------------|--------------|-------|
| ModularBrain | CMA-ES | Parameter-shift | 88% | Evolution best for quantum |
| QModularBrain | CMA-ES | Parameter-shift | 85% | Hybrid quantum-classical |
| MLPBrain | REINFORCE | Adam + LR schedule | 92% | Classic policy gradient |
| PPOBrain | Clipped PPO | Adam | 97% | Actor-critic with GAE |
| QMLPBrain | REINFORCE | Adam | 75% | Gradient-based works |
| SpikingBrain | Surrogate + REINFORCE | Adam | 63-78% | Surrogate gradients |

## Detailed Findings

### Quantum Architectures (ModularBrain, QModularBrain)

#### Recommended: CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

The December 2025 experiments demonstrated that evolutionary optimization significantly outperforms gradient-based methods for quantum circuits:

| Method | Success Rate | Variance |
|--------|-------------|----------|
| CMA-ES | 88% | Low |
| Parameter-shift gradients | 22% | High |

#### Why CMA-ES works better

1. **Shot noise resilience** - Quantum measurements have inherent stochasticity. CMA-ES uses population-based sampling that naturally averages over this noise.

2. **Barren plateau avoidance** - Gradient-based methods can get stuck in barren plateaus where gradients vanish. CMA-ES explores the landscape globally.

3. **Hyperparameter robustness** - CMA-ES self-adapts its covariance matrix, requiring less tuning than learning rates for gradient methods.

#### Configuration

```yaml
brain:
  name: qvarcircuit  # formerly 'modular'
  config:
    # No learning_rate needed - uses evolution

optimization:
  method: cmaes
  population_size: 20
  sigma: 0.5  # Initial step size
  max_generations: 500
```

### Classical Neural Networks (MLPReinforceBrain)

#### Recommended: REINFORCE with baseline

The standard REINFORCE algorithm with a learned baseline works well for classical networks:

| Method | Success Rate |
|--------|-------------|
| PPO | 97% |
| REINFORCE + baseline | 92% |
| Raw REINFORCE | 78% |

#### Why REINFORCE works

1. **Simplicity** - Fewer hyperparameters than actor-critic methods
2. **Stability** - The baseline reduces variance effectively
3. **Efficiency** - Single network, no value function needed

#### Configuration

```yaml
brain:
  name: mlpreinforce  # formerly 'mlp'
  config:
    hidden_dim: 64
    num_hidden_layers: 2
    learning_rate: 0.001
    baseline: 0.0
    baseline_alpha: 0.05  # Exponential moving average
    entropy_beta: 0.01    # Entropy regularization
    gamma: 0.99

gradient:
  method: clip
```

### PPO Brain (MLPPPOBrain)

#### Recommended: Clipped surrogate objective

PPO uses the clipped surrogate objective with GAE for advantage estimation. December 2025 benchmarks achieved **97.1% ± 1.2% success rate** on foraging small (20x20) with fast convergence (~14 episodes to 80% success).

#### Why PPO excels

1. **Stable updates** - Clipped objective prevents destructive policy updates
2. **Sample efficiency** - Multiple epochs per rollout improve data utilization
3. **Variance reduction** - GAE provides low-variance advantage estimates
4. **Fast convergence** - Learns effective policies in ~14 episodes

#### Configuration

```yaml
brain:
  name: mlpppo  # formerly 'ppo'
  config:
    actor_hidden_dim: 64
    critic_hidden_dim: 64
    clip_epsilon: 0.2      # Clipping parameter
    gae_lambda: 0.95       # GAE lambda
    value_loss_coef: 0.5   # Value function weight
    entropy_coef: 0.01     # Entropy bonus
    learning_rate: 0.0003
    num_epochs: 4          # Epochs per update
    num_minibatches: 4     # Minibatches per epoch
    rollout_buffer_size: 2048
    max_grad_norm: 0.5
```

### Spiking Neural Networks (SpikingReinforceBrain)

#### Recommended: Surrogate gradients + REINFORCE

Spiking networks require special handling due to non-differentiable spike functions:

| Task | Method | Success Rate |
|------|--------|-------------|
| Foraging | Surrogate + REINFORCE | 78% |
| Predator evasion | Surrogate + REINFORCE | 63% |

#### Configuration

```yaml
brain:
  name: spikingreinforce  # formerly 'spiking'
  config:
    hidden_size: 64
    num_steps: 10
    tau_mem: 10.0
    tau_syn: 5.0
    threshold: 1.0
    learning_rate: 0.001
    surrogate_gradient: fast_sigmoid  # Options: fast_sigmoid, atan, piece_wise
    beta: 5.0  # Surrogate gradient sharpness
```

## Selection Guidance

Use this decision tree to choose an optimization method:

```text
Is the brain quantum-based (QVarCircuitBrain, QQLearningBrain)?
├── YES → Use CMA-ES
│         - Evolution handles shot noise naturally
│         - Avoids barren plateaus
│         - Self-adapting hyperparameters
│
└── NO → Is it a spiking network?
         ├── YES → Use Surrogate Gradients + REINFORCE
         │         - Surrogate enables backprop through spikes
         │         - REINFORCE handles non-differentiable reward
         │
         └── NO → Classical MLP/PPO
                  ├── Want best performance? → PPO (Recommended)
                  │   - 97% success rate on foraging
                  │   - Fast convergence (~14 episodes)
                  │   - Stable training with clipped objective
                  │
                  └── Want simplicity? → REINFORCE with baseline
                      - Single network, fewer hyperparameters
                      - 92% success on foraging
```

## Hyperparameter Recommendations

### Learning Rates

| Architecture | Recommended LR | Range |
|-------------|----------------|-------|
| MLPBrain | 0.001 | 0.0001 - 0.01 |
| PPOBrain | 0.0003 | 0.0001 - 0.001 |
| SpikingBrain | 0.001 | 0.0001 - 0.01 |
| QMLPBrain | 0.01 | 0.001 - 0.1 |

### CMA-ES Parameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Population size | 20 | 4 + 3\*log(n_params) |
| Initial sigma | 0.5 | Start exploring broadly |
| Max generations | 500 | Increase for complex tasks |

### Entropy Coefficients

| Architecture | Entropy Coef | Notes |
|-------------|--------------|-------|
| MLPBrain | 0.01 | Encourages exploration |
| PPOBrain | 0.01 | Standard value |
| SpikingBrain | 0.005 | Less needed with spikes |

## Common Pitfalls

### Quantum Circuits

1. **Using gradients on noisy hardware** - Parameter-shift gradients amplify shot noise
2. **Learning rate too high** - Quantum parameters are angles, small changes matter
3. **Not using enough shots** - Low shot counts increase variance

### Classical Networks

1. **No baseline** - Raw REINFORCE has high variance
2. **Learning rate too high** - Causes policy collapse
3. **No entropy regularization** - Premature convergence

### Spiking Networks

1. **Wrong surrogate choice** - Fast sigmoid works best empirically
2. **Tau values too small** - Information doesn't propagate
3. **Threshold too high** - Neurons never fire

## Experimental Results

### Foraging Task (Small, 20x20)

| Method | Architecture | Success | Learning Speed |
|--------|-------------|---------|----------------|
| Clipped PPO | PPOBrain | 97% | 14 episodes |
| CMA-ES | ModularBrain | 88% | 120 episodes |
| REINFORCE | MLPBrain | 92% | 85 episodes |
| Surrogate | SpikingBrain | 78% | 150 episodes |
| Param-shift | ModularBrain | 22% | - (unstable) |

### Predator Evasion Task (Small, 20x20)

| Method | Architecture | Success | Survival |
|--------|-------------|---------|----------|
| CMA-ES | ModularBrain | 75% | 82% |
| REINFORCE | MLPBrain | 84% | 88% |
| Surrogate | SpikingBrain | 63% | 71% |

## Key Insight: Why CMA-ES for Quantum

The most significant finding from our experiments is that **evolutionary optimization outperforms gradient-based methods by 4x for quantum circuits** (88% vs 22% success rate).

This is because:

1. **Quantum measurements are stochastic** - Each circuit evaluation returns different results due to quantum shot noise. Gradient estimates become extremely noisy.

2. **Parameter-shift rule doubles evaluations** - The parameter-shift method requires 2 circuit evaluations per parameter, amplifying the noise problem.

3. **Barren plateaus** - Deep quantum circuits exhibit vanishing gradients in most of the parameter space.

4. **CMA-ES is noise-tolerant** - Evolution uses population-based sampling that naturally averages over noise, and doesn't require gradient computation.

For researchers new to quantum ML: **start with CMA-ES** for any quantum architecture, then only try gradient methods if you have a specific reason.
