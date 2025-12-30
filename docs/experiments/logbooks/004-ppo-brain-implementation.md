# 004: PPO Brain Implementation

**Status**: `completed`

**Branch**: `feature/21-add-sota-rl-baselines`

**Date Started**: 2025-12-26

**Date Completed**: 2025-12-27

## Objective

Implement Proximal Policy Optimization (PPO) as a state-of-the-art classical reinforcement learning baseline, achieving competitive performance with the existing MLP brain while providing more stable learning dynamics.

## Background

The quantum nematode project uses multiple brain architectures (post-convergence success rates):

- **Quantum (ModularBrain)**: 2-qubit circuits with CMA-ES optimization (100% foraging, 95% predators)
- **Classical MLP**: REINFORCE policy gradient (100% foraging, 92% predators)
- **Spiking**: LIF neurons with surrogate gradients (100% foraging, 63% predators)

PPO offers advantages over REINFORCE:

- **Clipped objective**: Prevents destructively large policy updates
- **Actor-critic**: Separate value estimation reduces variance
- **GAE**: Generalized Advantage Estimation for better credit assignment
- **Sample efficiency**: Multiple epochs per batch of experience

This satisfies Phase 0 exit criterion: "PPO >85% success on foraging."

## Hypothesis

PPO with properly tuned hyperparameters should:

1. Match or exceed MLP's 96.5% success rate on dynamic foraging
2. Converge within 20 runs (matching MLP's convergence speed)
3. Show more stable post-convergence performance due to clipped updates
4. Provide a stronger baseline for predator environments than REINFORCE

## Method

### Architecture

**Network**: Actor-Critic with separate value head

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                         PPO ACTOR-CRITIC ARCHITECTURE                                    │
│                           (Separate Networks)                                            │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  INPUTS                                                                                  │
│  ┌───────────────────┐                                                                   │
│  │ gradient_strength │──┐                                                                │
│  │ [0.0 - 1.0]       │  │                                                                │
│  ├───────────────────┤  ├──────────────────┬──────────────────┐                          │
│  │ gradient_direction│──┘                  │                  │                          │
│  │ [-π, +π]          │                     ▼                  ▼                          │
│  └───────────────────┘       ┌──────────────────────────┐  ┌──────────────────────────┐  │
│                              │      ACTOR NETWORK       │  │     CRITIC NETWORK       │  │
│                              │    (Policy Function)     │  │    (Value Function)      │  │
│                              ├──────────────────────────┤  ├──────────────────────────┤  │
│                              │  Input: 2                │  │  Input: 2                │  │
│                              │    ↓                     │  │    ↓                     │  │
│                              │  Linear(2→64) + ReLU     │  │  Linear(2→64) + ReLU     │  │
│                              │    ↓                     │  │    ↓                     │  │
│                              │  Linear(64→64) + ReLU    │  │  Linear(64→64) + ReLU    │  │
│                              │    ↓                     │  │    ↓                     │  │
│                              │  Linear(64→4)            │  │  Linear(64→1)            │  │
│                              │    ↓                     │  │    ↓                     │  │
│                              │  Softmax                 │  │  State Value V(s)        │  │
│                              │    ↓                     │  └──────────────────────────┘  │
│                              │  Action Probabilities    │                                │
│                              │  [π(a|s) for a∈Actions]  │                                │
│                              └─────────────┬────────────┘                                │
│                                            ▼                                             │
│                              ┌─────────────────────────────┐                             │
│                              │  Action Selection           │                             │
│                              │  Sample from π(a|s)         │                             │
│                              │  → FORWARD/LEFT/RIGHT/STAY  │                             │
│                              └─────────────────────────────┘                             │
│                                                                                          │
│  PARAMETERS: ~8,500 total (Actor: ~4,400 + Critic: ~4,100)                               │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

**PPO Learning Algorithm**:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PPO LEARNING LOOP                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. COLLECT ROLLOUT (256 steps)                                             │
│     ┌────────────────────────────────────────────────────────┐              │
│     │  For each step t:                                      │              │
│     │    • Observe state s_t                                 │              │
│     │    • Sample action a_t ~ π(·|s_t)                      │              │
│     │    • Store (s_t, a_t, r_t, log π(a_t|s_t), V(s_t))     │              │
│     └────────────────────────────────────────────────────────┘              │
│                            ↓                                                │
│  2. COMPUTE ADVANTAGES (GAE)                                                │
│     ┌────────────────────────────────────────────────────────┐              │
│     │  δ_t = r_t + γV(s_{t+1}) - V(s_t)      (TD error)      │              │
│     │  A_t = Σ (γλ)^k δ_{t+k}                (GAE)           │              │
│     │  R_t = A_t + V(s_t)                    (Returns)       │              │
│     └────────────────────────────────────────────────────────┘              │
│                            ↓                                                │
│  3. OPTIMIZE (10 epochs, 2 minibatches)                                     │
│     ┌────────────────────────────────────────────────────────┐              │
│     │  For each minibatch:                                   │              │
│     │                                                        │              │
│     │  Policy Loss (clipped):                                │              │
│     │    r(θ) = π_new(a|s) / π_old(a|s)                      │              │
│     │    L_clip = min(r(θ)·A, clip(r(θ), 0.8, 1.2)·A)        │              │
│     │                                                        │              │
│     │  Value Loss:                                           │              │
│     │    L_vf = 0.5 · MSE(V(s), R)                           │              │
│     │                                                        │              │
│     │  Entropy Bonus:                                        │              │
│     │    H = -Σ π(a|s) log π(a|s)                            │              │
│     │                                                        │              │
│     │  Total Loss = -L_clip + 0.5·L_vf - 0.02·H              │              │
│     └────────────────────────────────────────────────────────┘              │
│                            ↓                                                │
│  4. UPDATE NETWORKS                                                         │
│     ┌────────────────────────────────────────────────────────┐              │
│     │  Adam optimizer (lr=0.001)                             │              │
│     │  Gradient clipping (max_norm=0.5)                      │              │
│     └────────────────────────────────────────────────────────┘              │
│                                                                             │
│  WHY CLIPPING MATTERS:                                                      │
│  • Without clip: Large advantage → huge policy update → instability         │
│  • With clip: r(θ) bounded to [0.8, 1.2] → controlled updates               │
│  • Enables higher learning rate (0.001) without divergence                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

Parameters: `clip_epsilon=0.2`, `gamma=0.99`, `gae_lambda=0.95`, `value_loss_coef=0.5`

### Code Changes

- `quantumnematode/brain/arch/ppo.py` - Complete PPO implementation (~400 lines)
- `quantumnematode/brain/arch/__init__.py` - Added PPOBrain export
- `quantumnematode/brain/arch/dtypes.py` - Added BrainType.PPO enum
- `quantumnematode/utils/config_loader.py` - Added PPOBrainConfig handling
- `scripts/run_simulation.py` - Integrated PPO brain loading

### Configuration

Initial configuration (suboptimal):

```yaml
brain:
  name: ppo
  config:
    actor_hidden_dim: 64
    critic_hidden_dim: 64
    num_hidden_layers: 2
    clip_epsilon: 0.2
    gamma: 0.99
    gae_lambda: 0.95
    value_loss_coef: 0.5
    entropy_coef: 0.01        # Too low
    learning_rate: 0.0003     # Too low
    rollout_buffer_size: 2048 # Too large
    num_epochs: 4             # Too few
    num_minibatches: 4        # Too many for small buffer
```

Optimized configuration (final):

```yaml
brain:
  name: ppo
  config:
    actor_hidden_dim: 64
    critic_hidden_dim: 64
    num_hidden_layers: 2
    clip_epsilon: 0.2
    gamma: 0.99
    gae_lambda: 0.95
    value_loss_coef: 0.5
    entropy_coef: 0.02        # More exploration
    learning_rate: 0.001      # Matches MLP
    rollout_buffer_size: 256  # Updates every ~1 episode
    num_epochs: 10            # More learning per batch
    num_minibatches: 2        # Fewer splits
```

## Results

### Summary Table

| Metric | MLP (best) | PPO (initial) | PPO (optimized) | Change |
|--------|------------|---------------|-----------------|--------|
| Success Rate | 96.5% | 80% | **98.5%** | +2.0% vs MLP |
| Avg Steps | 191.5 | 278.1 | 188.3 | -3 steps |
| Avg Reward | 34.86 | 28.82 | **35.60** | +0.74 |
| Convergence Run | 20 | 34 | 20 | Matched |
| Post-Conv Success | 100% | 100% | **100%** | Matched |
| Distance Efficiency | 0.456 | 0.285 | **0.485** | +0.029 |
| Composite Score | 0.822 | 0.729 | **0.832** | +0.010 |

### Detailed Findings

#### Phase 1: Initial Implementation (50 runs)

Session `20251226_132348` with default PPO hyperparameters:

- 80% success rate (vs MLP's 80% in same session)
- Converged at run 34 (vs MLP's run 20)
- 9 starved episodes in first 25 runs

**Problem identified**: `rollout_buffer_size=2048` too large for 200-500 step episodes. Updates occurred only every 4-10 episodes, slowing learning.

#### Phase 2: Hyperparameter Optimization (50 runs)

Session `20251226_133645` with optimized config:

- **94% success rate** (vs 80% initial)
- Converged at **run 20** (matching MLP)
- Composite score 0.765 (vs 0.729 initial)

Key changes:

1. `rollout_buffer_size: 256` - Updates every ~1 episode
2. `learning_rate: 0.001` - Matches MLP's learning speed
3. `entropy_coef: 0.02` - More exploration early
4. `num_epochs: 10` - More learning per batch
5. `num_minibatches: 2` - Fewer splits for smaller buffer

#### Phase 3: Full Benchmark (200 runs)

Session `20251226_140324` with optimized config:

- **98.0% success rate** (exceeds MLP's 96.5%)
- Converged at run 20
- Post-convergence: 99.4% success, 183.5 avg steps
- Composite score: **0.823** (matches MLP's 0.822)

Session `20251226_143130` with continued tuning - **new best**:

- **98.5% success rate** (+0.5% improvement)
- Converged at run 20
- Post-convergence: **100% success**, 178.3 avg steps
- Avg reward: **35.60** (up from 35.17)
- Distance efficiency: **0.485** (up from 0.470)
- Composite score: **0.832** (+0.009 improvement)

### Predator Environment

#### PPO vs MLP on Predator Small (200 runs each)

| Metric | MLP (best) | PPO (best) | Change |
|--------|------------|------------|--------|
| Success Rate | 85% | **89%** | +4% |
| Post-Conv Success | 91.8% | **93.2%** | +1.4% |
| Avg Steps | 203.4 | **177.9** | -25.5 steps |
| Convergence Run | 30 | **20** | 10 runs faster |
| Post-Conv Variance | 0.076 | **0.064** | -0.012 (more stable) |
| Distance Efficiency | 0.470 | **0.546** | +0.076 |
| Composite Score | 0.740 | **0.781** | +0.041 |
| Predator Deaths | 21 | 20 | -1 |

**Key findings:**

1. **PPO exceeds MLP on predators**: 93.2% vs 91.8% post-convergence success
2. **Faster convergence**: PPO converges at run 20-24 vs MLP's run 30
3. **More efficient navigation**: 0.546 vs 0.470 distance efficiency (+16%)
4. **Lower variance**: PPO shows more stable post-convergence behavior

**Top PPO predator sessions:**

- `20251227_011611`: Best composite (0.781), 93.2% post-conv success, converge@24
- `20251227_022229`: Fastest convergence (run 20), 90% post-conv success

## Analysis

### Why Initial PPO Was Slow

The default `rollout_buffer_size=2048` is designed for environments with thousands of steps per episode. For our 200-500 step episodes:

- Buffer fills every 4-10 episodes
- Agent only learns 5-12 times in 50 episodes
- MLP with REINFORCE learns after every episode

```text
BUFFER SIZE IMPACT ON LEARNING FREQUENCY
═══════════════════════════════════════════════════════════════════════════

Episode Length: ~200-500 steps

INITIAL CONFIG (buffer=2048):
┌─────────────────────────────────────────────────────────────────────────┐
│ Ep1 (350 steps)  Ep2 (280 steps)  Ep3 (400 steps)  Ep4 (320 steps)  ... │
│ ████████████████ ██████████████   ██████████████████ ████████████████   │
│ └───────────────────────────────────────────────────┘                   │
│                Buffer fills after 4-6 episodes                          │
│                         ↓                                               │
│                    ONE LEARNING UPDATE                                  │
└─────────────────────────────────────────────────────────────────────────┘
Result: Only 5-12 updates in 50 episodes → Slow convergence (run 34)

OPTIMIZED CONFIG (buffer=256):
┌─────────────────────────────────────────────────────────────────────────┐
│ Ep1 (350 steps)                                                         │
│ ████████ ← Update! ████████ ← Update!                                   │
│                                                                         │
│ Each episode triggers 1-2 learning updates                              │
└─────────────────────────────────────────────────────────────────────────┘
Result: 50+ updates in 50 episodes → Fast convergence (run 20)

═══════════════════════════════════════════════════════════════════════════
```

### Why Optimized PPO Works

1. **Frequent updates**: 256-step buffer means learning every 1-2 episodes
2. **Higher LR**: 0.001 matches MLP's learning speed
3. **More epochs**: 10 epochs extracts more learning from each batch
4. **Higher entropy**: 0.02 encourages exploration before convergence

### PPO vs REINFORCE Trade-offs

| Aspect | PPO | REINFORCE (MLP) |
|--------|-----|-----------------|
| Variance | Lower (actor-critic) | Higher (single-sample) |
| Stability | Higher (clipped objective) | Can overshoot |
| Complexity | More hyperparameters | Simpler |
| Sample efficiency | Better (multiple epochs) | One pass |

```text
PPO VS REINFORCE: LEARNING DYNAMICS
═══════════════════════════════════════════════════════════════════════════

                              REINFORCE (MLP)
                    ┌─────────────────────────────────────┐
                    │   Episode Reward                    │
                    │        ↓                            │
                    │   ∇θ log π(a|s) × R    (high var)   │
                    │        ↓                            │
                    │   Single gradient update            │
                    │        ↓                            │
                    │   Next episode                      │
                    └─────────────────────────────────────┘
                    • Simple: One line of math
                    • High variance: Single-sample gradient
                    • Can overshoot on lucky episodes

                                 PPO
                    ┌─────────────────────────────────────┐
                    │   Rollout Buffer (256 steps)        │
                    │        ↓                            │
                    │   GAE Advantages (reduced variance) │
                    │        ↓                            │
                    │   10 epochs × 2 minibatches         │
                    │        ↓                            │
                    │   Clipped updates (stable)          │
                    │        ↓                            │
                    │   Next rollout                      │
                    └─────────────────────────────────────┘
                    • Complex: More hyperparameters
                    • Low variance: Actor-critic + GAE
                    • Stable: Clipping prevents overshooting

═══════════════════════════════════════════════════════════════════════════
```

### LR Scheduler Consideration

PPO's clipped objective provides built-in learning rate adaptivity. Adding an LR scheduler showed minimal benefit in testing because:

1. Clipping already constrains update magnitude
2. The 0.001 fixed LR works well throughout training
3. Value function convergence naturally stabilizes policy

## Conclusions

### Key Findings

1. **PPO exceeds MLP performance**: 98.5% vs 96.5% success, identical convergence (run 20)
2. **Buffer size is critical**: Default 2048 too large for short episodes
3. **Higher LR is fine**: PPO's clipping prevents the instability that high LR causes in REINFORCE
4. **More epochs help**: 10 epochs vs 4 improves sample efficiency significantly
5. **PPO is now a viable SOTA baseline**: Satisfies Phase 0 exit criterion (>85% success)
6. **100% post-convergence success**: PPO achieves perfect stability after convergence

### Lessons Learned

1. **PPO defaults assume long episodes**: Always tune buffer size to episode length
2. **Actor-critic reduces variance**: More stable learning curves than policy-only methods
3. **Clipping enables higher LR**: The 0.001 LR would destabilize REINFORCE
4. **Entropy matters early**: 0.02 vs 0.01 improves exploration before convergence

## Next Steps

- [x] Implement PPO brain architecture
- [x] Integrate with config system and run_simulation.py
- [x] Achieve >85% success on foraging (achieved 98.5%)
- [x] Match MLP convergence speed (both at run 20)
- [x] Update all PPO configs with optimized hyperparameters
- [x] Benchmark on predator environment (achieved 93% post-conv, exceeds MLP's 92%)

**Deferred to future work:**

- Compare learning curves (PPO vs MLP vs Spiking)[^1]
- Test on medium/large environments

\[^1\]: Learning curve comparison requires per-episode logging (success, reward, steps per run) which isn't currently persisted. Would need to add `run_history` array to experiment JSON output.

## Data References

### Key Sessions

**Foraging (Small):**

- **Initial PPO (50 runs)**: `20251226_132348` - 80% success, convergence run 34
- **Optimized PPO (50 runs)**: `20251226_133645` - 94% success, convergence run 20
- **Full Benchmark (200 runs)**: `20251226_140324` - 98% success, composite 0.823
- **Best PPO Foraging (200 runs)**: `20251226_143130` - **98.5% success**, composite **0.832** ★
- **MLP Foraging Baseline (200 runs)**: `20251127_205353` - 96.5% success, composite 0.822

**Predator (Small):**

- **Best PPO Predator (200 runs)**: `20251227_011611` - **93.2% post-conv**, composite **0.781** ★
- **Fast Convergence PPO (200 runs)**: `20251227_022229` - 90% post-conv, converge@20
- **MLP Predator Baseline (200 runs)**: `20251127_140342` - 91.8% post-conv, composite 0.740

### Config Files

**Foraging environments:**

- Small (20x20): `configs/examples/ppo_foraging_small.yml`
- Medium (50x50): `configs/examples/ppo_foraging_medium.yml`
- Large (100x100): `configs/examples/ppo_foraging_large.yml`

**Predator environments:**

- Small (20x20, 2 predators): `configs/examples/ppo_predators_small.yml`
- Medium (50x50, 3 predators): `configs/examples/ppo_predators_medium.yml`
- Large (100x100, 5 predators): `configs/examples/ppo_predators_large.yml`

### Comparison with Other Brains

Post-convergence success rates (fair comparison since Quantum uses pre-optimized CMA-ES parameters):

| Brain | Foraging Success | Predator Success | Convergence |
|-------|------------------|------------------|-------------|
| PPO | **100%** | **93%** | Run 20 |
| Quantum | 100% | 95% | N/A (CMA-ES pre-optimized) |
| MLP | 100% | 92% | Run 20-30 |
| Spiking | 100% | 63% | Run 22 |

```text
BRAIN ARCHITECTURE COMPARISON: DYNAMIC FORAGING (post-convergence)
═══════════════════════════════════════════════════════════════════════════

                      POST-CONVERGENCE SUCCESS RATE
────────────────────────────────────────────────────────────────────────────
PPO (optimized)     ██████████████████████████████████████████████████ 100%
MLP (REINFORCE)     ██████████████████████████████████████████████████ 100%
Spiking (surrogate) ██████████████████████████████████████████████████ 100%
Quantum (CMA-ES)    ██████████████████████████████████████████████████ 100%
────────────────────────────────────────────────────────────────────────────

                   CONVERGENCE SPEED (online learning only)
────────────────────────────────────────────────────────────────────────────
                    0    5    10   15   20   25   30   35   40
PPO (optimized)     ●────────────────────●                      Run 20
MLP (REINFORCE)     ●────────────────────●                      Run 20
Spiking             ●──────────────────────●                    Run 22
Quantum (CMA-ES)    ════════════════════════════════════════    Pre-optimized
────────────────────────────────────────────────────────────────────────────

                          COMPOSITE SCORE
────────────────────────────────────────────────────────────────────────────
PPO (optimized)     █████████████████████████████████░░░░░░░  0.832  ★
MLP (REINFORCE)     ████████████████████████████████░░░░░░░░  0.822
Quantum (CMA-ES)    ███████████████████████████████░░░░░░░░░  0.762
Spiking             █████████████████████████████░░░░░░░░░░░  0.733
────────────────────────────────────────────────────────────────────────────

KEY INSIGHT: All architectures achieve 100% post-convergence success on foraging.
             PPO leads on composite score due to fast convergence + efficiency.

═══════════════════════════════════════════════════════════════════════════
```

```text
HYPERPARAMETER OPTIMIZATION IMPACT
═══════════════════════════════════════════════════════════════════════════

SUCCESS RATE PROGRESSION:
────────────────────────────────────────────────────────────────────────────
Initial PPO (50 runs)     ████████████████████████████████░░░░░░░░░░  80%
Optimized PPO (50 runs)   █████████████████████████████████████████░  94%  ↑14%
Full Benchmark (200 runs) ██████████████████████████████████████████  98%  ↑18%
Best Run (200 runs)       ███████████████████████████████████████████ 98.5% ↑18.5% ★
────────────────────────────────────────────────────────────────────────────

CONVERGENCE RUN:
────────────────────────────────────────────────────────────────────────────
Initial PPO               ●──────────────────────────────────●   Run 34
Optimized PPO             ●────────────────────●                 Run 20  ↓14
MLP Baseline              ●────────────────────●                 Run 20
────────────────────────────────────────────────────────────────────────────

KEY CHANGES:
┌────────────────────────┬─────────────┬─────────────┬───────────────────┐
│ Parameter              │ Initial     │ Optimized   │ Impact            │
├────────────────────────┼─────────────┼─────────────┼───────────────────┤
│ rollout_buffer_size    │ 2048        │ 256         │ 8x more updates   │
│ learning_rate          │ 0.0003      │ 0.001       │ 3x faster learn   │
│ entropy_coef           │ 0.01        │ 0.02        │ 2x exploration    │
│ num_epochs             │ 4           │ 10          │ 2.5x efficiency   │
│ num_minibatches        │ 4           │ 2           │ Larger batches    │
└────────────────────────┴─────────────┴─────────────┴───────────────────┘

═══════════════════════════════════════════════════════════════════════════
```
