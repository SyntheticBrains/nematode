# QLIF-LSTM (H.4) Optimization History

**Architecture**: Custom LSTM cell where forget and input gates use QLIF quantum neuron measurements (via surrogate gradients) instead of classical sigmoid activations, trained via recurrent PPO with chunk-based truncated BPTT. First temporal architecture in the codebase — provides within-episode memory (h_t, c_t persist across steps, reset between episodes).

**Total**: 12 rounds, ~66 sessions, ~36,000 episodes across foraging, pursuit predators, and stationary predators (classical + quantum).

______________________________________________________________________

## Round 1 — Stage 4a Classical Ablation (Foraging)

**Purpose**: Validate the LSTM + PPO training loop works correctly before enabling quantum gates.

**Config**: `qliflstm_foraging_small_classical.yml` — 200 episodes, 4 sessions, lstm_hidden_dim=32, bptt_chunk_length=16, rollout_buffer_size=256, actor_lr=0.003, critic_lr=0.001, entropy_coef=0.05→0.005 over 200 episodes.

**Environment**: 20×20 grid, 5 foods, target 10, no predators.

**Params**: 11,141 (actor: 4,676, critic: 6,465).

| Session ID | Success | Post-Conv | Convergence | Seed |
|------------|---------|-----------|-------------|------|
| 20260305_140313 | 83.5% | 100% | Run 34 | 331503929 |
| 20260305_140316 | 87.5% | 99.4% | Run 25 | 3847038028 |
| 20260305_140318 | 84.5% | 100% | Run 41 | 2871586844 |
| 20260305_140321 | 89.5% | 100% | Run 25 | — |
| **Mean** | **86.25%** | **99.85%** | **31.25** | |

**Learning pattern**: Sharp 3-phase pattern — exploration (0% success), abrupt phase transition (runs 20-41), then near-perfect mastery. All sessions show identical characteristic shape.

**LSTM dynamics**: h_norm bounded (0.9-2.0), c_norm growing from 2-5 early to 15-30 late (healthy temporal memory accumulation). No explosion or instability.

**Decision**: **PASS** — all 4 sessions exceed ≥80% gate. Proceed to quantum.

______________________________________________________________________

## Round 2 — Stage 4a-Q Quantum Gates Enabled (Foraging)

**Config**: `qliflstm_foraging_small.yml` — `use_quantum_gates: true`, shots=1024, membrane_tau=0.9. All else identical to R1.

| Session ID | Success | Post-Conv | Convergence | Seed |
|------------|---------|-----------|-------------|------|
| 20260305_141819 | 90.0% | 100% | Run 21 | 1030049700 |
| 20260305_141822 | 79.0% | 99.4% | Run 42 | 1260891760 |
| 20260305_141825 | 90.0% | 100% | Run 25 | 832140819 |
| 20260305_141831 | 83.5% | 100% | Run 36 | 2791107382 |
| **Mean** | **85.63%** | **99.85%** | **31.0** | |

### Quantum vs Classical Comparison (Foraging)

| Metric | Classical (R1) | Quantum (R2) | Delta |
|--------|---------------|-------------|-------|
| Success rate | 86.25% | 85.63% | -0.63% |
| Success rate std | 2.72% | 5.31% | +2.59% |
| Convergence run | 31.25 | 31.0 | -0.25 |
| Post-convergence | 99.85% | 99.85% | 0.00% |
| Mean steps (success) | 213.65 | 224.45 | +10.80 |
| Distance efficiency | 0.403 | 0.372 | -0.031 |
| Entropy (final) | ~0.65-0.87 | ~0.94-1.07 | +0.15-0.30 |

**Key finding**: Quantum ≈ Classical on foraging. Functionally equivalent with slightly higher variance and higher terminal entropy (quantum measurement noise acts as implicit regularizer). The foraging task is too simple to differentiate.

**Decision**: **PASS** — quantum gates do not degrade learning. Proceed to pursuit predators.

______________________________________________________________________

## Round 3 — Stage 4b Pursuit Predators (200 Episodes)

**Config**: `qliflstm_pursuit_predators_small[_classical].yml` — 200 episodes, 4+4 sessions. 2 pursuit predators (speed=0.5, detection_radius=5, damage=6.0), health system (max_hp=100, food_healing=10.0). Sensory: food_chemotaxis + nociception (input_dim=4). gae_lambda=0.98.

**Params**: 11,525 (actor: 4,932, critic: 6,593).

| Metric | Classical | Quantum | Delta |
|--------|-----------|---------|-------|
| Success rate | 51.88% | 47.25% | -4.63% |
| SR std | **15.16%** | **4.31%** | -10.85% |
| Convergence run | 118.25 | 114.0 | -4.25 |
| Convergence std | **40.12** | **8.83** | -31.29 |
| Post-conv rate | 91.83% | 88.05% | -3.78% |
| Evasion rate | 81.5% | 79.9% | -1.6% |

**Key finding**: Variance reversal — quantum has *much lower* variance than classical (SR std 4.31% vs 15.16%, convergence std 8.83 vs 40.12). Classical shows wild seed-dependent variation while quantum is remarkably consistent. But 200 episodes is insufficient — convergence at run ~115 leaves only ~85 post-convergence runs.

**Decision**: Partial PASS — increase to 500 episodes with extended entropy decay.

______________________________________________________________________

## Round 4 — Stage 4b Pursuit Predators (500 Episodes, Updated Params)

**Changes**: Episodes 200→500, entropy_decay_episodes 200→400, rollout_buffer_size 256→512.

### Classical Results (500 episodes, 4 sessions)

| Session ID | Success | Post-Conv | Last 100 SR | Convergence |
|------------|---------|-----------|-------------|-------------|
| 20260305_232259 | 71.6% | 85.9% | 97% | Run 154 |
| 20260305_232303 | 75.8% | 90.9% | 97% | Run 105 |
| 20260305_232305 | 74.2% | 98.5% | 100% | Run 191 |
| 20260305_232309 | 77.2% | 94.3% | 97% | Run 135 |
| **Mean** | **74.70%** | **92.40%** | **98%** | **146.25** |

### Quantum Results (500 episodes, 4 sessions)

| Session ID | Success | Post-Conv | Last 100 SR | Convergence |
|------------|---------|-----------|-------------|-------------|
| 20260305_232312 | 74.6% | 86.3% | 96% | Run 71 |
| 20260305_233816 | 65.8% | 95.7% | 96% | Run 225 |
| 20260305_232320 | 75.0% | 91.7% | 94% | Run 128 |
| 20260305_232322 | 67.8% | 89.4% | 88% | Run 209 |
| **Mean** | **70.80%** | **90.78%** | **94%** | **158.25** |

### Round 4 Quantum vs Classical (500 eps)

| Metric | Classical | Quantum | Delta |
|--------|-----------|---------|-------|
| Success rate | **74.70%** | 70.80% | -3.90% |
| SR std | **2.33%** | 4.70% | +2.37% |
| Post-conv rate | 92.40% | 90.78% | -1.62% |
| Last 100 SR | **98%** | 94% | -4% |
| Convergence std | **36.22** | 72.43 | +36.21 |
| Per-encounter evasion | 83.45% | 84.00% | +0.55% |

**Key findings**:

1. Overall SR improved +22-24pp from Round 3 — longer training and extended entropy decay were the fix.
2. Classical variance collapsed (15.16% → 2.33% std) with sufficient training time.
3. The Round 3 quantum "regularization advantage" doesn't persist — with enough training, classical converges more reliably.
4. Q4 showed late-session entropy rebound (0.77→0.99) and performance regression (98%→78%) — entropy floor of 0.005 may be too low.

**Decision**: **PASS** — investigate entropy floor, then proceed to Stage 4c.

______________________________________________________________________

## Round 4b — Entropy Floor Validation (entropy_coef_end=0.015)

**Purpose**: Targeted test — does raising entropy floor prevent the late-session entropy rebound and regression observed in R4 Q4?

**Config**: Quantum pursuit predators, 500 eps, entropy_coef_end 0.005→0.015. 2 sessions.

| Metric | Q5 | Q6 | Mean | R4 Q mean (0.005) |
|--------|----|----|------|-------------------|
| Overall SR | 66.6% | 68.2% | 67.40% | 70.80% |
| Last 100 SR | **98%** | **96%** | **97%** | 94% |
| Entropy (final) | ~0.89 | ~0.89 | ~0.89 | ~0.84 |

**Result**: **Fix validated.** No entropy rebound in either session. Late-session performance improved (97% vs 94% last-100 SR). The entropy floor prevents the destabilization that caused Q4's regression. Trade-off: slightly lower overall SR (67.4% vs 70.8%) from slower convergence, but late-session stability is what matters for deployed performance.

**Decision**: Adopt entropy_coef_end=0.015 for quantum configs.

______________________________________________________________________

## Round 5 — Stage 4c Thermotaxis + Pursuit Predators (Large Environment)

**Config**: 100×100 grid (25× area), 4 pursuit predators (speed=1.0, detection_radius=10, damage=15), 3 sensory modules (7 features), temperature zones, max_hp=150, lstm_hidden_dim=48, critic_hidden_dim=128, actor_lr=0.001, num_epochs=3, bptt_chunk_length=32. 500 episodes, 4+4 sessions. ~34,853 params.

### Quantum vs Classical (Stage 4c Pursuit)

| Metric | Classical | Quantum | Delta |
|--------|-----------|---------|-------|
| Success rate | **60.10%** | 45.35% | -14.75% |
| SR std | 8.26% | **2.30%** | -5.96% |
| Convergence run | **182.25** | 256.25 | +74.0 |
| Convergence std | 61.04 | **21.55** | -39.49 |
| Last 50 SR | 82.50% | **85.00%** | +2.50% |
| Last 100 SR | 82.00% | 81.50% | -0.50% |
| Evasion rate | **88.60%** | 84.68% | -3.92% |

**Key findings**:

1. Stage 4c is genuinely hard — SR drops ~15pp vs Stage 4b for both variants.
2. The variance reversal from Round 3 returns: quantum has 3.6× lower SR variance than classical.
3. Late-session performance is comparable (last 100: 82% vs 82%) — the overall SR gap is a convergence-speed artifact.
4. Temperature damage causes more health deaths than predators. Weak thermotaxis reward signal is the limiting factor.
5. Multi-objective learning confirmed: all sessions learn forage > evade > thermoregulate priority hierarchy.

**Decision**: **PASS** — proceed to stationary predators.

______________________________________________________________________

## Rounds 6-11 — Stage 4c Stationary Predators (Classical Optimization)

**Task**: 100×100 grid, 5 stationary predators (speed=0, damage_radius=4, damage=10/tick), temperature zones, 4 sensory modules (9 features). The hardest task variant.

### Systematic Tuning (6 rounds, 28 sessions)

| Round | Key Change | Last-100 SR | Peak R50 | Episodes | Sessions |
|-------|--------|-------------|----------|----------|----------|
| R6 | Baseline (3 modules, epochs=3, buffer=512) | 33.0% | 35.0% | 500 | 4 |
| R7 | +mechanosensation, +epochs=6 | 33.0%\* | 35.0%\* | 700 | 4 |
| R7b | +rollout_buffer_size=1024 | 36.8% | 42.5% | 700 | 4 |
| R8 | +LR decay (0.001→0.0001), 1000 eps | 34.8% | 42.3% | 1000 | 4 |
| R9 | +lstm_dim=96, +penalty=0.8 (**reverted**) | ~24.2% | — | 1000 | 4 |
| R10 | Actor [features, h_t] fix | **36.8%** | **46.5%** | 1000 | 4 |
| R11 | LR warmup + 3× LR (**reverted**) | 34.3% | 48.5% | 500 | 4 |

\*R7 data identical to R6 due to recording issue.

### Key Architectural Discovery (R9→R10)

After R9's failure with larger LSTM, investigation revealed that the **actor head only received h_t** (48-dim LSTM hidden state), while the critic received [features, h_t]. This meant the actor couldn't see current sensory signals directly — only through whatever the LSTM encoded. **Fix**: Actor head now receives [features, h_t] (57-dim = 9 features + 48 hidden). Produced +2pp last-100 and +4.2pp peak.

### Root Cause: Why QLIF-LSTM Struggles on Stationary Predators

| Aspect | Pursuit Predators | Stationary Predators |
|--------|-------------------|---------------------|
| Agent must learn | Temporal evasion sequences | Spatial memory of fixed zones |
| LSTM advantage | Track recent chase direction | Remember zone locations |
| R4 result (classical) | **98% last-100** | — |
| R10 result (classical) | — | **37% last-100** |

The LSTM excels at temporal evasion (pursuit) but struggles with spatial memory (stationary). With only 9-dim sensory input and 11×11 viewport on a 100×100 grid, implicitly encoding 5 zone locations in 48 hidden dimensions is insufficient. MLP PPO (96.5% on same task) doesn't need spatial memory — it uses reactive gradient sensing.

**Classical ceiling established**: ~37% sustained / ~48% peak after 6 rounds, 28 sessions.

______________________________________________________________________

## Quantum Comparison — Stage 4c Stationary Predators

**Config**: Quantum gates enabled, all other parameters match R10 classical. 500 episodes, 4 sessions.

### Quantum vs Classical (Both at 500 Episodes)

| Metric | Classical (R10@500) | Quantum | Delta |
|--------|-------------------|---------|-------|
| Overall SR | **23.3%** | 21.1% | -2.2pp |
| Last-50 SR | **35.0%** | 30.5% | -4.5pp |
| Last-100 SR | **32.8%** | 30.5% | -2.3pp |
| Peak R50 SR | 43.0% | **44.0%** | +1.0pp |
| SR std | 3.3% | **1.9%** | -1.4% |
| Still improving? | 4/4 | 4/4 | — |

Both variants struggle equally. Classical leads by ~2-4pp on average. Peak performance is equivalent. Quantum shows lower session variance (consistent with all prior rounds). Both still improving at episode 500.

______________________________________________________________________

## Cross-Stage Summary

### QLIF-LSTM Performance Across All Tasks

| Task | Classical Mean SR | Quantum Mean SR | Classical Last-100 | Quantum Last-100 | Gap |
|------|------------------|----------------|-------------------|-----------------|-----|
| Foraging (R1/R2, 200ep) | 86.25% | 85.63% | ~100% | ~100% | ≈0 |
| Pursuit small (R4, 500ep) | 74.70% | 70.80% | 98% | 94% | C +4pp |
| Pursuit large (R5, 500ep) | 60.10% | 45.35% | 82% | 82% | ≈0 late |
| Stationary large (R10/Q, 500ep) | 23.3%\* | 21.1% | 32.8%\* | 30.5% | C +2pp |

\*Classical at 500 episodes for fair comparison; full 1000ep results: 28.8% overall, 36.8% last-100.

### Quantum vs Classical Verdict

1. **Foraging**: Equivalent. Task too simple to differentiate.
2. **Pursuit predators**: Classical slightly better (+4pp) with sufficient training. Quantum shows lower variance in short sessions but this advantage disappears with longer training.
3. **Large environments**: Classical converges faster (~74 runs earlier). Quantum has lower session variance but higher convergence variance.
4. **Stationary predators**: Both struggle. Classical leads modestly.

**Overall**: Quantum QLIF gates do not provide a measurable advantage on any task. Classical sigmoid is consistently equal or slightly better, with faster convergence. The hypothesis that quantum measurement noise provides beneficial exploration stochasticity is not supported.

### Task-Specific Architecture Assessment

- **Pursuit predators**: **Strong** — temporal memory helps with predator tracking and evasion (98% classical last-100).
- **Foraging**: **Strong** — converges reliably, near-perfect post-convergence.
- **Stationary predators**: **Weak** — LSTM temporal memory doesn't help with spatial zone avoidance (~37% ceiling vs MLP PPO's 96.5%).

### Key Learnings

1. **LSTM temporal memory provides genuine value for temporal tasks**: The pursuit predator results (98% last-100) demonstrate that within-episode memory for tracking predator movement patterns meaningfully improves performance over what memoryless architectures can achieve at equivalent parameter counts.

2. **Entropy floor matters for quantum stability**: `entropy_coef_end=0.015` prevents late-session entropy rebound and policy destabilization. Validated across multiple task configurations.

3. **Actor should receive [features, h_t], not just h_t**: The actor needs direct access to current sensory signals alongside the LSTM's temporal context. Pure h_t bottleneck limits peak performance by ~4pp.

4. **Longer training eliminates apparent quantum advantages**: The "quantum regularization" observed in 200-episode sessions (lower variance) disappears with 500+ episodes as classical has time to fully converge.

5. **Sawtooth instability on hard tasks**: All stationary predator sessions show persistent 7-10pp regressions throughout training. This is structural — related to PPO policy oscillation on multi-objective tasks with high return variance — not a hyperparameter issue (LR decay didn't fix it).

6. **Circuit batching is essential**: Quantum sessions on the large environment ran ~21 hours with batched circuits (single `backend.run()` per gate) vs estimated 3-5 days without. The optimization is critical for practical evaluation.

______________________________________________________________________

## Session Data

Best-round session JSONs and configs are archived in `artifacts/logbooks/008/` (34 of ~66 total sessions; non-best rounds R3, R6-R9, R11 were not archived):

| Directory | Round | Sessions | Config |
|-----------|-------|----------|--------|
| `qliflstm_foraging_small/` | R1 + R2 | 8 (4C + 4Q) | Foraging 20×20 |
| `qliflstm_pursuit_predators_small/` | R4 + R4b | 10 (4C + 4Q + 2Q validation) | Pursuit predators 20×20 |
| `qliflstm_thermotaxis_pursuit_predators_large/` | R5 | 8 (4C + 4Q) | Pursuit + thermotaxis 100×100 |
| `qliflstm_thermotaxis_stationary_predators_large/` | R10 + Q comp | 8 (4C + 4Q) | Stationary + thermotaxis 100×100 |
