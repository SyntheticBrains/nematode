## Context

QA-7 tests whether PQC (Parameterized Quantum Circuit) unitarity prevents catastrophic forgetting during sequential multi-objective training. This is the final quantum experiment at current environment complexity before the project pivots to environment enrichment.

The existing training infrastructure (`scripts/run_simulation.py`) runs a single-objective session: one config, one environment, N episodes. There is no built-in support for mid-session environment switching. However, the `QuantumNematodeAgent` already supports `reset_environment()` which recreates the environment while preserving the brain instance and its weights. The HybridQuantum multi-stage training pattern (Stage 1 → 2 → 3) demonstrates weight persistence across configs via file-based checkpointing, though each stage is a separate script invocation.

The five test architectures (QRH, CRH, HybridQuantum, HybridClassical, MLP PPO) all use PyTorch internally and support `state_dict()` / `load_state_dict()` through their respective implementations. There is no universal checkpoint interface on the `Brain` protocol — save/load is architecture-specific.

## Goals / Non-Goals

**Goals:**

- Implement a sequential training protocol that trains a brain on Objective A → B → C → A, preserving weights across transitions
- Measure backward forgetting, forward transfer, and plasticity retention at each transition
- Compare quantum architectures (QRH, HybridQuantum) against classical controls (CRH, HybridClassical, MLP PPO) with statistical rigour (4-8 seeds, t-tests)
- Export per-phase metrics to CSV for analysis and visualisation
- Reuse existing brain architectures, environments, and configs without modification

**Non-Goals:**

- Modifying any brain architecture implementation
- Modifying the environment or reward system
- Building a general-purpose continual learning framework (this is a one-off evaluation protocol)
- Implementing new visualisations beyond CSV export (matplotlib plots can be done post-hoc)
- Running on QPU hardware (simulator only for this evaluation)

## Decisions

### 1. New standalone script vs extending run_simulation.py

**Decision**: New standalone script `scripts/run_plasticity_test.py`

**Rationale**: `run_simulation.py` is 1100+ lines handling single-objective sessions with rich plotting, manyworlds mode, and incremental CSV writers. Extending it to handle multi-phase sequential training would add significant complexity to an already large file. A standalone script can be purpose-built for the sequential protocol, reusing the same agent/brain/environment construction utilities.

**Alternative considered**: Adding a `--plasticity-mode` flag to `run_simulation.py`. Rejected because the control flow is fundamentally different (multiple environment configs per session, metric snapshots between phases, aggregate comparison across architectures).

### 2. Environment switching mechanism

**Decision**: Reconstruct the `QuantumNematodeAgent` for each phase with the same brain instance but different environment config.

**Rationale**: The agent's `reset_environment()` method recreates the environment, but the agent constructor also sets up reward calculators, satiety managers, and other components that are environment-specific. Rather than partially reconstructing these internals, it's cleaner to build a new agent for each phase while passing the existing brain. This is exactly the pattern used in HybridQuantum's multi-stage training (Stage 1 → 2 → 3 each create fresh configs but load prior weights).

The brain survives across phases because it is constructed once and passed by reference. No serialisation needed between phases — the brain object persists in memory. Weight checkpoints are saved to disk at each transition for reproducibility and debugging.

### 3. Brain state checkpointing approach

**Decision**: Use disk-based checkpointing only (`torch.save` of `state_dict()` dicts) at each transition point. No in-memory snapshots via `brain.copy()`.

**Rationale**: `MLPPPOBrain.copy()` raises `NotImplementedError`, so we cannot rely on `copy()` across all five architectures. Disk checkpoints are sufficient — we save before each phase transition and can reload if needed for debugging or reproducibility. In-memory snapshots are unnecessary since eval blocks measure the *current* brain state, not a prior snapshot.

For architectures with multiple components (HybridQuantum has reflex + cortex + critic), we use their existing `save_*_weights()` / `load_*_weights()` methods. For simpler architectures (QRH, CRH, MLP PPO), we save the full `state_dict()` of their PyTorch modules plus optimizer state into a single checkpoint dict.

### 4. Eval mode mechanism

**Decision**: The plasticity script controls eval mode externally by running episodes without calling `learn()` or `post_process_episode()` on the brain. No modifications to brain internals needed.

**Rationale**: None of the 5 target brains implement an eval/train toggle. Rather than adding eval mode flags to each architecture (modifying existing code, violating the non-goal of no brain changes), the plasticity script's eval runner simply runs the agent episode loop in a "collect metrics only" mode — calling `run_brain()` for action selection but skipping all learning calls. Since the script controls the training loop, this is straightforward.

Before eval: save optimizer `state_dict()` and clear PPO buffer via `buffer.reset()`. After eval: restore optimizer state and clear buffer again to prevent eval experience from leaking into training updates.

### 5. Metric computation approach

**Decision**: Run a fixed-length evaluation block (50 episodes) at each transition point using the eval mode above, then resume training.

**Rationale**: Measuring performance *during* training conflates learning dynamics with task performance. A clean eval block at each transition gives us uncontaminated measurements.

**Full evaluation matrix** (which objectives are evaluated at which transition points):

| Transition Point | Eval on A (foraging) | Eval on B (pursuit) | Eval on C (thermo+pursuit) |
|---|---|---|---|
| Pre-training (random baseline) | Yes | Yes | — |
| Post-A (after foraging training) | Yes | Yes | — |
| Post-B (after pursuit training) | Yes | Yes | — |
| Post-C (after thermo+pursuit training) | Yes | — | Yes |
| Post-A' (after foraging retraining) | Yes | — | — |

Eval on A at every point tracks backward forgetting. Eval on B at pre-training and post-A measures forward transfer. Eval on C only at post-C (task competence). Eval on B/C at post-A' is omitted as it adds runtime without addressing the core hypothesis.

**Key metrics**:

- **Backward Forgetting (BF)** = `post_A_score - post_C_score_on_A` (how much A degrades after training B and C)
- **Forward Transfer (FT)** = `post_A_eval_on_B - random_baseline_on_B` (does A-training help B?)
- **Plasticity Retention (PR)** = `post_A'_convergence_rate / post_A_convergence_rate` (can it relearn A as fast?)

### 6. Evaluation episodes per phase vs training episodes

**Decision**: 200 training episodes per objective, 50 evaluation episodes at each transition.

**Rationale**: 200 episodes is sufficient for convergence on our environments based on prior experiments (QRH converges by episode ~100 on foraging, MLP PPO by ~50). 50 evaluation episodes provide a stable mean with reasonable variance. Total per seed: ~1250 episodes (200×4 training phases + 50×(5 A-evals + 2 B-evals + 1 C-eval) = 200×4 + 50×8 = 1200). Manageable for 5 architectures × 8 seeds = 40 runs.

### 7. Objective sequence and grid size

**Decision**: Foraging → Pursuit Predators → Thermotaxis+Pursuit → return to Foraging. All phases on 100×100 grid.

**Rationale**: This sequence increases complexity and tests progressively different skills:

- **A (Foraging)**: Pure food-seeking. Well-understood baseline, all architectures converge reliably.
- **B (Pursuit Predators)**: Adds evasion to foraging. Tests whether foraging knowledge transfers.
- **C (Thermotaxis+Pursuit)**: Adds temperature navigation. Most complex multi-objective task.
- **A' (Foraging return)**: Measures how much of the original skill was forgotten and how fast it can be relearned.

**Grid size**: 100×100 for all phases. Using a uniform grid eliminates grid-size confounds when comparing eval scores across transition points. The 100×100 grid is necessary for phase C (thermotaxis+pursuit is too crowded on 20×20) and provides better dynamic range for measuring forgetting — if 20×20 foraging is trivially easy (95%+), there's little room to detect forgetting differences between quantum and classical. Harder tasks amplify any plasticity differences. Prior benchmarks on 100×100 foraging-only don't exist, but this doesn't matter — we're comparing quantum vs classical forgetting deltas, not absolute performance against external baselines.

**Alternative considered**: 20×20 for A/B and 100×100 for C. Rejected because eval-on-A scores at different transition points would not be directly comparable if the A-eval grid differs from the C-training grid (different absolute difficulty levels confound the forgetting measurement).

### 8. Config structure

**Decision**: One YAML config per architecture defining all four phases, rather than separate configs per phase. Each phase includes both environment and reward config.

**Rationale**: Plasticity testing requires consistent hyperparameters across phases — same learning rate, same architecture dimensions, same training episodes. A single config file per architecture ensures consistency and makes it easy to reproduce. Phase-specific environment and reward parameters are embedded as sections within the config.

```yaml
# configs/studies/plasticity/qrh_plasticity.yml
brain:
  name: qrh
  config:
    # ... architecture params (same across all phases)

plasticity:
  training_episodes_per_phase: 200
  eval_episodes: 50
  seeds: [42, 123, 256, 512, 789, 1024, 2048, 4096]
  phases:
    - name: foraging
      environment: { grid_size: 100, foraging: {...}, predators: { enabled: false } }
      reward: { food_reward: 2.0, step_penalty: -0.01, ... }
    - name: pursuit_predators
      environment: { grid_size: 100, foraging: {...}, predators: { enabled: true, ... } }
      reward: { food_reward: 2.0, predator_penalty: -0.15, death_penalty: -10.0, ... }
    - name: thermotaxis_pursuit
      environment: { grid_size: 100, foraging: {...}, predators: {...}, thermotaxis: {...} }
      reward: { food_reward: 2.0, predator_penalty: -0.15, comfort_reward: 0.05, ... }
    - name: foraging_return
      environment: { grid_size: 100, foraging: {...}, predators: { enabled: false } }
      reward: { food_reward: 2.0, step_penalty: -0.01, ... }
```

### 9. Single architecture per invocation

**Decision**: The plasticity script runs one architecture per invocation. Cross-architecture comparison is a separate post-hoc step.

**Rationale**: Each architecture has different config requirements (HybridQuantum needs pre-trained weights, QRH has reservoir params, etc.). Running one at a time keeps the script simple and allows independent execution/parallelisation. The aggregate CSV from each run contains all the data needed for cross-architecture comparison. A lightweight post-hoc script (or manual analysis) combines the aggregate CSVs and computes forgetting ratios + t-tests.

### 10. Statistical comparison

**Decision**: Per-architecture mean ± std across seeds, two-sample t-test between quantum and classical pairs (QRH vs CRH, HybridQuantum vs HybridClassical), plus MLP PPO as overall classical baseline.

**Rationale**: Paired comparisons (quantum vs its classical ablation) control for architectural differences. The ≤50% forgetting threshold from the proposal translates to: `mean_BF_quantum / mean_BF_classical ≤ 0.5` with p < 0.05.

## Risks / Trade-offs

**[Risk] Some architectures may not converge within 200 episodes on 100×100 grids** → Mitigation: Prior benchmarks used 100×100 for pursuit/thermotaxis tasks. Foraging-only on 100×100 is untested but structurally simpler. Pre-test with 1 seed before running the full 8-seed campaign. If convergence is too slow, increase `training_episodes_per_phase` (the config makes this trivial to adjust).

**[Risk] Evaluation episodes may interfere with training state (optimizer momentum, learning rate schedules)** → Mitigation: Save and restore optimizer `state_dict()` before/after eval blocks. Clear PPO `buffer.reset()` before eval and after eval to prevent eval experience from leaking into training updates.

**[Risk] HybridQuantum's multi-stage training complicates the protocol** → Mitigation: Use HybridQuantum in Stage 3 (joint fine-tune) mode with pre-trained weights from existing artifacts. This way it behaves as a single unified model during the plasticity test, same as all other architectures.

**[Risk] MLPPPOBrain.copy() raises NotImplementedError** → Mitigation: Don't use `brain.copy()` at all. Use disk-based checkpointing only (`torch.save` of state dicts). This works uniformly for all architectures.

**[Trade-off] 50 eval episodes × 8 eval points adds ~33% overhead to total runtime** → Accepted: Clean measurement is worth the cost. Without eval blocks, training metrics conflate learning and forgetting dynamics.
