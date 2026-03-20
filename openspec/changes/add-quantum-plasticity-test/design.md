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

**Decision**: Use `brain.copy()` to snapshot the brain at each transition point, plus `torch.save(model.state_dict(), path)` for disk persistence.

**Rationale**: All five test architectures implement `copy()` (required by `Brain` protocol). For evaluation metrics, we need the state *before* each phase starts (to measure forgetting by comparing pre-phase A performance vs post-phase C performance on objective A). `copy()` gives us an in-memory snapshot. Disk checkpoints provide reproducibility.

For architectures with multiple components (HybridQuantum has reflex + cortex + critic), we use their existing `save_*_weights()` / `load_*_weights()` methods. For simpler architectures (QRH, CRH, MLP PPO), we save the full `state_dict()` of their PyTorch modules.

### 4. Metric computation approach

**Decision**: Run a fixed-length evaluation block (e.g., 50 episodes) at each transition point with the brain in eval mode (no learning), then resume training.

**Rationale**: Measuring performance *during* training conflates learning dynamics with task performance. A clean eval block at each transition gives us:

- **Pre-A baseline**: Eval on A before training starts (random policy baseline)
- **Post-A**: Eval on A after training on A (task A competence)
- **Post-B**: Eval on A after training on B (backward forgetting from B)
- **Post-C**: Eval on A after training on C (cumulative backward forgetting)
- **Post-A'**: Eval on A after retraining on A (plasticity retention — can it relearn?)

Plus equivalent evals on B and C at appropriate points for forward transfer measurement.

**Key metrics**:

- **Backward Forgetting (BF)** = `post_A_score - post_C_score_on_A` (how much A degrades after training B and C)
- **Forward Transfer (FT)** = `pre_B_score - random_baseline_on_B` (does A-training help B?)
- **Plasticity Retention (PR)** = `post_A'_convergence_rate / post_A_convergence_rate` (can it relearn A as fast?)

### 5. Evaluation episodes per phase vs training episodes

**Decision**: 200 training episodes per objective, 50 evaluation episodes at each transition.

**Rationale**: 200 episodes is sufficient for convergence on our environments based on prior experiments (QRH converges by episode ~100 on foraging, MLP PPO by ~50). 50 evaluation episodes provide a stable mean with reasonable variance. This gives a total of ~1050 episodes per seed (200×4 training + 50×5 eval blocks), which is manageable for 5 architectures × 8 seeds = 40 runs.

### 6. Objective sequence

**Decision**: Foraging → Pursuit Predators → Thermotaxis+Pursuit → return to Foraging.

**Rationale**: This sequence increases complexity and tests progressively different skills:

- **A (Foraging)**: Pure food-seeking. Well-understood baseline, all architectures converge reliably.
- **B (Pursuit Predators)**: Adds evasion to foraging. Tests whether foraging knowledge transfers.
- **C (Thermotaxis+Pursuit)**: Adds temperature navigation. Most complex multi-objective task.
- **A' (Foraging return)**: Measures how much of the original skill was forgotten and how fast it can be relearned.

Environment configs: small grid (20×20) for A and B, large grid (100×100) for C to match existing benchmark configs.

### 7. Config structure

**Decision**: One YAML config per architecture defining all four phases, rather than separate configs per phase.

**Rationale**: Plasticity testing requires consistent hyperparameters across phases — same learning rate, same architecture dimensions, same training episodes. A single config file per architecture ensures consistency and makes it easy to reproduce. Phase-specific environment parameters are embedded as sections within the config.

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
      environment: { grid_size: 20, foraging: {...}, predators: { enabled: false } }
    - name: pursuit_predators
      environment: { grid_size: 20, foraging: {...}, predators: { enabled: true, ... } }
    - name: thermotaxis_pursuit
      environment: { grid_size: 100, foraging: {...}, predators: {...}, thermotaxis: {...} }
    - name: foraging_return
      environment: { grid_size: 20, foraging: {...}, predators: { enabled: false } }
```

### 8. Statistical comparison

**Decision**: Per-architecture mean ± std across seeds, two-sample t-test between quantum and classical pairs (QRH vs CRH, HybridQuantum vs HybridClassical), plus MLP PPO as overall classical baseline.

**Rationale**: Paired comparisons (quantum vs its classical ablation) control for architectural differences. The ≤50% forgetting threshold from the proposal translates to: `mean_BF_quantum / mean_BF_classical ≤ 0.5` with p < 0.05.

## Risks / Trade-offs

**[Risk] Some architectures may not converge within 200 episodes on all objectives** → Mitigation: Use environment configs known to work for each architecture from prior benchmarks. If an architecture fails to converge on a phase, record it as a data point (inability to learn = relevant to plasticity analysis). Pre-test with 1 seed before running the full 8-seed campaign.

**[Risk] Evaluation episodes may interfere with training state (optimizer momentum, learning rate schedules)** → Mitigation: Save and restore optimizer state before/after eval blocks. For brains using PPO (QRH, CRH, HybridQuantum, HybridClassical, MLP PPO), clear the experience buffer before eval and after eval to prevent eval data from corrupting training updates.

**[Risk] HybridQuantum's multi-stage training complicates the protocol** → Mitigation: Use HybridQuantum in Stage 3 (joint fine-tune) mode with pre-trained weights from existing artifacts. This way it behaves as a single unified model during the plasticity test, same as all other architectures.

**[Risk] Grid size change between phases B (20×20) and C (100×100) may confound forgetting measurement** → Mitigation: Use small grid (20×20) for all phases in the primary analysis. Run a secondary analysis with the large grid for phase C only, reporting both results.

**[Trade-off] 50 eval episodes adds ~25% overhead to total runtime** → Accepted: Clean measurement is worth the cost. Without eval blocks, training metrics conflate learning and forgetting dynamics.
