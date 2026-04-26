## Why

Phase 5 (Evolution & Adaptation) requires evolving classical brain architectures (MLPPPOBrain, LSTMPPOBrain) — but the existing [scripts/run_evolution.py](scripts/run_evolution.py) is hardcoded to `QVarCircuitBrain` quantum circuit parameters. The QVarCircuit coupling lives in three places: `create_brain_from_config()` (line 204) instantiates QVarCircuitBrain explicitly with a type check; `run_episode()` (line 248) hard-codes QVarCircuitBrain-specific sensory inputs; and `evaluate_fitness()` (line 386) calls both. Every subsequent Phase 5 milestone (M1 predator-as-brain, M2 hyperparameter pilot, M3 Lamarckian, M4 Baldwin, M5 co-evolution, M6 transgenerational) needs a brain-agnostic loop that does not exist today.

Per the Phase 5 design decisions recorded in [2026-04-26-phase5-tracking/proposal.md](openspec/changes/2026-04-26-phase5-tracking/proposal.md), refined for M0:

- **No QVarCircuit backwards compatibility, no legacy preservation**: the existing QVarCircuit-only script and its smoke test are **deleted outright**. The user has explicitly decided that retaining legacy code under `scripts/legacy/` would tie M0 to suboptimal implementation choices. If quantum brain evolution is needed at a Phase 6 re-evaluation, a `QVarCircuitEncoder` will be added cleanly to the new framework's registry — not by resurrecting the old script. Git history preserves the old code for reference.
- **LSTMPPO + klinotaxis is the first-class biological brain** for headline scientific milestones (M4/M5/M6). M0 ships its encoder so M3 can target it without a follow-up framework PR.

This change replaces the QVarCircuit-only loop with a brain-agnostic framework that: serializes any classical brain (via the existing `WeightPersistence` protocol) into a flat genome, evolves it via the existing `CMAESOptimizer` / `GeneticAlgorithmOptimizer`, evaluates fitness across parallel workers, tracks parent→child lineage, and supports pickle resume for long-running campaigns.

M0 ships **frozen-weight fitness only** (`EpisodicSuccessRate` — runs the brain without calling `.learn()`). The learn-then-evaluate variant (`LearnedPerformanceFitness`) is M2's responsibility and is explicitly out of scope here.

## What Changes

### 1. New `quantumnematode.evolution` Module

Six files under `packages/quantum-nematode/quantumnematode/evolution/`:

- `genome.py` — `Genome` dataclass (`params: np.ndarray`, `genome_id`, `parent_ids`, `generation`, `birth_metadata`) plus a deterministic `genome_id_for(generation, index, parent_ids)` helper
- `brain_factory.py` — thin wrapper `instantiate_brain_from_sim_config(sim_config: SimulationConfig) -> Brain` that gathers all the arguments [`utils/brain_factory.setup_brain_model()`](packages/quantum-nematode/quantumnematode/utils/brain_factory.py#L51) needs (`shots`, `qubits`, `device`, `learning_rate`, `gradient_method`, `gradient_max_norm`, `parameter_initializer_config`) from a `SimulationConfig` and returns a fresh `Brain` instance. Encoders call this — they do not call `setup_brain_model` directly. Single source of truth for "how to build a fresh brain for evolution from a config"
- `encoders.py` — `GenomeEncoder` protocol whose methods take **the full `SimulationConfig`** (not just `BrainConfig`), since brain instantiation needs more than `BrainConfig` provides (`shots`, `device`, learning rate config, etc.). Signatures: `initial_genome(sim_config) -> Genome`, `decode(genome, sim_config) -> Brain`, `genome_dim(sim_config) -> int`. Plus two concrete encoders (`MLPPPOEncoder`, `LSTMPPOEncoder`) and `ENCODER_REGISTRY: dict[str, type[GenomeEncoder]]`. Encoders use the existing `WeightPersistence` protocol from [brain/weights.py](packages/quantum-nematode/quantumnematode/brain/weights.py). Components are discovered **dynamically** via `get_weight_components()` and filtered by a fixed **denylist** of non-genome state: `{"optimizer", "actor_optimizer", "critic_optimizer", "training_state"}`. This picks up *all* learned-weight components automatically — including conditional ones like MLPPPO's `gate_weights` (when `_feature_gating` is enabled) and LSTMPPO's `layer_norm` — and survives future component additions without encoder changes
- `fitness.py` — `FitnessFunction` protocol, `EpisodicSuccessRate` (frozen weights, no `.learn()` call), and a `FrozenEvalRunner` class that subclasses [`StandardEpisodeRunner`](packages/quantum-nematode/quantumnematode/agent/runners.py#L599) and overrides `_terminate_episode` to force `learn=False, update_memory=False` on every termination path while passing all other kwargs through unchanged (preserving the `food_history=...` sentinel). The standard runner defaults `learn=True` on success, so we can't just call `agent.run_episode()` directly — that would mutate weights between episodes and break the M0 frozen-weight contract. Composition over copy-paste: `FrozenEvalRunner` reuses the per-step loop and helper methods from the standard runner. Success detection uses `result.termination_reason == TerminationReason.COMPLETED_ALL_FOOD` (codebase convention; `EpisodeResult` has no `episode_success` attribute). The `evaluate()` method also pins determinism by applying its `seed` argument to all three RNG sources (env, brain numpy RNG, torch). ~60 LOC total
- `lineage.py` — `LineageTracker` writing a single CSV at `evolution_results/<session_id>/lineage.csv` with columns `generation, child_id, parent_ids, fitness, brain_type`. Append mode so resume works seamlessly
- `loop.py` — `EvolutionLoop` class with `run(*, resume_from)` method. Wraps the existing optimisers and provides parallel fitness eval (multiprocessing.Pool with the SIGINT-ignore worker pattern from the legacy script) plus pickle checkpoint/resume

### 2. New CLI Script

`scripts/run_evolution.py` (fresh, ~150 LOC) — a thin CLI wiring `EvolutionLoop` + encoder + fitness from YAML config. Same flag surface as the legacy script (`--config`, `--generations`, `--population`, `--episodes`, `--algorithm`, `--sigma`, `--parallel`, `--seed`, `--resume`, `--output-dir`) so muscle-memory transfers.

### 3. Legacy Script Deleted

Existing `scripts/run_evolution.py` is **deleted**. Its CI smoke test (`test_run_evolution_smoke` in [tests/quantumnematode_tests/test_smoke.py](packages/quantum-nematode/tests/quantumnematode_tests/test_smoke.py#L82)) is also deleted. Git history preserves both for archaeology. The legacy `configs/evolution/qvarcircuit_foraging_small.yml` is also removed since nothing consumes it. A future Phase 6 quantum re-evaluation will add quantum encoder support to the new framework cleanly, not by resurrecting legacy code.

### 4. Configuration Schema Extension

New `EvolutionConfig` Pydantic model added to [config_loader.py](packages/quantum-nematode/quantumnematode/utils/config_loader.py) with fields: `algorithm` (literal `cmaes`/`ga`), `population_size`, `generations`, `episodes_per_eval`, `sigma0`, GA-specific params, `parallel_workers`, `checkpoint_every`. Added to `SimulationConfig` (line 933+) as an optional field — existing scenario configs without an `evolution:` block are unaffected.

### 5. Two Pilot Configs

- `configs/evolution/mlpppo_foraging_small.yml` — minimal smoke target: `mlpppo` brain, 20×20 grid, oracle sensing, `evolution: {generations: 10, population_size: 8, episodes_per_eval: 3}`. The fastest possible end-to-end framework verification
- `configs/evolution/lstmppo_foraging_small_klinotaxis.yml` — copies brain hyperparameters verbatim from [configs/scenarios/foraging/lstmppo_small_klinotaxis.yml](configs/scenarios/foraging/lstmppo_small_klinotaxis.yml) (gru, lstm_hidden_dim=64, klinotaxis sensing, STAM enabled). Adds the same minimal `evolution:` block. This is the "first-class biological brain" smoke target

### 6. Tests

New directory `packages/quantum-nematode/tests/quantumnematode_tests/evolution/`:

- `test_brain_factory.py` — `instantiate_brain_from_sim_config` returns the correct brain type for `mlpppo` and `lstmppo` configs
- `test_encoders.py` — round-trip determinism (MLPPPO + LSTMPPO), genome dim correctness, `_episode_count` reset + LR sync guard, registry membership, dynamic component discovery (proves `gate_weights` and `layer_norm` are picked up), denylist exclusion
- `test_lineage.py` — CSV append correctness across generations, gen-0 empty parent_ids, 0-based generation indexing
- `test_fitness.py` — fitness in [0, 1] for an arbitrary genome; deterministic for fixed seed; `FrozenEvalRunner` never calls `brain.learn()` (including on the success path that the standard runner would default to `learn=True`); success detection uses `TerminationReason.COMPLETED_ALL_FOOD`
- `test_loop_smoke.py` — 3-generation MLPPPO run end-to-end, checkpoint resume, checkpoint key shape, unknown brain name error
- `test_config.py` — existing scenario configs load with `evolution=None`; `evolution:` block parses correctly; CLI overrides YAML (subprocess test)

CI smoke test added to existing `tests/quantumnematode_tests/test_smoke.py`: a new `@pytest.mark.smoke test_run_evolution_smoke_mlpppo` that runs the new `scripts/run_evolution.py` against `configs/evolution/mlpppo_foraging_small.yml` with minimal parameters (1 gen, pop 4, 2 episodes). The legacy `test_run_evolution_smoke` is **deleted** as part of legacy script removal.

## Capabilities

**Added**: `evolution-framework` (new) — five requirements covering the encoder protocol, lineage tracking, checkpoint/resume, configuration schema, and the encoder registry.

## Impact

**Code:**

- `packages/quantum-nematode/quantumnematode/evolution/{__init__,genome,brain_factory,encoders,fitness,lineage,loop}.py` — new module
- `scripts/run_evolution.py` — fresh CLI replaces legacy QVarCircuit-only script (legacy deleted, not preserved)
- `packages/quantum-nematode/quantumnematode/utils/config_loader.py` — `EvolutionConfig` model + optional field on `SimulationConfig`

**Configs:**

- `configs/evolution/mlpppo_foraging_small.yml` (new)
- `configs/evolution/lstmppo_foraging_small_klinotaxis.yml` (new)
- `configs/evolution/qvarcircuit_foraging_small.yml` — **deleted** (no consumer remains after legacy script removal)

**Tests:**

- `packages/quantum-nematode/tests/quantumnematode_tests/evolution/{__init__,test_brain_factory,test_encoders,test_lineage,test_fitness,test_loop_smoke,test_config}.py` (new)
- `packages/quantum-nematode/tests/quantumnematode_tests/test_smoke.py` — `test_run_evolution_smoke` deleted; `test_run_evolution_smoke_mlpppo` added

**Docs:**

- `openspec/changes/2026-04-26-phase5-tracking/tasks.md` — M0.1 → M0.14 sub-tasks marked `[x]` per the M-1 invariant
- `docs/roadmap.md` Phase 5 Milestone Tracker — M0 status updated per the M-1 invariant

## Breaking Changes

- The CLI surface for `scripts/run_evolution.py` is **replaced** without backwards compatibility. Anyone who runs `python scripts/run_evolution.py --config configs/evolution/qvarcircuit_foraging_small.yml` against the new script will fail because no `QVarCircuitEncoder` is registered.
- The legacy script is **deleted entirely** — there is no `scripts/legacy/` fallback. Git history (`git log -- scripts/run_evolution.py`) preserves the implementation for archaeology.
- The legacy `configs/evolution/qvarcircuit_foraging_small.yml` is **deleted** (no consumer remains).
- The CI smoke test `test_run_evolution_smoke` (against the QVarCircuit script) is **deleted**, replaced by `test_run_evolution_smoke_mlpppo` against the new framework.

These breaks are sanctioned by the Phase 5 decision recorded in `2026-04-26-phase5-tracking/proposal.md`. The user has explicitly declined to retain a legacy fallback so M0 is not constrained by suboptimal legacy choices.

## Backward Compatibility

- All existing scenario configs (`configs/scenarios/**/*.yml`) — unaffected. The `evolution:` block is optional.
- `WeightPersistence`, `CMAESOptimizer`, `GeneticAlgorithmOptimizer`, `EvolutionResult`, and `BRAIN_CONFIG_MAP` — all unmodified, used as-is by the new framework.
