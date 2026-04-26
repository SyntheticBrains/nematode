## Why

Phase 5 (Evolution & Adaptation) requires evolving classical brain architectures (MLPPPOBrain, LSTMPPOBrain) — but the existing [scripts/run_evolution.py](scripts/run_evolution.py) is hardcoded to `QVarCircuitBrain` quantum circuit parameters. The QVarCircuit coupling lives in three places: `create_brain_from_config()` (line 204) instantiates QVarCircuitBrain explicitly with a type check; `run_episode()` (line 248) hard-codes QVarCircuitBrain-specific sensory inputs; and `evaluate_fitness()` (line 386) calls both. Every subsequent Phase 5 milestone (M1 predator-as-brain, M2 hyperparameter pilot, M3 Lamarckian, M4 Baldwin, M5 co-evolution, M6 transgenerational) needs a brain-agnostic loop that does not exist today.

Per the Phase 5 design decisions recorded in [2026-04-26-phase5-tracking/proposal.md](openspec/changes/2026-04-26-phase5-tracking/proposal.md):

- **No QVarCircuit backwards compatibility**: the legacy script moves to `scripts/legacy/` unmaintained. Quantum brain support deferred to a Phase 6 quantum re-evaluation if warranted.
- **LSTMPPO + klinotaxis is the first-class biological brain** for headline scientific milestones (M4/M5/M6). M0 ships its encoder so M3 can target it without a follow-up framework PR.

This change replaces the QVarCircuit-only loop with a brain-agnostic framework that: serializes any classical brain (via the existing `WeightPersistence` protocol) into a flat genome, evolves it via the existing `CMAESOptimizer` / `GeneticAlgorithmOptimizer`, evaluates fitness across parallel workers, tracks parent→child lineage, and supports pickle resume for long-running campaigns.

M0 ships **frozen-weight fitness only** (`EpisodicSuccessRate` — runs the brain without calling `.learn()`). The learn-then-evaluate variant (`LearnedPerformanceFitness`) is M2's responsibility and is explicitly out of scope here.

## What Changes

### 1. New `quantumnematode.evolution` Module

Five files under `packages/quantum-nematode/quantumnematode/evolution/`:

- `genome.py` — `Genome` dataclass (`params: np.ndarray`, `genome_id`, `parent_ids`, `generation`, `birth_metadata`) plus a deterministic `genome_id_for(generation, index, parent_ids)` helper
- `encoders.py` — `GenomeEncoder` protocol (`initial_genome`, `decode`, `genome_dim`) plus two concrete encoders (`MLPPPOEncoder`, `LSTMPPOEncoder`) plus `ENCODER_REGISTRY: dict[str, type[GenomeEncoder]]`. Encoders use the existing `WeightPersistence` protocol from [brain/weights.py](packages/quantum-nematode/quantumnematode/brain/weights.py) — they serialize `{"policy", "value"}` for MLPPPO and `{"lstm", "actor", "critic"}` for LSTMPPO, deliberately excluding optimiser state and training counters
- `fitness.py` — `FitnessFunction` protocol plus `EpisodicSuccessRate` (frozen weights, no `.learn()` call). Lifts ~80 LOC from the per-episode loop pattern in [scripts/run_simulation.py](scripts/run_simulation.py) without rendering or CSV export
- `lineage.py` — `LineageTracker` writing a single CSV at `evolution_results/<session_id>/lineage.csv` with columns `generation, child_id, parent_ids, fitness, brain_type`. Append mode so resume works seamlessly
- `loop.py` — `EvolutionLoop` class with `run(*, resume_from)` method. Wraps the existing optimisers and provides parallel fitness eval (multiprocessing.Pool with the SIGINT-ignore worker pattern from the legacy script) plus pickle checkpoint/resume

### 2. New CLI Script

`scripts/run_evolution.py` (fresh, ~150 LOC) — a thin CLI wiring `EvolutionLoop` + encoder + fitness from YAML config. Same flag surface as the legacy script (`--config`, `--generations`, `--population`, `--episodes`, `--algorithm`, `--sigma`, `--parallel`, `--seed`, `--resume`, `--output-dir`) so muscle-memory transfers.

### 3. Legacy Script Preserved

Existing `scripts/run_evolution.py` moves to `scripts/legacy/run_evolution_qvarcircuit.py` — preserved as reference, no maintenance commitment, no test coverage requirement. This is per the recorded Phase 5 decision.

### 4. Configuration Schema Extension

New `EvolutionConfig` Pydantic model added to [config_loader.py](packages/quantum-nematode/quantumnematode/utils/config_loader.py) with fields: `algorithm` (literal `cmaes`/`ga`), `population_size`, `generations`, `episodes_per_eval`, `sigma0`, GA-specific params, `parallel_workers`, `checkpoint_every`. Added to `SimulationConfig` (line 933+) as an optional field — existing scenario configs without an `evolution:` block are unaffected.

### 5. Two Pilot Configs

- `configs/evolution/mlpppo_foraging_small.yml` — minimal smoke target: `mlpppo` brain, 20×20 grid, oracle sensing, `evolution: {generations: 10, population_size: 8, episodes_per_eval: 3}`. The fastest possible end-to-end framework verification
- `configs/evolution/lstmppo_foraging_small_klinotaxis.yml` — copies brain hyperparameters verbatim from [configs/scenarios/foraging/lstmppo_small_klinotaxis.yml](configs/scenarios/foraging/lstmppo_small_klinotaxis.yml) (gru, lstm_hidden_dim=64, klinotaxis sensing, STAM enabled). Adds the same minimal `evolution:` block. This is the "first-class biological brain" smoke target

### 6. Tests

New directory `packages/quantum-nematode/tests/quantumnematode_tests/evolution/`:

- `test_encoders.py` — round-trip determinism (MLPPPO + LSTMPPO), genome dim correctness, `_episode_count` reset guard
- `test_lineage.py` — CSV append correctness across generations
- `test_loop_smoke.py` — 3-generation MLPPPO run end-to-end + checkpoint resume

## Capabilities

**Added**: `evolution-framework` (new) — five requirements covering the encoder protocol, lineage tracking, checkpoint/resume, configuration schema, and the encoder registry.

## Impact

**Code:**

- `packages/quantum-nematode/quantumnematode/evolution/{__init__,genome,encoders,fitness,lineage,loop}.py` — new module
- `scripts/run_evolution.py` — fresh CLI replaces legacy QVarCircuit-only script
- `scripts/legacy/run_evolution_qvarcircuit.py` — moved from `scripts/run_evolution.py`, no further maintenance
- `packages/quantum-nematode/quantumnematode/utils/config_loader.py` — `EvolutionConfig` model + optional field on `SimulationConfig`

**Configs:**

- `configs/evolution/mlpppo_foraging_small.yml` (new)
- `configs/evolution/lstmppo_foraging_small_klinotaxis.yml` (new)
- `configs/evolution/qvarcircuit_foraging_small.yml` — left untouched; remains compatible with the legacy script under `scripts/legacy/`

**Tests:**

- `packages/quantum-nematode/tests/quantumnematode_tests/evolution/{__init__,test_encoders,test_lineage,test_loop_smoke}.py`

**Docs:**

- `openspec/changes/2026-04-26-phase5-tracking/tasks.md` — M0.1 → M0.14 sub-tasks marked `[x]` per the M-1 invariant
- `docs/roadmap.md` Phase 5 Milestone Tracker — M0 status updated per the M-1 invariant

## Breaking Changes

The CLI surface for `scripts/run_evolution.py` is **deliberately replaced** without backwards compatibility. Anyone who runs `python scripts/run_evolution.py --config configs/evolution/qvarcircuit_foraging_small.yml` against the new script will fail because no `QVarCircuitEncoder` is registered. The legacy behaviour is available at `python scripts/legacy/run_evolution_qvarcircuit.py` with identical flags. This break is sanctioned by the Phase 5 decision recorded in `2026-04-26-phase5-tracking/proposal.md`.

## Backward Compatibility

- All existing scenario configs (`configs/scenarios/**/*.yml`) — unaffected. The `evolution:` block is optional.
- The legacy quantum evolution path remains runnable via `scripts/legacy/run_evolution_qvarcircuit.py`; its config (`configs/evolution/qvarcircuit_foraging_small.yml`) is unchanged.
- `WeightPersistence`, `CMAESOptimizer`, `GeneticAlgorithmOptimizer`, `EvolutionResult`, and `BRAIN_CONFIG_MAP` — all unmodified, used as-is by the new framework.
