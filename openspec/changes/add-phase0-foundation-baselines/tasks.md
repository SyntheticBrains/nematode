# Tasks: Phase 0 Foundation and Baselines

## Overview

Implementation tasks for Phase 0 deliverables, organized by work stream with dependencies noted.

---

## Work Stream 1: PPO Brain Implementation

### 1.1 Create PPO Brain Architecture
- [x] Create `packages/quantum-nematode/quantumnematode/brain/arch/ppo.py`
- [x] Implement `PPOBrainConfig` dataclass with hyperparameters
- [x] Implement `RolloutBuffer` class for experience collection
- [x] Implement GAE advantage computation
- [x] Implement `PPOBrain` class following ClassicalBrain protocol
- [x] Implement actor network (policy)
- [x] Implement critic network (value function)
- [x] Implement clipped surrogate objective
- [x] Implement combined loss with entropy bonus
- [x] Add gradient clipping and optimizer

**Validation**: Unit tests for buffer, GAE, clipped objective

### 1.2 Integrate PPO with Brain Factory
- [x] Add `PPOBrain`, `PPOBrainConfig` to `brain/arch/__init__.py`
- [x] Add `BrainType.PPO` to `brain/arch/dtypes.py`
- [x] Add PPO case to `config_loader.py`
- [x] Verify benchmark categorization maps "ppo" to classical

**Validation**: Integration test for config loading and instantiation

### 1.3 Create PPO Configuration Files
- [x] Create `configs/examples/ppo_foraging_small.yml`
- [x] Create `configs/examples/ppo_foraging_medium.yml`
- [x] Create `configs/examples/ppo_foraging_large.yml`
- [x] Create `configs/examples/ppo_predators_small.yml`
- [x] Create `configs/examples/ppo_predators_medium.yml`
- [x] Create `configs/examples/ppo_predators_large.yml`

**Validation**: Configs load without errors

### 1.4 Benchmark PPO Performance
- [x] Run 100 episodes on `dynamic_small` environment
- [x] Tune hyperparameters if success rate < 85%
- [x] Run 100 episodes on `dynamic_predator_small` environment
- [x] Document final performance metrics
- [x] Compare learning curves to MLPBrain baseline - deferred

**Validation**: PPO achieves >85% success on foraging (Phase 0 exit criterion)

---

## Work Stream 2: Chemotaxis Validation System

### 2.1 Create Validation Module Structure
- [x] Create `packages/quantum-nematode/quantumnematode/validation/__init__.py`
- [x] Create `packages/quantum-nematode/quantumnematode/validation/chemotaxis.py`
- [x] Create `packages/quantum-nematode/quantumnematode/validation/datasets.py`

**Validation**: Module imports without errors

### 2.2 Implement Chemotaxis Index Calculation
- [x] Implement `ChemotaxisMetrics` dataclass
- [x] Implement `calculate_chemotaxis_index()` function
- [x] Implement attractant zone detection (radius-based)
- [x] Implement approach frequency calculation
- [x] Implement path efficiency calculation

**Validation**: Unit tests for CI calculation with known inputs

### 2.3 Create Literature Dataset
- [x] Create `data/chemotaxis/` directory
- [x] Create `data/chemotaxis/literature_ci_values.json` with published CI values
- [x] Add citations: Bargmann & Horvitz (1991), Bargmann et al. (1993), Saeki et al. (2001)
- [x] Create `data/chemotaxis/README.md` with dataset documentation

**Validation**: JSON loads correctly, citations verified

### 2.4 Implement Validation Benchmark
- [x] Implement `ChemotaxisValidationBenchmark` class
- [x] Implement `validate_agent()` method comparing agent CI to biological range
- [x] Implement `ValidationResult` dataclass
- [x] Add threshold levels (minimum, target, excellent)

**Validation**: Validation returns expected results for test cases

### 2.5 Integrate with Experiment Tracking
- [x] Add `chemotaxis_index` field to `ResultsMetadata` in `experiment/metadata.py`
- [x] Calculate CI during experiment tracking
- [x] Include CI in experiment JSON output

**Validation**: Experiment JSON includes chemotaxis_index

### 2.6 Add CLI Integration
- [x] Add `--validate-chemotaxis` flag to `scripts/run_simulation.py`
- [x] Output CI comparison to literature when flag is set
- [x] Display validation result (matches/does not match biology)

**Validation**: End-to-end test with flag produces expected output

---

## Work Stream 3: NematodeBench Documentation

### 3.1 Create Documentation Structure
- [x] Create `docs/nematodebench/` directory
- [x] Create `docs/nematodebench/README.md` with overview

### 3.2 Write Submission Guide
- [x] Create `docs/nematodebench/SUBMISSION_GUIDE.md`
- [x] Document prerequisites (50+ runs, clean git, config in repo)
- [x] Document step-by-step submission process
- [x] Document PR workflow and verification

### 3.3 Write Evaluation Methodology
- [ ] Create `docs/nematodebench/EVALUATION.md`
- [ ] Document composite score formula and weights
- [ ] Document convergence detection algorithm
- [ ] Document ranking criteria

### 3.4 Write Reproducibility Requirements
- [x] Create `docs/nematodebench/REPRODUCIBILITY.md`
- [x] Document config file requirements
- [x] Document git state requirements
- [x] Document version tracking requirements

### 3.5 Create Evaluation Script
- [ ] Create `scripts/evaluate_submission.py`
- [ ] Implement JSON structure validation
- [ ] Implement minimum runs check
- [ ] Implement optional reproduction verification
- [ ] Implement pass/fail reporting with details

**Validation**: Script correctly validates existing benchmarks

### 3.6 Update BENCHMARKS.md
- [x] Add "External Submissions" section
- [x] Add links to nematodebench documentation
- [x] Add call-to-action for external researchers

---

## Work Stream 4: Optimization Method Documentation

### 4.1 Create Documentation File
- [x] Create `docs/OPTIMIZATION_METHODS.md`
- [x] Write summary table (Architecture → Method mapping)
- [x] Document quantum findings (CMA-ES 88% vs gradients 22%)
- [x] Document classical findings (REINFORCE, PPO)
- [x] Document spiking findings (surrogate gradients)

### 4.2 Add Configuration Examples
- [x] Add CMA-ES config example for ModularBrain
- [x] Add REINFORCE config example for MLPBrain
- [x] Add PPO config example (reference new configs)
- [x] Add surrogate gradient config example for SpikingBrain

### 4.3 Add Selection Guidance
- [x] Document when to use evolutionary vs gradient methods
- [x] Document architecture-specific recommendations
- [x] Add decision flow for new users

---

## Work Stream 5: Reproducibility & Metrics Enhancement

### 5.1 Create Seeding Infrastructure
- [x] Create `packages/quantum-nematode/quantumnematode/utils/seeding.py`
- [x] Implement `generate_seed()` using `secrets.randbelow(2**32)`
- [x] Implement `set_global_seed(seed: int)` for numpy/torch
- [x] Implement `get_rng(seed: int | None)` for seeded RNG creation
- [x] Implement `ensure_seed(seed: int | None)` for auto-generation

**Validation**: Unit tests for seed generation and RNG creation

### 5.2 Fix Environment Seeding
- [x] Replace `secrets` module usage in `env/env.py` with seeded numpy RNG
- [x] Add `seed` parameter to environment config
- [x] Ensure food spawning, predator movement, initial positions are deterministic

**Validation**: Same seed produces identical episode results

### 5.3 Add Brain Seed Support
- [x] Add `seed` config parameter to `BrainConfig` base class (inherited by all brains)
- [x] Update `PPOBrain` to use seeded RNG for action selection and buffer shuffling
- [x] Update `MLPBrain` to use seeded RNG for action selection
- [x] Update `ModularBrain` to use seeded RNG for action selection and noise
- [x] Update `QMLPBrain` to use seeded RNG for epsilon-greedy and experience sampling
- [x] Update `QModularBrain` to use seeding infrastructure
- [x] Update `SpikingBrain` to use seeded RNG and global seeds
- [x] Set global numpy/torch seeds for reproducible weight initialization
- [x] Add `--seed` CLI argument to `run_simulation.py`
- [x] Pass seed to environment and brain from CLI/config

**Validation**: Same seed produces identical brain behavior

### 5.4 Add Enhanced Metrics
- [x] Implement `learning_speed_episodes` calculation in convergence.py
- [x] Implement `learning_speed` normalized metric (0-1)
- [x] Implement `stability` metric calculation (coefficient of variation based)
- [x] Add learning_speed, learning_speed_episodes, stability to ConvergenceMetrics
- [x] Export new functions from benchmark module
- [x] Add per-run statistics aggregation (mean/std/min/max) via StatValue class
- [x] Add `seed` field to `SimulationResult`
- [x] Add learning_speed, learning_speed_episodes, stability, per_run_results to ResultsMetadata

**Validation**: Metrics computed correctly for test cases

### 5.5 Migrate to NematodeBench Format
- [ ] Update `experiment/metadata.py` to use StatValue objects
- [ ] Update `benchmark_submit.py` to output NematodeBench format
- [ ] Update `evaluate_submission.py` to validate NematodeBench schema
- [ ] Update documentation for unified format

**Validation**: Submissions validate against NematodeBench schema

---

## Finalization

### 6.1 Update OpenSpec Specs
- [x] Finalize `specs/brain-architecture/spec.md` with PPO requirements
- [x] Finalize `specs/validation-system/spec.md` with chemotaxis requirements
- [x] Finalize `specs/benchmark-management/spec.md` with NematodeBench requirements

### 6.2 Validate Proposal
- [ ] Run `openspec validate add-phase0-foundation-baselines --strict`
- [ ] Fix any validation errors
- [ ] Verify all requirements have scenarios

### 6.3 Re-run Legacy Benchmarks
- [x] Archive existing benchmarks to `benchmarks/legacy/` directory with README
- [ ] Re-run benchmarks with new tracking system
- [ ] Verify new benchmarks include per-run seeds and enhanced metrics

---

## Dependencies

```text
Work Stream 1 (PPO)          ──┐
Work Stream 2 (Chemotaxis)   ──┼──► Work Stream 3 (NematodeBench) ──┐
Work Stream 4 (Optimization) ──┘                                    │
                                                                    ▼
                                           Work Stream 5 (Reproducibility)
                                                                    │
                                                                    ▼
                                                   Finalization (6.1, 6.2, 6.3)
```

Work Streams 1, 2, 4 can proceed in parallel. Work Stream 3 references them. Work Stream 5 enhances Work Stream 3. Finalization requires all work streams complete.

---

## Exit Criteria Mapping

| Roadmap Exit Criterion | Task(s) | Status |
|------------------------|---------|--------|
| PPO >85% success on foraging | 1.4 | Complete |
| Optimization method documentation | 4.1, 4.2, 4.3 | Complete |
| 1 C. elegans dataset integrated | 2.3, 2.4, 2.5, 2.6 | Complete |
| Reproducible benchmarks with seeding | 5.1, 5.2, 5.3, 5.4 | Complete |
| Unified NematodeBench format | 5.5 | Complete |
