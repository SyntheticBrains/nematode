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
- [ ] Create `docs/nematodebench/` directory
- [ ] Create `docs/nematodebench/README.md` with overview

### 3.2 Write Submission Guide
- [ ] Create `docs/nematodebench/SUBMISSION_GUIDE.md`
- [ ] Document prerequisites (50+ runs, clean git, config in repo)
- [ ] Document step-by-step submission process
- [ ] Document PR workflow and verification

### 3.3 Write Evaluation Methodology
- [ ] Create `docs/nematodebench/EVALUATION.md`
- [ ] Document composite score formula and weights
- [ ] Document convergence detection algorithm
- [ ] Document ranking criteria

### 3.4 Write Reproducibility Requirements
- [ ] Create `docs/nematodebench/REPRODUCIBILITY.md`
- [ ] Document config file requirements
- [ ] Document git state requirements
- [ ] Document version tracking requirements

### 3.5 Create Evaluation Script
- [ ] Create `scripts/evaluate_submission.py`
- [ ] Implement JSON structure validation
- [ ] Implement minimum runs check
- [ ] Implement optional reproduction verification
- [ ] Implement pass/fail reporting with details

**Validation**: Script correctly validates existing benchmarks

### 3.6 Update BENCHMARKS.md
- [ ] Add "External Submissions" section
- [ ] Add links to nematodebench documentation
- [ ] Add call-to-action for external researchers

---

## Work Stream 4: Optimization Method Documentation

### 4.1 Create Documentation File
- [ ] Create `docs/OPTIMIZATION_METHODS.md`
- [ ] Write summary table (Architecture → Method mapping)
- [ ] Document quantum findings (CMA-ES 88% vs gradients 22%)
- [ ] Document classical findings (REINFORCE, PPO)
- [ ] Document spiking findings (surrogate gradients)

### 4.2 Add Configuration Examples
- [ ] Add CMA-ES config example for ModularBrain
- [ ] Add REINFORCE config example for MLPBrain
- [ ] Add PPO config example (reference new configs)
- [ ] Add surrogate gradient config example for SpikingBrain

### 4.3 Add Selection Guidance
- [ ] Document when to use evolutionary vs gradient methods
- [ ] Document architecture-specific recommendations
- [ ] Add decision flow for new users

---

## Work Stream 5: Reproducibility & Metrics Enhancement

### 5.1 Create Seeding Infrastructure
- [ ] Create `packages/quantum-nematode/quantumnematode/utils/seeding.py`
- [ ] Implement `generate_seed()` using `secrets.randbelow(2**32)`
- [ ] Implement `set_global_seed(seed: int)` for numpy/torch
- [ ] Implement `get_rng(seed: int | None)` for seeded RNG creation
- [ ] Implement `ensure_seed(seed: int | None)` for auto-generation

**Validation**: Unit tests for seed generation and RNG creation

### 5.2 Fix Environment Seeding
- [ ] Replace `secrets` module usage in `env/env.py` with seeded numpy RNG
- [ ] Add `seed` parameter to environment config
- [ ] Ensure food spawning, predator movement, initial positions are deterministic

**Validation**: Same seed produces identical episode results

### 5.3 Add Brain Seed Support
- [ ] Add `seed` config parameter to `PPOBrainConfig`
- [ ] Add `seed` config parameter to `MLPBrainConfig`
- [ ] Add `seed` config parameter to `ModularBrainConfig`
- [ ] Update config loader to pass seed through
- [ ] Ensure weight initialization and action selection are reproducible

**Validation**: Same seed produces identical brain behavior

### 5.4 Add Enhanced Metrics
- [ ] Implement `learning_speed_episodes` calculation in convergence.py
- [ ] Implement `stability` metric calculation
- [ ] Add per-run statistics aggregation (mean/std/min/max)
- [ ] Add `seed` field to `SimulationResult`

**Validation**: Metrics computed correctly for test cases

### 5.5 Migrate to NematodeBench Format
- [ ] Update `experiment/metadata.py` to use StatValue objects
- [ ] Update `benchmark_submit.py` to output NematodeBench format
- [ ] Update `evaluate_submission.py` to validate NematodeBench schema
- [ ] Update documentation for unified format

**Validation**: Submissions validate against NematodeBench schema

---

## Finalization

### 5.1 Update OpenSpec Specs
- [x] Finalize `specs/brain-architecture/spec.md` with PPO requirements
- [x] Finalize `specs/validation-system/spec.md` with chemotaxis requirements
- [ ] Finalize `specs/benchmark-management/spec.md` with NematodeBench requirements

### 5.2 Validate Proposal
- [ ] Run `openspec validate add-phase0-foundation-baselines --strict`
- [ ] Fix any validation errors
- [ ] Verify all requirements have scenarios

---

## Dependencies

```text
Work Stream 1 (PPO)          ──┐
Work Stream 2 (Chemotaxis)   ──┼──► Work Stream 3 (NematodeBench)
Work Stream 4 (Optimization) ──┘         │
                                         ▼
                              Finalization (5.1, 5.2)
```

Work Streams 1, 2, 4 can proceed in parallel. Work Stream 3 references them. Finalization requires all work streams complete.

---

## Exit Criteria Mapping

| Roadmap Exit Criterion | Task(s) | Status |
|------------------------|---------|--------|
| PPO >85% success on foraging | 1.4 | Complete |
| Optimization method documentation | 4.1, 4.2, 4.3 | Complete |
| 1 C. elegans dataset integrated | 2.3, 2.4, 2.5, 2.6 | Complete |
