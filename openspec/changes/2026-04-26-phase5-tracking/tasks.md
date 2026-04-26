# Tasks: Phase 5 (Evolution & Adaptation) Milestone Tracker

This is the living checklist for all of Phase 5. Each milestone (M0–M7) has its own
OpenSpec change directory listed below. Each milestone PR MUST update this file
to mark sub-tasks complete as part of its diff.

**Status legend**: `[ ]` not started, `[x]` complete. Milestone-level "in progress" status lives in the **Status** header of each milestone section (matches the roadmap milestone tracker emoji column).

## M-1: Phase 5 Tracking Scaffold (THIS CHANGE)

**Branch**: `feat/phase5-tracking-scaffold`
**OpenSpec change**: `2026-04-26-phase5-tracking` (this directory)
**Status**: in progress

- [x] M-1.1 Create `openspec/changes/2026-04-26-phase5-tracking/proposal.md`
- [x] M-1.2 Create `openspec/changes/2026-04-26-phase5-tracking/design.md`
- [x] M-1.3 Create `openspec/changes/2026-04-26-phase5-tracking/tasks.md` (this file)
- [x] M-1.4 Update `docs/roadmap.md` Phase 5 row in Timeline Overview to `🟡 IN PROGRESS`
- [x] M-1.5 Add "Phase 5 Milestone Tracker" sub-section to `docs/roadmap.md` Phase 5 block
- [x] M-1.6 Validate change: `openspec validate --changes 2026-04-26-phase5-tracking --strict`
- [x] M-1.7 Run `uv run pre-commit run -a` clean
- [x] M-1.8 Open PR

## M0: Brain-Agnostic Evolution Framework (Fresh Build)

**OpenSpec change**: `2026-04-28-add-evolution-framework` (not yet created)
**Status**: not started
**Bio fidelity**: LOW
**Brain target**: MLPPPO (smoke) + LSTMPPO+klinotaxis (smoke)
**Dependencies**: M-1

- [ ] M0.1 Create `packages/quantum-nematode/quantumnematode/evolution/` module: `genome.py`, `encoders.py`, `lineage.py`, `loop.py`, `fitness.py`
- [ ] M0.2 Implement `GenomeEncoder` protocol + `MLPPPOEncoder` + `LSTMPPOEncoder` + `ENCODER_REGISTRY` (no `QVarCircuitEncoder`)
- [ ] M0.3 Implement `LineageTracker` writing `evolution_results/<session>/lineage.csv`
- [ ] M0.4 Implement `EvolutionLoop` class (fresh, not extracted from old script)
- [ ] M0.5 Implement `FitnessFunction` protocol with `EpisodicSuccessRate` (frozen weights, no `.learn()`). `LearnedPerformanceFitness` deferred to M2.
- [ ] M0.6 Move existing `scripts/run_evolution.py` → `scripts/legacy/run_evolution_qvarcircuit.py`
- [ ] M0.7 Write fresh `scripts/run_evolution.py` as a thin CLI wiring loop + encoder + fitness
- [ ] M0.8 Extend `config_loader.py` with an `evolution:` block schema
- [ ] M0.9 Create `configs/evolution/mlpppo_foraging_small.yml`
- [ ] M0.10 Create `configs/evolution/lstmppo_foraging_small_klinotaxis.yml`
- [ ] M0.11 Tests: encoder round-trip (MLPPPO + LSTMPPO), lineage CSV, smoke loop
- [ ] M0.12 Smoke run: 10-gen MLPPPO config, verify `best_params_*.json` round-trips
- [ ] M0.13 Smoke run: 10-gen LSTMPPO+klinotaxis config, verify `best_params_*.json` round-trips
- [ ] M0.14 Update this checklist + roadmap milestone tracker

## M1: Predator-as-Brain Refactor

**OpenSpec change**: `2026-05-05-add-learning-predators` (not yet created)
**Status**: not started
**Bio fidelity**: MEDIUM
**Brain target**: MLPPPO predator
**Dependencies**: M0

- [ ] M1.1 Create `env/predator_brain.py`: `PredatorBrain` protocol, `HeuristicPredatorBrain` adapter, `PredatorBrainParams` dataclass
- [ ] M1.2 Modify `Predator.update_position` to delegate to `self.brain.run_brain(params)` if brain set, else fall back to current behaviour
- [ ] M1.3 Extend `PredatorParams` with optional `brain_config`; `create_predators()` defaults to `HeuristicPredatorBrain`
- [ ] M1.4 Modify `MultiAgentSimulation` to expose per-predator metrics (`kills`, `prey_proximity_steps`, `distance_traveled`) in `EpisodeResult`
- [ ] M1.5 Tests: `tests/env/test_predator_brain.py`
- [ ] M1.6 Regression: 4-seed × 200-episode run on existing predator scenarios, agent survival rate within ±2pp of pre-refactor baseline
- [ ] M1.7 `uv run pytest -m smoke -v` passes
- [ ] M1.8 Update this checklist + roadmap milestone tracker

## M2: Hyperparameter Evolution Pilot

**OpenSpec change**: `2026-05-12-add-hyperparameter-evolution` (not yet created)
**Status**: not started
**Bio fidelity**: LOW
**Brain target**: MLPPPO + LSTMPPO+klinotaxis
**Dependencies**: M0
**Decision gate**: GO if either brain ≥3pp over hand-tuned baseline AND fitness still rising at gen 20

- [ ] M2.1 Implement `HyperparameterEncoder` in `evolution/encoders.py` (encodes config dict, not weights)
- [ ] M2.2 Implement `LearnedPerformanceFitness` in `evolution/fitness.py`
- [ ] M2.3 Create `configs/evolution/hyperparam_mlpppo_pilot.yml`
- [ ] M2.4 Create `configs/evolution/hyperparam_lstmppo_klinotaxis_pilot.yml`
- [ ] M2.5 Create campaign script `scripts/campaigns/phase5_m2_hyperparam.sh`
- [ ] M2.6 Run pilot: 20 gens × population 12 × 4 seeds × 2 brains
- [ ] M2.7 Publish `artifacts/logbooks/012/m2_hyperparam_pilot.md` with GO/PIVOT/STOP decision
- [ ] M2.8 Update this checklist + roadmap milestone tracker

## M3: Lamarckian Evolution Pilot

**OpenSpec change**: `2026-05-19-add-lamarckian-evolution` (not yet created)
**Status**: not started
**Bio fidelity**: MEDIUM
**Brain target**: MLPPPO (cheap) + LSTMPPO+klinotaxis (headline)
**Dependencies**: M0
**Decision gate**: GO to M4 if LSTMPPO+klinotaxis shows ≥10pp faster convergence AND F1 eval-only success > random init

- [ ] M3.1 Create `evolution/inheritance.py` with `LamarckianInheritance` strategy
- [ ] M3.2 Wire through `EvolutionLoop` via `--inheritance lamarckian` CLI flag
- [ ] M3.3 Encoder round-trip serialization for MLPPPO + LSTMPPO via `WeightPersistence`
- [ ] M3.4 Create `configs/evolution/lamarckian_mlpppo_pilot.yml`
- [ ] M3.5 Create `configs/evolution/lamarckian_lstmppo_klinotaxis_pilot.yml`
- [ ] M3.6 Run pilot: 30 gens × population 16 × 4 seeds × 2 brains
- [ ] M3.7 Publish `artifacts/logbooks/012/m3_lamarckian_pilot.md` with GO/PIVOT/STOP decision
- [ ] M3.8 Update this checklist + roadmap milestone tracker

## M4: Baldwin Effect Demonstration

**OpenSpec change**: `2026-06-02-add-baldwin-effect` (not yet created)
**Status**: not started (gated on M3 GO)
**Bio fidelity**: MEDIUM
**Brain target**: LSTMPPO+klinotaxis
**Dependencies**: M3 GO
**Decision gate**: GO if Baldwin cohort outperforms from-scratch AND learning-blocked control still improves

- [ ] M4.1 Implement `BaldwinInheritance` in `evolution/inheritance.py`
- [ ] M4.2 Implement learning-blocked F1 control cohort (config fields: `control_cohort_interval`, `control_cohort_fraction`)
- [ ] M4.3 Create `configs/evolution/baldwin_lstmppo_klinotaxis_full.yml`
- [ ] M4.4 Create `configs/evolution/baldwin_comparison_campaign.yml`
- [ ] M4.5 Run head-to-head: from-scratch vs Lamarckian (M3) vs Baldwin × 50 gens × 4 seeds
- [ ] M4.6 Publish `artifacts/logbooks/013/` with full Baldwin Effect findings
- [ ] M4.7 Update this checklist + roadmap milestone tracker

## M5: Co-Evolution Arms Race

**OpenSpec change**: `2026-06-16-add-coevolution` (not yet created)
**Status**: not started
**Bio fidelity**: HIGH
**Brain target**: LSTMPPO+klinotaxis prey, MLPPPO predator
**Dependencies**: M0, M1
**Decision gate**: GO if phenotypic cycling visible AND trait escalation monotonic over ≥30 gens

- [ ] M5.1 Create `evolution/coevolution.py` with `CoevolutionLoop`
- [ ] M5.2 Create `evolution/redqueen_metrics.py` with phenotypic cycling, trait escalation, fitness lag, coupled rate
- [ ] M5.3 Create `configs/evolution/coevolution_pilot.yml`
- [ ] M5.4 Create `configs/evolution/coevolution_full.yml`
- [ ] M5.5 Run 50+ gen × 4 seed campaign
- [ ] M5.6 Multi-cluster vs single-cluster transfer evaluation
- [ ] M5.7 Publish `artifacts/logbooks/014/` with Red Queen findings
- [ ] M5.8 Update this checklist + roadmap milestone tracker

## M6: Transgenerational Memory

**OpenSpec change**: `2026-06-30-add-transgenerational-memory` (not yet created)
**Status**: not started (gated on M3 GO + M4-or-M5)
**Bio fidelity**: HIGH
**Brain target**: LSTMPPO+klinotaxis
**Dependencies**: M3 GO + (M4 or M5) complete
**Decision gate**: GO if F1 retains ≥40% of F0 avoidance, F3 ≤10%

- [ ] M6.1 Create `agent/transgenerational_memory.py` with `TransgenerationalMemory` class
- [ ] M6.2 Hook into `prepare_episode()` as a prior on response distribution
- [ ] M6.3 Implement `inherit_from(parents, decay_factor)` transmission
- [ ] M6.4 Create `configs/evolution/transgenerational_pathogen_avoidance_lstmppo_klinotaxis.yml`
- [ ] M6.5 Run F0/F1/F2/F3 pathogen avoidance experiment (Posner replication design)
- [ ] M6.6 Publish `artifacts/logbooks/015/` with transgenerational findings
- [ ] M6.7 Update this checklist + roadmap milestone tracker

## M6.5: NEAT Architecture Evolution (OPTIONAL)

**OpenSpec change**: `2026-07-14-add-neat-evolution` (only if scheduled)
**Status**: not started (compute-budget dependent)
**Bio fidelity**: LOW
**Brain target**: MLPPPO

- [ ] M6.5.1 Integrate `neat-python` via `NeatEncoder` + `NeatGenome`
- [ ] M6.5.2 30-gen pilot, single environment
- [ ] M6.5.3 Logbook supplement to whichever logbook is current
- [ ] M6.5.4 Update this checklist + roadmap milestone tracker

## M7: Phase 5 Synthesis Logbook

**OpenSpec change**: `2026-07-21-add-phase5-evaluation` (not yet created)
**Status**: not started
**Dependencies**: M2–M6 complete (or explicitly dropped via gates)

- [ ] M7.1 Aggregate cross-milestone fitness curves and tables
- [ ] M7.2 Walk through each Phase 5 exit criterion with evidence
- [ ] M7.3 Document negative findings honestly (Phase 4 precedent)
- [ ] M7.4 Phase 6 quantum re-evaluation trigger recommendation
- [ ] M7.5 Publish `artifacts/logbooks/016/synthesis.md`
- [ ] M7.6 Update `docs/roadmap.md` Phase 5 status → COMPLETE; record exit criterion outcomes
- [ ] M7.7 Archive `2026-04-26-phase5-tracking` alongside `2026-07-21-add-phase5-evaluation`
