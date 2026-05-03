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

**OpenSpec change**: `2026-04-28-add-evolution-framework` — archived in the M0 PR (see `openspec/changes/archive/2026-04-27-2026-04-28-add-evolution-framework/`); the `evolution-framework` capability is now under `openspec/specs/`
**Status**: complete (archived)
**Bio fidelity**: LOW
**Brain target**: MLPPPO (smoke) + LSTMPPO+klinotaxis (smoke)
**Dependencies**: M-1

- [x] M0.1 Create `packages/quantum-nematode/quantumnematode/evolution/` module: `genome.py`, `brain_factory.py`, `encoders.py`, `fitness.py`, `lineage.py`, `loop.py`
- [x] M0.2 Implement `GenomeEncoder` protocol + `MLPPPOEncoder` + `LSTMPPOEncoder` + `ENCODER_REGISTRY` (no `QVarCircuitEncoder`)
- [x] M0.3 Implement `LineageTracker` writing `evolution_results/<session>/lineage.csv`
- [x] M0.4 Implement `EvolutionLoop` class (fresh, not extracted from old script)
- [x] M0.5 Implement `FitnessFunction` protocol with `EpisodicSuccessRate` (frozen weights, no `.learn()`). `LearnedPerformanceFitness` deferred to M2.
- [x] M0.6 Delete existing `scripts/run_evolution.py`, `configs/evolution/qvarcircuit_foraging_small.yml`, and `test_run_evolution_smoke` (no `scripts/legacy/` fallback — git history is the archive)
- [x] M0.7 Write fresh `scripts/run_evolution.py` as a thin CLI wiring loop + encoder + fitness
- [x] M0.8 Extend `config_loader.py` with an `evolution:` block schema
- [x] M0.9 Create `configs/evolution/mlpppo_foraging_small.yml`
- [x] M0.10 Create `configs/evolution/lstmppo_foraging_small_klinotaxis.yml`
- [x] M0.11 Tests: encoder round-trip (MLPPPO + LSTMPPO), lineage CSV, smoke loop (48 unit tests)
- [x] M0.12 Smoke run: MLPPPO config end-to-end via `test_run_evolution_smoke_mlpppo` CI smoke (3.8 s)
- [x] M0.13 Smoke run: LSTMPPO+klinotaxis config end-to-end via `test_loop_smoke.py` unit tests (run real episodes with seeded determinism)
- [x] M0.14 Update this checklist + roadmap milestone tracker

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

**OpenSpec change**: `2026-04-27-add-hyperparameter-evolution` (archived as `2026-04-28-2026-04-27-add-hyperparameter-evolution`)
**Status**: complete. Four arms GO: MLPPPO+oracle +5.5pp (saturated), LSTMPPO+klinotaxis foraging +7.5pp (saturated), LSTMPPO+klinotaxis+predator under CMA-ES (M2.11) **+47.0pp on 4 seeds, non-saturated** with 3/4 reaching 0.92 best fitness and 1/4 dead-zone failure, and LSTMPPO+klinotaxis+predator under TPE (M2.12) **+79.0pp on 4 seeds**, with seed 43's CMA-ES dead zone rescued (0.000 → 1.000) and mean +32pp over CMA-ES on the same brain/sensing/schema/budget. RQ1 closed: M3 default optimiser is TPE. Roadmap row flipped to ✅ complete
**Bio fidelity**: LOW
**Brain target**: MLPPPO + LSTMPPO+klinotaxis
**Dependencies**: M0
**Decision gate**: GO if either brain ≥3pp over hand-tuned baseline AND fitness still rising at gen 20

- [x] M2.1 Implement `HyperparameterEncoder` in `evolution/encoders.py` (encodes config dict, not weights)
- [x] M2.2 Implement `LearnedPerformanceFitness` in `evolution/fitness.py`
- [x] M2.3 Create `configs/evolution/hyperparam_mlpppo_pilot.yml`
- [x] M2.4 Create `configs/evolution/hyperparam_lstmppo_klinotaxis_pilot.yml`
- [x] M2.5 Create campaign script(s) under `scripts/campaigns/`. The MLPPPO PR ships `phase5_m2_hyperparam_mlpppo.sh`; the LSTMPPO PR ships `phase5_m2_hyperparam_lstmppo_klinotaxis.sh` + matching baseline script. Per-brain split (rather than a single combined script) lets PR 3 ship cleanly even if PR 2's pilot decides STOP/PIVOT
- [x] M2.6 Run pilot: MLPPPO 20 gens × pop 12 × 4 seeds (PR 2 / re-run in PR 3); LSTMPPO+klinotaxis 20 gens × pop 12 × 2 seeds (PR 3). Both arms decide **GO** (MLPPPO +5.5pp, LSTMPPO+klinotaxis +7.5pp); recorded per-arm in logbook 012
- [x] M2.7 Publish combined logbook under `artifacts/logbooks/012/`. Single integrated logbook covers both arms (Part 1 = MLPPPO from PR 2, Part 2 = LSTMPPO+klinotaxis from PR 3 + bug-fix narrative) at `docs/experiments/logbooks/012-hyperparam-evolution-mlpppo-pilot.md` + supporting appendix; combined rather than per-arm because the bug-fix story spans both
- [x] M2.8 Update this checklist + roadmap milestone tracker
- [x] M2.9 Fix three M2 framework bugs surfaced by the LSTMPPO+klinotaxis arm investigation (PR 3): (a) `_build_agent` missed `max_body_length`, so `agent.reset_environment()` between episodes silently switched every episode-1+ to body=6; (b) `instantiate_brain_from_sim_config` skipped `apply_sensing_mode`, so non-oracle envs (klinotaxis, derivative, temporal) ran with oracle modules; (c) `LearnedPerformanceFitness.evaluate` and `EpisodicSuccessRate.evaluate` did not reseed per-episode (`set_global_seed(derive_run_seed(...))` + `agent.env.seed/rng` patch), so every reset rebuilt the same env layout. Each fix mirrors the canonical pattern in `scripts/run_simulation.py`. Three regression tests pinned the fixes (`test_build_agent_threads_max_body_length`, `test_instantiate_brain_translates_klinotaxis_modules`, `test_instantiate_brain_oracle_modules_unchanged`); existing `test_lstmppo_encoder_roundtrip` updated to no longer depend on bug (b). MLPPPO arm re-run under fixed framework (decision unchanged: GO +5.5pp); the `artifacts/logbooks/012/m2_hyperparam_pilot/` archive replaces PR #134's now-superseded data
- [x] M2.10 Add optional warm-start fitness to `LearnedPerformanceFitness` (PR 3): new `evolution.warm_start_path: Path | None` field on `EvolutionConfig`, plus a YAML-load-time validator that rejects warm-start configs whose `hyperparam_schema` includes architecture-changing fields (`actor_hidden_dim`, `lstm_hidden_dim`, `rnn_type`, etc. — `_ARCHITECTURE_CHANGING_FIELDS` in `config_loader.py`). When set, each genome's brain loads weights from the checkpoint AFTER `encoder.decode` and BEFORE the K train phase; missing path raises `FileNotFoundError` on first eval. Spec delta extends the existing `Learned-Performance Fitness` requirement with one paragraph + 3 scenarios. Six unit tests cover the happy path, validator rejection, validator allow (non-arch schema), arch-allowed-without-warm-start, mocked load, and missing-path error. Feature is shipped but unused in this PR — both M2 pilots run from-scratch — and is intended for M3+ work where harder fitness landscapes warrant fine-tuning a baseline rather than training from random init
- [x] M2.11 Add a third M2 arm — **LSTMPPO + klinotaxis + pursuit predators**. Done: pilot YAML at `configs/evolution/hyperparam_lstmppo_klinotaxis_predator_pilot.yml` (same 6-field schema as M2.4's LSTMPPO+klinotaxis arm; only env block differs — predator + nociception + health blocks copied from `configs/scenarios/pursuit/lstmppo_small_klinotaxis.yml`); two campaign scripts at `scripts/campaigns/phase5_m2_hyperparam_lstmppo_klinotaxis_predator{,_baseline}.sh`; archived under `artifacts/logbooks/012/m2_hyperparam_lstmppo_klinotaxis_predator_pilot/`. **Decision: GO +47.0pp** on 4 seeds (pilot mean 0.640 vs baseline 0.170). 3 of 4 seeds reach 0.92 best fitness; seed 43 stuck at 0.000 (dead-zone failure: actor_lr clipped at lower bound 1e-5 + entropy ≈ 3e-4 → brain can't update or explore). Inter-seed std 0.378 — first M2 arm with non-saturated landscape; CMA-ES is genuinely climbing a fitness gradient. Reveals a CMA-ES-on-narrow-landscape failure mode (1/4 = 25% dead-trajectory rate) that M2.12 (Optuna/TPE) will test directly and M3's Lamarckian inheritance can also improve over. After this lands, multi-modality arms (thermotaxis/aerotaxis per logbook 010) are deferred to M3+ scope
- [x] M2.12 Close RQ1: added an Optuna/TPE optimiser adapter (`OptunaTPEOptimizer` in [`packages/quantum-nematode/quantumnematode/optimizers/evolutionary.py`](../../../packages/quantum-nematode/quantumnematode/optimizers/evolutionary.py); `evolution.algorithm: tpe` selector via `Literal[...]` extension in `config_loader.py`; new `genome_bounds(sim_config)` method on `GenomeEncoder` protocol so gradient-free samplers can declare per-genome-dim ranges; `_build_optimizer` in `scripts/run_evolution.py` threads bounds through and rejects unbounded encoders for TPE). Re-ran the M2.11 predator-arm pilot under TPE — same brain + sensing + 6-field schema + K=50/L=25 + 4 seeds 42-45, only optimiser changed. **Decision: TPE wins on both criteria**. Mean final fitness 0.960 (TPE) vs 0.640 (CMA-ES) = +32pp, far above the +5pp gate. Seed 43 (CMA-ES dead zone: stuck at 0.000 across 20 gens) rescued to best fitness 1.000 under TPE. Wall: ~55 min total (4 seeds × ~14 min/seed). RQ1 closed inline — TPE is now first-class via this PR's spec delta + tests + adapter, so the original "open `<DATE>-add-optuna-optimizer`" plan is satisfied without a separate change. Spec delta added an "Optimiser Portfolio" requirement covering TPE selectability and the `genome_bounds` contract; 7 unit tests cover bounds validation, seed reproducibility, double-ask rejection, length-mismatch, result-before-tell sentinel. M3 starts on the predator config + `algorithm: tpe`.

## M3: Lamarckian Evolution Pilot

**OpenSpec change**: `2026-05-02-add-lamarckian-evolution` (archived in this PR; deltas synced into `openspec/specs/evolution-framework/spec.md`)
**Status**: complete. Speed gate +5.25 gens (lamarckian mean gen-to-0.92 = 4.50 vs control = 9.75; passes by 1.3× the +4 threshold). All 4 lamarckian seeds reach best fitness 1.00; control tops at 0.88-0.96 with seed 42 saturated at 0.88. Population mean climbs to 0.83-0.90 sustained vs control's 0.05-0.50. Inheritance rescues TPE-unlucky seed 42 — direct analogue of M2.12 rescuing M2.11's seed-43 dead-zone. Cross-schema check rules out the 4-vs-6-field schema simplification as a confounder (worth ~0 gens). 18 LSTMPPO trained tensors round-trip bit-exact through `save_weights → load_weights`. **M4 (Baldwin Effect) starts on this configuration.** See logbook 013
**Bio fidelity**: MEDIUM
**Brain target**: LSTMPPO+klinotaxis (predator arm only — M2 saturated arms can't measure inheritance signal)
**Dependencies**: M0, M2 (TPE optimiser from M2.12)
**Decision gate**: GO to M4 if LSTMPPO+klinotaxis shows ≥10pp faster convergence AND F1 eval-only success > random init — translated to the predator arm as: GO if mean_gen_lamarckian_to_092 + 4 ≤ mean_gen_control_to_092 AND mean_gen2_lamarckian ≥ mean_gen3_control (gen-2 reference, not gen-1, because gen-0-from-scratch is identical between arms by construction)

Scope changed from the original 8 sub-tasks (which assumed MLPPPO + LSTMPPO arms with a `2026-05-19-` date prefix and CMA-ES base) to the actual M3 scope after planning (predator arm only, TPE base optimiser, single-elite-broadcast, within-experiment from-scratch control instead of MLPPPO companion arm). The expanded checklist:

- [x] M3.1 Create `evolution/inheritance.py` with `InheritanceStrategy` Protocol + `NoInheritance` + `LamarckianInheritance` strategies
- [x] M3.2 Wire through `EvolutionLoop` via `--inheritance {none,lamarckian}` CLI flag with CLI guard rejecting `inheritance + --fitness success_rate` mismatch at startup
- [x] M3.3 Per-genome warm-start via `LearnedPerformanceFitness`'s new `warm_start_path_override` + `weight_capture_path` kwargs; checkpoint via `save_weights`/`load_weights` (round-trip bit-exact for LSTMPPO trained tensors); two-phase GC bounds disk usage to `2 * inheritance_elite_count` files
- [x] M3.4 ~~Create `configs/evolution/lamarckian_mlpppo_pilot.yml`~~ — DROPPED. M2 saturated arms (MLPPPO+oracle, LSTMPPO foraging) saturate at 1.000 from gen 1 and can't measure inheritance signal. Predator-arm-only scope decided during planning
- [x] M3.5 Create `configs/evolution/lamarckian_lstmppo_klinotaxis_predator_pilot.yml` (renamed from `lamarckian_lstmppo_klinotaxis_pilot.yml`) + `lamarckian_lstmppo_klinotaxis_predator_control.yml` (within-experiment from-scratch control) + matching campaign scripts under `scripts/campaigns/phase5_m3_*.sh`. Schema dropped `rnn_type` + `lstm_hidden_dim` (architecture-changing fields incompatible with per-genome warm-start; validator rejects)
- [x] M3.6 Run pilot: 4 seeds × 20 gens × pop 12 × K=50/L=25 under TPE base optimiser. Both arms re-run baseline (`scripts/campaigns/phase5_m2_hyperparam_lstmppo_klinotaxis_predator_baseline.sh` under M3 revision; reproduces M2.11's published 0.15/0.16/0.15/0.22 mean 0.170 — confirms run_simulation.py path is unchanged). Pre-pilot smoke (3 gens × pop 6 × seed 42) ran on both arms before committing to the 2-hour full-pilot wall
- [x] M3.7 Publish `artifacts/logbooks/013/m3_lamarckian_pilot/` (per-seed lamarckian + control + baseline + final-gen winner checkpoints) and `docs/experiments/logbooks/013-lamarckian-inheritance-pilot.md` (main + supporting appendix at `supporting/013/lamarckian-inheritance-pilot-details.md`). Aggregator at `scripts/campaigns/aggregate_m3_pilot.py` produces `summary.md` + `convergence.png` + `convergence_speed.csv`. Decision gate translated to two metrics: speed (+5.25 gens, PASS) and floor (corrected gen-2 reference: +0.42pp population mean, PASS)
- [x] M3.8 Update this checklist + roadmap milestone tracker (this file + `docs/roadmap.md`)
- [x] M3.9 Aggregator + within-experiment control YAML/script — added during planning so the lamarckian-vs-control comparison is confounder-free (no Python/dep/machine drift between M2.12's results and M3's). Cross-schema check (M3-control vs M2.12 6-field TPE: +0.25 gens speed margin) empirically rules out the 4-field-vs-6-field schema simplification as a confounder
- [x] M3.10 Pre-pilot smoke (task group 9b in the M3 OpenSpec change) — added during planning to validate framework correctness at real K=50/L=25 + parallel=4 scale before committing to the ~2-hour full-pilot wall. Both arms passed all mechanical assertions in ~85s + ~65s

## M4: Baldwin Effect Demonstration

**OpenSpec change**: `2026-05-03-add-baldwin-evolution` (archived in this PR; deltas synced into `openspec/specs/evolution-framework/spec.md`)
**Status**: complete. **STOP ❌**. Speed gate FAIL — Baldwin mean gen-to-0.92 (8.50) exactly matches the M3-control rerun (8.50), margin +0.00 vs need ≥2. Genetic-assimilation gate FAIL — F1 innate-only mean 0.000 across all 4 seeds, vs need >baseline (0.170) +0.10. Comparative gate trivially PASS. Baldwin Effect does not manifest as evolutionary acceleration on this codebase + arm. The two new evolvable knobs were genuinely explored by TPE (weight_init_scale 0.57-1.33; entropy_decay_episodes 1022-1562) but produce no exploitable signal. Lamarckian rerun reproduces M3 exactly (mean 4.50, all 4 seeds at 1.00) — confirms the M4 framework changes (kind() Protocol + early-stop + weight_init_scale brain field) are byte-equivalent for the M3 path. The negative result isolates the source of M3's +5.25-gen acceleration to bit-exact trained-weight transfer, not to elite-ID lineage flow. See logbook 014. M5 (co-evolution) unblocked independently; M6 (transgenerational memory) needs Lamarckian (or new mechanism) as substrate, not Baldwin
**Bio fidelity**: MEDIUM
**Brain target**: LSTMPPO+klinotaxis
**Dependencies**: M3 GO
**Decision gate**: GO if all three: (a) Speed (Baldwin vs control): mean_gen_baldwin_to_092 + 2 ≤ mean_gen_control_to_092; (b) Genetic-assimilation (F1 vs hand-tuned baseline): mean_f1_baldwin > mean_baseline + 0.10; (c) Comparative (Baldwin vs Lamarckian): mean_gen_baldwin_to_092 ≤ mean_gen_lamarckian_to_092 + 4

Scope changed from the original 7 sub-tasks (which assumed an interleaved learning-blocked F1 control cohort) to the actual M4 scope after planning (post-hoc F1 evaluator instead). Plus a richer learnability schema with two new evolvable knobs (`weight_init_scale`, `entropy_decay_episodes`) and a new `--early-stop-on-saturation N` loop flag (the wishlist item below was upgraded from "optional" to "shipped"). The expanded checklist:

- [x] M4.1 Implemented `BaldwinInheritance` in `evolution/inheritance.py` with the `kind() -> Literal["none", "weights", "trait"]` Protocol extension (option (a) from the original task draft). The two-guard split in `loop.py` (lineage-tracking via `_inheritance_records_lineage()` for both Lamarckian and Baldwin; weight-IO GC via `_inheritance_active()` for Lamarckian only) cleanly resolves the "Baldwin returns None checkpoint paths but should still record lineage" tension the M3 plan flagged
- [x] M4.2 ~~Implement learning-blocked F1 control cohort (config fields: `control_cohort_interval`, `control_cohort_fraction`)~~ — SCOPE CHANGED during planning. Replaced with a post-hoc evaluator at `scripts/campaigns/baldwin_f1_postpilot_eval.py` (per design Decision 6 of the OpenSpec change). Keeps the loop unchanged and makes F1 a clean forensic step. F1 result: 0.000 across all 4 seeds — the elite hyperparameter genome alone produces no useful behaviour without K=50 training (no genetic assimilation observed)
- [x] M4.3 Created `configs/evolution/baldwin_lstmppo_klinotaxis_predator_pilot.yml` (renamed from `baldwin_lstmppo_klinotaxis_full.yml`). 6-field schema (M3 control's 4 + `weight_init_scale` + `entropy_decay_episodes`); `inheritance: baldwin`; `early_stop_on_saturation: 5`
- [x] M4.4 ~~Create `configs/evolution/baldwin_comparison_campaign.yml`~~ — replaced with three campaign scripts under `scripts/campaigns/phase5_m4_*.sh` (Baldwin pilot + Lamarckian rerun + control rerun) + the F1 post-pilot evaluator + the 4-way aggregator at `scripts/campaigns/aggregate_m4_pilot.py`. The single-YAML "comparison campaign" pattern doesn't generalise once you have three arms with three different configs
- [x] M4.5 Ran 3-arm head-to-head: Baldwin pilot + Lamarckian rerun + control rerun under TPE, 4 seeds (42-45) × up to 20 gens with `--early-stop-on-saturation 5` cutting wall to ~67 min total. **Verdict: STOP** per the Status header above. Lamarckian rerun reproduces M3 exactly (Lamarckian: [3, 4, 4, 7] gen-to-092 mean 4.50 vs M3 published [3, 4, 4, 7]; all 4 seeds at 1.00). Worker-tuple consideration from the original task draft: M3's 11-element tuple was sufficient for M4 — Baldwin doesn't need new metadata threading, since the elite ID flows through the existing `_selected_parent_ids` array per design Decision 2
- [x] M4.6 Published `artifacts/logbooks/014/m4_baldwin_pilot/{baldwin,lamarckian,control,baseline,summary}/` (per-seed best_params.json + history.csv + lineage.csv + checkpoint.pkl + Lamarckian's surviving final-gen elite under `inheritance/`) and `docs/experiments/logbooks/014-baldwin-inheritance-pilot.md` (main + supporting appendix at `supporting/014/baldwin-inheritance-pilot-details.md`). Aggregator at `scripts/campaigns/aggregate_m4_pilot.py` produces `summary.md` + `convergence.png` + `convergence_speed.csv`
- [x] M4.7 Updated this checklist + `docs/roadmap.md` (M4 row flipped to ✅ complete with STOP verdict summary)
- [x] M4.8 Pre-pilot smoke (task group 7 in the M4 OpenSpec change) — added during planning to validate framework correctness at real K=50/L=25 scale before committing to the ~70-min full-pilot wall. Baldwin smoke at 3 gens × pop 6 × seed 42 ran in ~90s; verified lineage gen-0 rows have empty `inherited_from`, gen-1+ rows share the prior-gen elite ID, NO `inheritance/` directory created. Early-stop smoke verified the counter walks correctly through `[0.0, 0.0, 0.5, 0.0, 0.0]` with N=2 → fires after gen 5
- [x] M4.9 Two new evolvable knobs + new brain field (task group 3 in the M4 OpenSpec change) — `weight_init_scale: float = Field(default=1.0, ge=0.1, le=5.0)` added to `LSTMPPOBrainConfig`. Scales the orthogonal-init `gain` for the actor's hidden Linears + critic's Linears (the actor's small-init output `gain=0.01` is preserved as the standard PPO trick; the LSTM/GRU module is unaffected). 8 unit tests cover byte-equivalence at scale=1.0, exact 2× scaling at scale=2.0, validator boundary handling. `entropy_decay_episodes` already existed in the brain config; M4 just exposes it via the Baldwin pilot's hyperparam_schema

The "M4 wishlist before kickoff" item from the original tracker (`--early-stop-on-saturation N` flag) was UPGRADED from optional to shipped: `EvolutionConfig.early_stop_on_saturation: int | None = None`, CLI override `--early-stop-on-saturation N`, persisted in the checkpoint pickle (CHECKPOINT_VERSION 2 → 3) so resume preserves the saturation counter. 4 unit tests in `test_early_stop.py` cover the gen-1 bootstrap, monotonic improvement, counter reset on strict improvement, and resume preservation. Saved roughly half the per-seed wall on saturating arms (none of the 12 arm-seed combos reached the 20-gen budget).

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

## Phase 5 Research Questions

Open research questions surfaced during Phase 5 milestones that don't fit cleanly under any single milestone but are worth tracking. Each question has a concrete trigger condition for escalation; nothing here commits us to work upfront.

### RQ1: Optimiser portfolio re-evaluation

**Status**: closed — TPE wins decisively. M3 default optimiser is TPE.
**Trigger**: M2.11 (predator arm) surfaced a CMA-ES dead-zone failure mode (seed 43 stuck at 0.000 across 20 generations because CMA-ES converged on `actor_lr` clipped at lower bound + `entropy_coef` ≈ 0). 1/4 dead-trajectory rate is exactly the kind of pathology TPE's tree-structured prior + bounded uniform sampling are designed to avoid.
**Outcome (M2.12)**: TPE mean final fitness 0.960 vs CMA-ES 0.640 = **+32pp** (gate was +5pp; passed 6×); seed 43's dead zone rescued from 0.000 → 1.000 (4 of 4 seeds at >0.92 under TPE vs 3 of 4 under CMA-ES). TPE is now first-class via the `OptunaTPEOptimizer` adapter shipped in M2.12 — no separate `<DATE>-add-optuna-optimizer` change needed. See logbook 012 § Part 4 for the full comparison + mechanism analysis (bounded uniform prior + per-trial sampling avoids CMA-ES's variance-collapse-onto-bound failure mode).

The framework's current optimisers (CMA-ES, GA) were chosen in earlier roadmap phases for brain-weight evolution at large genome dim (n=9k–47k), where CMA-ES diagonal mode is genuinely the right tool. M2's hyperparameter-evolution use case is fundamentally different — n=7–20, few-hundred-evaluation budgets, mixed-type genomes (float / int / bool / categorical), conditional parameter dependencies. The broader ML community standard for that regime is Bayesian Optimisation (BO) or Tree-structured Parzen Estimator (TPE) — Optuna's default. Compared to CMA-ES at the M2 scale, TPE/BO handles categoricals natively (no bin-plateau), handles log-scale floats natively, and is generally more sample-efficient.

We shipped M2.1-M2.11 with CMA-ES because (a) it's the path of minimum framework change from M0, and (b) M2 is a pilot — its job is to exercise the framework and produce a GO/PIVOT/STOP signal, not to find globally-optimal configs. M2.11 produced both the GO signal and a concrete CMA-ES failure mode worth probing.

**Escalation rule (now actionable)**: M2.12 runs the **predator arm** pilot config (same brain, same sensing, same 6-field schema, same 4 seeds, same K=50/L=25 budget) under Optuna's TPE sampler. **Decision criteria**:

- If TPE's mean final fitness across seeds is ≥5pp higher than CMA-ES's 0.640 (i.e. TPE ≥0.690), AND/OR TPE rescues seed 43's dead-zone trajectory: open a follow-up change `<DATE>-add-optuna-optimizer` to make Optuna a first-class optimiser; M3 uses TPE.
- If TPE is comparable or worse: close RQ1; CMA-ES stays the default and Optuna isn't worth the dependency. M3 uses CMA-ES.

**Related future need**: M6.5 (NEAT topology evolution) inherently requires a NEAT-specific optimiser, not CMA-ES or BO. So an "optimiser portfolio" will exist in the Phase 5 future regardless of this RQ's outcome — the M2-driven question is whether to add one optimiser earlier (BO/TPE for hyperparameters) or wait until M6.5 to expand the portfolio.

**Recorded by**: M2 OpenSpec change `2026-04-27-add-hyperparameter-evolution` (`design.md` § Considered Alternatives — Optimiser choice for hyperparameter evolution); follow-up tracked as M2.12 above.
