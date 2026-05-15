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

**OpenSpec change**: `add-learning-predators` (archived on PR merge)
**Status**: ✅ complete. PredatorBrain Protocol + HeuristicPredatorBrain adapter + per-predator metrics shipped with **zero behavioural change** verified at two levels: 23 byte-equivalence unit tests over the `{STATIONARY, PURSUIT} × speed ∈ {0.5, 1.0, 2.0}` parameter grid (step-by-step position equality + RNG-state equality across 1000-step horizons), AND 80/80 metric-cell pre/post deltas exactly 0.0 across the regression-baseline campaign (20 (config, seed) cells × 4 metrics; multi-agent + single-agent arms × MLPPPO + LSTMPPO scenarios). M5 unblocked. See logbook 016
**Bio fidelity**: MEDIUM
**Brain target**: heuristic predator (M5 will introduce learnable predator brains via the PredatorBrain Protocol seam)
**Dependencies**: M0 ✅

- [x] M1.1 Created `env/predator_brain.py`: `PredatorBrain` Protocol, `HeuristicPredatorBrain` adapter, `PredatorBrainParams` frozen dataclass, `PredatorAction` enum, `PredatorBrainConfig` runtime dataclass
- [x] M1.2 `Predator.update_position` delegates to `self.brain.run_brain(params)`; legacy `_update_pursuit` / `_update_random` deleted; new `_apply_action` helper owns kinematics. Frozen-branch invariant (`chase_target` + `is_pursuing` resolved once per call) preserves byte-equivalence on multi-step movement at speed > 1.0
- [x] M1.3 `PredatorParams.brain_config` field; `_initialize_predators` defaults to `HeuristicPredatorBrain` via `_build_predator_brain` dispatcher; `_make_predator` factory centralises construction at all three sites; YAML schema rejects unknown kinds at load time
- [x] M1.4 `MultiAgentEpisodeResult` exposes `per_predator_kills` / `per_predator_prey_proximity_steps` / `per_predator_distance_traveled`. Kill attribution: closest covering predator (Manhattan) with lex tie-break on `predator_id`; defensive global-closest fallback for residual-HP edge cases
- [x] M1.5 Tests at `tests/env/test_predator_brain.py` (21) + `test_predator_brain_byte_equivalence.py` (23) + `test_predator_brain_config.py` (15) + `test_multi_agent.py::TestPerPredatorMetrics` (6) = **65 new tests**; full env + multi_agent suite passes 369 (was 305, +64 net new — one redundant test in `test_predator_brain.py` dropped during lint cleanup)
- [x] M1.6 Regression: 4-seed × 200-episode multi-agent + 4-seed × 100-episode single-agent campaigns × 4 metrics on commits `73684213` (pre-M1) and `3d45e75c` (M1 head). All 80 metric-cells show exactly 0.0 delta. See `artifacts/logbooks/016-predator-brain-refactor/{baseline_pre,baseline_post}.csv`
- [x] M1.7 `uv run pytest -m smoke -v` clean (22 tests pass)
- [x] M1.8 This checklist updated; `docs/roadmap.md` Phase 5 milestone tracker M1 row flipped to ✅ complete

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

**OpenSpec change**: `add-baldwin-evolution` (NOT archived in this PR — kept open as the substrate for M4.5 and as the historical record of the audit + redesign plan)
**Status**: complete (framework shipped) — **INCONCLUSIVE ⚠️** on the science. The literal aggregator verdict was STOP (Speed gate FAIL margin +0.00; Genetic-assimilation gate FAIL margin -0.17; Comparative gate PASS), but a post-pilot audit found three blocking design flaws that mean the gates compared confounded populations rather than the Baldwin mechanism vs no-Baldwin: (A1) **schema-shift confounder** — Baldwin's 6-field schema produces a different TPE prior than Control's 4-field schema, so Baldwin's gen-0 starting populations are systematically -0.14pp weaker than Control's before any inheritance signal can fire (mean gen-0 best 0.580 vs 0.720); (A2) **F1 test is biologically incoherent** — Baldwin Effect predicts learning-acceleration, not innate-without-learning; the F1 evaluator measured the latter (random-LSTM-no-training); (A3) **F1 baseline is apples-to-oranges** — 0 episodes of learning vs 100 training episodes of the hand-tuned baseline can't be compared. Plus n=4 seeds is too few for ±2-gen statistical detection. The framework changes (kind() Protocol method, BaldwinInheritance strategy, two-guard loop split, weight_init_scale brain field, --early-stop-on-saturation flag) are validated and shipped in this PR. Lamarckian rerun reproduces M3 exactly (mean 4.50, all 4 seeds at 1.00) — confirms framework changes are byte-equivalent for the M3 path. **M4.5 (the proper Baldwin re-evaluation: matched 6-field schemas across both arms, learning-acceleration F1 test, n≥8 seeds, possibly architecture-changing innate-bias knobs) is recorded as the follow-up.** M5 (co-evolution) unblocked independently; M6 (transgenerational memory) deferred until M4.5 settles the Baldwin question. See logbook 014 § Audit + § Decision
**Bio fidelity**: MEDIUM
**Brain target**: LSTMPPO+klinotaxis
**Dependencies**: M3 GO
**Decision gate**: GO if all three: (a) Speed (Baldwin vs control): mean_gen_baldwin_to_092 + 2 ≤ mean_gen_control_to_092; (b) Genetic-assimilation (F1 vs hand-tuned baseline): mean_f1_baldwin > mean_baseline + 0.10; (c) Comparative (Baldwin vs Lamarckian): mean_gen_baldwin_to_092 ≤ mean_gen_lamarckian_to_092 + 4

Scope changed from the original 7 sub-tasks (which assumed an interleaved learning-blocked F1 control cohort) to the actual M4 scope after planning (post-hoc F1 evaluator instead). Plus a richer learnability schema with two new evolvable knobs (`weight_init_scale`, `entropy_decay_episodes`) and a new `--early-stop-on-saturation N` loop flag (the wishlist item below was upgraded from "optional" to "shipped"). The expanded checklist:

- [x] M4.1 Implemented `BaldwinInheritance` in `evolution/inheritance.py` with the `kind() -> Literal["none", "weights", "trait"]` Protocol extension (option (a) from the original task draft). The two-guard split in `loop.py` (lineage-tracking via `_inheritance_records_lineage()` for both Lamarckian and Baldwin; weight-IO GC via `_inheritance_active()` for Lamarckian only) cleanly resolves the "Baldwin returns None checkpoint paths but should still record lineage" tension the M3 plan flagged
- [x] M4.2 ~~Implement learning-blocked F1 control cohort (config fields: `control_cohort_interval`, `control_cohort_fraction`)~~ — SCOPE CHANGED during planning. Replaced with a post-hoc evaluator at `scripts/campaigns/baldwin_f1_postpilot_eval.py` (per design Decision 6 of the OpenSpec change). The evaluator works correctly (script + smoke verified) but the AUDIT (M4.10) found the F1 test design itself is biologically incoherent for Baldwin and apples-to-oranges vs the hand-tuned baseline. F1 yielded 0.000 across all 4 seeds, but this is a test-design failure, not a Baldwin-mechanism finding. M4.5 will redesign F1 as a learning-acceleration test
- [x] M4.3 Created `configs/evolution/baldwin_lstmppo_klinotaxis_predator_pilot.yml` (renamed from `baldwin_lstmppo_klinotaxis_full.yml`). 6-field schema (M3 control's 4 + `weight_init_scale` + `entropy_decay_episodes`); `inheritance: baldwin`; `early_stop_on_saturation: 5`. AUDIT (M4.10) found the schema choice introduces a confounder vs the M3 4-field control; M4.5 will run a matched 6-field control alongside Baldwin
- [x] M4.4 ~~Create `configs/evolution/baldwin_comparison_campaign.yml`~~ — replaced with three campaign scripts under `scripts/campaigns/phase5_m4_*.sh` (Baldwin pilot + Lamarckian rerun + control rerun) + the F1 post-pilot evaluator + the 4-way aggregator at `scripts/campaigns/aggregate_m4_pilot.py`. The single-YAML "comparison campaign" pattern doesn't generalise once you have three arms with three different configs
- [x] M4.5 Ran 3-arm head-to-head: Baldwin pilot + Lamarckian rerun + control rerun under TPE, 4 seeds (42-45) × up to 20 gens with `--early-stop-on-saturation 5` cutting wall to ~67 min total. Lamarckian rerun reproduces M3 exactly (Lamarckian: [3, 4, 4, 7] gen-to-092 mean 4.50 vs M3 published [3, 4, 4, 7]; all 4 seeds at 1.00) — confirming framework byte-equivalence. Literal aggregator output: STOP. AUDIT (M4.10) downgraded to INCONCLUSIVE because the gates compared confounded populations. Worker-tuple consideration from the original task draft: M3's 11-element tuple was sufficient for M4 — Baldwin doesn't need new metadata threading, since the elite ID flows through the existing `_selected_parent_ids` array per design Decision 2
- [x] M4.6 Published `artifacts/logbooks/014/m4_baldwin_pilot/{baldwin,lamarckian,control,baseline,summary}/` (per-seed best_params.json + history.csv + lineage.csv + checkpoint.pkl + Lamarckian's surviving final-gen elite under `inheritance/`) and `docs/experiments/logbooks/014-baldwin-inheritance-pilot.md` (main + supporting appendix at `supporting/014/baldwin-inheritance-pilot-details.md`). Aggregator at `scripts/campaigns/aggregate_m4_pilot.py` produces `summary.md` + `convergence.png` + `convergence_speed.csv`. Logbook + appendix updated post-audit to document the INCONCLUSIVE verdict + audit findings + M4.5 redesign plan
- [x] M4.7 Updated this checklist + `docs/roadmap.md` (M4 row marked ✅ framework shipped, ⚠️ INCONCLUSIVE on the science with audit summary)
- [x] M4.8 Pre-pilot smoke (task group 7 in the M4 OpenSpec change) — added during planning to validate framework correctness at real K=50/L=25 scale before committing to the ~70-min full-pilot wall. Baldwin smoke at 3 gens × pop 6 × seed 42 ran in ~90s; verified lineage gen-0 rows have empty `inherited_from`, gen-1+ rows share the prior-gen elite ID, NO `inheritance/` directory created. Early-stop smoke verified the counter walks correctly through `[0.0, 0.0, 0.5, 0.0, 0.0]` with N=2 → fires after gen 5
- [x] M4.9 Two new evolvable knobs + new brain field (task group 3 in the M4 OpenSpec change) — `weight_init_scale: float = Field(default=1.0, ge=0.1, le=5.0)` added to `LSTMPPOBrainConfig`. Scales the orthogonal-init `gain` for the actor's hidden Linears + critic's Linears (the actor's small-init output `gain=0.01` is preserved as the standard PPO trick; the LSTM/GRU module is unaffected). 8 unit tests cover byte-equivalence at scale=1.0, exact 2× scaling at scale=2.0, validator boundary handling. `entropy_decay_episodes` already existed in the brain config; M4 just exposes it via the Baldwin pilot's hyperparam_schema. AUDIT (M4.10) found `weight_init_scale` may not be the optimal innate-bias knob for a K=50 budget — its effect is largely washed out by gradient descent within 50 episodes. M4.5 will try architecture-changing knobs (allowed under Baldwin per design Decision 3) instead
- [x] M4.10 Post-pilot audit (added during PR review). Found three blocking design flaws that mean the M4 pilot did not measure what the gates claimed to measure: (A1) **schema-shift confounder** — Baldwin's 6-field schema produces a different TPE prior than Control's 4-field schema, so Baldwin gen-0 starting populations are systematically -0.14pp weaker than Control's (mean gen-0 best 0.580 vs 0.720); (A2) **F1 test biologically incoherent** — measures innate-without-learning, not learning-acceleration which is what Baldwin Effect actually predicts; (A3) **F1 baseline apples-to-oranges** — 0 episodes of learning vs 100 training episodes can't be compared. Plus n=4 seeds is statistically underpowered for ±2-gen detection (A4), and the chosen knobs may not be optimal for K=50 (A5). Verdict downgraded from STOP to INCONCLUSIVE; framework changes ship as-is; OpenSpec change `add-baldwin-evolution` NOT archived in this PR — kept open as substrate for M4.5 and historical record. See logbook 014 § Audit + § Decision

The "M4 wishlist before kickoff" item from the original tracker (`--early-stop-on-saturation N` flag) was UPGRADED from optional to shipped: `EvolutionConfig.early_stop_on_saturation: int | None = None`, CLI override `--early-stop-on-saturation N`, persisted in the checkpoint pickle (CHECKPOINT_VERSION 2 → 3) so resume preserves the saturation counter. 4 unit tests in `test_early_stop.py` cover the gen-1 bootstrap, monotonic improvement, counter reset on strict improvement, and resume preservation. Saved roughly half the per-seed wall on saturating arms (none of the 12 arm-seed combos reached the 20-gen budget).

## M4.5: Baldwin Effect Demonstration (proper retry — iteration step 1)

**OpenSpec change**: `add-baldwin-retry` (archived in this PR as `2026-05-03-add-baldwin-retry`)
**Status**: complete (audit closures + structural finding) — **iteration step 1 of N** (M4.6 is the follow-up)
**Bio fidelity**: MEDIUM
**Brain target**: LSTMPPO+klinotaxis (same as M4)
**Dependencies**: M4 framework changes shipped

The M4 audit (task M4.10 above) identified three blocking design flaws + two significant issues in the original Baldwin evaluation. M4.5 redesigned the comparison to fix all five — and the clean run revealed a structural finding requiring an additional iteration (M4.6, see below).

- [x] M4.5.1 Equalised Baldwin vs Control schemas: ran an 8-field control alongside the 8-field Baldwin pilot (M4's 4 hyperparam knobs + M4's 2 innate-bias knobs + 2 NEW arch knobs per add-baldwin-retry design Decision 1) — same TPE prior at gen 0 → identical starting populations across arms. **Audit A1 closed PERFECTLY: |Δ| = 0.0000** (gen-0 mean best_fitness 0.715 for both arms; verified bit-identical lineage CSV rows). The aggregator's pre-flight check at the 0.05 tolerance trivially passed.
- [x] M4.5.2 F1 redesigned as a learning-acceleration test. Per pilot seed: elite genome from `best_params.json` → train K' = 10 episodes → measure success rate; compare to a schema-prior baseline genome (deterministic per-seed sample from each schema entry's prior via `HyperparameterEncoder.initial_genome(sim_config, rng=np.random.default_rng(seed))`) trained the same way. Both arms go through `LearnedPerformanceFitness.evaluate` with the same per-seed seed (identical env trajectory; only genome differs). **Audit A2 + A3 addressed.** K'=10 produced all-zeros across all 8 seeds (too small a budget for this task); K'=25 (Risk 4 fallback per design) produced a real signal (mean elite 0.110 vs baseline 0.000) but the elites are bit-identical to Control's elites, so the F1 signal isn't a Baldwin-vs-Control comparison.
- [x] M4.5.3 Increased n to 8 seeds (42-49). SE on gen-to-092 dropped from ~1.0-1.3 gens (n=4) to ~0.6-0.9 gens (n=8) — sufficient for the speed-gate ±2 threshold to be 2-3σ. **Audit A4 addressed.**
- [x] M4.5.4 Added `actor_hidden_dim` + `actor_num_layers` ALONGSIDE M4's `weight_init_scale` + `entropy_decay_episodes` (head-to-head 8-field schema per M4.5.1). Forensic: TPE explored ALL 8 fields broadly; `actor_hidden_dim` consistently lands at 149-251 (above the default 64) — supports M4 audit A5's hypothesis that arch knobs matter under K=50. But this exploration happens IDENTICALLY in Baldwin and Control, so it's verdict-orthogonal. **Audit A5 partially addressed** (arch knobs ARE explored, NOT pinned at defaults).
- [x] M4.5.5 Gate thresholds calibrated against M4.5's own arm reruns rather than M4's published numbers (since M4.5 uses an 8-field schema, M4's 8.50 reference doesn't apply directly). Speed gate `+2`, F1 gate `+0.05` (tightened from M4's `+0.10` because both arms now include K' training — within-test comparison with symmetric noise), comparative gate `+4` retained.
- [x] M4.5.6 Ran the redesigned 4-arm pilot (Baldwin + 8-field Control + Lamarckian rerun (4-field, n=8) + F1 post-pilot eval at K'=10 and K'=25). Aggregator emitted STOP per its 3-gate logic (Speed FAIL margin +0.00; F1 FAIL signal_delta +0.000 at K'=10; Comparative PASS margin +0.38). Published **logbook 015** (this logbook spans the iteration: M4.5 documented as iteration step 1; M4.6 will append step 2). Total pilot wall ~4h 6min (3 arms in parallel + F1 + aggregator).
- [x] M4.5.7 Update tracker + roadmap with the M4.5 verdict; flip M4 from INCONCLUSIVE to whichever the data supports — **resolved by M4.6.7** (M4 row reframed to "✅ closed (3 iterations) — STOP verdict; substrate constraint identified per Fernando 2018 / Chiu 2024 literature; M5 may surface Baldwin signal serendipitously via co-evolving predator non-stationarity").

**Iteration step 1 verdict** (logbook 015): Audit A1-A5 all addressed; framework's current Baldwin abstraction proven structurally null vs Control (5 conditions ⇒ bit-identical genome populations, verified empirically). M4.6 will implement an abstraction with selection feedback and re-evaluate.

**Reusable infrastructure shipped** (consumable by M4.6 and beyond):

- Redesigned F1 evaluator (`scripts/campaigns/baldwin_f1_postpilot_eval.py`) — paired K'-train + L-eval, schema-prior baseline, append-mode CSV
- 4-way aggregator (`scripts/campaigns/aggregate_baldwin_retry_pilot.py`) — schema-equalisation pre-flight + 3 gates + INCONCLUSIVE override
- 8-field pilot configs (Baldwin + matching Control)
- Campaign scripts at n=8 seeds (42-49)
- 22 unit tests covering F1 evaluator + aggregator gate logic

## M4.6: Baldwin Effect Demonstration (iteration step 2 — STOP closure)

**OpenSpec change**: none (M4.6 ships as STOP closure with no framework code changes — investigation + analysis + documentation only)
**Status**: ✅ closed — STOP verdict (3-iteration arc concluded)
**Bio fidelity**: n/a (closed iteration)
**Brain target**: n/a
**Dependencies**: M4.5 ships ✅

After M4.5's structural finding (the framework's `inherited_from` is metadata-only and doesn't feed back into TPE's selection), M4.6 was scoped as "implement an abstraction with selection feedback and re-evaluate." A planning + smoke phase ruled out the candidate abstractions on substrate grounds — the constraint isn't the abstraction, it's the single-task fitness landscape. M4.6 closes the Baldwin question for Phase 5.

- [x] M4.6.1 Enumerate abstraction candidates (informed by Fernando 2018 + Chiu 2024 literature search and a structured options-comparison sub-agent pass). Three candidates surfaced: **B3** learning-gain composite fitness (`fit_K=50 + α · (fit_K=50 - fit_K=25)`), **B6** cost-of-learning penalty (`fit - α · episodes_to_threshold`), and **truncated-K hybrid** (Baldwin selects on `fit_K=25`, Control on `fit_K=50`). All three change the scalar handed to TPE's `tell()`, which would mechanically break M4.5's bit-identity result.
- [x] M4.6.2 Pre-flight smoke (`tmp/m4_6_b3_smoke.py` → `artifacts/logbooks/015/m4_6_smoke/`). Measured per-genome (fit_K=10, fit_K=25, fit_K=50) triples for the 12-genome gen-0 population deterministically reproduced from the M4.5 Baldwin pilot's seed 42 (~3min wall). Result: **Spearman(fit_K=50, Δ_25→50) = 1.000** → B3 is provably rank-preserving, composite is a near-no-op for TPE. Non-zero genomes by K: K=10: 0/12, K=25: 1/12, K=50: 3/12 → truncated-K and B6 both lose TPE bootstrap (selection gradient collapses on near-zero variance).
- [x] M4.6.3 Substrate diagnosis: smoke evidence + literature converge on the **substrate** being the constraint, not the abstraction. Single-task K=50 PPO has no Baldwin axis — optimal strategy is "innate-task-specific behaviour," opposite of what Baldwin selects for. Published Baldwin demonstrations on hyperparameter/initial-weight evolution (Fernando 2018 *Meta-Learning by the Baldwin Effect*, Chiu 2024 *Baldwinian PINN*) **both use task distributions** to create the selection pressure for general learners.
- [x] M4.6.4 Cost-of-fix assessment: Path A-minimal (multi-task aggregation wrapper + env-variant configs + validator + tests) is ~5 days impl + ~9-14h pilot wall. Path A-proper (per-layer init scales for richer Baldwin axis) adds ~2 more days. Path A-Fernando-faithful (full initial-weight evolution) adds ~5 more days and re-opens RQ1 (TPE-vs-CMA-ES). All paths yield an incremental scientific contribution (reproducing Fernando 2018's published result on a TPE substrate) rather than novel research.
- [x] M4.6.5 STOP decision: Baldwin question closed for Phase 5. Reasoning documented in logbook 015 § Iteration step 2 § Decision: (a) marginal scientific value is incremental not novel; (b) no technical blocker (M5/M6 don't depend on a Baldwin GO); (c) M5's co-evolving predators intrinsically introduce task variation, may surface Baldwin serendipitously; (d) the 5-day Path A-minimal investment is better spent on M5 directly; (e) the 3-iteration arc produces a defensible STOP rather than an open-ended punt.
- [x] M4.6.6 Logbook 015 updated with iteration step 2 (smoke evidence + substrate diagnosis + STOP decision + Conclusions/Next Steps reframed to span both iterations); status flipped from `IN PROGRESS` to `CLOSED — STOP after iteration step 2 (M4.6)`.
- [x] M4.6.7 M4 row in this checklist + roadmap reframed from "iterative IN PROGRESS" to "✅ closed (3 iterations) — STOP verdict; substrate constraint identified". M5/M6 dependencies un-deferred (M5 proceeds without Baldwin; M6 uses Lamarckian per the original roadmap dependency text). Optional **M4.7** added below as deferred follow-up.

**Iteration step 2 verdict** (logbook 015 § Iteration step 2 § Decision): STOP. The Baldwin question is closed for Phase 5 after three iterations (M4 audit → M4.5 audit closures + structural finding → M4.6 substrate-constraint diagnosis). Each iteration narrowed the diagnosis. Future revisit gated on M5's outcome (see M4.7 below).

## M4.7 (deferred): Multi-task Baldwin Effect retry

**OpenSpec change**: none (deferred milestone; scaffold only if/when triggered)
**Status**: 🔲 deferred — gated on M5's outcome
**Bio fidelity**: HIGH (would address the substrate constraint identified in M4.6)
**Brain target**: LSTMPPO+klinotaxis or whatever M5 settles on
**Dependencies**: M5 ships

Optional follow-up that revisits the Baldwin Effect with **task-distribution selection pressure** (the published mechanism from Fernando 2018, Chiu 2024). Would require a multi-task fitness aggregation wrapper + env-variant configs (~5-7 days impl + pilot + overhead).

**Trigger conditions** (any one is sufficient):

- M5 (co-evolution) produces multi-task aggregation infrastructure that makes M4.7 a small marginal investment (≤2 days incremental work)
- M5 doesn't surface a Baldwin signal serendipitously AND demonstrating Baldwin rises in priority for a future Phase 5/6 narrative
- A separate research question (e.g. M6 transgenerational memory) requires multi-task pressure infrastructure that M4.7 could share

**Non-trigger**: do NOT scaffold M4.7 just to "complete" the Baldwin question. The 3-iteration STOP verdict is scientifically defensible by reference to published literature; reopening should be motivated by either infrastructure savings or a new research question.

## M5: Co-Evolution Arms Race

**OpenSpec change**: `2026-06-16-add-coevolution` (not yet created)
**Status**: next (post-M1) — headline scientific milestone post-M4
**Bio fidelity**: HIGH
**Brain target**: LSTMPPO+klinotaxis prey, MLPPPO predator
**Dependencies**: M0 ✅, M1
**Decision gate (softened, disjunctive)**: GO if **(a) phenotypic cycling visible OR (b) measurable trait escalation in at least one trait dim** over ≥30 gens. (Original conjunctive "cycling AND monotonic escalation" softened post-M4 replan: [Sci Reports 2026 — apparent decoupling of dual-trait predator-prey dynamics](https://www.nature.com/articles/s41598-026-50762-1) finds that "rapid prey evolution in multidimensional trait space can stabilize population dynamics and yield stable ecological equilibrium, even as traits continue to undergo persistent coevolutionary cycles" — i.e. stable populations and cycling traits can coexist, so requiring monotonic escalation rules out a known-realistic regime.)

**Literature grounding**: [Sakana Digital Red Queen 2025](https://pub.sakana.ai/drq/) (self-play Red Queen via continual adaptation to changing objective), [Entropy 2021 MARL co-evolution](https://www.mdpi.com/1099-4300/23/4/461) (equilibrium-policy MARL replicates real predator-prey trait oscillations), [Entropy 2019 deep-RL co-evolution](https://www.mdpi.com/1099-4300/21/8/773) (predator RL ability stabilises ecosystem coexistence). Our LSTMPPO+klinotaxis prey + MLPPPO predator setup is more biologically grounded than the cited published work because the prey side carries Phase 4's validated bilateral-sensing + recurrent-state machinery.

- [x] M5.1 Create `evolution/coevolution.py` with `CoevolutionLoop` (shipped in PR 3 via `add-coevolution-arms-race`; lives at `packages/quantum-nematode/quantumnematode/evolution/coevolution.py`)
- [x] M5.2 Create `evolution/redqueen_metrics.py` with phenotypic cycling, trait escalation, fitness lag, coupled rate (shipped in PR 3; lives at `packages/quantum-nematode/quantumnematode/evolution/redqueen_metrics.py`)
- [x] M5.3 Create `configs/evolution/coevolution_pilot.yml` (shipped in PR 4 as two arms: `coevolution_pilot_arm_a.yml` heuristic-imitation pretrain + `coevolution_pilot_arm_b.yml` cold-start)
- [x] M5.4 Create `configs/evolution/coevolution_full.yml` (shipped in PR 4)
- [x] M5.5 Run 50+ gen × 4 seed campaign — **superseded by PR 6's screening pivot** (NOT executed; deferred). The originally planned 2-seed × 30-gen pilot expanded into a 13-screen lever sweep (plus an R1 re-audit) when the canonical pilot's verdict gate produced ambiguous signal under the K-block-elite series. No candidate produced strict Red Queen entanglement (own-vs-cross lag delta landed at +0.017 to +0.024 across the 8 screens that contributed full champion-archive snapshots, including the GA optimiser fallback design.md D2 reserved for this regime). **Pilot recommendation: STOP M5.** Full 50+ gen × 4 seed run not executed; the screening evidence is decisive (saving ~30 wall-hours). See [logbook 017](../../../docs/experiments/logbooks/017-coevolution-arms-race.md) for the screening trajectory + STOP verdict rationale + methodology contributions.
- [x] M5.6 Multi-cluster vs single-cluster transfer evaluation — skipped under STOP M5 (transfer evaluation was a post-pilot ablation; the pilot's STOP makes it moot).
- [x] M5.7 Secondary Baldwin instrumentation — **formally dropped**. Readout requires per-gen elite snapshots from the 4-seed full-run lineage CSV, which was not produced under the screening-pivot trajectory. Re-scoping to screening data would lose statistical power and muddle the protocol. M4.7 stays armed but unmotivated by M5 evidence; definitive Baldwin closure requires a dedicated hyperparameter-evolution milestone. See [logbook 017 § M5.7 dropped](../../../docs/experiments/logbooks/017-coevolution-arms-race.md).
- [x] M5.8 Publish `artifacts/logbooks/017/` with Red Queen findings — done. [`docs/experiments/logbooks/017-coevolution-arms-race.md`](../../../docs/experiments/logbooks/017-coevolution-arms-race.md) covers the screen sweep, lag-matrix methodology, fair-test methodology, STOP M5 verdict + rationale, and M6.5 NEAT as the natural Red Queen follow-up. Artefact stash under [`artifacts/logbooks/017-coevolution-arms-race/`](../../../artifacts/logbooks/017-coevolution-arms-race/) — per-screen post-hoc summaries + champion archives + summary CSVs (fair-test, verdict, lag matrix) + c2 lag-matrix log (the analysis helper scripts themselves are recoverable from PR #153 commits, not stashed under the logbook).
- [x] M5.9 Update this checklist + roadmap milestone tracker — done. Phase 5 milestone tracker M5 row flipped to ❌ STOP in `docs/roadmap.md`.

## M6: Transgenerational Memory

**OpenSpec change**: `2026-06-30-add-transgenerational-memory` (not yet created)
**Status**: pending (gated on M3 GO ✅ + M4-or-M5; M3 already cleared, M4 closed STOP, M5 pending). Recommend beginning design once M5's pilot is launched and stable
**Bio fidelity**: HIGH
**Brain target**: LSTMPPO+klinotaxis
**Dependencies**: M3 GO ✅ + (M4 closed STOP / M5 pending — gate satisfied once M5 ships, or partially overlap once M5 pilot is launched and stable)
**Decision gate**: GO if F1 retains ≥40% of F0 avoidance, F3 ≤10%

**Highest-novelty remaining Phase 5 milestone.** Post-M4-replan elevation: 2025 mechanism literature converges on the project's deliverable-5 design with multiple new papers in the year since the original roadmap was drafted. **Recommend beginning design once M5's pilot is launched and stable** (M5 pilots run for hours; M6 design + spec scaffolding can happen in that window) rather than serial-after-M5-completes — but committing M6 implementation only after M5's verdict so design choices can incorporate any M5 findings.

**Literature grounding (2023-2025, all post-Posner-2023)**:

- [Posner et al. 2023 — Inheritance of associative memories and acquired cellular changes in C. elegans (Nat Commun)](https://www.nature.com/articles/s41467-023-39804-8) — original deliverable-5 inspiration; small-RNA-mediated transmission of behavioural associations across generations
- [SET-24 + HCF-1 maintain transgenerational epigenetic memory (Nat Commun 2025)](https://www.nature.com/articles/s41467-025-68200-7) — SET domain protein essential for germline immortality + small-RNA-mediated silencing
- [Molecular Requirements for C. elegans TEI of Pathogen Avoidance (bioRxiv Jan 2025)](https://www.biorxiv.org/content/10.1101/2025.01.21.634111v1) — Pv1 + Pfs1 small RNAs → maco-1 silencing → daf-7 in ASI neuron → avoidance behaviour. **Maps directly onto the F0/F1/F2/F3 pathogen avoidance experiment in M6.5**
- [Nucleus-independent transgenerational small-RNA inheritance (Sci Adv)](https://www.science.org/doi/10.1126/sciadv.adj8618) — cytoplasmic inheritance independent of nuclear factors; suggests the heritable substrate doesn't need a nucleus-equivalent in our model
- [eLife 2025 review — Transgenerational Epigenetic Inheritance: twists and turns of learned avoidance](https://elifesciences.org/articles/109427)

**Substrate readiness**: M3's `WeightPersistence` + `LamarckianInheritance` (heritable substrate proven bit-exact across 18 LSTMPPO trained tensors) + Phase 1's predator/pathogen-avoidance reward machinery + Phase 4's multi-agent infrastructure → most building blocks already shipped.

- [ ] M6.1 Create `agent/transgenerational_memory.py` with `TransgenerationalMemory` class
- [ ] M6.2 Hook into `prepare_episode()` as a prior on response distribution
- [ ] M6.3 Implement `inherit_from(parents, decay_factor)` transmission
- [ ] M6.4 Create `configs/evolution/transgenerational_pathogen_avoidance_lstmppo_klinotaxis.yml`
- [ ] M6.5 Run F0/F1/F2/F3 pathogen avoidance experiment (Posner 2023 replication design; cross-reference 2025 mechanism literature for biological plausibility checks on the inheritance decay schedule)
- [ ] M6.6 Publish `artifacts/logbooks/018/` with transgenerational findings (017 is M5's; 016 is M1's predator-brain refactor)
- [ ] M6.7 Update this checklist + roadmap milestone tracker

## M6.5: NEAT Architecture Evolution (OPTIONAL)

**OpenSpec change**: `2026-07-14-add-neat-evolution` (only if scheduled)
**Status**: not started (compute-budget dependent; cost has dropped, see below)
**Bio fidelity**: LOW
**Brain target**: MLPPPO

**Cost note (post-M4 replan)**: [TensorNEAT (GPU-accelerated NEAT, 2024-2025)](https://arxiv.org/pdf/2504.08339) materially lowers M6.5's cost vs the original `neat-python` plan — TensorNEAT manages a range of neural networks and simultaneously optimises topology + weights on GPU. A 2025 NEAT-vs-PPO comparison ([Engineering Journal](https://engj.org/index.php/ej/article/view/4615)) reported NEAT +25% over PPO on inventory tasks at 1k generations, suggesting NEAT is competitive for our scale. Switching the integration target from `neat-python` to TensorNEAT is recommended if M6.5 is scheduled.

- [ ] M6.5.1 Integrate **TensorNEAT** (preferred; falls back to `neat-python` if GPU unavailable) via `NeatEncoder` + `NeatGenome`
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
- [ ] M7.5 Publish `artifacts/logbooks/019/synthesis.md` (016 = M1 predator-brain; 017 = M5 co-evolution; 018 = M6 transgenerational; 019 = M7 phase synthesis)
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
