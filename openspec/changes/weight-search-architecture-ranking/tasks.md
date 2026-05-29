# Tasks: Weight-Search Architecture Ranking on the Existing Grid Substrate

> **Dependency**: this change consumes the `feedforwardga` brain shipped by `add-neat-weights-brain`. Section 2 (substrate readiness) verifies the dependency before Phase 4 cells launch.

## 0. Phase 0 — Predator-sensing convergence investigation (T4.0g)

Runs concurrently with Section 2 below. Output: canonical sensor + reward locked in writing in design.md AND in the Phase 0 scratchpad.

> **Implementation note (post-shipping)**: the task numbering below preserves the original plan structure. Two divergences shipped: (i) the planned "B0.5 2×2 cross" expanded into the actual B0.5 (canonical sensors + new reward) + B0.6 (sparse_fix sensor + new reward, orthogonal stack), executed as part of Tasks 0.6 + the extended validation under Task 0.7; (ii) two implementation bugs were uncovered + fixed mid-Phase 0 (composite STAM-channel recognition; `predator_lateral_gradient` silent silencing under new-biology configs) and are documented in design.md § "Phase 0 canonical-variant selection". Composite did NOT win (the conditional MODIFIED delta in Task 0.4 was therefore NOT triggered). See [supporting/025/phase-0/README.md](../../../docs/experiments/logbooks/supporting/025-weight-search-architecture-ranking/phase-0/README.md) for the full ranking + per-seed data.

- [x] 0.1 Create the Phase 0 evaluation scratchpad at `tmp/evaluations/weight-search-architecture-ranking/phase-0/phase-0_scratchpad.md` (co-located under the change's scratchpad directory for findability) mirroring the format established by `tmp/evaluations/predator-sensing-biology-smoke/predator-sensing-biology-smoke_scratchpad.md` (purpose, configs table, results, hypothesis tracking).
- [x] 0.2 **B0.1 Matched-compute baseline**. Run legacy `nociception_klinotaxis` vs new two-channel biology on MLPPPO at canonical Phase 4 budget (≥ 500 ep, n=4 seeds). Use `--theme headless` (per project memory feedback for batch runs of `scripts/run_simulation.py`). Quantify the gap on last-25 mean success, foods/ep, predator-deaths/ep.
- [x] 0.3 **B0.2 Sparse-signal ablation**. Implement a sensor-module variant `predator_mechanosensation_klinotaxis_sparse_fix` that injects `predator_distal_concentration` into the `predator_contact_intensity` field when `predator_contact_zone == ContactZone.NONE`. Sub-steps:
  - Register the module in `ModuleName` enum at [`packages/quantum-nematode/quantumnematode/brain/modules.py`](../../../packages/quantum-nematode/quantumnematode/brain/modules.py) (existing enum entries at lines ~119-124 for the canonical two-channel modules).
  - Add the new module to the STAM-dim inference table at [`packages/quantum-nematode/quantumnematode/brain/modules.py:1409`](../../../packages/quantum-nematode/quantumnematode/brain/modules.py#L1409) (`_infer_stam_dim_from_modules`). The just-archived `fix-predator-sensing-biology` change explicitly flagged this as a pitfall — failing to update inference produces shape-mismatch errors on the first forward pass.
  - Smoke-test that `BrainParams.stam` input gets the correct dim under the new module list.
  - Matched-compute eval (n=4 seeds at canonical budget) and record in scratchpad.
- [x] 0.4 **B0.3 Redundancy ablation**. Implement a single-channel composite `predator_biology_klinotaxis` module emitting `[intensity, zone_as_angle, distal_concentration, dconcentration_dt]` (4-dim). Same sub-steps as Task 0.3: register in `ModuleName` enum + STAM-dim inference table + smoke-test STAM dim + matched-compute eval. If Phase 0 selects this variant as canonical, ALSO ship a `MODIFIED Requirements` delta to `predator-sensing-biology` spec's "Two-Channel Predator-Sensing Model" requirement per the change's `specs/predator-sensing-biology/spec.md` relationship clause. **Composite did NOT win** (16.0% ± 7.3 vs legacy 67% — structurally inferior, dropped `lateral_gradient` by design); MODIFIED delta therefore NOT shipped.
- [x] 0.5 **B0.4 Reward-shape ablation**. Author a `distal_chemo_penalty + binary_contact_damage_trigger` reward variant in [`packages/quantum-nematode/quantumnematode/agent/reward_calculator.py`](../../../packages/quantum-nematode/quantumnematode/agent/reward_calculator.py) (env-coupled, follows the `gradient_proximity` precedent at lines ~115-128 of reading env state directly). Smoke + matched-compute eval against `gradient_proximity`. Shipped as `reward_mode: distal_chemo_contact_trigger`.
- [x] 0.6 **B0.5 sensor×reward joint mapping**. Cross best-sensor × {legacy reward, new reward} at matched compute. Map the joint sensor×reward space. **Realised as**: canonical-sensors × new-reward (B0.5 = 81.0% ± 5.0, winner) + sparse_fix-sensor × new-reward (B0.6 orthogonal stack = 78.0% ± 6.9, extended validation under Task 0.7). The post-Bug-1-fix re-runs of A2/B0.3/B0.5 (12 additional canonical-budget runs) supply the remaining cells of the 2×2 space.
- [x] 0.7 **B0.6 Canonical selection**. Lock canonical sensor + reward in writing in this change's design.md (new section `## Phase 0 canonical-variant selection`). Update `specs/predator-sensing-biology/spec.md` Phase 0 requirements scenario "Canonical variant locked before Phase 4 C-cells launch" with the chosen names. If the composite variant (Task 0.4) wins, also ship the MODIFIED delta to "Two-Channel Predator-Sensing Model". **Locked**: canonical two-channel sensors (`predator_mechanosensation_klinotaxis` + `predator_chemosensation_klinotaxis`) + `reward_mode: distal_chemo_contact_trigger`.
- [x] 0.8 Promote Phase 0 forensics into `docs/experiments/logbooks/supporting/025-weight-search-architecture-ranking/phase-0/` (per-variant summary CSVs, comparison plot). Update the logbook draft.
- [x] 0.9 **Unplanned: implementation-bug fixes uncovered during Phase 0**.
  - Composite STAM-channel recognition (commit `c25588a1`): `agent/stam.py:resolve_active_channels` did not recognise the composite `predator_biology_klinotaxis` module name, falling through to the legacy `predator` fallback channel and crashing the first forward pass with a shape-mismatch (env-side STAM dim 7 vs brain-side STAM dim 9). Fix: explicit name-match activates both `predator_mechano` and `predator_distal` channels for the composite. 5 regression tests added in `test_stam.py`.
  - `predator_lateral_gradient` silent silencing (commit `65a5b517`): `agent.py:_compute_temporal_data` gated `predator_lateral_gradient` population on the legacy `nociception_mode == KLINOTAXIS` only. New-biology configs set `predator_distal_mode: klinotaxis` (a different knob); the lateral-gradient field was therefore silently None and the chemo channel's directional `angle` feature collapsed to 0.0. **This single bug explained ~all of the 44pp Step A convergence gap** (post-fix A2 went 23% → 65%, closing the gap to legacy). Fix: dual gate (legacy OR new-biology). 3 regression tests added in `test_agent.py::TestPredatorLateralGradientPopulation`.

## 1. Phase 1 — Substrate readiness

### 1a. FeedforwardGABrain integration verification

- [ ] 1a.1 Verify `add-neat-weights-brain` change is merged; pull latest `main`.
- [ ] 1a.2 Run a single short smoke via `uv run python scripts/run_evolution.py --config configs/evolution/feedforwardga_foraging_small.yml --seed 2026 --generations 5`. Confirm the brain instantiates through the L1 registry without per-architecture branches and produces standard evolution-framework artefacts. Note: `scripts/run_evolution.py` always runs headless; there is no `--theme` flag.

### 1b. ConnectomePPOBrain predator-gains projection

- [x] 1b.1 Implement the canonical `predator_gains` projection in [`packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py`](../../../packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py) per the spec's `Connectome PPO Predator Sensor Projection` requirement. Target neurons use the bilateral-suffixed canonical forms from Cook 2019:
  - Distal: `ASHL`, `ASHR`, `ASIL`, `ASIR` (4 targets; 2 features → gain matrix shape `(2, 4)`)
  - Anterior contact: `ALML`, `ALMR`, `AVM` (3 targets — AVM unilateral; 2 features → `(2, 3)`)
  - Posterior contact: `PLML`, `PLMR` (2 targets; 2 features → `(2, 2)`)
  - Lateral contact: degenerate, reuses anterior + posterior gains scaled by 0.5 (no separate matrix)
  - Document the bilateral broadcast convention inline (one column per L/R member with identical init, OR shared column broadcast). **Shipped**: option (a) one independent learnable column per L/R member, identical zero-mean init. Lets PPO updates discover L/R asymmetries that real *C. elegans* sensory pairs develop through experience. Gated by `enable_predator_projection: bool = False` (config field) so foraging-only configs allocate zero predator-related `nn.Parameter` objects and consume byte-identical RNG-stream order vs pre-projection builds.
- [x] 1b.2 Add unit tests at `packages/quantum-nematode/tests/quantumnematode_tests/brain/test_connectome_predator_projection.py` covering each scenario in the spec requirement: distal routes to ASHL+ASHR+ASIL+ASIR with shape `(2, 4)`; anterior to ALML+ALMR+AVM with shape `(2, 3)`; posterior to PLML+PLMR with shape `(2, 2)`; lateral degenerate routing reuses anterior + posterior gains at half-weight; inert without predator inputs; no cross-contamination with food projection / chemical-synapse mask. **Shipped**: 17 tests, all passing; full `tests/quantumnematode_tests/brain/arch/` suite (1140 tests) still green post-additive changes.
- [x] 1b.3 Smoke-run `connectomeppo_small_klinotaxis.yml` (no predator inputs) and verify the Gate 1 R2b baseline result is preserved: last-25 mean klinotaxis success ≥ R2b's reported baseline at the equivalent seed − 3 percentage points. NOT byte-identical (adding `nn.Parameter` tensors perturbs PyTorch RNG-stream consumption order during `__init__` which can shift downstream `nn.init.normal_` draws on existing parameters by float-ulp amounts). If a > 3pp regression appears, the projection has accidentally affected the food path — diagnose before proceeding to Phase 1c. **Result**: 500-episode run on `connectomeppo_small_low_entropy_klinotaxis.yml` (R2b's config) at seed 2026 produced 92.0% overall success + 100% last-25 — byte-identical to logbook 023's reported R2b numbers (Δ = 0pp). The `enable_predator_projection: false` default skips the predator parameter allocation entirely, so the food path's RNG-stream order is byte-preserved.

### 1c. Connectome-specific predator + thermotaxis configs (gated by Phase 0 + 1b)

- [ ] 1c.1 Author `configs/scenarios/pursuit/connectomeppo_small_predator_biology_klinotaxis.yml` consuming Phase 0's canonical sensor + reward + the new `predator_gains` projection. Mirror the structure of `configs/scenarios/foraging/connectomeppo_small_klinotaxis.yml`.
- [ ] 1c.2 Author `configs/scenarios/thermal_foraging/connectomeppo_small_isothermal_klinotaxis.yml` mirroring the `mlpppo_small_isothermal_oracle.yml` structure but with the connectome brain + klinotaxis sensing.

### 1d. Combined-behaviour small-klinotaxis configs per architecture (gated by Phase 0)

- [ ] 1d.1 Create the new directory `configs/scenarios/foraging_predator_thermal/` (the directory name is pre-decided in design.md Decision 8; parallels the existing `oxygen_thermal_pursuit/` noun-modifier compound).
- [ ] 1d.2 Author the four combined configs in their respective directories per design.md Decision 8's split:
  - `configs/scenarios/foraging_predator_thermal/mlpppo_small_combined_klinotaxis.yml`
  - `configs/scenarios/foraging_predator_thermal/lstmppo_small_combined_klinotaxis.yml`
  - `configs/scenarios/foraging_predator_thermal/connectomeppo_small_combined_klinotaxis.yml`
  - `configs/evolution/feedforwardga_small_combined_klinotaxis.yml` (separate directory because GA configs require an `evolution:` block and are launched via `scripts/run_evolution.py`)
    Each integrates food + predator + thermotaxis (no aerotaxis per `phase6-tracking/design.md` Decision 5). Reference shape for the PPO configs: `configs/scenarios/oxygen_thermal_pursuit/mlpppo_large_oracle.yml` (with food + thermal + aerotaxis + predators); strip the aerotaxis primitives, scale down to small-klinotaxis-mode. Reference shape for the GA config: `configs/evolution/feedforwardga_foraging_small.yml` (from the `add-neat-weights-brain` dependency), extended with the predator + thermal env blocks.
- [ ] 1d.3 Author the C1 (foraging-only) and C2 (foraging + predator) curriculum smokes per architecture. May reuse existing `configs/scenarios/foraging/` and `configs/scenarios/pursuit/` configs verbatim where the existing config matches what C1/C2 needs.

### 1e. Sample-config canonicalisation (T4.0e/f)

- [ ] 1e.1 Promote the low-entropy klinotaxis config: copy the contents of `configs/scenarios/foraging/connectomeppo_small_low_entropy_klinotaxis.yml` (R2b reference run, `entropy_coef=0.005`) into the canonical `configs/scenarios/foraging/connectomeppo_small_klinotaxis.yml`. After the copy the canonical file's `brain.config.entropy_coef` MUST be `0.005` (NOT `0.02` — the pre-existing value to-be-overwritten). **KEEP** the `connectomeppo_small_low_entropy_klinotaxis.yml` file in place — it is referenced from logbook 023 R2 (the high-entropy drift case is the load-bearing evidence for why R2b exists; deleting the file breaks the link from a published logbook to a reproducible artefact). Update the header comment on the renamed canonical file to note "promoted from low_entropy variant on `<date>`; the high-entropy variant at `_low_entropy_klinotaxis.yml` is retained for logbook 023 R2 reproducibility. `entropy_coef=0.005` is the locked canonical value per design.md § Decision 8 / Task 3.4 (T4.0d)."
- [ ] 1e.2 Update `configs/scenarios/foraging/connectomeppo_small_frozen_control_klinotaxis.yml` to match the canonical `entropy_coef=0.005` (irrelevant under `freeze_updates: true` but kept for diff cleanliness).
- [ ] 1e.3 Verify `configs/scenarios/foraging/mlpppo_small_klinotaxis.yml` has the provenance comment in its header (added at T2). Add if missing.

### 1f. Synchronise phase6-tracking with the integrated-C3 scope change

- [ ] 1f.1 Amend `openspec/changes/phase6-tracking/tasks.md` § T4 row structure to reflect the integrated-C3 pattern (4 architectures × 1 integrated C3 each + curriculum smokes, NOT 4 architectures × 3 per-behaviour cells). "Synchronous" here means **applied in the same commit/PR as Section 3's MCC commitment (Task 3.6)** — Task 1f.1 is listed in Phase 1 only because the substrate-readiness phase is the natural home for cross-change coordination work, not because it blocks on Phase 1 sub-tasks; the actual sequencing dependency is Task 3.6, and reviewers should expect 1f.1 + 1f.2 + 3.6 to land together in one PR rather than as three separate commits. The point is that the phase6-tracking tracker reads correctly throughout this change's implementation, NOT only at Phase 5 closeout.
- [ ] 1f.2 Amend `openspec/changes/phase6-tracking/design.md` § Decision 6 G3.a wording: the existing "All 12 MUST cells in T7 (4 families × 3 behaviours) … n ≥ 4 seeds per cell" language assumes T7 inherits the 12-cell pattern from T4. With this change collapsing T4 to integrated-C3 and T7 inheriting that pattern, G3.a SHALL be widened to "All 4 MUST architecture integrated-C3 cells in T7 (n ≥ 4 seeds per cell) with per-behaviour-component sub-metrics extracted per the architecture-comparison-protocol capability." The n ≥ 4 seed floor MUST be preserved verbatim from the original wording — this change only widens the cell-shape framing (12 per-behaviour → 4 integrated), it does NOT relax the Phase 5 statistical bar.
- [ ] 1f.3 Re-run `openspec validate phase6-tracking --strict` after the amendments in 1f.1 + 1f.2 and verify clean. If validation surfaces structural issues (e.g. a tasks.md row format requirement), iterate until clean before proceeding to Section 2.

## 2. Phase 2 — Compute pre-flight (T4.0a)

- [ ] 2.1 Run one canonical-budget klinotaxis run per architecture (4 runs × n=1 seed) on the cheapest cell (foraging-only C1). Use `--theme headless` for `scripts/run_simulation.py` invocations.
- [ ] 2.2 Record per-architecture per-run wall-time AND the assumed parallelism factor (e.g. "8 parallel workers on this machine") in `tmp/evaluations/weight-search-architecture-ranking/weight-search-architecture-ranking_scratchpad.md` (create if not yet present). Both numbers are load-bearing for Task 2.4's trigger arithmetic.
- [ ] 2.3 Extrapolate to full sweep using the formula `wall_clock_weeks = (total_runs × per_run_hours) / (parallelism_factor × hours_per_week)`. Total runs: Phase 0 (~24) + Phase 2 (4) + Phase 4 curriculum (24) + Phase 4 ablations (20) + Phase 4.5 promotions (variable; 6 per promotion) ≈ 72 without promotions per design.md Decision 8. Reasonable `hours_per_week` is 168 (24/7 compute) or 40 (working-hours-only); pick + document.
- [ ] 2.4 If wall-clock projection diverges > 2× from phase6-tracking Decision 1's "4-6 weeks" estimate, draft an amendment to phase6-tracking design.md Decision 1 (include the new wall-clock estimate + the per-run wall-time + parallelism factor + total-run-count breakdown — all three are evidence, not just total time). The trigger SHALL fire on the wall-clock metric NOT on per-run wall-time alone (if per-run time stays flat but parallelism assumption fails, the wall-clock blow-up still triggers the amendment). If within 2×, document the projection in this change's design.md as Phase 2 evidence.

## 3. Phase 3 — Planning decisions land in design.md

- [ ] 3.1 **T4.0a** Add compute-budget section to this change's design.md with the Phase 2 projection. Reference (and link to) any phase6-tracking Decision 1 amendment.
- [ ] 3.2 **T4.0b** Add per-cell seed-count section with floor n ≥ 4 per Phase 5 inheritance + per-cell rationale if n > 4 (M4.5 precedent: n = 8 if SE on primary metric > 0.5× gate threshold at n=4).
- [ ] 3.3 **T4.0c** Add connectome sensor-projection + motor-readout section documenting the canonical food + predator projections (food shipped at T2; predator from Phase 1b with the bilateral-suffixed neuron names per design.md Decision 8).
- [ ] 3.4 **T4.0d** Add entropy-schedule section locking `entropy_coef=0.005` for all `T4.connectome.*` cells; note per-architecture configs whether others need similar scrutiny.
- [ ] 3.5 **T4.0e/f** Add config-canonicalisation note (already executed in 1e; documented here for traceability).
- [ ] 3.6 **MCC commitment**: confirm Decision 2 (BH-FDR within-pass) lands in design.md AND in the `specs/architecture-comparison-protocol/spec.md` paired-seed-statistics requirement BEFORE Phase 4 launches. Land this in the same commit/PR as Tasks 1f.1 + 1f.2 (the phase6-tracking T4 row-structure + G3.a wording amendments) so the tracker and the MCC commitment stay in sync.

## 4. Phase 4 — Curriculum + integrated comparison sweep

### 4a. Create Phase 4 evaluation scratchpad

- [ ] 4a.1 Create `tmp/evaluations/weight-search-architecture-ranking/weight-search-architecture-ranking_scratchpad.md` with sections: purpose, per-architecture status table, per-cell metrics, decisions taken mid-sweep.

### 4b. C1 + C2 + C3 per architecture (use `nematode-run-experiments` skill where parallel groups fit)

For each architecture (MLPPPO, LSTMPPO, FeedforwardGA, ConnectomePPO):

- Run C1 (foraging-only smoke, n=1). Record metrics.

- Run C2 (foraging + predator smoke, n=1). Record metrics.

- **Before launching C3**: check the Decision 7 tuning trigger. **Default = use the global reward weights** (inherit from the closest existing config) — NO per-arch tuning. If the C2 result shows foraging-or-predator < 50% of at least one other arch's C2 result on the same metric, escalate: run C2 with a second seed (n=2 total for that arch) to confirm the imbalance isn't noise, then pick per-arch weights and document them in design.md (new section `## Per-architecture reward weights for C3`) with the C2 numbers + rationale. Freeze the weights for C3.

- Run C3 (foraging + predator + thermotaxis, n=4 seeds). Record metrics.

- [ ] 4b.1 MLPPPO: C1 → C2 → freeze reward weights → C3.

- [ ] 4b.2 LSTMPPO: C1 → C2 → freeze reward weights → C3.

- [ ] 4b.3 FeedforwardGA: C1 → C2 → freeze reward weights → C3.

- [ ] 4b.4 ConnectomePPO: C1 → C2 → freeze reward weights → C3.

### 4c. Ablations (on C3 substrate)

- [ ] 4c.1 **Strict-mask vs soft-prior** on the connectome C3 cell — second connectome C3 run with `chemical_mask_mode: "soft_prior"`. Same n=4 seeds.
- [ ] 4c.2 **Per-family predator reward-shape ablation** — for each architecture, run the C3 cell with the non-canonical reward variant chosen in Phase 0. 4 architectures × 1 ablation cell × n=4 seeds = 16 additional runs.

## 5. Phase 4.5 — Architecture-promotion gate

- [ ] 5.1 Per SHOULD/MAY candidate (quantum, spiking, reservoir, hybrid), document a GO or SKIP verdict in this change's design.md (new section `## Phase 4.5 architecture-promotion gate`). Include rationale citing the (a) compute-fit, (b) roadmap-relevance, (c) headline-impact criteria from design.md Decision 3.
- [ ] 5.2 For each GO verdict: queue and run the C1 + C2 + C3 cells for that architecture per Section 4b (with the same C2-results-inform-C3-reward-weights pattern). Run any ablations from Section 4c.

## 6. Phase 5 — Cross-architecture analysis + logbook

- [ ] 6.1 Author analysis script under `scripts/analysis/weight_search_architecture_ranking.py` consuming the C3 (and any Phase 4.5 promoted) per-cell CSV exports. Extract the reusable inner computation pattern (paired-seed delta → one-sided Wilcoxon → 80% bootstrap CI with seeded RNG, 1000 resamples) from `scripts/campaigns/aggregate_m613_pilot.py:329-418` (which is M6.13-specific with a 4-tuple-keyed dict) into a generic `_paired_seed_wilcoxon_bootstrap(deltas: list[float]) -> dict` helper. Add BH-FDR correction across the **realised** active test set per Decision 2 (the realised set may be smaller than the planned set if Phase 4 risk-mitigation drops an architecture).
- [ ] 6.2 Run analysis. Output: per-pair-per-metric delta + p + BH-q + bootstrap CI as CSVs under `tmp/evaluations/weight-search-architecture-ranking/analysis/`.
- [ ] 6.3 Generate per-behaviour-component ranking tables (foraging success, predator survival, isotherm-tracking) + overall combined ranking.
- [ ] 6.4 Generate the connectome verdict: per-behaviour wins/ties/losses vs each other architecture.
- [ ] 6.5 Promote analysis CSVs + summary tables + plots into `docs/experiments/logbooks/supporting/025-weight-search-architecture-ranking/`.
- [ ] 6.6 Author the logbook at `docs/experiments/logbooks/025-weight-search-architecture-ranking.md`. Reference `supporting/*` paths only (never `tmp/*`). Cover: methodology summary, per-architecture C1/C2/C3 results, ablation results, cross-architecture ranking, connectome verdict, env-upgrade-delta baseline data (the publishable intermediate result T7 will measure against).
- [ ] 6.7 Pause for user review of evaluations + Gate 3 verdict-implication discussion BEFORE finalising the logbook (per memory feedback).
- [ ] 6.8 Update `openspec/changes/phase6-tracking/tasks.md` T4 rows to reflect this change's outcomes. Tick T4.0a-g; tick the integrated-C3 rows (which Phase 1f.1 amended into the tracker at change start).
- [ ] 6.9 Update `docs/roadmap.md` Phase 6 Tranche Tracker T4 row.
- [ ] 6.10 Amend `openspec/changes/phase6-tracking/design.md § Decision 6` MCC default (Holm-Bonferroni → BH-FDR) AND confirm the G3.a wording amendment from Task 1f.2 still reflects the realised T4 outcomes. T7's inheritance of BH-FDR is captured in this amendment.

## 7. Phase 7 — Pre-merge verification

- [ ] 7.1 Run `openspec validate weight-search-architecture-ranking --strict` and verify clean.
- [ ] 7.2 Run targeted `uv run pre-commit run --files <changed-files>` and verify clean. Targeted runs during iteration; full suite (`pre-commit run -a`) before push (per memory feedback).
- [ ] 7.3 Audit staged files for `/Users/`, `/home/`, `C:\\Users\\` absolute path leakage (per project memory feedback). Sanitise any found.
- [ ] 7.4 Audit any > 100 KB artefacts under `docs/experiments/logbooks/supporting/025-weight-search-architecture-ranking/` against `.gitattributes` LFS rules (per project memory feedback). Promote to LFS if needed.
- [ ] 7.5 Ask user before pushing the branch or opening a PR.
