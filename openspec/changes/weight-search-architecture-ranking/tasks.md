# Tasks: Weight-Search Architecture Ranking on the Existing Grid Substrate

> **Dependency**: this change consumes the `feedforwardga` brain shipped by `add-neat-weights-brain`. Section 2 (substrate readiness) verifies the dependency before Phase 4 cells launch.

## 0. Phase 0 — Predator-sensing convergence investigation (T4.0g)

Runs concurrently with Section 2 below. Output: canonical sensor + reward locked in writing in design.md AND in the Phase 0 scratchpad.

- [ ] 0.1 Create the Phase 0 evaluation scratchpad at `tmp/evaluations/predator-convergence-investigation/predator-convergence-investigation_scratchpad.md` mirroring the format established by `tmp/evaluations/predator-sensing-biology-smoke/predator-sensing-biology-smoke_scratchpad.md` (purpose, configs table, results, hypothesis tracking).
- [ ] 0.2 **B0.1 Matched-compute baseline**. Run legacy `nociception_klinotaxis` vs new two-channel biology on MLPPPO at canonical Phase 4 budget (≥ 500 ep, n=4 seeds). Use `--theme headless`. Quantify the gap on last-25 mean success, foods/ep, predator-deaths/ep.
- [ ] 0.3 **B0.2 Sparse-signal ablation**. Implement a sensor-module variant `predator_mechanosensation_klinotaxis_sparse_fix` that injects `predator_distal_concentration` into the `predator_contact_intensity` field when `predator_contact_zone == ContactZone.NONE`. Register through the existing module registry. Smoke + matched-compute eval.
- [ ] 0.4 **B0.3 Redundancy ablation**. Implement a single-channel composite `predator_biology_klinotaxis` module emitting `[intensity, zone_as_angle, distal_concentration, dconcentration_dt]` (4-dim). Smoke + matched-compute eval against the parallel two-channel default.
- [ ] 0.5 **B0.4 Reward-shape ablation**. Author a `distal_chemo_penalty + binary_contact_damage_trigger` reward variant in `reward_calculator.py` (env-coupled, follows the `gradient_proximity` precedent of reading env state directly). Smoke + matched-compute eval against `gradient_proximity`.
- [ ] 0.6 **B0.5 2×2 cross**. Best-sensor × {legacy reward, new reward} at matched compute. Map the joint sensor×reward space.
- [ ] 0.7 **B0.6 Canonical selection**. Lock canonical sensor + reward in writing in this change's design.md (new section `## Phase 0 canonical-variant selection`). Update `specs/predator-sensing-biology/spec.md` Phase 0 requirements scenario "Canonical variant locked before Phase 4 C-cells launch" with the chosen names.
- [ ] 0.8 Promote Phase 0 forensics into `docs/experiments/logbooks/supporting/0XX-weight-search-architecture-ranking/phase-0/` (per-variant summary CSVs, comparison plot). Update the logbook draft.

## 1. Phase 1 — Substrate readiness

### 1a. FeedforwardGABrain integration verification

- [ ] 1a.1 Verify `add-neat-weights-brain` change is merged; pull latest `main`.
- [ ] 1a.2 Run a single short smoke via `uv run python scripts/run_evolution.py --config configs/scenarios/foraging/feedforwardga_small_klinotaxis.yml --theme headless --generations 5 --seed 2026`. Confirm the brain instantiates through the L1 registry without per-architecture branches and produces standard evolution-framework artefacts.

### 1b. ConnectomePPOBrain predator-gains projection

- [ ] 1b.1 Implement the canonical `predator_gains` projection in `packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py` per the spec's `Connectome PPO Predator Sensor Projection` requirement (ASH/ASI distal + ALM/AVM anterior + PLM posterior + lateral at half-weight on ALM+PLM).
- [ ] 1b.2 Add unit tests at `packages/quantum-nematode/tests/quantumnematode_tests/brain/test_connectome_predator_projection.py` covering each scenario in the spec requirement (distal routes to ASH+ASI; anterior to ALM+AVM; posterior to PLM; lateral degenerate routing; inert without predator inputs; no cross-contamination with food projection / chemical-synapse mask).
- [ ] 1b.3 Smoke-run `connectomeppo_small_klinotaxis.yml` (no predator inputs) and verify the Gate 1 R2b baseline result is unchanged byte-for-byte — the predator projection must be inert when no predator signal exists.

### 1c. Connectome-specific predator + thermotaxis configs (gated by Phase 0 + 1b)

- [ ] 1c.1 Author `configs/scenarios/pursuit/connectomeppo_small_predator_biology_klinotaxis.yml` consuming Phase 0's canonical sensor + reward + the new `predator_gains` projection. Mirror the structure of `configs/scenarios/foraging/connectomeppo_small_klinotaxis.yml`.
- [ ] 1c.2 Author `configs/scenarios/thermal_foraging/connectomeppo_small_isothermal_klinotaxis.yml` mirroring the `mlpppo_small_isothermal_oracle.yml` structure but with the connectome brain + klinotaxis sensing.

### 1d. Combined-behaviour small-klinotaxis configs per architecture (gated by Phase 0)

- [ ] 1d.1 Decide the directory for combined-behaviour configs — most likely a new `configs/scenarios/combined_klinotaxis/` directory (vs reusing `oxygen_thermal_pursuit/` which already includes aerotaxis). Document the choice in design.md.
- [ ] 1d.2 Author `{mlpppo,lstmppo,connectomeppo,feedforwardga}_small_combined_klinotaxis.yml` configs. Each integrates food + predator + thermotaxis (no aerotaxis). Reference shape: `configs/scenarios/oxygen_thermal_pursuit/mlpppo_large_oracle.yml`.
- [ ] 1d.3 Author the C1 (foraging-only) and C2 (foraging + predator) curriculum smokes per architecture. May reuse existing `configs/scenarios/foraging/` and `configs/scenarios/pursuit/` configs verbatim where the existing config matches what C1/C2 needs.

### 1e. Sample-config canonicalisation (T4.0e/f)

- [ ] 1e.1 Rename `configs/scenarios/foraging/connectomeppo_small_low_entropy_klinotaxis.yml` → `configs/scenarios/foraging/connectomeppo_small_klinotaxis.yml` (overwriting the high-entropy variant). Archive the previous high-entropy variant by deleting (git history preserves it).
- [ ] 1e.2 Update `configs/scenarios/foraging/connectomeppo_small_frozen_control_klinotaxis.yml` to match the canonical entropy_coef (irrelevant under `freeze_updates: true` but kept for diff cleanliness).
- [ ] 1e.3 Verify `configs/scenarios/foraging/mlpppo_small_klinotaxis.yml` has the provenance comment in its header (added at T2). Add if missing.

## 2. Phase 2 — Compute pre-flight (T4.0a)

- [ ] 2.1 Run one canonical-budget klinotaxis run per architecture (4 runs × n=1 seed) on the cheapest cell (foraging-only C1). Use `--theme headless`.
- [ ] 2.2 Record wall-time per architecture in `tmp/evaluations/weight-search-architecture-ranking/weight-search-architecture-ranking_scratchpad.md` (create if not yet present).
- [ ] 2.3 Extrapolate to full sweep: 4 architectures × (C1 + C2 + n=4 × C3) + ablations. Calculate total wall-time projection.
- [ ] 2.4 If projection diverges > 2× from phase6-tracking Decision 1's "4-6 weeks" estimate, draft an amendment to phase6-tracking design.md Decision 1 (include the new estimate + the empirical basis). If within 2×, document the projection in this change's design.md as Phase 2 evidence.

## 3. Phase 3 — Planning decisions land in design.md

- [ ] 3.1 **T4.0a** Add compute-budget section to this change's design.md with the Phase 2 projection. Reference (and link to) any phase6-tracking Decision 1 amendment.
- [ ] 3.2 **T4.0b** Add per-cell seed-count section with floor n ≥ 4 per Phase 5 inheritance + per-cell rationale if n > 4 (M4.5 precedent: n = 8 if SE on primary metric > 0.5× gate threshold at n=4).
- [ ] 3.3 **T4.0c** Add connectome sensor-projection + motor-readout section documenting the canonical food + predator projections (food shipped at T2; predator from Phase 1b).
- [ ] 3.4 **T4.0d** Add entropy-schedule section locking `entropy_coef=0.005` for all `T4.connectome.*` cells; note per-architecture configs whether others need similar scrutiny.
- [ ] 3.5 **T4.0e/f** Add config-canonicalisation note (already executed in 1e; documented here for traceability).
- [ ] 3.6 **MCC commitment**: confirm Decision 2 (BH-FDR within-pass) lands in design.md AND in the `specs/architecture-comparison-protocol/spec.md` paired-seed-statistics requirement BEFORE Phase 4 launches.

## 4. Phase 4 — Curriculum + integrated comparison sweep

### 4a. Create Phase 4 evaluation scratchpad

- [ ] 4a.1 Create `tmp/evaluations/weight-search-architecture-ranking/weight-search-architecture-ranking_scratchpad.md` with sections: purpose, per-architecture status table, per-cell metrics, decisions taken mid-sweep.

### 4b. C1 + C2 + C3 per architecture (use `nematode-run-experiments` skill where parallel groups fit)

- [ ] 4b.1 MLPPPO: C1 → C2 → C3. Record metrics after each.
- [ ] 4b.2 LSTMPPO: C1 → C2 → C3.
- [ ] 4b.3 FeedforwardGA: C1 → C2 → C3.
- [ ] 4b.4 ConnectomePPO: C1 → C2 → C3.

### 4c. Ablations (on C3 substrate)

- [ ] 4c.1 **Strict-mask vs soft-prior** on the connectome C3 cell — second connectome C3 run with `chemical_mask_mode: "soft_prior"`. Same n=4 seeds.
- [ ] 4c.2 **Per-family predator reward-shape ablation** — for each architecture, run the C3 cell with the non-canonical reward variant chosen in Phase 0. 4 architectures × 1 ablation cell = 4 additional cells.

## 5. Phase 4.5 — Architecture-promotion gate

- [ ] 5.1 Per SHOULD/MAY candidate (quantum, spiking, reservoir, hybrid), document a GO or SKIP verdict in this change's design.md (new section `## Phase 4.5 architecture-promotion gate`). Include rationale citing the (a) compute-fit, (b) roadmap-relevance, (c) headline-impact criteria from design.md Decision 3.
- [ ] 5.2 For each GO verdict: queue and run the C1 + C2 + C3 cells for that architecture per Section 4b. Run any ablations from Section 4c.

## 6. Phase 5 — Cross-architecture analysis + logbook

- [ ] 6.1 Author analysis script under `scripts/analysis/weight_search_architecture_ranking.py` consuming the C3 (and any Phase 4.5 promoted) per-cell CSV exports. Reuse `scripts/campaigns/aggregate_m613_pilot.py:compute_cross_arm_delta_stats` (lines 329-418) for the paired-seed Wilcoxon + bootstrap CI. Add BH-FDR correction across the active test set per Decision 2.
- [ ] 6.2 Run analysis. Output: per-pair-per-metric delta + p + BH-q + bootstrap CI as CSVs under `tmp/evaluations/weight-search-architecture-ranking/analysis/`.
- [ ] 6.3 Generate per-behaviour-component ranking tables (foraging success, predator survival, isotherm-tracking) + overall combined ranking.
- [ ] 6.4 Generate the connectome verdict: per-behaviour wins/ties/losses vs each other architecture.
- [ ] 6.5 Promote analysis CSVs + summary tables + plots into `docs/experiments/logbooks/supporting/0XX-weight-search-architecture-ranking/`.
- [ ] 6.6 Author the logbook at `docs/experiments/logbooks/0XX-weight-search-architecture-ranking.md`. Reference `supporting/*` paths only (never `tmp/*`). Cover: methodology summary, per-architecture C1/C2/C3 results, ablation results, cross-architecture ranking, connectome verdict, env-upgrade-delta baseline data (the publishable intermediate result T7 will measure against).
- [ ] 6.7 Pause for user review of evaluations + Gate 3 verdict-implication discussion BEFORE finalising the logbook (per memory feedback).
- [ ] 6.8 Update `openspec/changes/phase6-tracking/tasks.md` T4 row to reflect this change's outcomes. Tick T4.0a-g and the 12-cell rows (re-shaped as 4-cell-curriculum rows per Decision 1).
- [ ] 6.9 Update `docs/roadmap.md` Phase 6 Tranche Tracker T4 row.
- [ ] 6.10 Amend `openspec/changes/phase6-tracking/design.md § Decision 6` (and reference T7's inheritance of BH-FDR) per the open-question from this change's design.md.

## 7. Pre-merge verification

- [ ] 7.1 Run `openspec validate weight-search-architecture-ranking --strict` and verify clean.
- [ ] 7.2 Run `uv run pre-commit run --files <changed-files>` and verify clean. Targeted runs during iteration; full suite (`pre-commit run -a`) before push (per memory feedback).
- [ ] 7.3 Audit staged files for `/Users/`, `/home/`, `C:\\Users\\` absolute path leakage (per project memory feedback). Sanitise any found.
- [ ] 7.4 Audit any > 100 KB artefacts under `docs/experiments/logbooks/supporting/0XX-weight-search-architecture-ranking/` against `.gitattributes` LFS rules (per project memory feedback). Promote to LFS if needed.
- [ ] 7.5 Ask user before pushing the branch or opening a PR.
