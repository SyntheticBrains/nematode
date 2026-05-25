# Proposal: Weight-Search Architecture Ranking on the Existing Grid Substrate

## Why

Phase 6 Tranche 4 (T4) per `openspec/changes/phase6-tracking/tasks.md` is the first cross-architecture comparison sweep of the project: four MUST architecture families (connectome-constrained PPO, MLP-PPO, LSTM/GRU-PPO, feed-forward GA) evaluated head-to-head on the existing discrete-grid substrate with corrected predator biology, each instantiated against all three Phase 6 behaviours simultaneously (food chemotaxis + predator evasion + thermotaxis active in one integrated simulation). The output is a publishable intermediate result — an architecture ranking on the existing grid — that becomes the baseline for the T4-vs-T7 env-upgrade delta (a load-bearing Phase 6 finding per [phase6-tracking/design.md § Decision 1](../phase6-tracking/design.md)).

T4 carries seven planning sub-tasks (T4.0a–T4.0g) that must land in writing before any comparison cell runs: compute-budget extrapolation, per-cell seed-count rationale, connectome sensor-projection ablation choices, entropy-schedule lock, config canonicalisation cleanups, and the predator-sensing convergence-rate investigation deferred from `fix-predator-sensing-biology` (Modelling caveat 6 of its archived design.md). T4.0g in particular is owed empirical work: the new two-channel predator biology learned substantially slower than the legacy single-channel `nociception_klinotaxis` at matched 100-episode budgets (MLPPPO 3% vs 51%; LSTMPPO 0% vs 7%), and T4 owns the matched-compute evaluation + sensor + reward ablations under canonical T4 budget.

T4 closes no gate by itself but produces three of Gate 3's four criteria's evidence base: G3.a (12 cells at the Phase 5 statistical bar — but here re-shaped as 4 architectures × 1 integrated multi-behaviour cell each + per-behaviour-component sub-metrics extracted from the integrated runs, per the user's "all behaviours in one config" decision), G3.b (where does the wild-type connectome rank?), and partial G3.c context (the grid baseline the env-upgrade delta is measured against). Gate 3 itself closes at T7.

## What Changes

### Phase 0 — Predator-sensing convergence investigation (T4.0g)

Investigation-only; runs concurrently with Phase 1a/1b. Owns the empirical question the just-merged predator-sensing change deferred.

- Run new vs legacy predator-sensor matched-compute baseline at canonical T4 budget (≥ 500 ep, n=4 seeds) on MLPPPO.
- Ablate the "sparse contact signal" hypothesis with a variant injecting distal sulfolipid concentration into the mechano-strength field when not in contact.
- Ablate the "channel redundancy" hypothesis with a single-channel composite `predator_biology` module (4-dim feature vector vs the current two parallel 3-dim modules).
- Ablate reward shape: existing `gradient_proximity` vs `distal_chemo_penalty + binary_contact_damage_trigger`.
- 2×2 best-sensor × {legacy reward, new reward} grid to map the joint space.
- Lock canonical predator-evasion sensor encoding + reward shape for the comparison cells. Output: logbook section + 1-2 canonical config templates.

### Phase 1 — Substrate readiness

- **B1.a** Verify the `feedforwardga` brain (shipped by the `add-neat-weights-brain` dependency) integrates with the existing simulation-launcher path through the L1 plugin registry.
- **B1.b** Implement the canonical `predator_gains` projection in `ConnectomePPOBrain` (deferred from `fix-predator-sensing-biology` per its design.md Decision T3.8; ASH/ASI distal + ALM/AVM anterior + PLM posterior, per the Bargmann literature). One projection, documented; ablation is contingent on Phase 4 evidence per [phase6-tracking/design.md § Decision 7](../phase6-tracking/design.md).
- **B1.c** Author the connectome-specific predator + thermotaxis configs (gated by Phase 0's canonical sensor/reward output).
- **B1.d** Author small-sized klinotaxis-mode combined-behaviour configs for all four architectures (food + predator + thermotaxis active in one config; aerotaxis explicitly excluded per [phase6-tracking/design.md § Decision 5](../phase6-tracking/design.md)). Reference existing combined configs are large-oracle-mode only (`configs/scenarios/oxygen_thermal_pursuit/mlpppo_large_oracle.yml`); the small klinotaxis variants don't exist yet.
- **B1.e** Sample-config canonicalisation (T4.0e/f): promote `connectomeppo_small_low_entropy_klinotaxis.yml` (R2b reference run, `entropy_coef=0.005`) into the canonical slot and archive the high-entropy variant.

### Phase 2 — Compute pre-flight (T4.0a)

- One canonical-budget klinotaxis run per family (4 runs × n=1 seed) on the cheapest cell. Extrapolate wall-time to the full sweep. Amend [phase6-tracking/design.md § Decision 1](../phase6-tracking/design.md) if projection diverges by > 2×.

### Phase 3 — Planning decisions land in design.md

Document the load-bearing T4 decisions in this change's design.md:

- **T4.0a** Compute-budget table from Phase 2 + Decision 1 amendment if needed.
- **T4.0b** Per-cell seed count (floor n ≥ 4; rationale if higher).
- **T4.0c** Sensor-projection + motor-readout choices for `ConnectomePPOBrain` (food projection already shipped at T2; predator projection from Phase 1b).
- **T4.0d** Entropy schedule for connectome PPO (`entropy_coef=0.005` per logbook 023 R2b).
- **T4.0e/f** Config canonicalisation cleanups (from Phase 1e).
- **MCC strategy commitment**: BH-FDR within-pass, pre-committed before any Phase 4 cell launches; T7 inherits the strategy. This is a deliberate amendment to Phase 6 Decision 6's "default Holm-Bonferroni" — see this change's design.md for the rationale.

### Phase 4 — Curriculum + integrated comparison sweep

Per architecture (4 architectures: connectome, mlp_ppo, lstm_gru_ppo, feedforward_ga):

- **C1** Foraging-only smoke (n=1 seed, short budget).
- **C2** Foraging + predator smoke (n=1 seed).
- **C3** Foraging + predator + thermotaxis primary cell (n ≥ 4 seeds). The integrated, ranked configuration.

Plus ablations on C3:

- Connectome strict-mask vs soft-prior (per [phase6-tracking/design.md § Decision 7](../phase6-tracking/design.md)).
- Per-family predator reward-shape ablation (coupled to Phase 0's 2×2 outcome).

Curriculum rationale: per-architecture sensor-wiring failures surface in C1/C2 (cheap, single-seed) before the expensive n=4 C3 cells launch. Net cost over "combined only" is ~2 cheap runs per architecture; far less than re-running an n=4 C3 because of an undiagnosed pipeline bug.

### Phase 4.5 — Architecture-promotion gate

Explicit decision point between Phase 4 and Phase 5. Decide whether to promote SHOULD/MAY architectures from [phase6-tracking/design.md § Decision 4](../phase6-tracking/design.md) (quantum, spiking, reservoir, hybrid) into the comparison before publishing, based on Phase 4 C3 results in hand + remaining compute budget. Output: a written GO/SKIP verdict per candidate architecture, landed in this change's design.md.

### Phase 5 — Cross-architecture analysis + logbook

- Paired-seed Wilcoxon + bootstrap CIs across the C3 primary cells, reusing `scripts/campaigns/aggregate_m613_pilot.py:compute_cross_arm_delta_stats` as the starting point.
- Apply BH-FDR across the active test set per Phase 3's MCC commitment.
- Per-behaviour-component verdict for the connectome row (foraging success / predator survival / thermotaxis isotherm-tracking, extracted from the integrated C3 runs).
- Strict-mask vs soft-prior delta on C3.
- Predator reward-shape ablation analysis.
- Logbook publication at `docs/experiments/logbooks/0XX-weight-search-architecture-ranking.md`. Supporting data (CSVs, per-cell summary tables, plots) at `docs/experiments/logbooks/supporting/0XX-weight-search-architecture-ranking/`. The logbook references only `supporting/*`, never `tmp/*` (per user instruction: `tmp/` doesn't persist).
- Update `openspec/changes/phase6-tracking/tasks.md` T4 row + `docs/roadmap.md` Phase 6 Tranche Tracker T4 row.

## Capabilities

### New Capabilities

- `architecture-comparison-protocol`: methodology guarantees for the Phase 4 cross-architecture comparison sweep — curriculum-then-integrated cell structure, paired-seed statistics, BH-FDR multiple-comparisons correction, per-behaviour-component metric extraction from integrated runs, scratchpad-and-supporting-files persistence discipline.

### Modified Capabilities

- `connectome-ppo-brain`: adds the canonical predator-sensor projection (`predator_gains`: ASH/ASI distal + ALM/AVM anterior + PLM posterior) deferred from `fix-predator-sensing-biology` per its design.md Decision T3.8. No change to existing food-projection, strict-mask, gap-junction, or motor-readout requirements.
- `predator-sensing-biology`: locks the canonical predator-evasion sensor encoding + reward shape that Phase 0's investigation settles. The existing two-channel sensor modules + ContactZone primitives remain unchanged; this delta documents the chosen Phase 4 sensor/reward variant as the canonical one (with the alternative variants explored in Phase 0 captured as ablations in the logbook, not as new requirements).

## Impact

- **Depends on**: `add-neat-weights-brain` change merged (this change's Phase 1a verifies the `feedforwardga` brain is registry-consumable; Phase 4's GA C-cells consume configs that instantiate it).
- **New code**: Phase 1b adds the `predator_gains` projection to `ConnectomePPOBrain` (~50-100 LOC + tests). Phase 0 may add 1-2 ablation sensor-module variants under `quantumnematode/brain/modules.py` (each ~30-50 LOC + tests). Phase 5 may add an analysis script under `scripts/analysis/` that consumes the cross-cell evaluation CSVs.
- **New configs**: ~12 new small-klinotaxis configs (4 architectures × C1/C2/C3) + connectome-specific predator/thermotaxis configs + Phase 0 ablation configs. All under `configs/scenarios/`.
- **No new external dependencies.** Reuses existing PPO, GA, evolution-loop, env, and statistical-test infrastructure.
- **Compute footprint**: 4 architectures × (2 smokes + n=4 seeds) = ~24 simulation runs at canonical budget + ablations + Phase 0 sub-investigations. Phase 2 quantifies the wall-time projection.
- **Logbook output**: one new logbook entry at `docs/experiments/logbooks/0XX-weight-search-architecture-ranking.md` + supporting/ directory.
- **Downstream**: feeds Gate 3 evidence base (G3.a, G3.b, partial G3.c). T7 (Phase 6's L2 re-run on the upgraded substrate) measures the env-upgrade delta against this change's grid baseline.
- **Backward compatibility**: existing configs and brains are not modified except for the additive `predator_gains` projection on `ConnectomePPOBrain` (which is new optional behaviour gated by config — Phase 4 cells turn it on, Gate 1's klinotaxis-only configs are unaffected).
