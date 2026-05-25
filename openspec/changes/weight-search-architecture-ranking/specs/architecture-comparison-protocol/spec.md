## ADDED Requirements

### Requirement: Curriculum-Then-Integrated Cell Structure

The cross-architecture comparison SHALL evaluate each architecture across a three-cell curriculum: a foraging-only smoke (C1, n=1 seed, short budget), a foraging+predator smoke (C2, n=1 seed), and an integrated foraging+predator+thermotaxis primary cell (C3, n ≥ 4 seeds per planning decision T4.0b). Only the C3 cells carry the ranked comparison; C1 and C2 SHALL be treated as de-risking smokes whose failure SHALL block launching the corresponding C3 for that architecture until the failure is diagnosed.

#### Scenario: C1 smoke runs before C2 smoke before C3 cell for each architecture

- **GIVEN** an architecture queued for the comparison sweep
- **WHEN** the architecture's C3 (primary) cell is queued to launch
- **THEN** the architecture's C1 smoke SHALL have completed without error
- **AND** the architecture's C2 smoke SHALL have completed without error
- **AND** C3 SHALL NOT launch until both smokes are green

#### Scenario: C3 is the primary cell with n ≥ 4 seeds; C1 and C2 are throwaway

- **WHEN** the comparison sweep records per-architecture results
- **THEN** the architecture ranking + paired-seed statistics SHALL be computed from C3 results only
- **AND** C1 and C2 results SHALL be retained as smoke verification only (no statistical aggregation, no ranking impact)

### Requirement: Integrated Three-Behaviour Configuration in C3

The C3 primary cell SHALL run all three Phase 6 behaviours simultaneously in one integrated environment: food chemotaxis + predator evasion + thermotaxis active in the same simulation. Aerotaxis SHALL NOT be enabled (per [phase6-tracking design.md § Decision 5](../../../phase6-tracking/design.md) the Phase 6 behaviour set is fixed at three). Per-behaviour performance components SHALL be extracted from the integrated runs (e.g. foods collected, predator survival rate, isotherm-tracking metric) for the ranking analysis.

#### Scenario: C3 config enables all three Phase 6 behaviours simultaneously

- **WHEN** a C3 config is parsed
- **THEN** the resulting `SimulationConfig` SHALL have foraging enabled (food sources present)
- **AND** predators enabled (count ≥ 1)
- **AND** thermotaxis active (thermal sources present in the env)
- **AND** aerotaxis disabled (no oxygen primitives in the env config)

#### Scenario: Per-behaviour components are extracted from the integrated C3 run for analysis

- **WHEN** a C3 cell completes
- **THEN** the analysis pipeline SHALL extract per-episode foods-collected, predator-survival-rate, and isotherm-tracking metric from the single integrated run
- **AND** the ranking SHALL be reported both on the combined overall metric AND on the per-behaviour components, so the "where does architecture X rank?" question can be answered per behaviour as well as overall

### Requirement: Paired-Seed Statistics with BH-FDR Multiple-Comparisons Correction

The C3 cross-architecture analysis SHALL compute paired-seed deltas with one-sided Wilcoxon signed-rank tests and 80% bootstrap CIs (1000 resamples, seeded RNG) for each architecture pair on each per-behaviour component and on the combined metric. The resulting p-values SHALL be corrected via Benjamini-Hochberg FDR at α=0.05 across the **realised** active test set within this change's Phase 4 evaluation (the realised set MAY be smaller than the planned set if Phase 4 risk-mitigation drops an architecture mid-stream). The MCC strategy SHALL be committed in this change's design.md before any Phase 4 cell launches; mid-Phase 4 strategy changes SHALL be forbidden.

**Implementation note for the analysis script.** The existing utility `compute_cross_arm_delta_stats` at `scripts/campaigns/aggregate_m613_pilot.py:329-418` is M6.13-specific (its dict key is `(arm, seed, fundus_idx)` and it averages F1+ retention specifically). The analysis script in this change SHALL extract the reusable inner computation pattern (paired-seed delta → one-sided Wilcoxon → 80% bootstrap CI with seeded RNG, 1000 resamples) into a generic helper (e.g. `_paired_seed_wilcoxon_bootstrap(deltas: list[float]) -> dict`) that operates on a flat list of per-seed deltas, NOT directly call the M6.13 function. The bootstrap CI level (80%) and resample count (1000) constants from `aggregate_m613_pilot.py` SHALL be carried forward to preserve methodological consistency across the project.

#### Scenario: Paired-seed Wilcoxon + bootstrap CI is computed per architecture pair per metric

- **GIVEN** four architecture C3 cells (connectome, mlp_ppo, lstm_gru_ppo, feedforward_ga) with n ≥ 4 paired seeds each
- **WHEN** the analysis script runs
- **THEN** for each pair of architectures (A, B) and each metric M, the script SHALL compute the per-seed delta `metric(A, seed) - metric(B, seed)`
- **AND** report the mean delta, the one-sided Wilcoxon p-value (alternative: A > B), and the 80% bootstrap CI of the mean delta
- **AND** the bootstrap RNG SHALL be seeded (deterministic across re-runs)

#### Scenario: BH-FDR correction applied across the active test set

- **GIVEN** a set of N paired-comparison p-values from the analysis above
- **WHEN** the analysis script applies multiple-comparisons correction
- **THEN** the script SHALL apply Benjamini-Hochberg FDR at α=0.05 across all N tests within Phase 4
- **AND** report both the raw p-value and the BH-adjusted q-value per comparison
- **AND** the active test set SHALL include all per-pair per-metric tests across the C3 cells (4 architectures × 4 component metrics × C(4,2) = 6 pairs = up to 24 tests if all four architectures complete and four metrics are reported)

#### Scenario: MCC strategy is pre-committed and immutable mid-Phase 4

- **WHEN** the change's design.md is finalised before any Phase 4 cell launches
- **THEN** the design.md SHALL document the BH-FDR-at-α=0.05 commitment with rationale
- **WHEN** Phase 4 cells launch
- **THEN** the analysis script's MCC strategy SHALL match the design.md commitment
- **AND** any subsequent change to the MCC strategy mid-Phase 4 SHALL be rejected (the commitment lasts until Phase 4 closes)

### Requirement: Architecture-Promotion Gate Between Phase 4 and Phase 5

A written architecture-promotion gate (Phase 4.5) SHALL land between Phase 4 cell completion and Phase 5 analysis publication. The gate SHALL decide, per SHOULD/MAY architecture candidate from [phase6-tracking design.md § Decision 4](../../../phase6-tracking/design.md) (quantum, spiking, reservoir, hybrid), whether to promote that architecture into the comparison before publishing. The decision per candidate SHALL be GO (promote, run additional cells before Phase 5) or SKIP (do not promote; document the rationale). The verdict per candidate SHALL be landed in this change's design.md as a written decision moment, not a silent extension or contraction of scope.

#### Scenario: Phase 4.5 records a per-candidate verdict in design.md

- **GIVEN** all Phase 4 C3 cells complete (4 MUST architectures × 1 C3 each)
- **WHEN** the Phase 4.5 gate runs
- **THEN** this change's design.md SHALL be amended with a `## Phase 4.5 architecture-promotion gate` section
- **AND** the section SHALL list each SHOULD/MAY candidate (quantum, spiking, reservoir, hybrid) with a GO or SKIP verdict and the rationale
- **AND** SKIP rationales SHALL reference the criteria (compute fit, roadmap relevance, headline impact) and where applicable the Phase 6 Decision 4 SHOULD/MAY classification + deferral mechanism

#### Scenario: Promoted architectures run additional cells before Phase 5

- **GIVEN** a Phase 4.5 GO verdict for candidate architecture X
- **WHEN** Phase 5 analysis is queued
- **THEN** architecture X's C1 + C2 + C3 cells SHALL be launched and complete before the cross-cell analysis runs
- **AND** the MCC active test set SHALL include the additional pairs introduced by X's C3 cell

### Requirement: Logbook Supporting Data Persistence Discipline

Per-cell raw artefacts (per-cell summary CSVs, plots, supporting tables) that the change's logbook references SHALL live under `docs/experiments/logbooks/supporting/<NNN>-weight-search-architecture-ranking/`. The change's logbook SHALL NOT reference any path under `tmp/` (`tmp/` artefacts do not persist across machine state). In-flight working forensics MAY live in `tmp/evaluations/weight-search-architecture-ranking/...` as scratchpads, but anything the logbook references SHALL first be promoted into `supporting/`.

#### Scenario: Logbook references resolve to supporting/\* paths only

- **WHEN** the change's logbook is published
- **THEN** any file-path reference (relative or absolute) in the logbook body SHALL resolve to a path under `docs/experiments/logbooks/supporting/`, `docs/experiments/`, `openspec/`, `scripts/`, `packages/`, or another permanent repository directory
- **AND** the logbook SHALL NOT reference any path under `tmp/`

#### Scenario: Scratchpad artefacts are promoted to supporting/ before logbook publication

- **GIVEN** the Phase 4 scratchpad at `tmp/evaluations/weight-search-architecture-ranking/weight-search-architecture-ranking_scratchpad.md` and the Phase 0 scratchpad at `tmp/evaluations/<phase-0-topic>/<phase-0-topic>_scratchpad.md`
- **WHEN** the change's logbook is being authored
- **THEN** any per-cell artefact the logbook needs to cite SHALL be copied into `docs/experiments/logbooks/supporting/<NNN>-weight-search-architecture-ranking/` before the logbook references it
- **AND** the scratchpads themselves MAY remain in `tmp/` (working forensics) but SHALL NOT be cited from the logbook
