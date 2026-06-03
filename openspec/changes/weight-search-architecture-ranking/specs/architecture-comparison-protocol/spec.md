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

### Requirement: Convergence-Aware Budget and Plateau-Performance Metric

The episode budget for the C3 primary cells SHALL be set so that every evaluated architecture reaches a plateau (converges) on the cell, so the comparison is plateau-vs-plateau rather than arbitrary-cutoff-vs-arbitrary-cutoff.

Two distinct convergence operations are used (they are NOT the same test):

- **Budget-setting convergence** (Phase 2 pre-flight): a run is converged for budget-selection purposes when its trailing-window success rate is stable — last-25-mean success within ±5 percentage points of last-100-mean success. This test selects the C3 budget (it set 1000 episodes: the recurrent architectures plateau by 1000ep but not 500ep).
- **Ranked-metric plateau detection** (Phase 5 analysis): the ranked metric is **`post_convergence_success_rate`** — the full-clear (`COMPLETED_ALL_FOOD`) rate averaged over the post-convergence plateau, where the plateau onset is found by `detect_convergence` ([`packages/quantum-nematode/quantumnematode/benchmark/convergence.py`](../../../../packages/quantum-nematode/quantumnematode/benchmark/convergence.py)): the earliest 10-run window whose success-rate variance is below 0.05 AND whose mean success exceeds 0.5, requiring ≥ 30 total runs; the metric averages success from that onset to the end of the run.

The ranked metric SHALL be `post_convergence_success_rate`, NOT a fixed last-N window mean. The fixed-window choice was deliberately rejected during implementation because the evaluated arms have very different warm-up lengths — the from-scratch spiking and quantum arms have long dead-exploration warm-ups on this lethal cell, so a fixed last-N (e.g. last-25) window would mis-measure the slow-igniting arms relative to the fast learners; ranking on the detected post-convergence plateau is the fair comparison. (Overall `success_rate`, which includes the warm-up, is retained alongside in the per-seed export for reference.) For GA cells the analogue is the evolved-champion full-clear rate over a frozen eval. Convergence is distinct from success — an architecture MAY converge to a low plateau (its ceiling, a valid finding) or a high one.

**Sample-efficiency reporting (realised scope).** The primary cross-architecture ranking is on asymptotic plateau performance (`post_convergence_success_rate`). Sample efficiency / warm-up length is reported **descriptively** — the per-500-episode full-clear-rate trajectory in the logbook illustrates the long-warm-up-then-ignition shape of the slow arms — rather than as a computed per-architecture "episodes-to-90%-of-plateau" metric. This is a deliberate realised simplification: the asymptotic ranking is the load-bearing result, with the warm-up trajectory as qualitative context. A future pass (e.g. the T7 re-run) MAY add the computed sample-efficiency dimension.

#### Scenario: C3 budget is set to the slowest-converging architecture

- **GIVEN** the set of architectures to be compared on the C3 cell
- **WHEN** the C3 episode budget is chosen
- **THEN** the budget SHALL be at least the convergence point (per the ±5pp trailing-window test) of the slowest-converging architecture in the set
- **AND** the Phase 2 pre-flight evidence for that budget SHALL be recorded (the weight-search-architecture-ranking change sets it to 1000 episodes based on its pre-flight: recurrent architectures — LSTMPPO, connectome — plateau by 1000ep but not by 500ep)

#### Scenario: Non-plateau triggers a budget extension and rerun

- **WHEN** a C3 cell's run does NOT satisfy the convergence test at its budget (last-25 mean still diverges from last-100 mean by more than ±5pp, indicating it is still climbing)
- **THEN** that is a trigger to extend the episode (or generation) budget for that architecture and rerun until it reaches its plateau
- **AND** because the ranked metric is the post-convergence plateau (`post_convergence_success_rate`), the comparison is plateau-vs-plateau even when arms reach their plateaus at different episode budgets — a uniform episode budget across arms is therefore NOT required, provided every arm's run has reached its plateau (the convergence detector confirms this per run; an arm that never converges is excluded / flagged, not mis-compared against a fixed cutoff). (Realised: the four MUST cells ran at 1000ep; the from-scratch promoted arms — spiking, equivariant-quantum — ran up to 4000ep for their longer warm-ups; all are compared on their detected plateaus.)
- **AND** when a SHOULD/MAY architecture is added after the initial budget is set, its budget SHALL be set to its own convergence point (extended as needed); the plateau metric normalises across budgets, so the already-converged cells are NOT force-rerun at a uniform budget

#### Scenario: Asymptotic plateau performance is the ranked metric; warm-up is reported descriptively

- **WHEN** the Phase 5 ranking is computed
- **THEN** each architecture's C3 result SHALL report `post_convergence_success_rate` (the detected-plateau full-clear rate) as the ranked asymptotic metric
- **AND** the warm-up / sample-efficiency dimension SHALL be reported descriptively (the per-500-episode clear-rate trajectory) rather than as a computed episodes-to-90%-of-plateau number
- **AND** the ranking narrative MAY note where a "converged to a higher plateau" claim is distinct from a "converged faster" observation

### Requirement: Paired-Seed Statistics with BH-FDR Multiple-Comparisons Correction

The C3 cross-architecture analysis SHALL compute paired-seed deltas with one-sided Wilcoxon signed-rank tests and 80% bootstrap CIs (1000 resamples, seeded RNG) for each architecture pair. The **BH-FDR family** (the set of p-values corrected together via Benjamini-Hochberg FDR at α=0.05) is the cross-architecture pairwise comparisons on the **primary ranked metric** (`post_convergence_success_rate`) across the realised architecture set. The realised set is the four MUST families plus any Phase 4.5 promotions — realised as **7 architectures → C(7,2) = 21 pairs** (it MAY differ from the planned four-family set: larger if Phase 4.5 promotes SHOULD/MAY architectures, smaller if Phase 4 risk-mitigation drops one). The per-behaviour sub-metrics (foraging foods, predator-evasion rate, thermal-comfort) and the connectome wins/ties/losses verdict are reported as **descriptive** paired-seed deltas (Wilcoxon p + 80% bootstrap CI) and are NOT folded into the BH-FDR family — the family is held to the single headline ranking metric so the correction stays interpretable. The MCC strategy SHALL be committed in this change's design.md before any Phase 4 cell launches; mid-Phase 4 strategy changes SHALL be forbidden.

**Implementation note for the analysis script.** The existing utility `compute_cross_arm_delta_stats` at `scripts/campaigns/aggregate_m613_pilot.py:329-418` is M6.13-specific (its dict key is `(arm, seed, fundus_idx)` and it averages F1+ retention specifically). The analysis script in this change SHALL extract the reusable inner computation pattern (paired-seed delta → one-sided Wilcoxon → 80% bootstrap CI with seeded RNG, 1000 resamples) into a generic helper (e.g. `_paired_seed_wilcoxon_bootstrap(deltas: list[float]) -> dict`) that operates on a flat list of per-seed deltas, NOT directly call the M6.13 function. The bootstrap CI level (80%) and resample count (1000) constants from `aggregate_m613_pilot.py` SHALL be carried forward to preserve methodological consistency across the project.

**Rationale for 80% bootstrap CIs (α=0.20).** The 80% CI level is set explicitly by `CROSS_ARM_BOOTSTRAP_CI_LEVEL = 0.80` in [`scripts/campaigns/aggregate_m613_pilot.py:98`](../../../../scripts/campaigns/aggregate_m613_pilot.py#L98) and [`scripts/campaigns/aggregate_m69_pilot.py:90`](../../../../scripts/campaigns/aggregate_m69_pilot.py#L90) (both with the inline comment `80% CI ⇒ alpha=0.20`); the M6.13 + M6.9 pilots established the precedent. The choice is a deliberate trade-off: narrower intervals than 95% CIs increase precision and discriminative power for exploratory paired-seed comparisons at the n≥4 sample sizes Phase 5 inherited as a floor, at the cost of reduced coverage probability. Carrying it forward here preserves comparability with prior project analyses and signals that this change's outputs are exploratory ranking evidence (intended to feed Gate 3, not to support stand-alone confirmatory claims). If a Gate 3 reviewer requests 95% CIs for a specific headline claim, the analysis script can rerun with `CROSS_ARM_BOOTSTRAP_CI_LEVEL = 0.95` against the same archived per-seed deltas — the choice is reversible at analysis time.

#### Scenario: Paired-seed Wilcoxon + bootstrap CI is computed per architecture pair per metric

- **GIVEN** the realised architecture C3 cells (the four MUST families connectome / mlp_ppo / lstm_gru_ppo / feedforward_ga, plus any Phase 4.5 promotions — realised: 7 architectures) with n ≥ 4 paired seeds each (n = 8 in the realised run)
- **WHEN** the analysis script runs
- **THEN** for each pair of architectures (A, B) on the primary ranked metric, the script SHALL compute the per-seed delta `metric(A, seed) - metric(B, seed)`
- **AND** report the mean delta, the one-sided Wilcoxon p-value (alternative: A > B), and the 80% bootstrap CI of the mean delta
- **AND** the bootstrap RNG SHALL be seeded (deterministic across re-runs)

#### Scenario: BH-FDR correction applied across the active test set

- **GIVEN** the set of N paired-comparison p-values on the primary ranked metric (`post_convergence_success_rate`) across the realised architecture set
- **WHEN** the analysis script applies multiple-comparisons correction
- **THEN** the script SHALL apply Benjamini-Hochberg FDR at α=0.05 across all N pairwise tests within Phase 4
- **AND** report both the raw p-value and the BH-adjusted q-value per comparison
- **AND** the active test set SHALL be the cross-architecture pairwise comparisons on the primary metric across the realised set (realised: 7 architectures → C(7,2) = 21 pairs); the per-behaviour sub-metric deltas and the connectome verdict are reported descriptively (uncorrected) alongside, not folded into the FDR family

#### Scenario: MCC strategy is pre-committed and immutable mid-Phase 4

- **WHEN** the change's design.md is finalised before any Phase 4 cell launches
- **THEN** the design.md SHALL document the BH-FDR-at-α=0.05 commitment with rationale
- **WHEN** Phase 4 cells launch
- **THEN** the analysis script's MCC strategy SHALL match the design.md commitment
- **AND** any subsequent change to the MCC strategy mid-Phase 4 SHALL be rejected (the commitment lasts until Phase 4 closes)

### Requirement: Architecture-Promotion Gate Between Phase 4 and Phase 5

A written architecture-promotion gate (Phase 4.5) SHALL land between Phase 4 cell completion and Phase 5 analysis publication. The gate SHALL decide, per SHOULD/MAY architecture candidate from [phase6-tracking design.md § Decision 4](../../../phase6-tracking/design.md) (quantum, spiking, reservoir, hybrid), whether to promote that architecture into the comparison before publishing. The decision per candidate SHALL be GO (promote, run additional cells before Phase 5) or SKIP (do not promote; document the rationale). The verdict per candidate SHALL be landed in this change's design.md as a written decision moment, not a silent extension or contraction of scope.

The gate MAY additionally evaluate a candidate family that is NOT in the Decision 4 SHOULD/MAY table. If such an off-list family is promoted (GO), its verdict SHALL record that adding it is a [phase6-tracking § Decision 4](../../../phase6-tracking/design.md) amendment event — a scope note flagging that the Decision 4 table should absorb the family at the next synthesis — so the off-list promotion is explicit, not silent. (Realised: CfC, a liquid / closed-form-continuous-time recurrent network, was promoted GO off the Decision 4 list with such a scope note.)

#### Scenario: Phase 4.5 records a per-candidate verdict in design.md

- **GIVEN** all Phase 4 C3 cells complete (4 MUST architectures × 1 C3 each)
- **WHEN** the Phase 4.5 gate runs
- **THEN** this change's design.md SHALL be amended with a `## Phase 4.5 architecture-promotion gate` section
- **AND** the section SHALL list each SHOULD/MAY candidate (quantum, spiking, reservoir, hybrid) with a GO or SKIP verdict and the rationale
- **AND** any promoted family NOT in the Decision 4 SHOULD/MAY table SHALL also appear with a GO/SKIP verdict and a Decision-4 scope note (realised: CfC, GO, off-list)
- **AND** SKIP rationales SHALL reference the criteria (compute fit, roadmap relevance, headline impact) and where applicable the Phase 6 Decision 4 SHOULD/MAY classification + deferral mechanism. **Operational definition of headline impact**: a candidate architecture has headline impact if, given the Phase 4 C3 results in hand, its plausible C3 performance range (per a back-of-envelope estimate from its Phase 2 forecast or its closest existing baseline) would change which architecture tops the ranking on ≥ 1 per-behaviour component. A candidate with no plausible scenario for topping any component-level ranking has no headline impact.

#### Scenario: Promoted architectures run additional cells before Phase 5

- **GIVEN** a Phase 4.5 GO verdict for candidate architecture X
- **WHEN** Phase 5 analysis is queued
- **THEN** architecture X's C1 + C2 + C3 cells SHALL be launched and complete before the cross-cell analysis runs
- **AND** the MCC active test set SHALL include the additional pairs introduced by X's C3 cell

### Requirement: Controlled Attribution for a Promoted Structured-Prior Architecture

When a promoted architecture carries a non-trivial inductive bias (e.g. a quantum circuit, or a hard-coded symmetry / equivariance prior) and lands at or near the top of the C3 ranking, its apparent advantage SHALL be attributed via matched-capacity control arms BEFORE any architecture-specific advantage is claimed in the logbook. A raw rank is not a claim of advantage: the headline payload for such an architecture is the **controlled-attribution delta**, not its position in the ranking.

The control set SHALL isolate each candidate source of advantage at matched capacity:

- A **fair classical control** that reproduces the architecture's inductive bias in a conventional substrate at matched parameter capacity (e.g. for an equivariant quantum actor: a classical actor with the same equivariance prior and a comparable parameter count). The promoted-arch-minus-fair-control delta isolates the genuinely-exotic component (e.g. the quantum circuit) from the inductive bias it shares with the control.
- A **structure-ablation control** at matched capacity that removes the structural prior (e.g. drops the symmetry / equivariance) while holding capacity fixed. The structure-present-minus-structure-ablated delta isolates the structural prior's contribution.

A control that is weaker than a plain baseline (e.g. a starved sub-capacity MLP) SHALL NOT be used to claim an advantage — a positive delta against an under-capacity control is an artifact, not a result, and the logbook SHALL flag it as such if it is reported.

#### Scenario: A leading structured-prior arm reports controlled-attribution deltas, not just its rank

- **GIVEN** a Phase 4.5-promoted architecture with a non-trivial inductive bias that lands at or near the top of the C3 ranking
- **WHEN** the logbook reports its result
- **THEN** the logbook SHALL report the promoted-arch-minus-fair-classical-control delta (isolating the exotic component) AND the structure-present-minus-structure-ablated delta (isolating the prior), each with paired-seed Wilcoxon p + bootstrap CI
- **AND** any advantage claim SHALL be supported by those controlled deltas, not by the raw rank or by a delta against an under-capacity control
- **AND** if a delta against an under-capacity control is shown, the logbook SHALL flag it as an artifact

#### Scenario: Realised quantum-arm attribution

- **GIVEN** the equivariant-quantum arm (a bilateral-Z₂-equivariant parameterised quantum circuit) was promoted and led the C3 ranking
- **WHEN** its attribution is computed
- **THEN** the control set SHALL include: an unstructured-quantum arm; a thin classical-equivariant arm (an under-capacity control, flagged as such); a matched-capacity rich classical-equivariant arm (the fair control); and a matched-capacity rich classical non-equivariant arm (the structure-ablation control) — the latter two implemented via the `classical_rich` / `classical_symmetrise` flags on the equivariant-quantum brain
- **AND** the realised verdict SHALL be recorded: quantum minus fair-classical = −1.9 (ns — **no quantum advantage**); the +24.6 delta against the thin control is an artifact; matched-capacity symmetry deltas +1.5 (classical) / +2.4 (quantum) are both ns (**no significant symmetry effect**)

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

### Requirement: Per-architecture C3 reward-weight tuning discipline

Each architecture's C3 cell SHALL run with the documented global reward weights (inherited from the closest existing reference config, e.g. `oxygen_thermal_pursuit/mlpppo_large_oracle.yml`'s reward block, scaled appropriately for the small variant) by default; per-architecture divergence is the exception, not the rule. Per-architecture reward weights MAY diverge from the global default only when BOTH (a) the architecture's C2 (foraging + predator) smoke result shows its foraging-or-predator metric below 50% of the same metric on at least one other architecture's C2 result, AND (b) the imbalance is supported by ≥ n=2 C2 seeds (a single C2 run is insufficient evidence). When tuning is triggered, the chosen per-arch reward weights SHALL be documented in the change's design.md (under a `## Per-architecture reward weights for C3` section) BEFORE the n≥4 C3 cell launches for that architecture, including the rationale, the C2 numbers that triggered tuning, and the chosen weights. Once C3 launches for an architecture, the reward weights for that architecture's C3 cell SHALL be frozen — no mid-C3 retuning is permitted, even if early-episode metrics look bad.

#### Scenario: Default reward weights used when tuning trigger does not fire

- **GIVEN** an architecture whose C2 smoke result shows its foraging-and-predator metrics within 50% of every other architecture's C2 result
- **WHEN** the C3 cell config is authored
- **THEN** the reward block SHALL match the global default reward weights inherited from the reference config
- **AND** no per-arch entry SHALL be added to the `## Per-architecture reward weights for C3` section of design.md for this architecture

#### Scenario: Tuning trigger requires both C2 imbalance AND a second seed

- **GIVEN** an architecture whose first C2 seed shows a foraging-or-predator metric below 50% of at least one other architecture's first-seed C2 result
- **WHEN** the implementer considers picking per-architecture reward weights
- **THEN** a second C2 seed for that architecture SHALL be run before any per-arch weights are committed
- **AND** the per-arch weights SHALL be committed only if the n=2 C2 mean confirms the >50% imbalance (a noisy single-seed result that doesn't replicate SHALL fall back to the global default)

#### Scenario: Pre-C3 documentation lands before n=4 C3 launches

- **GIVEN** the tuning trigger has fired for an architecture (n=2 C2 imbalance confirmed)
- **WHEN** the n≥4 C3 cell for that architecture is queued to launch
- **THEN** design.md SHALL contain a `## Per-architecture reward weights for C3` section entry for that architecture
- **AND** the entry SHALL include the rationale, the C2 numbers that triggered tuning, and the chosen weights
- **AND** C3 SHALL NOT launch for that architecture until the documentation is in place

#### Scenario: No mid-C3 retuning

- **GIVEN** a C3 cell that has begun execution for an architecture (any seed)
- **WHEN** mid-run metrics suggest the reward weights are imbalanced
- **THEN** the reward weights for any remaining seeds of that architecture's C3 cell SHALL remain frozen at the pre-launch values
- **AND** any reward-weight change SHALL be deferred to a post-Phase-4 follow-up change (the no-retuning rule lasts until Phase 4 closes)
