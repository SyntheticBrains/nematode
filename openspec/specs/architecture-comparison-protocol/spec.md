# architecture-comparison-protocol Specification

## Purpose

Defines the methodology for fair, statistically-rigorous cross-architecture brain comparisons: the curriculum-then-integrated cell structure, the convergence-aware plateau-performance ranking metric (`post_convergence_success_rate`), paired-seed one-sided Wilcoxon + bootstrap CIs with BH-FDR multiple-comparisons correction, the architecture-promotion gate, the controlled-attribution requirement for promoted structured-prior (e.g. quantum / equivariant) architectures, and the logbook supporting-data persistence discipline.

*(Widened 2026-09-20.)* It also holds the **comparison methodology that applies whatever is being compared** — how a control coupled to its own runs is declared, how a capacity change crossed with a structure contrast is read, when a committed baseline may be reused, how a shipped result carries an uncontrolled confound, and what a replication may and may not rewrite. Eleven such requirements were consolidated here from `plasticity-evaluation`, where eighteen Phase 7 milestone changes had registered them because it was the nearest plasticity-named capability; they are not specific to a plasticity rule, and the two wiring-contrast requirements Phase 7 registered here directly set the precedent. See `openspec/changes/archive/2026-09-20-consolidate-plasticity-methodology/` for the full 44-row redistribution.

## Requirements

### Requirement: Curriculum-Then-Integrated Cell Structure

The cross-architecture comparison SHALL evaluate each architecture across a three-cell curriculum: a foraging-only smoke (C1, n=1 seed, short budget), a foraging+predator smoke (C2, n=1 seed), and an integrated foraging+predator+thermotaxis primary cell (C3, n ≥ 4 seeds per planning decision T4.0b). Only the C3 cells carry the ranked comparison; C1 and C2 SHALL be treated as de-risking smokes whose failure SHALL block launching the corresponding C3 for that architecture until the failure is diagnosed. The smoke-only status of C1 and C2 applies to the **architecture ranking**. A **wiring contrast** — the same substrate under two wirings — run on a single-behaviour cell at full seed count is governed by the wiring-contrast requirements below and is not a ranking result.

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

#### Scenario: A wiring contrast on a single-behaviour cell is not a ranking result

- **GIVEN** a wiring contrast run on a single-behaviour cell at full seed count
- **WHEN** its statistics are computed
- **THEN** it SHALL be reported under the wiring-contrast requirements, and SHALL NOT enter or alter
  the architecture ranking

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

- **Budget-setting convergence** (pre-flight): a run is converged for budget-selection purposes when its trailing-window success rate is stable — last-25-mean success within ±5 percentage points of last-100-mean success. This test selects the C3 budget (it set 1000 episodes for the grid-substrate ranking: the recurrent architectures plateau by 1000ep but not 500ep).
- **Ranked-metric plateau detection** (analysis): the ranked metric is **`post_convergence_success_rate`** — the full-clear (`COMPLETED_ALL_FOOD`) rate averaged over the post-convergence plateau, where the plateau onset is found by `detect_convergence` ([`packages/quantum-nematode/quantumnematode/experiment/convergence.py`](../../../../packages/quantum-nematode/quantumnematode/experiment/convergence.py)). Plateau detection SHALL be **level-agnostic**: convergence (the policy has stopped improving) SHALL be decoupled from the absolute success level, so a converged plateau is detected at any band (e.g. a stable 40% plateau and a stable 95% plateau are both detected). Convergence SHALL be established by **absence of trend** — comparing the final trailing block of runs against the immediately-preceding block of equal size and declaring converged when their mean full-clear rates agree within a band calibrated to block sampling noise (NOT by requiring a low-variance near-homogeneous window, which on binary outcomes is reachable only near 100% and so mis-classifies stable intermediate plateaus as non-converged). The plateau onset SHALL be the start of the final region whose smoothed (rolling-mean) success rate has reached the converged level (excluding the warm-up climb, and not anchoring on a transient touch of the band mid-climb), and the metric averages raw full-clear success from that onset to the end of the run, requiring ≥ 30 total runs. A run still trending at its budget SHALL return no plateau (flagged, not mis-scored).

The ranked metric SHALL be `post_convergence_success_rate`, NOT a fixed last-N window mean. The fixed-window choice was deliberately rejected because the evaluated arms can have very different warm-up lengths — the from-scratch spiking and quantum arms have long dead-exploration warm-ups on this lethal cell, so a fixed last-N (e.g. last-25) window would mis-measure the slow-igniting arms relative to the fast learners; ranking on the detected post-convergence plateau is the fair comparison. The full-window-mean (plateau mean over a post-warmup tail) SHALL be retained alongside as an **agreement cross-check**: for a homogeneous-warm-up arm set it MUST agree with the detected `post_convergence_success_rate` within noise, bounding the risk that plateau detection introduces bias. (Overall `success_rate`, which includes the warm-up, is also retained in the per-seed export for reference.) For GA cells the analogue is the evolved-champion full-clear rate over a frozen eval. Convergence is distinct from success — an architecture MAY converge to a low plateau (its ceiling, a valid finding) or a high one; the ranked metric SHALL measure that plateau level faithfully whether it is high or low.

**Sample-efficiency reporting (realised scope).** The primary cross-architecture ranking is on asymptotic plateau performance (`post_convergence_success_rate`). Sample efficiency / warm-up length is reported **descriptively** — the per-500-episode full-clear-rate trajectory in the logbook illustrates the long-warm-up-then-ignition shape of the slow arms — rather than as a computed per-architecture "episodes-to-90%-of-plateau" metric. This is a deliberate realised simplification: the asymptotic ranking is the load-bearing result, with the warm-up trajectory as qualitative context. A future pass MAY add the computed sample-efficiency dimension.

#### Scenario: C3 budget is set to the slowest-converging architecture

- **GIVEN** the set of architectures to be compared on the C3 cell
- **WHEN** the C3 episode budget is chosen
- **THEN** the budget SHALL be at least the convergence point (per the ±5pp trailing-window test) of the slowest-converging architecture in the set
- **AND** the pre-flight evidence for that budget SHALL be recorded (e.g. the grid-substrate ranking set 1000 episodes based on its pre-flight: recurrent architectures — LSTMPPO, connectome — plateau by 1000ep but not by 500ep)

#### Scenario: Non-plateau triggers a budget extension and rerun

- **WHEN** a C3 cell's run does NOT satisfy the convergence test at its budget (the final trailing block still differs from the preceding block by more than the calibrated band, indicating it is still climbing)
- **THEN** that is a trigger to extend the episode (or generation) budget for that architecture and rerun until it reaches its plateau
- **AND** because the ranked metric is the post-convergence plateau (`post_convergence_success_rate`), the comparison is plateau-vs-plateau even when arms reach their plateaus at different episode budgets — a uniform episode budget across arms is therefore NOT required, provided every arm's run has reached its plateau (the convergence detector confirms this per run; an arm that never converges is excluded / flagged, not mis-compared against a fixed cutoff)
- **AND** when a SHOULD/MAY architecture is added after the initial budget is set, its budget SHALL be set to its own convergence point (extended as needed); the plateau metric normalises across budgets, so the already-converged cells are NOT force-rerun at a uniform budget

#### Scenario: Asymptotic plateau performance is the ranked metric; warm-up is reported descriptively

- **WHEN** the ranking is computed
- **THEN** each architecture's C3 result SHALL report `post_convergence_success_rate` (the detected-plateau full-clear rate) as the ranked asymptotic metric
- **AND** the warm-up / sample-efficiency dimension SHALL be reported descriptively (the per-500-episode clear-rate trajectory) rather than as a computed episodes-to-90%-of-plateau number
- **AND** the ranking narrative MAY note where a "converged to a higher plateau" claim is distinct from a "converged faster" observation

#### Scenario: Level-agnostic plateau detection ranks a sub-saturation learnable band

- **GIVEN** a C3 cell whose difficulty is locked to a learnable sub-saturation band, so converged architectures plateau across a wide range of full-clear rates (e.g. ≈35–80%) rather than near 100%
- **WHEN** `post_convergence_success_rate` is computed per seed
- **THEN** a stable intermediate plateau (e.g. an architecture that completes the full-clear ~45% of episodes from a converged policy) SHALL be detected as converged and SHALL report its plateau rate (≈45%) averaged over the full post-onset region, NOT silently fall back to a fixed last-N-run window nor be mislabeled non-converged
- **AND** an architecture whose run is still trending at its budget SHALL be flagged as non-converged (extend-and-rerun), not ranked on the noisy fallback window
- **AND** the detected per-seed plateau rate SHALL agree, within sampling noise, with the full-window-mean cross-check

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

### Requirement: A wiring contrast is run on a cell matched to the behaviour under claim

Where a comparison asks whether a biological wiring is load-bearing, the cell it is measured on
SHALL be matched to the behaviour the claim is about, and the record SHALL state which behaviours
the cell demands. A contrast measured only on a multi-objective cell SHALL NOT be generalised beyond
that cell's demand, and where the substrate's measured deficit is concentrated in one component of
such a cell, that component SHALL be separated before a null on the cell is generalised to the
wiring. A verdict already committed on such a cell stands as committed, scoped to its cell.

#### Scenario: A multi-objective result is scoped to its cell

- **GIVEN** a wiring contrast measured on a cell demanding several behaviours
- **WHEN** the result is recorded
- **THEN** it SHALL be scoped to that cell's demand, and SHALL NOT be reported as a general
  statement about the wiring

#### Scenario: The component carrying the deficit is separated

- **GIVEN** a substrate whose measured deficit on a multi-objective cell is concentrated in one
  component
- **WHEN** a wiring contrast on that cell returns a null
- **THEN** the null SHALL NOT be generalised beyond that cell until the contrast has been run on a
  cell without that component, and the committed verdict SHALL stand scoped to its cell

### Requirement: A wiring contrast is gated on the cell showing learning

Each arm of a wiring contrast SHALL be paired with its own floor on the same seeds — the same
substrate with the optimiser disabled — and the contrast SHALL NOT be read unless the arm carrying
the claim beats that floor by the registered test. Where it does not, the result SHALL be recorded
as a finding about the platform and SHALL license no conclusion about the wiring.

A ceiling threshold SHALL be registered before any data exists, and where both arms of a contrast
reach it the contrast SHALL be recorded as unresolvable on that cell, with the remedy named in
advance rather than chosen after the outcome is known.

#### Scenario: A contrast whose arm did not learn is not read

- **GIVEN** a wiring contrast whose claim-carrying arm does not beat its own frozen floor
- **WHEN** the family is scored
- **THEN** the cell's verdict SHALL be that no learning occurred, and the contrast SHALL NOT be
  assigned a wiring verdict

#### Scenario: A saturated cell is named, not re-tuned

- **GIVEN** both arms of a contrast at or above the registered ceiling threshold
- **WHEN** the family is scored
- **THEN** the cell SHALL be recorded as unresolvable and the registered remedy applied, and the
  recipe SHALL NOT be adjusted until the arms separate

#### Scenario: A minimum effect is registered beside significance

- **GIVEN** a paired contrast at a sample size where a rank test fires on the consistency of the
  sign rather than the size of the shift
- **WHEN** the primary is registered
- **THEN** a minimum effect SHALL be registered with it, and a significant result below that minimum
  SHALL be recorded as such and SHALL license nothing on its own

### Requirement: An effect present on one cell and absent on another is attributed only after the cells' differences are separated

Where a wiring contrast is positive on one cell and null on another, the record SHALL enumerate every
respect in which the two cells differ, and SHALL NOT attribute the difference in outcome to one of
them while the others stand untested. Where a respect cannot be isolated by the arms available, the
record SHALL name it as unseparated rather than leaving the attribution implied.

#### Scenario: Cells differing in several respects are enumerated before attribution

- **GIVEN** a wiring contrast positive on one cell and null on another
- **WHEN** the difference is recorded
- **THEN** every respect in which the cells differ SHALL be stated
- **AND** no single respect SHALL be named as the cause while others are untested

#### Scenario: An unseparated factor is named rather than implied

- **GIVEN** two candidate explanations that the available arms cannot distinguish
- **WHEN** the result is recorded
- **THEN** both SHALL be reported as live, and the arm that would separate them SHALL be named
- **AND** the record SHALL NOT present either as established

#### Scenario: A difficulty manipulation that fails to discriminate is a fact about the manipulation

- **GIVEN** a cell whose difficulty was raised to make a contrast measurable
- **WHEN** both arms reach the registered ceiling on that cell
- **THEN** the result SHALL be recorded as a property of the manipulation, not of the wiring, and the
  registered remedy SHALL be applied once without tuning the recipe further

### Requirement: A wiring contrast under a learner that does not write the wiring states what it is about

Where a structure contrast is run under a learner that leaves the substrate's own weights fixed, the
record SHALL state that the substrate enters as **fixed features** and not as something the rule
adapts, and SHALL state which claim the result bears on. Such a result SHALL NOT be reported as
satisfying a deliverable whose condition is that the substrate's own weights are plastic.

#### Scenario: The fixed tensors are compared before and after the run

- **GIVEN** a contrast whose premise is that the substrate's own weights do not change
- **WHEN** the result is scored
- **THEN** those tensors SHALL be compared against a control in which nothing learned, the drift
  SHALL be recorded with the result, and the comparison SHALL cover every scored seed
- **AND** any non-zero drift, or drift evidence missing for any scored seed, SHALL return **void** --
  a substrate that moved is not a fixed substrate, and "it could not be checked" is not "it held"

#### Scenario: The learner's relationship to the substrate is recorded with the contrast

- **GIVEN** a wiring contrast run under a learner that does not write the substrate's weights
- **WHEN** the result is recorded
- **THEN** the record SHALL state which tensors the learner writes and which it leaves fixed
- **AND** it SHALL state that the contrast is about the substrate as fixed features

#### Scenario: A positive result is not read as the plastic-substrate deliverable

- **GIVEN** a deliverable whose condition is that the substrate's own weights are plastic
- **WHEN** a contrast under a learner that leaves them fixed returns positive
- **THEN** the record SHALL state that the deliverable's condition remains unmet
- **AND** the positive SHALL be reported as a claim about the substrate's fixed features

### Requirement: A campaign whose null carries a registered consequence states its power in advance

Where a null result would trigger a registered consequence — closing a phase, retiring a programme,
or standing as a claim about the substrate — the registration SHALL state, before the campaign runs,
what effect size the design can detect and against which comparator. Where a comparator's own effect
size is on the record, the power against it SHALL be computed and stated.

#### Scenario: The power arithmetic is registered before the run

- **GIVEN** a campaign whose null outcome carries a registered consequence
- **WHEN** its protocol is registered
- **THEN** the registration SHALL state the seed count, the detectable effect size, and the power
  against the comparator the result will be read beside
- **AND** where the design is underpowered against that comparator, the registration SHALL say so
  rather than leave it to be discovered in the result

#### Scenario: An underpowered null is not recorded as a clean null

- **GIVEN** a campaign underpowered against its comparator
- **WHEN** it returns a null
- **THEN** the record SHALL state the null as underpowered against that comparator
- **AND** it SHALL NOT be reported as evidence that the effect is absent

### Requirement: A control drawn from the same seeds as its runs states that coupling

Where a contrast's control condition is generated from the same seed that generates the run it is
compared against — a rewired graph, a shuffled label, a permuted input — the record SHALL state that
the control and the run are coupled, and SHALL state which of the two a later panel varies. Where two
results share their control conditions, the record SHALL NOT describe them as independent
replications of each other.

#### Scenario: The coupling is recorded with the result

- **GIVEN** a control generated from the run seed rather than from a seed of its own
- **WHEN** the result is recorded
- **THEN** the record SHALL state that the control and the run vary together
- **AND** it SHALL state what a later panel on fresh seeds would and would not separate

#### Scenario: Results sharing controls are not independent replications

- **GIVEN** two results whose control conditions are drawn from overlapping seed sets
- **WHEN** they are cited together
- **THEN** the record SHALL state the overlap
- **AND** neither SHALL be described as an independent replication of the other

### Requirement: A replication uses the original instrument unmodified

Where a result is re-run to test whether it holds, the replication SHALL be scored by the same
analysis the original was scored by, without modification. Where the original instrument cannot score
the replication, that SHALL be recorded as a limitation of the replication rather than resolved by
writing a new instrument.

#### Scenario: The instrument is not rewritten for the replication

- **GIVEN** a committed analysis that produced a result
- **WHEN** that result is replicated
- **THEN** the replication SHALL be scored by that analysis unmodified
- **AND** any new code SHALL be confined to preparing its inputs and reporting its outputs

#### Scenario: A failure to replicate is not attributed to the instrument

- **GIVEN** a replication scored by the original instrument
- **WHEN** it does not reproduce the original result
- **THEN** the record SHALL state that the reading is unchanged and the evidence differs
- **AND** the original result SHALL be withdrawn or qualified on the record rather than defended

### Requirement: A multi-panel replication fixes the disagreement case in advance

Where a replication covers more than one panel, the registration SHALL state before it runs how a
disagreement between panels is read. A split SHALL be reported as a split, and SHALL NOT be resolved
toward whichever panel supports the original claim.

#### Scenario: A split is reported as a split

- **GIVEN** a replication of two or more panels
- **WHEN** some panels replicate and others do not
- **THEN** each panel SHALL be reported against the registered branches on its own
- **AND** the pooled reading SHALL be withheld, the split recorded as evidence about the original
  claim's scope

### Requirement: A capacity manipulation crossed with a structure contrast is read as an interaction

Where an experiment changes a learner's capacity — the parameter count of a readout, a layer width,
the number of adapted tensors — in order to ask whether a structure effect was hidden by that
capacity, the record SHALL read the **interaction** between capacity and structure as the primary,
and SHALL NOT read a capacity main effect as evidence about the structure. A larger parameter set
learning faster is a fact about the parameter set.

#### Scenario: The capacity change is crossed rather than compared across campaigns

- **GIVEN** a structure contrast that returned null at one capacity
- **WHEN** the question is whether that capacity hid the structure effect
- **THEN** the design SHALL run both capacities against **both** levels of the structure contrast
- **AND** the primary contrast SHALL be the interaction, stated as such before the campaign runs
- **AND** a capacity main effect SHALL be reported beside the interaction and never in place of it

#### Scenario: The capacities are matched at initialisation

- **GIVEN** two capacities of the same learner compared on the same seeds
- **WHEN** the arms are constructed
- **THEN** the record SHALL state what differs between them and what does not, covering the random
  draws consumed, the initial policy, and any control arm claimed to be shared
- **AND** where an arm is claimed to be unaffected by the capacity change, that invariance SHALL be
  asserted by test rather than assumed, and a failure SHALL stop the campaign

#### Scenario: An analytic equivalence is not treated as run-level identity

- **GIVEN** two configurations shown to compute the same function in exact arithmetic
- **WHEN** that equivalence is used to justify running one of them in place of the other
- **THEN** the record SHALL establish that the two agree at the precision the runs use, not only
  analytically
- **AND** where they differ at that precision in a system whose trajectory depends on the difference,
  the configurations SHALL be run separately and the equivalence SHALL be reported as a statement
  about the computed function rather than about the runs

#### Scenario: A null interaction states the panel's sensitivity

- **GIVEN** an interaction contrast, whose per-seed variance exceeds that of either single contrast
- **WHEN** the interaction is not significant
- **THEN** the record SHALL report it as no interaction detected **at the panel's sensitivity**, with
  that sensitivity computed from observed per-seed spread and registered before the campaign ran
- **AND** that spread SHALL come from a source frozen before the campaign — a pilot on disjoint seeds,
  or prior committed data — and SHALL NOT be computed from the campaign's own results, which would
  make a pre-registered sensitivity a function of the outcome it is meant to bound
  *(clarified 2026-09-20 at the consolidation's PR review; the requirement was otherwise moved
  verbatim from `plasticity-evaluation`, and this is the one clause that differs)*
- **AND** it SHALL NOT be reported as excluding an interaction of unspecified size

### Requirement: A metric is chosen for the contrast it must support, and a departure is registered with its reason

Where a campaign departs from the metric a committed instrument used for a comparable contrast, the
record SHALL state the departure and its reason **before the campaign runs**, and SHALL report the
committed instrument's metric beside the chosen one.

#### Scenario: A censored metric is not used for a difference of differences

- **GIVEN** a metric that is right-censored at a horizon
- **AND** a contrast formed as a difference between two differences across cells of a design
- **WHEN** the censoring rate is not known to be equal across those cells
- **THEN** that metric SHALL NOT be the primary for that contrast
- **AND** where it is reported, its censoring SHALL be counted **per cell** rather than pooled

#### Scenario: The departed-from metric is reported beside the chosen one

- **GIVEN** a campaign that changed its primary metric from the one a committed instrument used
- **WHEN** the result is recorded
- **THEN** both metrics SHALL be reported
- **AND** where they disagree, the record SHALL state the disagreement rather than reporting only the
  primary

### Requirement: A feature ablation on a positive structure result registers a minimum effect as a decision rule

Where an experiment removes one feature of a substrate to ask whether a previously established
structure effect survives, the record SHALL register, before the campaign runs, the smallest
reduction in that effect that will be read as the feature carrying it — stated as a fraction of the
established effect — and SHALL NOT read a significant reduction below that minimum as the feature
carrying the effect.

#### Scenario: The minimum is a fraction of the effect being ablated

- **GIVEN** an established structure effect of a known size
- **WHEN** an ablation is registered against it
- **THEN** the minimum reduction SHALL be stated as a fraction of that size, with the power to detect
  it computed from observed spread and registered beside it

#### Scenario: Significant below the minimum is not carrying

- **GIVEN** an ablation whose interaction is significant but smaller than the registered minimum
- **WHEN** the reading is assigned
- **THEN** it SHALL be reported as inconclusive at the panel's sensitivity, with the observed size and
  the minimum both stated
- **AND** it SHALL NOT be reported as the feature carrying the effect

#### Scenario: A removal that moves the frozen substrate is qualified

- **GIVEN** an ablation whose frozen floors differ from the baseline's frozen floors, or whose
  learning arms' gains over those floors are smaller than the baseline's on every level of the
  structure contrast
- **WHEN** the ablation reads as carrying the effect
- **THEN** the reading SHALL state that the substrate's operating point or learnability moved and
  SHALL be reported as carrying-or-saturating, or carrying-or-unlearnable, rather than as carrying
- **AND** the floor comparison SHALL be registered before the campaign runs

#### Scenario: Ablations are read separately

- **GIVEN** more than one ablation of the same established effect in one campaign
- **WHEN** their readings are assigned
- **THEN** each SHALL be read on its own, and a difference between them SHALL be reported as the
  finding rather than averaged into one reading

### Requirement: A committed baseline is reused only under a parsed-field identity check

Where a campaign reuses committed runs from an earlier campaign as one cell of a contrast, the record
SHALL establish that a run produced now reproduces a committed run on every field the analysis reads,
and SHALL re-run the baseline in full where any field differs. *(Renamed 2026-09-19 from "a
byte-identity check": the check compares every **parsed field**, at one seed per reused arm, which is
not byte equality of logs, exports, weights or configuration. The obligation is unchanged; the name now
says what it verifies.)*

#### Scenario: Reuse is licensed by a re-run, not by argument

- **GIVEN** committed runs proposed as a baseline for a new contrast
- **WHEN** anything in the execution path has changed since they ran — code, flags, environment
- **THEN** at least one seed per reused arm SHALL be re-run under the new path and compared to the
  committed log on every parsed field
- **AND** the record SHALL state the comparison's result as the evidence for reuse

#### Scenario: There is no partial reuse

- **GIVEN** a parsed-field identity check in which any field differs for any reused arm
- **WHEN** the campaign is planned
- **THEN** the whole baseline SHALL be re-run under the new path
- **AND** no committed run SHALL be mixed with re-run ones in the same contrast

### Requirement: A positive result carrying an inherited learner setting is re-read at a calibrated operating point before a synthesis cites it

Where a positive structure result was obtained under a learner setting inherited from a different
capacity or substrate rather than calibrated for the arms it ran on, and a later campaign shows the
setting moves the outcome, the record SHALL re-read the result's **registered primary** at the
calibrated setting before the result enters a phase synthesis, and SHALL carry the outcome as a
condition beside the committed verdict rather than as a rewrite of it.

#### Scenario: The re-read is the registered primary at one setting, not a new question

- **GIVEN** a committed positive whose primary was a crossed interaction
- **WHEN** it is re-read at the calibrated setting
- **THEN** the primary SHALL be the same interaction at that setting, with the cells already measured
  there reused only under the committed baseline-reuse requirement — one seed per reused arm re-run on
  the current path and compared on **every field the analysis parses** — and only the missing cells run
- **AND** the record SHALL state what that check establishes and what it does not: it establishes that
  the current path reproduces the committed run on every quantity any analysis in the programme reads,
  at the seed checked; it is **parsed-field identity, not byte equality** of logs, exports, weights or
  configuration, and it does not extend to the seeds left unchecked
- **AND** where a parsed field cannot be compared because the committed side no longer holds the
  artefact it derives from, that field SHALL be named as uncompared rather than counted as matching
- **AND** a minimum effect SHALL be registered as a fraction of the committed effect, with the
  reading that a significant result below it receives named before the runs, **and registered for
  both directions** where the reading is two-sided

#### Scenario: A headline form that is already known not to hold is stated before the runs

- **GIVEN** a committed positive with a registered primary and a more striking form it happened to
  take (a sign flip, a lead at one level)
- **WHEN** committed data already shows that form absent at the calibrated setting
- **THEN** the design SHALL say so before the runs, and SHALL read the registered primary
- **AND** a positive SHALL be reported as the weaker claim it is, never as the headline form
  reproduced

#### Scenario: The verdict is conditioned, not rewritten

- **GIVEN** any reading of the re-read
- **WHEN** the record is written
- **THEN** the committed verdict SHALL stand as read at its own setting
- **AND** the tracker, the original logbook and the roadmap SHALL each carry the condition as a dated
  note in the same place the verdict is cited
- **AND** the synthesis SHALL state the condition in the same sentence as the claim

#### Scenario: The re-read does not decide the setting and does not ablate against it

- **GIVEN** a re-read showing both arms learn better at the calibrated setting
- **WHEN** the record draws consequences
- **THEN** it SHALL NOT declare either setting correct, and SHALL leave the choice to the calibration
  of the rung that next runs there
- **AND** no ablation SHALL be read against the calibrated baseline inside the same campaign that
  establishes it

### Requirement: A shipped result with an uncontrolled confound carries it as a standing condition

Where a result ships with a confound its own record registered but did not control, the confound SHALL
be carried as a **standing condition** stated in the same sentence as the claim at every citation
site, and the control SHALL be named in the successor phase's opening scope rather than left as an
open caveat.

#### Scenario: The condition travels with the claim

- **GIVEN** a shipped result whose record names an uncontrolled confound
- **WHEN** the result is cited in a synthesis, a tracker, a roadmap or a later record
- **THEN** the condition SHALL appear in the same sentence as the claim, not in a separate caveat
  section
- **AND** the citation SHALL NOT state the effect size without it

#### Scenario: An external precedent for the confound raises its standing

- **GIVEN** published work that applies the missing control to the same kind of claim
- **WHEN** the close assesses the confound
- **THEN** that work SHALL be named, and the confound SHALL be treated as load-bearing rather than
  residual
- **AND** the control SHALL be scheduled ahead of further results that would inherit the confound

#### Scenario: A control needing a design decision opens a phase rather than closing one

- **GIVEN** a missing control whose specification is itself ambiguous
- **WHEN** the close schedules it
- **THEN** the ambiguity SHALL be stated as the reason it is the successor phase's work
- **AND** the close SHALL NOT report the control as a small remaining task

### Requirement: A claim that two arms share an initialisation is verified by test

Where a contrast's credibility rests on two arms starting from the same initialisation — the same drawn values, the same per-unit scale, the same policy — the record SHALL identify the quantity claimed to be shared and SHALL establish it by a test that compares the constructed arms, not by an argument about how the random stream is consumed. Where a quantity is claimed to be shared and is not asserted by such a test, the contrast SHALL be reported as matched by construction only, and the claim SHALL name what was not checked.

#### Scenario: The shared quantity is asserted against constructed arms

- **GIVEN** an initialisation-sharing mode whose purpose is to make two arms comparable
- **WHEN** the mode is added
- **THEN** a test SHALL construct both arms at one seed and assert the shared quantity directly — the identical value on every element present in both, or the identical multiset per unit, as the mode claims
- **AND** the test SHALL assert bitwise identity on **both axes the mode could disturb**: across the two arms at one mode, and across modes for one arm on every parameter the mode does not claim to change. Naming only one axis leaves the other free to move unnoticed

#### Scenario: A construction argument is not evidence of sharing

- **GIVEN** a claim that two arms share an initialisation because they consume the same random stream
- **WHEN** the claim is recorded
- **THEN** it SHALL be treated as a hypothesis until a test compares the constructed arms, since a shared stream establishes only that the same values were drawn and not where they landed
- **AND** a contrast relying on the unverified claim SHALL state that its matching is by construction

#### Scenario: More than one definition of sharing exists

- **GIVEN** a structure contrast in which no single definition of "the same initialisation" is uniquely correct, because the manipulation changes which elements exist
- **WHEN** the control is designed
- **THEN** each defensible definition SHALL be run as its own arm and named in the record
- **AND** a result SHALL NOT be reported as "under shared initialisation" without naming which definition produced it

### Requirement: A swept level is shown to reach the learner it is set on

Where a campaign varies a pinned setting in order to report how a result depends on it, the record SHALL establish that each level of that setting is read by the learner the arm runs, and SHALL NOT treat an accepted configuration value as evidence that it took effect. A setting that is declared, validated and then never read produces an arm that is indistinguishable from a swept one in its configuration, in its logs and in its score, and a sweep is the one design whose entire output is a claim about settings.

#### Scenario: The level is asserted to reach the learner

- **GIVEN** a sweep arm that differs from its parent in one setting
- **WHEN** the arm is registered
- **THEN** a test SHALL establish that the setting changes what the configured learner computes or updates, at the level the arm sets
- **AND** where the setting is declared on a shared configuration but read only by some learners, the record SHALL name which learners read it and SHALL confine the sweep to those

#### Scenario: A setting accepted by configuration is not evidence of a manipulation

- **GIVEN** a setting that passes its declared bounds and any load-time validation
- **WHEN** the arm runs under a learner that does not consume it
- **THEN** the arm SHALL NOT be reported as a level of that setting
- **AND** a surface SHALL NOT include a level whose only evidence of taking effect is that the configuration accepted it

#### Scenario: A key silently dropped is named rather than inferred

- **GIVEN** a configuration mechanism that discards unrecognised keys with a warning rather than an error
- **WHEN** a sweep templates one setting across more than one architecture
- **THEN** the record SHALL state, per architecture, whether the setting is declared on that architecture
- **AND** an architecture on which it is not declared SHALL be excluded from the sweep rather than run and reported as swept

### Requirement: A pin is chosen on the learner's own gate, never on the contrast it will carry

Where a pilot sweeps a setting in order to fix the value a later contrast runs at, the record SHALL
register the selection rule before the pilot runs, and that rule SHALL depend only on whether the
learner learns at each level — every arm of the later contrast against its own frozen floor, and
whether the level sits above the instrument's saturation ceiling — and on a stated default. It
SHALL NOT depend on the contrast the later campaign exists to read. A value picked because it shows
the contrast best has built the answer into the operating point, and fresh seeds downstream do not
undo that: they re-measure the contrast at a point chosen for its size.

#### Scenario: The rule is fixed before the pilot and names its default

- **GIVEN** a pilot that fixes a later contrast's setting
- **WHEN** it is registered
- **THEN** its launch record SHALL state the selection rule, the default value, and the tie-break,
  before any pilot seed runs
- **AND** the rule SHALL name, for the case where no level passes, what the later campaign does

#### Scenario: A level where the contrast could not be read is not chosen

- **GIVEN** a pilot level where one arm of the later contrast fails its floor, or both arms saturate
- **WHEN** the value is selected
- **THEN** that level SHALL NOT be chosen
- **AND** where no level passes, the later campaign SHALL NOT run the contrast at a failed level; the
  arm SHALL close with a status, naming the gate that failed

#### Scenario: The contrast is recorded but not consulted

- **GIVEN** a pilot that runs both arms of the later contrast at every level
- **WHEN** its value is selected
- **THEN** the contrast at each level SHALL be recorded descriptively, with its interval
- **AND** the selection SHALL NOT read it, and the record SHALL say so beside the chosen value

#### Scenario: A contrast that moves across the pilot is carried, not resolved

- **GIVEN** a pilot level whose contrast interval excludes zero on the side opposite to the default
  level's
- **WHEN** the record draws consequences
- **THEN** the movement SHALL be carried as a registered condition of the later contrast
- **AND** it SHALL NOT be reported as a finding at pilot scale
