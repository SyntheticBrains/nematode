## Overview

Phase 6 (Connectome Substrate & Architecture Comparison) is a four-layer build (L0 connectome substrate → L1 architecture-plugin → L2 PPO weight search → L3 NEAT topology search) plus continuous-physics + Rung 2 chemical gradients + corrected ASH/ADL nociception + ≥1 real-worm validation, with three mid-phase decision gates. The roadmap (`docs/roadmap.md` § Phase 6, approximately lines 569-740) is the canonical strategic document; this tracking change is its sub-task working artefact. Phase 6 spans ~6-10 months and many AI sessions. This design records the cross-session decisions whose re-litigation would consume disproportionate session budget, so each subsequent milestone change can pick up the framing without re-deriving it.

The decisions below were refined post-roadmap with the help of a parallel L0-planning session that surfaced four scope-level pushbacks against the roadmap's initial framing — captured here as Decisions 1, 3, 4, and 7. A subsequent self-critique iteration tightened the gate criteria (quantitative thresholds rather than prose), split the original T5 into two tranches (continuous-2D + parity vs. Rung 2 gradients) for cleaner gate placement, and tightened Decision 7's connection-type taxonomy.

## Goals / Non-Goals

**Goals:**

- A future AI session can answer "what's the next Phase 6 tranche?" by reading two files (this `tasks.md` and the roadmap Phase 6 block).
- Each Phase 6 milestone PR has a single canonical place to mark progress (this `tasks.md`).
- Seven Phase 6 design decisions (tranching with explicit ordering, connectome data source, L1 plugin parity as real refactor work, tightened MUST architecture-family scope, fixed behavioural scope, mid-phase gate discipline with quantitative thresholds, L2 connectome-substrate semantics including connection-type taxonomy) are recorded once, not re-derived per session.
- The three mid-phase gate decisions are indexed from this change so a reader can see, at a glance, where each go/no-go landed in writing.
- The tracking artefact decays gracefully: when the Phase 6 synthesis logbook publishes, this change archives alongside it.

**Non-Goals:**

- Real-time progress dashboards (the roadmap status table + this tracker's status headers are enough).
- Automated milestone status (humans/agents update the checklist manually as part of milestone PRs).
- Replacing per-tranche / per-milestone OpenSpec changes — those still happen; this scaffold is *additional* coordination.
- Inventing the Phase 6 strategy — the strategy lives in the roadmap. This change records the *tracking discipline* + the sub-roadmap-level scope decisions around it.
- Pre-deciding implementation details that belong inside per-tranche changes (see § What This Change Explicitly Does Not Decide below).

## Design Decisions

> **Mid-phase checkpoint amendments (2026-06-04, post-T4 / Logbook 025).** After the L2 first pass closed ([Logbook 025](../../../docs/experiments/logbooks/025-weight-search-architecture-ranking.md)), a roadmap checkpoint re-weighed the remaining tranches (T5–T9) against the T4 evidence and a fresh scan of 2023–2026 public research. Five pre-fire amendments landed (Gates 2 and 3 had not fired):
>
> - **Decision 4** — CfC (liquid / closed-form continuous-time) promoted SHOULD→MUST (co-top T4 performer; the NCP/LTC/CfC lineage is C. elegans-derived — Lechner et al. *Nat. MI* 2020, Hasani et al. *Nat. MI* 2022). 5 MUST families now.
> - **Decision 7** — peptidergic/extra-synaptic deferral reframed from "negligible" to an *explicit primary limitation* (Ripoll-Sánchez et al. 2023 *Neuron*; Dag et al. 2025 *PRX Life* shows the C. elegans functional signalling network diverges from the anatomical one); the learnable-gap-junction ablation elevated from footnote to a planned T8 topology variant (gap junctions are plastic — Bhattacharya & Hobert 2019 *Cell*; Choi et al. 2020 *Nat. Commun.*).
> - **T6 (env fidelity)** — dynamic-diffusion PDE descoped to a stretch goal; signal-type-specific *static* Fick gradients retained; effort reallocated to an adaptive-threshold/biphasic sensor (the field standard — Kato et al. 2014 *Neuron*, Levy & Bargmann 2020 *Neuron* — log-concentration is an under-powered special case). No published C. elegans chemotaxis model uses a live ∂C/∂t solve.
> - **T7 (L2 re-run)** — intent re-scoped: per Logbook 025's reactive-regime limitation, the substrate upgrade is not expected to change the ranking (RQ5 delta likely ≈ null, still a finding); validation leads with behavioural chemotaxis metrics (Pierce-Shimomura 1999; Iino & Yoshida 2009), not Ca²⁺ correlation matrices; the GA cell (0% at T4) is repair-or-drop.
> - **T8 (NEAT)** — degree-preserving rewired-null controls + matched init/budget added (Dhiman 2026 — connectome topological advantage does not survive proper controls); TensorNEAT confirmed as the tool (GECCO 2024 Best Paper) with a pinned-commit caveat.
>
> Framing anchors added for T9: Lappalainen et al. 2024 *Nature* (the connectome-constrained-network precedent) and Beiran & Litwin-Kumar 2025 *Nat. Neurosci.* (connectome under-constrains dynamics → report solution degeneracy across seeds).

### Decision 1: Tranching policy — Phase 6 has nine tranches with deliberate ordering, not a monolithic L0 → L1 → L2 → L3 block

Phase 6 is decomposed into nine tranches. The ordering matters: corrected ASH/ADL nociception precedes the L2 first pass; the env-upgrade work is split across two tranches (T5 platform refactor + T6 env fidelity) so each has its own scope and so Gate 2 closes cleanly against the platform-refactor work; an L2 re-run on the fully-upgraded substrate (T7) follows so the env-upgrade delta is itself a finding.

| Tranche | Scope | Roadmap layer | Approx duration | Gate trigger |
|---|---|---|---|---|
| 1 | L0 connectome ingest — Cook 2019 via OpenWorm `cect`, vendored data, cross-validated against Witvliet 2021, smoke-test forward pass, no env wiring | L0 | 2-3 weeks | — |
| 2 | L1 plugin refactor + connectome-as-brain wired through existing grid env | L1 | 3-5 weeks | **Gate 1** — basic PPO-on-connectome trainable on existing grid |
| 3 | Corrected ASH/ADL contact-based nociception (owed correctness work per Logbook 011) | env-correctness | 1-2 weeks | — |
| 4 | L2 initial pass — MUST architectures × 3 behaviours, grid-world substrate, corrected nociception | L2 (first pass) | 4-6 weeks | — |
| 5 | Platform refactor — continuous-2D coordinates + continuous-action heads on existing MUST brains; plugin-parity verification | env-upgrade (platform) | 3-4 weeks | **Gate 2** — L1 plugin parity primary checks: ≤ 6 files touched + no per-architecture branches when adding a new architecture family during this work; engineer-hours documented but not load-bearing |
| 6 | Env fidelity — signal-type-specific *static* Fick gradients + an adaptive-threshold/biphasic chemosensory sensor (dynamic-diffusion PDE descoped to a stretch goal 2026-06-04) | env-upgrade (fidelity) | 3-4 weeks | — |
| 7 | L2 re-run on fully-upgraded substrate; real-worm validation (behavioural-chemotaxis-first); SHOULD/MAY architectures evaluated opportunistically | L2 (final) | 4-6 weeks | **Gate 3** — L2 results across MUST set in hand |
| 8 | L3 NEAT topology search on upgraded substrate, incl. degree-preserving rewired-null controls | L3 | 6-10 weeks | — |
| 9 | Phase 6 synthesis logbook | — | 1-2 weeks | — |

Total: 27-42 weeks — within the roadmap's 6-10-month aspirational range at the lower end, with realistic slack. The original eight-tranche framing bundled T5 + T6 together; splitting them out matches the Phase 5 precedent of iterating substrate-refactor work in separate milestones (M4 → M4.5 → M4.6; M6 → M6.9+ → M6.13) rather than landing a "this absorbs scope creep" mega-tranche.

**Why this ordering** (load-bearing rationale; deviating requires amending this change):

- **L0 before L1**: the L0 Risk-mitigation pivot ("if c302 takes > 2 months, drop to hand-curated subset") needs Tranche 1's evidence in isolation. Bundling L0 with L1 buries the substrate-import diagnosis under a plugin-design diagnosis.
- **Corrected ASH/ADL (T3) before L2 first pass (T4)**: predator evasion is one of three Phase 6 behaviours; the corrected nociception is owed-correctness work per Logbook 011. Doing it after L2 first pass means rerunning every predator-evasion L2 cell.
- **L2 first pass (T4) before any env upgrade (T5 + T6)**: T4 produces a publishable intermediate result — connectome on the existing grid substrate, directly comparable to Phase 5's grid-world baseline. Without T4, the first L2 result has two confounded variables (new substrate + new env).
- **Platform refactor (T5) before env fidelity (T6)**: T5 is the platform-level refactor (continuous coordinates, continuous-action heads on every MUST brain) — clean refactor scope with a verifiable parity-test outcome, cleanly anchored to Gate 2. T6 is env-fidelity work (Rung 2 diffusion + log-concentration adaptation) on top of the new continuous substrate; bundling them buries the parity outcome under fidelity-tuning noise. The split also lets T5 ship even if T6 needs extra iteration (the Phase 5 M4-M4.6 pattern).
- **Env upgrades (T5 + T6) between L2 first pass (T4) and L2 re-run (T7)**: this is the load-bearing ordering choice. The env-upgrade delta (continuous-2D + continuous-action + Rung 2, T4→T7) becomes its own finding — "how much does the substrate upgrade change the architecture ranking?". Combining T4 and T7 into one L2 pass loses that delta entirely.
- **Continuous-action heads in T5, not T4**: every existing PPO-family brain is discrete-action (4-action `DEFAULT_ACTIONS` per `packages/quantum-nematode/quantumnematode/brain/actions.py:8-31`). Gaussian-policy heads are a meaningful refactor; they live with the continuous-2D physics work in T5, not gating T4. T4 keeps the discrete 4-action substrate.
- **Real-worm validation in T7, not standalone**: the validation needs defensible behavioural numbers from the fully-upgraded substrate before comparing to Bargmann chemotaxis indices or escape latencies. Running it earlier (against the grid baseline or against the partially-upgraded T5 substrate) compares against a known-low-fidelity intermediate.
- **L3 (T8) after T7**: NEAT topology search on the upgraded substrate uses the L2-final results as its baseline. Running T8 against the grid substrate would invalidate its conclusions the moment T5/T6/T7 ship.

**Alternative considered.** A single mega-change `add-phase6` covering all of L0+L1+L2+L3 (the roadmap's implicit framing). Rejected because it would be impossible to review, would defeat the per-gate decision discipline that Decision 6 below relies on, and would collapse the deliberate T4-vs-T7 env-delta finding into a single confounded L2 sweep.

**Alternative considered (and rejected once already).** An eight-tranche framing in which T5 bundled continuous-2D + continuous-action heads + Rung 2 gradients + log-concentration adaptation + plugin-parity verification into one 6-8-week tranche. Rejected on iteration after self-critique flagged it as the most likely scope-creep target (six substrate changes in one tranche; explicitly self-described as "absorbs scope creep risk"). The current split keeps T5 narrowly scoped to platform refactor + parity (3-4 weeks, single verifiable outcome) and T6 narrowly scoped to env fidelity (3-4 weeks, can iterate if Rung 2 adaptation kinetics need M4-style refinement).

### Decision 2: Connectome data source — Cook 2019 hermaphrodite via OpenWorm `cect` is L0 primary; c302/NeuroML is deferred to an export path

The L0 connectome import has two plausible primary sources:

- **Option A (chosen): OpenWorm `cect` / ConnectomeToolbox.** Python-native; ships Cook et al. 2019 hermaphrodite + Witvliet 2021 + multiple other datasets behind one API; widely cited; format is already-parsed adjacency/connectivity matrices, not raw NeuroML XML.
- **Option B (rejected as primary): NeuroML 2 / c302.** The OpenWorm c302 pipeline lives upstream of `cect` and is the canonical format for Sibernetic body-physics interop. Useful as an *export* target if/when Phase 6 needs to hand connectomes to Sibernetic for behavioural-fidelity validation — but using it as the primary *import* path means Phase 6 absorbs the NeuroML parsing + c302 metadata + element-tree schema complexity before it can validate a single neuron count.

**Why this matters now and not inside `add-connectome-substrate`.** The Risk-mitigation row "L0 c302 import takes > 2 months → drop to hand-curated subset" assumes c302 is the import path. With Option A as primary, the failure mode tightens to "Cook 2019 via `cect` doesn't expose the metadata we need" — a much narrower diagnosis, and one where the hand-curated-subset pivot becomes a deliberate substrate-engineering decision rather than a c302-format escape valve. Recording the choice here means `add-connectome-substrate` can frame its scope around Option A from the start; it does not have to argue for it.

**Cross-validation strategy**: Witvliet 2021 nerve-ring subset against Cook 2019 hermaphrodite (~50 lines of pandas — cheap; ships inside Tranche 1).

**Export path remains in scope.** If a later Phase 6 sub-task needs to hand a topology to OpenWorm Sibernetic for body-physics validation, NeuroML/c302 returns as the export format. That's a different concern from import-time data quality and lives in the Future Directions arc.

### Decision 3: L1 plugin parity is real refactor work, not "we already have a registry"

The codebase today has a `setup_brain_model()` dispatcher at `packages/quantum-nematode/quantumnematode/utils/brain_factory.py` (459 lines; 19 elif branches over the `BrainType` enum, with per-architecture config plumbing inline for each) plus a `Brain` Protocol at `packages/quantum-nematode/quantumnematode/brain/arch/_brain.py:346-368` (with surface `run_brain` / `update_memory` / `prepare_episode` / `post_process_episode` / `copy`). So "swap in another brain" works today *if* the new brain is one of the existing 19. The roadmap's "≤ 1 week to add a new architecture" plugin-parity test is NOT met by the current dispatcher style — adding a new architecture today touches the enum, the dispatcher, the per-family config class, and the YAML loader at minimum, plus tests and a config example.

L1's job is twofold:

1. **Refactor dispatcher → registry pattern.** Decorator-registration or entry-points style; the exact pattern is the L1 tranche's call (see § What This Change Explicitly Does Not Decide below).
2. **Factor *topology* out from *learning rule*.** Today every `BrainType` entry is a fused (topology + rule) bundle (e.g. `LSTMPPOBrain` = LSTM topology + PPO rule). L0 connectome data needs to be a *topology* that PPO, spiking, and NEAT-evolved-topology+PPO can all consume — so L1 must factor topology away from the learning-rule bundle. This is the abstraction the connectome substrate breaks.

**What L1 does NOT change.** The Brain Protocol surface (`run_brain` / `update_memory` / `prepare_episode` / `post_process_episode` / `copy`) is the right plugin contract and does not need to change. L1 is *registry + topology/rule factoring*, not Protocol redesign.

**Migration regression bar (T2 commits to this).** Phase 5 M1's PredatorBrain refactor shipped with 23 byte-equivalence tests + 80/80 metric-cell deltas at exactly 0.0 against the regression-baseline campaign. T2's migration of the existing 19 architectures behind the new registry must meet a comparable bar: a documented regression that every existing brain's training curve is byte-equivalent (or within seeded-RNG noise tolerance documented up-front) on at least one smoke config per architecture, both pre- and post-refactor. The plugin-parity test (Gate 2) verifies forward-looking parity; the migration regression verifies backward-looking parity.

**Why this matters as a tracker-level decision.** A future session reading "L1 is the architecture-plugin interface" and looking at the existing `Brain` Protocol could reasonably conclude L1 is already done. Recording explicitly that L1 = registry refactor + topology/rule factoring + 19-architecture migration with regression bar prevents that misread and prevents L1's OpenSpec change from trying to "just add a connectome subclass" against the existing dispatcher.

### Decision 4: Architecture-family MUST set tightened from eight to four

The roadmap's initial MUST set is eight families (connectome-constrained, MLP-PPO, LSTM/GRU-PPO, spiking, reservoir, quantum, hybrid, NEAT-evolved). With the parallel L0-planning session we tightened this to four MUSTs + two SHOULDs + three MAYs:

| Family | Existing impl | Verdict | Rationale |
|---|---|---|---|
| Connectome-constrained | Not yet (Tranches 1-2) | **MUST** | Headline. The whole point of Phase 6. |
| MLP-PPO | `MLPPPOBrain` | **MUST** | Strongest classical baseline; cheapest run; sanity anchor. |
| LSTM/GRU-PPO | `LSTMPPOBrain` | **MUST** | Strongest temporal baseline (Phase 3 reached 94% L500); matched-capacity comparator for connectome (both have recurrent state). |
| NEAT-evolved | Not yet (Tranche 8) | **MUST** | L3's whole point — answers "is the connectome a local optimum?" Can't drop. |
| CfC (liquid / closed-form continuous-time) | `CfCPPOBrain` (added at T4) | **MUST** *(promoted 2026-06-04)* | Co-top T4 performer (84.4%, statistically tied for #1). The NCP/LTC/CfC lineage (Lechner et al. *Nat. MI* 2020; Hasani et al. *Nat. MI* 2022) is *literally derived from the C. elegans connectome* — the most worm-relevant non-connectome comparator in the field. Promoted from the Phase-4.5 opportunistic set so it is a first-class row in the T7 upgraded-substrate pass and the T8 topology comparison. |
| Quantum | `QVarCircuitBrain` et al. | **SHOULD** | Phase 2's 300-session campaign carries forward as baseline reference per the Architecture-Comparison Protocol. RQ4 settled negative at T4 under controlled attribution (Logbook 025); one quantum row at continuous-physics complexity at T7 tests only whether higher complexity crosses the quantum-advantage threshold. |
| Spiking | `SpikingReinforceBrain` | **SHOULD** | Bridge to Phase 7 L4 STDP. But Phase 0's 73.3% on much easier tasks isn't a strong precedent; demoted so Phase 6 doesn't gate on spiking-on-connectome training. If it doesn't train cleanly, document and move on. Phase 7 L4 (STDP — spiking's native plasticity rule) is where spiking-on-connectome actually belongs. |
| Reservoir (`QRH`, `CRH`) | Yes | **MAY** | Phase 2 preserved QRH's +9.4pp pursuit advantage at low absolute performance. One row if cheap; not worth blocking on. |
| Hybrid quantum-classical | `HybridQuantum`, `HybridClassical` | **MAY** | Phase 2 SOTA finding survives as baseline reference. One row if cheap. |
| Transformer | Not yet | **MAY** | Unchanged from roadmap. |

**Budget impact.** *Original (per-behaviour) framing:* MUST × 3 behaviours × 4 seeds = 48 runs per L2 pass. *Realised framing (post-T4):* T4 collapsed to one integrated-C3 cell per family (n=8) with per-behaviour sub-metrics extracted, so the per-L2-pass budget is one integrated-C3 cell per MUST family. **With CfC promoted (2026-06-04), the MUST set is 5 families → 5 integrated-C3 cells per L2 pass (T4 + T7).** CfC already ran at T4 (its T4 cell stands; no re-run needed there); the promotion adds a CfC cell to the T7 upgraded-substrate pass and a CfC row to the T8 topology comparison. SHOULD/MAY architectures remain opportunistic in Tranche 7.

**Why this matters as a tracker-level decision.** It's tempting to add a ninth architecture family mid-Phase-6 because some new variant looks interesting — Phase 0-3 added 19 architectures total under exactly this pressure. Phase 6's value proposition is the *comparison*, which requires that the set of compared rows is stable across the sweep. Expansion mid-phase invalidates already-completed L2/L3 results for the rows that ran first.

**Mechanism.** A new family proposed mid-Phase-6 must amend *this* tracking change (a follow-up commit to `proposal.md` + `design.md` + `tasks.md`), not be added inside an individual milestone change. Promoting a SHOULD/MAY family to MUST follows the same mechanism (spec Requirement 4 below). The amend forces the project to look at the cross-family budget impact (each new MUST family is +3 behaviours × 4 seeds × 2 L2 passes = +24 runs, before NEAT topology search) before the family is added.

### Decision 5: Behavioural scope is fixed at three (klinotaxis, thermotaxis, predator evasion)

The same anti-scope-creep logic as Decision 4, applied on the behaviour axis. Phase 6 commits to three behaviours; aerotaxis / pheromones / multi-agent dynamics are explicitly *deferred*.

**Why this matters as a tracker-level decision.** Each behaviour added is +1 axis on the L2 sweep, +1 axis on the L3 sweep, and +1 set of training configs and analysis scripts per architecture family. A fourth behaviour added mid-phase pushes Phase 6 past 10 months without strengthening the headline platform claim.

**Mechanism.** Same as Decision 4 — a new behaviour must amend this tracking change. The amend forces the cross-tranche budget impact to be looked at before the behaviour is added to any milestone change.

### Decision 6: Mid-phase gate discipline — every gate produces a written go/no-go decision in the relevant tranche's logbook, with quantitative pass criteria pre-registered here

The roadmap defines three mid-phase decision gates (see `docs/roadmap.md` § Phase 6 § Mid-phase decision gates), mapped to tranche boundaries per Decision 1. Each gate has explicit numerical pass criteria, in the spirit of Phase 5's M2/M3/M4 quantitative gates (which were the reason those milestones' STOP verdicts were defensible).

#### Amendment mechanism (critical — read before reading the criteria below)

> **Normative requirements lives in spec.md.** The enforceable rules for this Amendment mechanism (definition of "the gate fires", pre-fire vs post-fire amendment discipline, separate-amendment-commit requirement, prohibition on post-fire recalibration, logbook-recording rule) are codified in [`specs/phase6-tracking/spec.md` § Requirement 3](../specs/phase6-tracking/spec.md) under scenarios titled to map back to this "Amendment mechanism" section. This section is rationale only — read it to understand *why* the rules exist; read the spec to learn *what* the rules are.

Some of the criteria below were necessarily set without empirical calibration — particularly G1.c's quantitative threshold (no prior PPO-on-connectome data exists) and G2's parity-test wall-clock + tolerance numbers (no baseline measurement against the current dispatcher has been taken). Mid-Phase-6 recalibration is explicitly allowed, under the discipline summarised below:

- **What "the gate fires" means** (the operative definition for pre-fire vs post-fire): the gate fires the moment the triggering tranche's logbook PR is *opened* with the gate-decision section drafted. Anything *before* that — including a separate prior PR that lands a Decision 6 amendment a day earlier — is pre-fire. Anything *after* — including same-PR diffs that try to amend Decision 6 alongside the gate-decision draft — is post-fire. This definition rules out the loophole where one commit silently bundles a criterion amendment with the verdict that depends on it.
- Pre-fire recalibration happens via a separate commit to this design.md before the triggering tranche's logbook PR opens with the gate-decision section drafted. The commit names the criterion being recalibrated, the in-flight evidence motivating the change, and (where applicable) the alternative criterion that's replacing it. The intent is to keep amendments visible and separate from the verdicts they affect, rather than bundled into the same diff.
- Post-fire recalibration (gate was failing, threshold lowered to pass it) is treated as goalpost-moving. The spec scenario at [`specs/phase6-tracking/spec.md` § Requirement 3](../specs/phase6-tracking/spec.md) (gate decision lacks quantitative criterion evaluation) treats the gate as not-yet-decided in that case, blocking the next tranche's PR until the original criterion's verdict is recorded.
- When a gate decision evaluates against a pre-fire amended criterion, the triggering tranche's logbook records both the original criterion and the amended one — so a reader can see what changed and why.

This is the difference between "we calibrated G1.c after T1 showed the random-policy baseline was wrong" (the allowed pre-fire case; commit lands before T2 closes) and "G1.c was about to fail so we lowered the threshold" (the post-fire case that the spec blocks; gate stays open). The discipline avoids the failure mode where rational up-front weakening (setting every threshold deliberately low to avoid amendment friction) substitutes for empirically-grounded calibration.

#### Gate 1 (Tranche 2 close, month ~2-3 cumulative): L0 import working in anger

Pass criteria (all required for GO):

- (G1.a) Connectome substrate loaded via `cect`; neuron count = 302 (or the documented hand-curated subset N); cross-validation against Witvliet 2021 nerve-ring subset shipped per T1.4.
- (G1.b) The L1 plugin registry instantiates both an existing MUST brain (MLP-PPO) and the connectome-constrained brain through the same code path; no per-brain dispatcher branches in the simulation loop.
- (G1.c) PPO-on-connectome on klinotaxis (smoke config, single seed) trains for ≥ 100 episodes without NaNs or policy collapse, with two paired-control comparisons:
  - **Learning-signal check**: mean episode return over the last 25 episodes ≥ mean episode return from an **architecture-matched frozen-random-weights forward-pass control** (same connectome topology, same random-weights initialisation, no PPO updates, same env, same episode budget, same seed) by a margin of ≥ 10% of the frozen-random control's return. This is the actually-relevant null for "does PPO weight tuning add signal over the same connectome topology without learning". A pure-random-action baseline is *not* the right null on klinotaxis because gradient-bias alone (drift toward attractive fields) clears 1.5× random.
  - **Anti-collapse check**: mean return over the last 25 episodes is strictly greater than mean return over the first 25 episodes (monotonic improvement signal). "No NaNs" doesn't catch policy collapse to a constant action; this does.
- (G1.d) Migration regression per Decision 3 demonstrates byte-equivalence (or seeded-RNG-noise-tolerance documented at T2 start) on one smoke config each for MLP-PPO + LSTM/GRU-PPO (the two MUST brains that have pre-existing Phase 0-5 configs). The connectome-constrained brain is NOT in scope for byte-equivalence — it has no pre-existing baseline; it establishes its own fresh baseline in T2.6 and is evaluated against G1.c above.

PIVOT trigger: G1.a fails on the full 302-neuron import → hand-curated-subset pivot per the roadmap risk row; record the subset choice in the T1 logbook; Gate 1 re-evaluates against the subset.

STOP trigger: G1.c fails after the diagnostic sequence (topology density / reward shaping / RNG seeds) AND the hand-curated subset pivot has also failed G1.c → the finding is "PPO-on-the-real-connectome requires further substrate work" and Phase 6 either scopes a substrate-engineering tranche before retrying or stops.

#### Gate 2 (Tranche 5 close, month ~4-5 cumulative): L1 plugin parity in practice

Pass criteria. G2.b and G2.c are the **primary** load-bearing parity checks (objectively measurable from the post-refactor codebase). G2.a (engineer-hours) is documented but **not load-bearing** — see note below. G2.d is a substrate-floor check.

- (G2.b) **PRIMARY**. Files-touched count for adding a new architecture through the L1 plugin interface is ≤ 6 files (registry registration + brain implementation + config class + config example + smoke test + docs). The original 4-file claim in Decision 3 is the floor; ≤ 6 leaves room for a continuous-action-head adapter file under the T5 platform refactor. Counted by running `git diff --name-only` against the addition commit.
- (G2.c) **PRIMARY**. No per-architecture branches in the simulation loop or training loop after the addition. Code-review verdict, recorded in the T5 logbook.
- (G2.a) **DOCUMENTED, NOT LOAD-BEARING**. During T5, at least one new architecture is added through the L1 plugin interface (revival of a Phase 0-3 family not in the MUST set, e.g. spiking via SpikingReinforceBrain, or a documented hypothetical new architecture). Engineer-hours for the addition are recorded in the T5 logbook for future reference, but the "≤ 5 working days" target is intentionally not load-bearing because no pre-refactor baseline measurement exists — the post-refactor measurement is the baseline for future additions, not a verdict on this one. The roadmap's "≤ 1 week" framing carries forward as informal expectation; if the actual measurement is dramatically higher (e.g. > 3 weeks), that's diagnostic information feeding the G2.b/G2.c review, not a separate gate failure.
- (G2.d) **FLOOR CHECK** (substrate change didn't break the architecture). Continuous-2D + continuous-action heads operational on at least connectome + MLP-PPO; smoke training run on klinotaxis (single seed, same episode budget as T4 baseline) converges to **mean episode return ≥ 50% of T4's per-architecture grid-substrate baseline mean episode return for the same architecture**. The 50% floor is deliberately wide because grid-discrete-action and continuous-2D-continuous-action are not directly comparable even at architectural parity (the action space changes the return scale); this is a "substrate change didn't break training" check, not an apples-to-apples ranking check. The actual upgraded-substrate ranking lands in T7 / G3.

PIVOT trigger: G2.b OR G2.c fails → L1 refactor pivot per the roadmap risk row; T5 amends to add 2-4 weeks of additional L1 refactor work before re-evaluating Gate 2.

STOP trigger: after the L1 refactor pivot, the plugin interface is still fundamentally incompatible with one or more MUST architecture families (most likely the connectome-constrained family if topology/rule factoring doesn't generalise as Decision 3 assumes) → the finding is "the connectome substrate requires a different interface abstraction than the existing 19 architectures share" and Phase 6 scopes a connectome-specific interface, accepting the parity test as not met.

#### Gate 3 (Tranche 7 close, month ~7-8 cumulative): L2 results across architectures, real-worm validation in hand

Pass criteria (all required for GO):

- (G3.a) All 5 MUST architecture integrated-C3 cells in T7 (n ≥ 4 seeds per cell) ship at the Phase 5 statistical bar *(was 4; CfC promoted to MUST 2026-06-04 — see Decision 4)*, with per-behaviour-component sub-metrics (foraging success / predator survival / thermotaxis isotherm-tracking) extracted from each integrated run per the `architecture-comparison-protocol` capability — paired-seed Wilcoxon vs. a documented baseline (MLP-PPO-on-grid serves where no architecture-specific baseline exists), bootstrap 95% CIs, n ≥ 4 seeds per cell. *(Amended 2026-05-30 by the `weight-search-architecture-ranking` change: the original "12 MUST cells (4 families × 3 behaviours)" wording assumed T7 inherits the 12-cell per-behaviour pattern from T4; with T4 collapsed to 4 integrated-C3 cells + extracted sub-metrics, T7 inherits that shape. This widens the cell-SHAPE framing only — the n ≥ 4 seed floor is preserved verbatim and the Phase 5 statistical bar is unchanged.)* **Multiple-comparisons strategy**: **BH-FDR within-pass**, applied separately to T4 and T7 (they test different substrate hypotheses), committed in the `weight-search-architecture-ranking` change's design.md + its `architecture-comparison-protocol` spec before any T4 cell launches. This **resolves** the prior "default Holm-Bonferroni, choice deferred to T4 planning" note: T4 planning selected BH-FDR for higher statistical power given the dependent tests across architectures. The test set per pass is the active comparison set (the integrated-C3 cells + extracted per-behaviour sub-metric tests + ablations), not the old 12 per-behaviour cells. **Note**: neither Holm-Bonferroni nor BH-FDR is Phase 5 precedent — Phase 5 used paired-seed Wilcoxon + bootstrap CIs + n ≥ 4 without family-wise correction; introducing BH-FDR for Phase 6 reflects the larger comparison-cell count. *(Confirmed post-realisation 2026-06-03 by the `weight-search-architecture-ranking` change's Task 6.10: the T4 first pass ([Logbook 025](../../../docs/experiments/logbooks/025-weight-search-architecture-ranking.md)) shipped with BH-FDR across the realised pairwise set (7 families, 21 pairs); G3.a's T7 inheritance of the strategy stands unchanged. This is an additive post-realisation confirmation, not a recalibration — Gate 3 has not fired (T7 is future) and no threshold changed.)*
- (G3.b) Connectome-constrained results land in the ranking — strict-mask cells produce a clear "wins / ties / loses" verdict against each other MUST family on each behaviour. "Failed to train, no result" is a STOP signal, not a tie.
- (G3.c) Env-upgrade delta analysis shipped — head-to-head ranking comparison between T4-grid and T7-upgraded for the integrated-C3 cells (+ extracted per-behaviour sub-metrics) present in both passes (RQ5) *(was "4 cells"; CfC promoted to MUST 2026-06-04 and also ran at T4, so up to 5 cells are comparable across passes — the GA/`neat_weights` cell is conditional per its T7 repair-or-drop decision)*. The delta is the load-bearing reason for the deliberate T4/T7 split.
- (G3.d) Real-worm validation shipped — chosen target (chemotaxis index / escape latency / Ca²⁺ correlation matrix) and chosen architecture's model output compared quantitatively against published real-worm data with reported confidence intervals.

PIVOT-scope trigger: partial coverage — some MUST cells didn't ship (most likely 1-2 cells of the 12) → Phase 6a/6b sub-phase split per the roadmap. The natural cut is "Phase 6a = T1–T7, ship what's done; Phase 6b = T8 NEAT + T9 synthesis, defer until coverage is plugged". This change amends to reflect 6a's exit criteria and 6b's deferred scope.

PIVOT-scope trigger (alternative): Phase 6 overshoots 10 months cumulative wall-clock despite T7 being technically complete → same 6a/6b split for delivery reasons (e.g. start Phase 7 L4 plasticity work without waiting for L3 NEAT).

STOP trigger: G3.a fails — fewer than half the MUST cells reach the statistical bar after the diagnostic sequence (T7 risk-mitigation pivot) → publishable negative result; Phase 7 L4 plasticity inherits the substrate-engineering question.

Each gate is the decision boundary between a tranche and its successor. Each must produce a **written** go/no-go decision in the relevant tranche's published logbook (NOT in `tasks.md`, which moves to `archive/` post-merge and becomes hard to amend) — not a silent continuation, not "the next milestone just started so I guess Gate 1 passed". The `tasks.md` here indexes those decisions by linking to them once they land.

**Why.** Phase 5's M4/M5/M6.x STOP verdicts were valuable specifically because the diagnosis was written down at the gate point — substrate constraint, architecture asymmetry, wrong abstraction. The same discipline applied at Phase 6's mid-phase gates protects against the failure mode where a layer "kind of works" and the project slides into the next layer without the underlying gate being clearly passed or pivoted. Phase 5's gates were quantitative (M2 "≥3pp over hand-tuned baseline AND fitness still rising at gen 20"; M3 "mean_gen_lamarckian_to_092 + 4 ≤ mean_gen_control_to_092"; M4 three-clause AND-gate with `+2 / +0.10 / +4` thresholds); the Phase 6 gates here match that quantitative bar but with the explicit recalibration mechanism above for the criteria that couldn't be empirically grounded up-front.

**Pivot path is part of the gate.** Each gate has a documented pivot (the Risk-mitigation rows + the PIVOT triggers above). A gate that triggers its pivot is a written decision — it produces an amended scope and a new tranche definition. "Gate FAILED → pivot to hand-curated subset" is a successful gate outcome by this design; only an undocumented slide past the gate is a failure of discipline.

### Decision 7: L2 connectome-substrate semantics — strict-mask is the headline, with explicit connection-type taxonomy

What does "PPO-on-connectome" mean concretely, given that LSTMPPO-GRU + klinotaxis + nociception already reaches 94% L500 on pursuit-predators-large per Logbook 009?

- **Today's LSTMPPO baseline** = generic LSTM (a few hundred neurons), fully-connected dense connectivity learned by gradient descent.
- **L2 on connectome substrate** = 302 named neurons wired according to Cook 2019, with PPO tuning weights along the connections defined by the connectome — *but only the connections of the right type*, per the taxonomy below.

#### Connection-type taxonomy

Cook 2019 (and `cect` exposes this distinction) reports three categories of inter-neuron signalling, which Phase 6 treats differently:

| Connection type | Cook 2019 representation | Phase 6 treatment | Rationale |
|---|---|---|---|
| **Chemical synapse** | Directed; weighted by synapse count | **Subject to strict-mask**. PPO tunes a learnable scalar weight per synapse; non-existent (count=0) synapses are pinned to zero. This is what "strict-mask" means in the headline run. | Chemical synapses are where learning lives in real *C. elegans* (Hebbian-style plasticity, neuromodulator gating). PPO weight tuning is the closest L2 analogue; the wild-type adjacency is the topology we're testing. |
| **Gap junction** | Undirected; reported by `cect` with a per-junction count (a stand-in for conductance) | **Fixed-weight, non-learnable; weight = Cook 2019 synapse count (the physiologically-informed signal)**. Gap junctions are passive ohmic couplings physiologically; their conductance is set by physiology, not by learning. They participate in the forward pass as fixed bidirectional couplings with weight equal to the reported count (a junction with count = 0 is non-existent; a junction with high count has higher fixed weight). Unit-weight-1.0 is a documented ablation if needed for diagnostic purposes, not the headline. **Learnable gap junctions are a planned T8 topology variant** *(elevated 2026-06-04, was a Risk-8 contingency only)*: real C. elegans electrical synapses are plastic and behaviourally consequential (Bhattacharya & Hobert 2019 *Cell*; Choi et al. 2020 *Nat. Commun.*), and a contemporary C. elegans locomotion model makes gap-junction weights learnable under anatomical-proportion constraints (Cao et al. 2025, bioRxiv). The learnable-gap-junction variant is a cheap, well-motivated test of one candidate cause of the connectome's predator-evasion gap; running it as a documented variant does NOT change the strict-mask *headline* (which stays chemical-synapses-tuned, gap-junctions-fixed), per Risk 8 below. | Letting PPO tune gap junctions would be biologically wrong (real worms can't learn-tune their gap junctions on a per-episode timescale) AND would partially obviate the strict-mask claim (a "fixed topology" with learnable electrical couplings has a different inductive bias than "fixed topology, learnable chemical weights only"). Synapse-count-as-weight (rather than unit-1.0) is the headline because it's the only physiologically-informed signal `cect` exposes. |
| **Extra-synaptic / peptidergic / monoaminergic** | Not edge-shaped (volume transmission via diffusible neuromodulators; CeNGEN gene-expression data needed to materialise it) | **Out of Phase 6 scope — but an EXPLICIT PRIMARY LIMITATION, not negligible** *(reframed 2026-06-04)*. Not in the L0 data model; not in the L2 architecture; flagged in the T9 synthesis as a known large missing layer that bounds every connectome verdict. | Materialising this layer requires CeNGEN receptor-class metadata + a diffusion model for neuromodulator concentration — both are Phase 7 L4 plasticity work per the roadmap. **However**, the 2023–2025 literature has moved the wireless connectome from "interesting extra" toward "essential for function": the neuropeptidergic connectome is dense and links neurons isolated in the wired graph (Ripoll-Sánchez et al. 2023 *Neuron*), the worm is a multiplex network (Bentley et al. 2016 *PLOS Comput. Biol.*), and the *functional* signalling network empirically diverges from the anatomical one in this exact organism (Dag et al. 2025 *PRX Life*). The connectome's mid-pack T4 rank (esp. behind on predator evasion) must therefore be interpreted as a property of the *wired-chemical-synapse-only* model — the missing neuromodulatory layer is a candidate cause alongside "no recurrent memory", not a footnote. |

#### Dual chemical + gap-junction connections between the same neuron pair

Many *C. elegans* neuron pairs (e.g. AVA ↔ AVB) are connected by both a chemical synapse AND a gap junction. The L0 data model represents these as **two separate edges with type metadata** (one chemical-synapse edge, one gap-junction edge between the same neuron pair) rather than a single edge with two weight attributes. This keeps the strict-mask claim well-defined: PPO tunes the chemical-synapse edge weight; the gap-junction edge weight is fixed by Decision 7's gap-junction rule above. T1.2 implements this representation; downstream architecture-plugin consumers in T2 iterate edges by type.

#### Strict-mask vs. soft-prior

Two design choices live inside L2 weight handling for chemical synapses; both ship in the same sweep:

- **Strict-mask (default, headline run)**: chemical-synapse connectivity fixed to Cook 2019 adjacency; PPO tunes weights only along existing edges; non-existent edges pinned to zero. Tests whether the wild-type topology *can* support the behaviour.
- **Soft-prior (documented ablation)**: chemical-synapse counts initialise weights; PPO is free to grow new connections during training. Tests whether the topology is a useful *starting point* even if not a constraint.

The headline L2 result uses **strict-mask** because that's the claim the connectome-ranking question wants: "the wild-type wiring supports behaviour X under PPO weight search." The soft-prior runs ship as a documented ablation in the same L2 sweep — they distinguish "the connectome's topology is right" from "the connectome's wiring is a useful prior even if not a constraint." Phase 5's existing LSTMPPO + klinotaxis + nociception result becomes one row of the comparison sweep (not work to redo) — it answers a different question ("how does an unconstrained LSTM compare?").

#### Sensor-projection and motor-readout choices are T4-scope ablations, not separate MUST families

There are real choices to be made about how environmental observations map onto named sensory neurons (ASE / AWA / AWC for chemotaxis, AFD for thermotaxis, ASH/ADL for nociception per T3) and how motor outputs are read off named motor neurons (pooled vs. per-neuron). These are NOT separate MUST architecture-family rows under Decision 4 — they are ablations inside the connectome-constrained row, scoped to T4's design phase. The L2 OpenSpec change documents the choice; if a sensor-projection ablation produces a result interesting enough to merit a dedicated row in T7, that's an amendment to this design (per Decision 4's mechanism), not an organic mid-phase expansion.

**Why this matters as a tracker-level decision.** Without this taxonomy, "PPO on the connectome" is ambiguous between at least four things (strict-mask chemical-only, strict-mask all-edges, soft-prior chemical-only, plus the gap-junction-as-learnable variant). Each is a different claim. Recording strict-mask on chemical synapses with fixed gap junctions and no peptidergic layer as the headline means L2's OpenSpec change doesn't have to re-argue the claim, and Gate 3 evaluates against a single defined benchmark.

## What This Change Explicitly Does Not Decide

To keep this tracking change's authority bounded, the following are deliberately deferred to the relevant per-tranche OpenSpec changes:

- **The L1 registry implementation pattern** (decorator vs entry-points vs config-driven) — the L1 tranche (Tranche 2) makes this call when it scopes the refactor.
- **The exact Cook 2019 sub-file selection** (adjacency XLSX vs synapse-list CSV vs another `cect`-exposed format) — the L0 tranche (Tranche 1) makes this call after looking at what `cect` actually exposes. NOTE: per Decision 7's taxonomy, T1 must expose chemical synapses AND gap junctions as separately-typed connections in the data model; the sub-file choice may be constrained by that requirement.
- **The continuous-action policy parameterisation** (Gaussian vs Beta vs Tanh-squashed Gaussian) — the env-upgrade platform tranche (Tranche 5) makes this call when continuous-action heads are designed.
- **The real-worm validation dataset selection** (Bargmann chemotaxis indices vs mechanosensation escape latencies vs Kavli/Janelia Ca²⁺ correlation matrices) — the L2-final / validation tranche (Tranche 7) makes this call when the upgraded-substrate behavioural numbers are in hand.
- **The compute budget estimate per L2 cell** — meaningfully estimable only once T1+T2 produce one real PPO-on-connectome wall-time. T4 planning records the estimate then; if it diverges from the per-tranche duration in Decision 1's table by > 2×, Decision 1 amends.
- **The per-cell seed count for T4 / T7** — `n ≥ 4` is the Phase 5 inherited floor; T4 planning may raise this to n=8 if the per-cell variance warrants it (Phase 5 M4.5 precedent), recording the rationale per cell.

If any of these decisions becomes load-bearing for the cross-tranche plan (e.g. the L0 sub-file choice forces a particular L1 topology representation), the relevant per-tranche change should record the decision *and* amend this design.md to lift the decision up to the tracker-level.

## Tracking Strategy

Three artefacts answer "where are we in Phase 6?":

1. **`openspec/changes/phase6-tracking/tasks.md`** — sub-task checklist updated by every Phase 6 milestone PR.
2. **`openspec/changes/<phase-6-milestone>/`** — per-tranche / per-milestone proposal/tasks/design/specs, archived on milestone merge.
3. **`docs/roadmap.md` Phase 6 section** — tranche-level status, updated as part of every Phase 6 milestone PR.

A future AI session orients by:

- Reading the roadmap Phase 6 block first (per-tranche current status via the Phase 6 Tranche Tracker sub-section).
- Reading this `tasks.md` for sub-task granularity.
- Reading active `openspec/changes/<milestone>/` if a specific milestone is in flight.
- Reading the latest published `docs/experiments/logbooks/0XX/` if a milestone has completed evaluation — this is where gate decisions live (per Decision 6).

## Maintenance

- Every Phase 6 milestone PR updates `tasks.md` (mark sub-tasks complete) and `docs/roadmap.md` Phase 6 Tranche Tracker (one-line status update).
- This change does not archive until the Phase 6 synthesis logbook (Tranche 9) ships.
- If Phase 6 deviates substantially from this plan (e.g. a tranche is dropped, gate criteria are softened, an architecture family is promoted or added, tranche ordering is changed), update `tasks.md` + `proposal.md` + this `design.md` to reflect reality — the checklist is descriptive, not aspirational. Git history is the audit trail of the change.

## Risks

1. **The checklist drifts from reality if PRs forget to update it.** Mitigation: each per-milestone OpenSpec change's `tasks.md` includes "update phase6-tracking tasks.md" as an explicit sub-task; spec Requirement 1 makes this a requirement.
2. **A mid-phase gate slides by without a written decision.** Mitigation: spec Requirement 3 makes the written gate decision a hard requirement; this `tasks.md` has explicit Gate 1 / Gate 2 / Gate 3 checkboxes that cannot be ticked without a link to the decision logbook; Decision 6 above pre-registers the quantitative pass criteria so "kind of GO" is not available.
3. **The fixed architecture-family scope (Decision 4) gets expanded informally inside individual milestone changes — either adding a ninth family or promoting a SHOULD/MAY to MUST.** Mitigation: spec Requirement 4 makes promotion / addition an amendment-blocking event; reviewers of per-tranche PRs check the proposed architecture set against this `design.md` Decision 4 table.
4. **The deliberate tranche ordering (Decision 1) gets reshuffled informally — most likely combining T4 and T7 into one L2 sweep on the upgraded substrate (losing the env-upgrade delta finding) or re-merging T5 and T6 back into a single env-upgrade tranche.** Mitigation: spec Requirement 5 makes re-ordering an amendment-blocking event with explicit rationale required; this design's Decision 1 "Why this ordering" paragraph is the load-bearing reference.
5. **Tranche 3 (corrected ASH/ADL) gets skipped because predator-evasion appears to work under the existing incorrect model.** Mitigation: Logbook 011 already flagged the existing nociception model as biologically wrong; the tracker's T3 sub-tasks remain visible and unticked, and the L2 first pass (T4) predator-evasion cells declare T3 as a hard dependency.
6. **Tranche 6 (Rung 2 env fidelity) needs M4-style iteration (Rung 2.5, Rung 2.6) and slips schedule.** Mitigation: T5 and T6 are deliberately separate so T5's parity-test outcome (Gate 2) isn't blocked by T6 iteration; if T6 iterates, the env-upgrade delta T7 vs T4 just measures a slightly narrower upgrade scope and the design.md amends to note the deferred fidelity work.
7. **Phase 6 overshoots 10 months and the Phase 6a/6b sub-phase split (roadmap Risk-mitigation row) is needed.** Mitigation: that split is a documented pivot, not a tracker failure. If triggered, this change amends to reflect 6a's exit criteria and 6b's deferred scope; archive happens on 6a's synthesis logbook publication and 6b inherits a fresh tracking change if scope warrants. The natural 6a/6b cut is after T7 (Phase 6a = T1–T7; Phase 6b = T8 NEAT + T9 synthesis).
8. **Decision 7's connection-type taxonomy (chemical-only strict-mask, fixed gap junctions, no peptidergic layer) turns out to be empirically wrong — e.g. the chemical-synapse-only model can't learn, and gap junctions need to participate in the gradient.** Mitigation: this is Gate 1's G1.c probe and the L2 risk-mitigation pivot — diagnostic sequence first (topology density / reward shaping). The right second-line response is to **take the STOP** (Phase 5 discipline: M4 / M5 / M6.x all closed STOP rather than amending their gates to pass). Enabling learnable gap-junction weights is available as a *documented deviation from the headline claim*, not a substitute for it — if invoked, the Phase 6 synthesis must explicitly note that the connectome-strict-mask claim weakened to "chemical synapses + tunable gap junctions" and the connectome-vs-evolved comparison interprets accordingly. The deviation amendment is allowed by Decision 6's recalibration mechanism (it amends Decision 7 before Gate 1 fires, on the basis of T1+T2 evidence), but it is NOT a way to make a failing gate pass after the fact.
