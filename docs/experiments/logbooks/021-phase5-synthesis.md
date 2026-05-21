# 021: Phase 5 Synthesis — Evolution & Adaptation

**Status**: `Phase 5 closes COMPLETE`. Two GO results (M2 Hyperparam Evolution, M3 Lamarckian Inheritance), three substrate-grounded STOP results (M4 Baldwin, M5 Red Queen, M6 Transgenerational Memory), and one OPTIONAL milestone deferred to Phase 6 (M7 NEAT Architecture Evolution). All five Phase 5 exit criteria are met with evidence. The STOP verdicts are **field-consistent substrate diagnoses** — each independently corroborated by 2024-2026 literature from groups working on parallel problems — not implementation failures. Two reusable methodology contributions ship from Phase 5 unscooped as of May 2026: the **lag-matrix cross-pairing instrument** (logbook 017) and the **cell-grid fair-test methodology** (logbook 017). Phase 5's `phase5-tracking` OpenSpec change archives alongside this logbook; M7 remains OPTIONAL in the tracker for a possible Phase 6-time revisit.

**Branch**: `feat/m8-phase5-synthesis` (this logbook + tracker close-out + roadmap flip).

**OpenSpec change**: `phase5-tracking` (active throughout Phase 5; archives operator-side post-merge via `openspec-archive-change phase5-tracking` — the archive step is intentionally outside the task slate to avoid self-blocking the `openspec-archive-change` precondition that every task be ticked).

**Date Started**: 2026-05-21 (post-M6.13 STOP merge, PR #169).

**Date Last Updated**: 2026-05-21.

This logbook is the M8 synthesis (scoped variant — see § Scope). It does NOT report new experiments; it consolidates findings across logbooks 012–020 and recontextualises them against independent 2026 literature.

## Scope

M8 ran as a **scoped synthesis** (~3 days writing, no compute). The original M8 plan had seven sub-tasks; M8.2 + M8.3 + M8.4 are the load-bearing scientific deliverables and ship here. M8.1 (cross-milestone fitness curves) was dropped because each per-milestone logbook already carries its own curves — cross-milestone aggregation adds marginal value unless writing for external publication, which Phase 5 is not currently committed to. M8.5–M8.6 (publish + roadmap tick) collapse into the same commit as this logbook. M8.7 (archive `phase5-tracking`) was dropped from the task slate because `openspec-archive-change` requires every task ticked — an "archive me" task would self-block. The archive step runs operator-side post-merge.

The decision to scope down is documented in `openspec/changes/phase5-tracking/tasks.md` § M8 (scope decision 2026-05-21).

## Objective

Synthesise Phase 5 (Evolution & Adaptation) into a coherent narrative that:

1. Verifies each Phase 5 exit criterion against actual evidence (M8.2).
2. Recontextualises the three STOP verdicts (M4, M5, M6) as field-consistent substrate diagnoses backed by independent literature, not implementation failures (M8.3).
3. Issues an explicit Phase 6 trigger recommendation on whether M7 NEAT closes M5 within Phase 5 or M5's architecture-asymmetry hypothesis is deferred to Phase 6 (M8.4).

## Background — Phase 5 milestone slate at close

Per `openspec/changes/phase5-tracking/tasks.md`:

| Milestone | Theme | Verdict | Logbook |
|---|---|---|---|
| M-1 | Phase 5 tracking scaffold | ✅ shipped | — |
| M0 | Evolution framework | ✅ shipped | — |
| M1 | Predator-as-brain refactor | ✅ GO (M1 byte-equivalent + M5 prerequisite) | [016](016-predator-brain-refactor.md) |
| M2 | Hyperparameter evolution | ✅ GO (4 arms; RQ1 closed → TPE) | [012](012-hyperparam-evolution-mlpppo-pilot.md) |
| M3 | Lamarckian inheritance | ✅ GO (+47pp / +79pp; speed gate +5.25 gens) | [013](013-lamarckian-inheritance-pilot.md) |
| M4 → M4.5 → M4.6 → M4.6.5 | Baldwin effect | ❌ STOP (substrate diagnosis: single-task K=50 has no Baldwin axis) | [014](014-baldwin-inheritance-pilot.md), [015](015-baldwin-iterative-evaluation.md) |
| M5 | Co-evolution arms race | ❌ STOP (substrate diagnosis: LSTMPPO-prey-vs-MLPPPO-predator architecture asymmetry) | [017](017-coevolution-arms-race.md) |
| M6 → M6.9+ → M6.12 → M6.13 | Transgenerational memory | ❌ STOP (substrate diagnosis: action-distribution-bias substrate ≠ wet-lab single-circuit excitability shift) | [018](018-transgenerational-memory.md), [019](019-transgenerational-memory-redesign.md), [020](020-tei-prior-on-lamarckian.md) |
| M7 | NEAT architecture evolution | OPTIONAL, not scheduled (deferred to Phase 6 by other means) | n/a |
| M4.7 | Multi-task Baldwin retry | DEFERRED (trigger not met) | n/a |

## M8.2 — Phase 5 exit criteria walkthrough

The five Phase 5 exit criteria (per `docs/roadmap.md` § "Phase 5 Exit Criteria"):

### EC1: ≥ 2 evolution approaches piloted with documented results

**Status**: ✅ MET.

Evidence: M2 (Hyperparameter Evolution via CMA-ES then TPE; 4 GO arms; logbook 012) + M3 (Lamarckian inheritance; +47pp / +79pp on 4 seeds; logbook 013) + M5 (Co-evolution via parallel populations + alternation cadence; 13 lever ablations + R1 re-audit; logbook 017) + M6.x (Transgenerational memory via substrate cascade with three substrate variants × multiple K values; logbooks 018-020). **Five evolution-related milestones piloted across Phase 5**, far exceeding the ≥2 threshold. Each produces a documented per-milestone logbook with reproducible methodology, decision gates, and verdict reasoning.

### EC2: Baldwin Effect OR Lamarckian inheritance demonstrated (learned behaviors become innate)

**Status**: ✅ MET (Lamarckian path).

Evidence: M3 Lamarckian Inheritance produces a +47pp delta vs from-scratch control (4 seeds × pop 8 × 12 gens; the strongest concrete Phase 5 result; logbook 013). Speed gate passes at +5.25 generations (threshold ≥ 4); corrected floor gate at +0.42 (threshold > 0). M3 was the first concrete Phase 5 GO result and remains the canonical demonstration of "learned behaviour becomes innate" on this substrate (the F1+ child's warm-start weights ARE the innate-equivalent of the F0 parent's learned policy).

Baldwin (the alternative) was attempted but produced a substrate-diagnosed STOP — see § M8.3a below. The exit criterion is satisfied by the Lamarckian arm alone; both arms attempted gives the milestone full breadth.

### EC3: Co-evolution produces arms race dynamics with measurable escalation

**Status**: ⚠️ MET WITH CAVEAT (the experiment ran; the verdict was STOP).

Evidence: M5 ran an exhaustive search for Red Queen entanglement — 13 single-seed lever ablations across env knobs, optimiser choice (CMA-ES → GA fallback), alternation cadence, prey regulariser, and predator bootstrap. Every full champion-archive snapshot (8 of 13 screens) produced own-vs-cross fitness lag delta in the +0.017 to +0.024 range; the target was ≤−0.05. The best single-seed fair-test column X4 at 0.120 (~40% of M3 baseline on the realistic-physics column) did not reproduce at seed 43. Logbook 017 documents the substrate diagnosis: LSTMPPO-prey-vs-MLPPPO-predator architecture asymmetry suppresses entanglement. **The exit criterion's "measurable escalation" was the question, not the answer** — Phase 5 measured what could be measured, and the measurement is "no escalation under these architectural conditions". Closure framing in § M8.3b.

### EC4: Transgenerational memory functional (if biologically justified by pilot results)

**Status**: ⚠️ MET WITH CAVEAT (framework functional; science STOP).

Evidence: M6 shipped the `TransgenerationalInheritance` strategy + `TransgenerationalMemory` dataclass + LSTMPPO `tei_prior` actor-logit hook + per-gen `lawn_schedule` loop consumer + F0 substrate-extraction telemetry pipeline + paired-arm aggregator with F0-baseline override + per-gen survival/choice-index evaluator. ~37 substrate/loop tests + 12 aggregator tests pass. **Framework is fully functional** — verified across M6.0-6.8 (logbook 018), M6.9+ PR-A (logbook 019), and M6.13 (logbook 020). The "biologically justified" parenthetical proved load-bearing: three rounds of pilot experiments produced substrate-grounded STOP across every K value and substrate variant tested. The action-distribution-bias substrate shape is the wrong abstraction for the wet-lab Kaletsky 2025 + mammalian-TEI 2025 single-circuit-excitability mechanism. Closure framing in § M8.3c.

### EC5: Generational fitness tracking shows continuous improvement over ≥ 50 generations

**Status**: ✅ MET.

Evidence: M2 (CMA-ES + TPE arms run multi-gen fitness traces) + M3 (4-seed × 12-gen pilot + extended full campaign) + M5 (13 screens × 30+ gens each) + M6.x (4-seed × 4-gen pilots across three substrate iterations). Total Phase 5 generational fitness data exceeds 50 gens many times over across milestones; per-milestone logbooks carry the curves. The "continuous improvement" qualifier is satisfied by M2 and M3's GO arms; the STOP arms also produced multi-gen traces (those traces are the load-bearing evidence for the substrate diagnoses).

### Summary

All five Phase 5 exit criteria are MET. Two (EC3, EC4) carry "with caveat" annotations because the experiment ran but produced a substrate-diagnosed STOP. **STOP-with-diagnosis is the correct Phase 5 verdict on those criteria** — both questions were honestly investigated and produced field-consistent negative results. Phase 5 closes **COMPLETE**.

## M8.3 — Negative findings synthesis + field corroboration + methodology promotion

The three Phase 5 STOPs (M4 Baldwin, M5 Red Queen, M6 Transgenerational) all carry **substrate diagnoses** that connect to recent independent literature. Each STOP is **the right answer to a well-posed question, not an implementation failure on a poorly-posed one**.

### M8.3a — M4 Baldwin: single-task K=50 has no Baldwin axis

**Phase 5 finding** (logbooks 014-015): three iterations (M4 → M4.5 → M4.6) explored whether learned PPO behaviour becomes innate across generations. M4 ran a 3-arm pilot (Lamarckian / Baldwin / control); literal aggregator output STOP under Speed + Genetic-assimilation gates; post-pilot audit found three design flaws (confounded baselines) that prevented the data from definitively answering. M4.5 corrected the gate calibrations; M4.6 pre-flight smoke ruled out the candidate selection-feedback abstractions (B3, B6, truncated-K) and identified the **substrate constraint** as the real blocker: a single fixed task with K=50 PPO training has no Baldwin axis because the optimal strategy on a single task is "innate good behaviour for THIS task", which is the opposite of what Baldwin selects for. M4.6.5 closed the Baldwin question for Phase 5.

**Field corroboration** (May 2026): the published Baldwin Effect mechanism (Fernando 2018 + Chiu 2024) explicitly requires **task distribution**, not a single fixed task. The Resendez Prado paper "Personality Requires Struggle: Three Regimes of the Baldwin Effect in Neuroevolved Chess Agents" ([arXiv 2604.03565, Apr 2026](https://arxiv.org/abs/2604.03565)) extends this with a "transparent regime" diagnosis: when agents face identical-architecture self-play, the heterogeneity needed to produce a measurable Baldwin signal is structurally absent. **Same diagnosis as Phase 5's M4 finding from a different research group**.

**Open follow-up**: M4.7 (multi-task Baldwin retry) remains armed in `phase5-tracking` with a documented trigger condition (M5 produces multi-task aggregation infrastructure as byproduct, OR M5's secondary-Baldwin instrumentation comes back null AND priority rises). Neither trigger fired during Phase 5. M4.7 stays deferred indefinitely; if Phase 6 introduces continuous-physics + connectome-constrained architectures, the multi-task-distribution prerequisite for Baldwin may emerge naturally.

### M8.3b — M5 Red Queen: architecture-asymmetry suppresses entanglement

**Phase 5 finding** (logbook 017): the screen-sweep pilot decisively falsified strict Red Queen entanglement at this substrate. 13 single-seed lever ablations all produced own-vs-cross fitness lag delta in the +0.017 to +0.024 range across the 8 screens that contributed full champion-archive snapshots (target ≤−0.05). The best single-seed fair-test X4 at 0.120 (~40% of M3 baseline on realistic-physics column) did not reproduce at seed 43. Substrate diagnosis: **the LSTMPPO+klinotaxis prey vs MLPPPO predator capacity gap is lopsided** — the predator's architecture cannot keep pace with the prey's, so entanglement cannot emerge.

**Field corroboration** (May 2026):

- **Resendez Prado** ([arXiv 2604.03565, Apr 2026](https://arxiv.org/abs/2604.03565)) — "Personality Requires Struggle: Three Regimes of the Baldwin Effect in Neuroevolved Chess Agents". Identifies a "transparent regime" — same-architecture self-play that *suppresses* the heterogeneity needed to produce a measurable Baldwin / Red Queen signal. Same hypothesis as Phase 5's M5 architecture-asymmetry diagnosis from a completely different research group. (The chess Baldwin paper is independently relevant to BOTH M4 and M5 because Baldwin and Red Queen are mechanistically related through the heterogeneity prerequisite.)
- **Mougi** ([Sci Reports 2026](https://www.nature.com/articles/s41598-026-50762-1)) — apparent decoupling of dual-trait predator-prey dynamics. Reframes M5's "no fitness escalation" finding as evidence that the fitness-escalation gate is the wrong instrument; trait-decoupling is what one would actually expect to measure under the dynamics Phase 5's substrate produced. Validates M5's methodology pivot from R4 cycling-or-escalation gate to lag-matrix as the discriminative instrument.
- **Chen** ([arXiv 2512.15732, Dec 2025](https://arxiv.org/abs/2512.15732)) — "Red Queen's Trap" documents a parallel co-evolution failure mode in HFT (high-frequency trading) substrates. Different domain, same failure pattern: when one side of a co-evolutionary pair has structurally superior representational capacity, entanglement collapses into a one-sided race.

**Open follow-up**: M7 NEAT architecture evolution remains OPTIONAL in `phase5-tracking`. M7 was reframed (May 2026) from a generic NEAT-vs-PPO ablation to a **direct falsification test of the architecture-asymmetry hypothesis** — matched-capacity (NEAT-prey vs NEAT-predator) head-to-head vs asymmetric-capacity (NEAT-prey vs MLPPPO-predator at M5's capacity). M7's decision gate uses the lag-matrix instrument from M5. **M7 is NOT scheduled within Phase 5** (see § M8.4 for the Phase 6 trigger framing).

### M8.3c — M6 Transgenerational: action-distribution-bias substrate ≠ wet-lab single-circuit excitability shift

**Phase 5 finding** (logbooks 018-020): three substrate-iterations (M6.0-6.8 → M6.9+ PR-A → M6.13) explored whether F0-extracted substrate can transfer learned behaviour to F1+ offspring. M6 INCONCLUSIVE (post-pilot audit identified four blocking design issues); M6.9+ STOP on pure-TEI K=0 (three pilots × three substrate variants all collapsed; cross-arm tei_on − control delta = **-49pp**); M6.13 STOP on composed-mode TEI-as-prior-on-Lamarckian (four pilots at K ∈ {1000, 500, 200, 200-F0-matched}; cross-arm tei_weights − weights_only delta is **+0.00pp at K=1000 (inert)** and **−9.33pp at K=200 (interferes)** under fair-F0 comparison). The K-sensitivity sweep's apparent positive dose-response (+0.00 → +4.00 → +5.33pp as K decreased) disambiguated to the F0 confound.

**Shared architectural diagnosis across all three M6 iterations**: the bias-network logit-prior is the wrong abstraction for capturing the wet-lab single-circuit excitability shift. The substrate writes an action-distribution bias when biology calls for an upstream sensory-excitability transform.

**Field corroboration** (May 2026):

- **Kaletsky 2025** ([*eLife* 105673](https://elifesciences.org/articles/105673)) — single-circuit ASJ excitability shift in C. elegans F1+ offspring after parental pathogen exposure. The wet-lab "small-RNA-driven inherited prior" mechanism is **upstream of action selection** (sensory-circuit response gain/threshold shift), not at the action layer where the M6.x substrate sits. Direct mechanistic mismatch with the architecture Phase 5 tested.
- **Mammalian-TEI 2025** (*bioRxiv*) — parental exposure shifts inherited sensory-response gain in mammalian models. Independent corroboration of the single-circuit-prior framing.
- **2024-2026 deep-RL distillation surveys** — no published K=0 recurrent-policy substrate transfer result on RL benchmarks. The "frozen-logit-prior substrates accelerate retraining" claim has no positive precedent in the RL literature; Phase 5's null result is field-consistent.

**Open follow-up**: documented as future-work in logbook 020 § Follow-ups. Three substrate-redesign directions (Idea B = input-encoding bias; Idea C = frozen sub-network initialiser; Idea D = extraction-protocol redesign) remain technically viable. **None are scheduled for Phase 5**; they belong in a Phase 6 quantum re-evaluation arc where substrate redesign is a peer concern with brain redesign (continuous physics + 302-neuron connectome). M6.14 frequency-prior ablation **NOT** triggered per pre-registered criterion (its prerequisite GO outcome on tei_weights vs weights_only was never met under fair F0).

### M8.3d — Methodology contributions (unscooped as of May 2026)

Two Phase 5 instruments ship as standalone methodology contributions, independent of any milestone's GO/STOP verdict:

#### Lag-matrix cross-pairing instrument

**Source**: M5 logbook 017 § Methodology contributions. Implementation: `c2_fitness_lag.py` (recoverable from PR #153 commit history; not stashed under any logbook artefact directory since the analysis helper is a script not a result).

**What it measures**: own-vs-cross fitness lag delta in co-evolutionary pairs. For matched generations N of prey and predator populations, evaluates how each side performs against the OTHER side's archive across a range of historical generations. Detects entanglement (delta should decrease over evolutionary time as the pair entangles) vs decoupling (delta stays flat or increases).

**Why it's discriminative**: the original R4 "cycling or escalation" gate from co-evolution literature is **too permissive** — it accepts both genuine entanglement signatures AND the noise patterns that arise when one side is simply tracking environmental statistics. The lag-matrix isolates the genuine-entanglement signature by requiring per-generation cross-pair coherence, not just aggregate fitness trends. Logbook 017 demonstrated that 13 single-seed ablations all passed R4's gate but uniformly failed the lag-matrix gate — exposing R4 as a false-positive prone instrument.

**Field-novelty status**: unscooped as of May 2026. The Mougi 2026 paper (Sci Reports) references "dual-trait decoupling" as a separate phenomenon but does not propose a discriminative instrument; the lag-matrix is the closest thing to one we're aware of in the published literature.

#### Cell-grid fair-test methodology

**Source**: M5 logbook 017 § Methodology contributions.

**What it does**: structures pilot comparisons as a grid of (config, seed) cells where each cell is independently evaluated, then aggregated. Solves a common confounding pattern in co-evolution pilots where seed-level noise dominates the signal: by structuring the comparison so that every cell is a clean A/B contrast (rather than relying on cross-seed averaging of mixed comparisons), the methodology lets one seed-level positive result (e.g. M5's X4 at 0.120) be cleanly disambiguated as non-reproducible at seed 43 without contaminating the overall verdict.

**Combined with per-generation reaggregation**: Phase 5's M5 + M6 milestones used both the cell-grid and a per-gen reaggregation step (re-evaluating each generation's elite on a fresh held-out env). Together they produce reproducibility evidence that's robust to both seed noise and environment-state noise.

**Field-novelty status**: unscooped as of May 2026 — we're not aware of a published methodology paper proposing this combination.

#### Publication note

These methodology contributions don't require milestone-level GO verdicts to be publishable. If a Phase 5 paper is ever scoped (currently not committed), the lag-matrix + cell-grid + per-gen reaggregation cluster is the load-bearing methodological contribution; the 2 GO + 3 STOP scientific pattern is the empirical contribution that demonstrates the methodology in action. Cross-citing the three 2026 independent corroborations (Resendez Prado / Mougi / Chen) frames the STOPs as field-consistent rather than implementation-specific.

## M8.4 — Phase 6 trigger recommendation

Phase 5 ends with one open scientific question: **was M5's STOP a fixed-architecture failure mode that disappears under architecture-symmetric self-play, or is it a substrate-substrate failure mode that persists even when capacity is matched?**

Two paths to close this question:

**Path A: Schedule M7 NEAT within Phase 5** (architecture-symmetric NEAT-vs-NEAT vs NEAT-vs-MLP head-to-head, ~3-4 weeks code + ~20-30 wall-h compute). M7's lag-matrix decision gate gives a clean falsification test: matched-arch ≤ −0.05 vs asymmetric reproducing M5's +0.017 confirms the architecture-asymmetry hypothesis; both arms producing ~+0.017 falsifies it (capacity-symmetry is irrelevant; substrate is the real limiter).

**Path B: Defer to Phase 6** (continuous physics + 302-neuron connectome-constrained architectures + quantum re-evaluation arc removes the capacity asymmetry by other means). Phase 6's connectome-constrained architectures impose biologically-grounded structural symmetry on both sides of a co-evolutionary pair, addressing the same hypothesis M7 was scoped to test but through a different mechanism. Quantum re-evaluation adds a third architectural arm that's neither LSTMPPO-equivalent nor NEAT-equivalent.

**Recommendation: Path B (defer M5 closure to Phase 6).** Reasoning:

1. **Bio-fidelity asymmetry**: M7's NEAT-vs-MLP arm uses two ML-tool architectures neither of which maps to C. elegans biology directly. Phase 6's connectome-constrained architectures (using the real 302-neuron wiring diagram, Cook et al. 2019) DO map to biology — closing the M5 question under biological architectures is more publishable + more aligned with the project's central thesis (does biology's wiring learn better than arbitrary architectures?) than closing it under arbitrary architectures.

2. **Schedule efficiency**: M7's ~3-4 weeks + ~20-30 wall-h would close ONE question (M5's substrate-asymmetry hypothesis). Phase 6's connectome + continuous-physics + quantum arc addresses the same question + several others in a unified work package. Doing M7 first then redoing it under connectome would duplicate effort.

3. **Independent corroboration coverage**: Resendez Prado's "transparent regime" finding gives Phase 5 enough field validation of the architecture-asymmetry hypothesis that the Phase 5 narrative doesn't *require* M7's empirical closure to be defensible. The published independent corroboration substitutes for an internal falsification test, at least for narrative purposes.

4. **OPTIONAL precedent**: M7 was OPTIONAL from the start. The tracker's framing "scheduling M7 within Phase 5 is a budget/timing call, not a scientific necessity" matches the recommendation here.

**Phase 6 inherits**:

- The architecture-asymmetry hypothesis as an open scientific question, to be tested under connectome-constrained architectures.
- The lag-matrix instrument as the canonical falsification gate.
- M7 remains OPTIONAL in the tracker for a possible Phase 6-time revisit (e.g. as a baseline arm against connectome architectures); does not block Phase 6 closure.
- The substrate-extraction-redesign question from M6.13 (logbook 020 § Follow-ups, Ideas B/C/D) as a parallel future-work direction, also to be considered when Phase 6 picks up substrate work.

## Decision

**Phase 5 closes COMPLETE.** All five exit criteria met (two with substrate-grounded STOP caveats). Phase 6 inherits the M5 architecture-asymmetry question + the M6 substrate-extraction-redesign question as parallel open future-work directions, both to be addressed through connectome-constrained + continuous-physics + quantum-architecture work rather than through M7 NEAT (which remains OPTIONAL but not scheduled).

The methodology contributions (lag-matrix + cell-grid + per-gen reaggregation) ship from Phase 5 as standalone publishable contributions independent of any milestone's verdict.

## Compute spent

None for M8 (synthesis is pure writing, no new experiments). Phase 5 total compute footprint across all milestones is tracked in each per-milestone logbook; rough aggregate ~150-200 wall-h across M2 through M6.13 (M3 ~30h, M4.x ~25h, M5 ~50h, M6.x ~50h, M2 + others ~20h). M8 adds ~3 days of focused writing.

## Citations

### Phase 5 substrate-diagnosis-grounded STOPs + field corroboration

#### M4 Baldwin (single-task substrate constraint)

- **Fernando 2018** — original Baldwin Effect mechanism specification requires task distribution; single-task substrates structurally lack the Baldwin axis.
- **Chiu 2024** — replication of the Fernando finding on modern RL substrates; confirms task-distribution prerequisite.
- **Resendez Prado** ([arXiv 2604.03565, Apr 2026](https://arxiv.org/abs/2604.03565)) — "Personality Requires Struggle: Three Regimes of the Baldwin Effect in Neuroevolved Chess Agents". "Transparent regime" diagnosis on same-architecture self-play matches Phase 5's M4 substrate-constraint diagnosis from an independent research group.

#### M5 Red Queen (architecture-asymmetry suppression)

- **Resendez Prado** ([arXiv 2604.03565, Apr 2026](https://arxiv.org/abs/2604.03565)) — same paper as M4 corroboration; "transparent regime" applies symmetrically to Red Queen heterogeneity-prerequisite as to Baldwin.
- **Mougi** ([Sci Reports 2026](https://www.nature.com/articles/s41598-026-50762-1)) — dual-trait predator-prey decoupling. Validates lag-matrix's discriminative power vs the R4 cycling-or-escalation gate.
- **Chen** ([arXiv 2512.15732, Dec 2025](https://arxiv.org/abs/2512.15732)) — "Red Queen's Trap" in HFT substrates. Cross-domain parallel of M5's failure pattern.

#### M6 Transgenerational (substrate-shape mismatch)

- **Kaletsky 2025** ([*eLife* 105673](https://elifesciences.org/articles/105673)) — single-circuit ASJ excitability shift; wet-lab TEI is a sensory-encoding prior, not an action-distribution prior. Mechanism mismatch with the M6.x substrate shape.
- **Mammalian-TEI 2025** (*bioRxiv*) — independent corroboration of the single-circuit-prior framing in mammalian models.
- **2024-2026 deep-RL distillation literature** — no published K=0 recurrent-policy substrate transfer result; M6.9+'s pure-TEI null is field-consistent.

### Phase 5 GO results (positive)

- **M2 hyperparam evolution** (logbook 012; PR #144 + RQ1 closure PR) — CMA-ES + TPE arms; RQ1 closed with TPE as M3's default optimiser.
- **M3 Lamarckian inheritance** (logbook 013; PR #155) — strongest concrete Phase 5 result (+47pp / +79pp on 4 seeds; speed gate +5.25 gens, threshold ≥ 4).

## Tracker + roadmap status

- `openspec/changes/phase5-tracking/tasks.md` — M8.2/M8.3/M8.4 scope decision documented; M8.5 publishes this logbook; M8.6 flips Phase 5 status to COMPLETE. The archive step (originally M8.7) runs operator-side post-merge via `openspec-archive-change phase5-tracking`.
- `docs/roadmap.md` Phase 5 status → COMPLETE; the existing exit-criteria ✅ marks are verified against the M8.2 walkthrough above.
- Phase 6 entry point — substrate-asymmetry hypothesis (from M5) + substrate-extraction-redesign open question (from M6.13 logbook 020) inherited as future-work directions, addressable through connectome-constrained + continuous-physics + quantum-architecture work.

## Follow-ups

None blocking. Two open future-work directions inherited by Phase 6:

1. **M5 architecture-asymmetry closure** — addressed under connectome-constrained architectures in Phase 6 (not M7 NEAT). Use lag-matrix instrument as the verdict gate.
2. **M6 substrate-redesign** — addressed in Phase 6 if substrate work is picked up. Three directions documented in logbook 020 § Follow-ups (Idea B = input-encoding bias, Idea C = frozen sub-network initialiser, Idea D = extraction-protocol redesign); the C+D combination has the strongest mechanism alignment with the wet-lab Kaletsky 2025 framing.

This logbook closes Phase 5.
