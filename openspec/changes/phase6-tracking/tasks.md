# Tasks: Phase 6 (Connectome Substrate & Architecture Comparison) Tranche Tracker

This is the living checklist for all of Phase 6. Each tranche (T1–T8) has its own
OpenSpec change directory listed below as it is created. Each Phase 6 milestone PR
MUST update this file to mark sub-tasks complete as part of its diff.

**Status legend**: `[ ]` not started, `[x]` complete. Tranche-level "in progress"
status lives in the **Status** header of each tranche section (matches the roadmap
Phase 6 Tranche Tracker emoji column).

## Phase 6 Tranche Map

Phase 6 has eight tranches with deliberate ordering. See
[design.md § Decision 1](design.md) for the load-bearing rationale (why T3 precedes
T4; why T5 sits between T4 and T6; why T6 holds real-worm validation; why T7 runs
against the upgraded substrate).

| Tranche | Scope | Roadmap layer | Approx duration | Gate trigger |
|---|---|---|---|---|
| 1 | L0 connectome ingest — Cook 2019 via OpenWorm `cect`, vendored data, cross-validated against Witvliet 2021, smoke-test forward pass, no env wiring | L0 | 2-3 weeks | — |
| 2 | L1 plugin refactor + connectome-as-brain wired through existing grid env | L1 | 3-5 weeks | **Gate 1** — basic PPO-on-connectome trainable on existing grid |
| 3 | Corrected ASH/ADL contact-based nociception (owed correctness work per Logbook 011) | env-correctness | 1-2 weeks | — |
| 4 | L2 initial pass — MUST architectures × 3 behaviours, grid-world substrate, corrected nociception | L2 (first pass) | 4-6 weeks | — |
| 5 | Env upgrades — Rung 2 dynamic Fick's-law + log-concentration chemosensory adaptation + continuous 2D + continuous-action heads on existing brains | env-upgrade | 6-8 weeks | **Gate 2** — L1 plugin parity test (adding a 9th architecture ≤ 1 week) verified during this work |
| 6 | L2 re-run on upgraded substrate; real-worm validation; SHOULD/MAY architectures evaluated opportunistically | L2 (final) | 4-6 weeks | **Gate 3** — L2 results across MUST set in hand |
| 7 | L3 NEAT topology search on upgraded substrate | L3 | 6-10 weeks | — |
| 8 | Phase 6 synthesis logbook | — | 1-2 weeks | — |

Total: 27-42 weeks. L4 (biologically-plausible plasticity) is deferred to Phase 7
and does not appear as a Phase 6 tranche.

## P6-0: Phase 6 Tracking Scaffold (THIS CHANGE)

**Branch**: `feat/phase6-tracking`
**OpenSpec change**: `phase6-tracking` (this directory)
**Status**: in progress

- [x] P6-0.1 Create `openspec/changes/phase6-tracking/proposal.md`
- [x] P6-0.2 Create `openspec/changes/phase6-tracking/design.md`
- [x] P6-0.3 Create `openspec/changes/phase6-tracking/tasks.md` (this file)
- [x] P6-0.4 Create `openspec/changes/phase6-tracking/specs/phase6-tracking/spec.md`
- [x] P6-0.5 Update `docs/roadmap.md` Phase 6 row in Timeline Overview to `🟡 IN PROGRESS`
- [x] P6-0.6 Add Phase 6 Tranche Tracker sub-section to `docs/roadmap.md` Phase 6 block
- [x] P6-0.7 Annotate `docs/roadmap.md` Phase 6 architecture-families table with MUST/SHOULD/MAY per [design.md § Decision 4](design.md)
- [x] P6-0.8 Annotate `docs/roadmap.md` Phase 6 layered-platform table with the env-upgrade tranche (T5) note
- [ ] P6-0.9 Validate change: `openspec validate phase6-tracking --strict`
- [ ] P6-0.10 Run targeted `uv run pre-commit run --files <changed>` clean
- [ ] P6-0.11 Open PR

## Tranche 1 — L0 Connectome Substrate

**OpenSpec change**: `add-connectome-substrate` (not yet created — first Phase 6 milestone change)
**Status**: 🔲 not started
**Roadmap layer**: L0
**Approx duration**: 2-3 weeks
**Bio fidelity**: HIGH (real wiring is the substrate)
**Dependencies**: P6-0 (this scaffold)
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § The layered platform § L0

T1 imports the *C. elegans* 302-neuron connectome (Cook et al. 2019 hermaphrodite via
OpenWorm `cect` per [design.md § Decision 2](design.md)) and validates it against an
independent reference (Witvliet 2021 nerve-ring subset). No plugin work, no training,
no env wiring in this tranche — the deliverable is "a real connectome loaded,
validated, vendored, and forward-passable" plus the data model that L1 will conform
to.

- [ ] T1.1 Choose and document the import library (decision pre-recorded: `cect` / ConnectomeToolbox; see [design.md § Decision 2](design.md)). Pin the dependency version; record the upstream dataset identifiers. The exact Cook 2019 sub-file selection (adjacency XLSX vs synapse-list CSV) is deliberately left to this tranche to decide — see [design.md § What This Change Explicitly Does Not Decide](design.md).
- [ ] T1.2 Build the connectome data model (neurons, synapses, gap junctions, neurotransmitter labels, anatomical roles — sensory / interneuron / motor). The data model is the topology interface every architecture plugin will conform to in T2.
- [ ] T1.3 Import the Cook 2019 hermaphrodite connectome. Verify neuron count (302), synapse count (within published bounds), gap-junction count, and neuron-name conformance to standard *C. elegans* naming.
- [ ] T1.4 Cross-validate against Witvliet et al. 2021 nerve-ring subset (~50 lines of pandas). Document where the two datasets agree, where they diverge (developmental stage + lineage-tracing differences are expected), and which divergences would materially affect downstream learning.
- [ ] T1.5 Vendor the connectome data (no network access at training time). Document the import provenance: upstream dataset DOI / commit, `cect` version, any transformations applied during import, any neurons / synapses dropped or merged, any metadata gaps.
- [ ] T1.6 Forward-pass smoke test: instantiate a trivial PPO weight set on the connectome topology, run a single forward pass, verify shapes / no NaNs. Not a training run — a "does the substrate respond to inputs" sanity check.
- [ ] T1.7 Unit + smoke tests for the data model and import pipeline; CI integration.
- [ ] T1.8 Update this checklist + `docs/roadmap.md` Phase 6 Tranche Tracker T1 row.
- [ ] T1.9 Publish T1 logbook (suggested: `docs/experiments/logbooks/0XX-connectome-substrate.md`). Feeds into Gate 1's evidence base (full Gate 1 decision lands at T2 close).

**T1 risk-mitigation pivot (per roadmap)**: if `cect` import / dataset access proves harder than expected (format incompatibility, missing metadata, unclear synaptic-weight provenance), drop to a hand-curated subset of the Cook 2019 connectome — sensory-interneuron-motor subgraph for the three target behaviours, ~50-100 neurons. The pivot decision is itself a written gate-style decision; the T1 OpenSpec change documents it.

## Tranche 2 — L1 Plugin Refactor + Connectome-as-Brain on Existing Grid

**OpenSpec change**: `add-architecture-plugin-interface` (not yet created)
**Status**: 🔲 not started
**Roadmap layer**: L1
**Approx duration**: 3-5 weeks
**Bio fidelity**: n/a (interface refactor)
**Dependencies**: T1 closed
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § The layered platform § L1

T2 does the L1 work per [design.md § Decision 3](design.md): refactor the existing
`setup_brain_model()` dispatcher into a registry pattern, factor *topology* out from
*learning rule* so the L0 connectome topology can be consumed by PPO, spiking, and
NEAT-topology+PPO alike, and wire the connectome-as-brain through the existing grid
env end-to-end. The `Brain` Protocol surface (`run_brain` / `update_memory` /
`prepare_episode` / `post_process_episode` / `copy` at
`packages/quantum-nematode/quantumnematode/brain/arch/_brain.py:346-368`) does NOT
change.

T2 closes Gate 1.

Sub-tasks are coarse-grained; the L1 OpenSpec change elaborates them.

- [ ] T2.1 Audit the existing `setup_brain_model()` dispatcher at `packages/quantum-nematode/quantumnematode/utils/brain_factory.py` and the `BrainType` enum. Identify every per-architecture branch and every file touched when a new architecture is added (currently ~4 files per addition: enum, dispatcher, config class, YAML loader).
- [ ] T2.2 Refactor the dispatcher into a registry pattern. The exact pattern (decorator-registration vs entry-points vs config-driven) is deliberately deferred to this tranche per [design.md § What This Change Explicitly Does Not Decide](design.md).
- [ ] T2.3 Factor *topology* out from *learning rule* in the registry-registered architectures. The L0 connectome data model is the new topology API; PPO, spiking, and NEAT-evolved-topology+PPO must be able to consume it without bespoke wiring.
- [ ] T2.4 Migrate the existing 19 architectures behind the new registry. The migration is mechanical for the 17 already-built MUST/SHOULD/MAY families; LSTMPPO + MLPPPO go first as the architectures that will carry the first L2 runs.
- [ ] T2.5 Wire the connectome-as-brain through the existing grid env via the new plugin interface. Train weights with PPO on klinotaxis (Phase 4's substrate carries forward). This is the first closed-loop learning on the real *C. elegans* connectome — the platform claim, on the existing grid (the upgraded-substrate version lands in T6).
- [ ] T2.6 Plugin-parity test: a documented procedure (or CI integration test) that demonstrates adding a hypothetical 9th architecture is ≤ 1 week of work — counting LOC, files touched, tests required. NOTE: Gate 2's "≤ 1 week" verdict comes during T5, not here — T2 establishes the test methodology and runs it once; T5 re-runs it against the env-upgrades work as a real-world verification.
- [ ] T2.7 Documentation: plugin-developer guide; "how to add a new architecture family in ≤ 1 week" walkthrough.
- [ ] T2.8 Update this checklist + `docs/roadmap.md` Phase 6 Tranche Tracker T2 row.
- [ ] T2.9 Publish T2 logbook (suggested: `docs/experiments/logbooks/0XX-l1-plugin-interface.md`). Required reading material for Gate 1.

**T2 risk-mitigation pivot (per roadmap)**: if multiple architecture families need bespoke plumbing and the interface accumulates per-architecture branches, pause architecture-sweep work and spend 2-4 weeks refactoring L1 toward genuine plugin parity. Better to delay L2/L3 results than to ship a "platform" that isn't one.

## Gate 1 (closes Tranche 2, month ~2-3 cumulative): L0 import working on the L1 plugin interface?

**Trigger**: T2 closed; connectome-as-brain trainable via the new plugin interface on the existing grid env.
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § Mid-phase decision gates § Gate 1.
**Decision must be written in**: the T2 OpenSpec change's logbook or `tasks.md`. This tracker links to the decision once it lands.

- [ ] **Gate 1 decision recorded**: connectome substrate loaded, validated, and basic-MLP-PPO + first connectome-as-brain training run trainable on the existing grid env via the L1 plugin interface (GO) — OR — full 302-neuron import infeasible and the hand-curated subset pivot is the path forward, with documented subset choice (PIVOT) — OR — both paths infeasible at substrate level (STOP).
- [ ] **Gate 1 decision link**: [add link to the OpenSpec change / logbook where the decision is recorded]

## Tranche 3 — Corrected ASH/ADL Contact-Based Nociception

**OpenSpec change**: `fix-ash-adl-contact-nociception` (placeholder; not yet created)
**Status**: 🔲 not started
**Roadmap layer**: env-correctness (precedes L2 first pass per [design.md § Decision 1](design.md))
**Approx duration**: 1-2 weeks
**Bio fidelity**: HIGH (corrects an existing biologically-wrong model)
**Dependencies**: T2 closed (Gate 1 GO or PIVOT)
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § Continuous environment + sensory physics § Mechanosensation

Real *C. elegans* nociception is contact-based mechanosensation (ASH/ADL neurons),
not chemosensory at distance. Phase 4's Logbook 011 flagged this; T3 fixes it before
the L2 first pass (T4) so the L2 predator-evasion cells run against the corrected
model from the start. Doing this after T4 would mean rerunning every predator-evasion
L2 cell.

- [ ] T3.1 Implement ASH/ADL contact-based mechanosensation with realistic sensory ranges scaled to worm body length.
- [ ] T3.2 Document the corrected model against the existing (incorrect) Phase 4 model; quantify the behavioural difference under matched conditions on the existing grid env.
- [ ] T3.3 Update existing predator-evasion configs / smoke tests to consume the corrected sensor; deprecate the chemosensory-at-distance code path or feature-flag it for archaeology only.
- [ ] T3.4 Unit + smoke tests for the corrected sensor; CI integration.
- [ ] T3.5 Update this checklist + `docs/roadmap.md` Phase 6 Tranche Tracker T3 row.
- [ ] T3.6 Brief T3 logbook entry (may live as a section in `docs/experiments/logbooks/0XX-l2-first-pass.md` rather than its own logbook — the fix is small, the verification belongs alongside the L2 cells that depend on it).

## Tranche 4 — L2 Initial Pass on Grid Substrate (MUST architectures × three behaviours)

**OpenSpec change**: `add-l2-first-pass` (placeholder; not yet created)
**Status**: 🔲 not started
**Roadmap layer**: L2 (first pass)
**Approx duration**: 4-6 weeks
**Bio fidelity**: depends on architecture family (HIGH for connectome-constrained; lower for others); grid substrate caps env-fidelity
**Dependencies**: T2 closed (L1 plugin), T3 closed (corrected ASH/ADL for predator evasion cells)
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § The layered platform § L2 + § Architecture-comparison protocol

T4 is the L2 weight-search sweep across the four MUST architecture families
([design.md § Decision 4](design.md)) × three behaviours
([design.md § Decision 5](design.md)) on the **existing grid substrate** with
corrected nociception (T3), using **strict-mask** semantics for the connectome rows
([design.md § Decision 7](design.md)). Phase 5 statistical bar carries forward
(paired-seed Wilcoxon + bootstrap CIs + n ≥ 4 seeds per condition).

T4 produces a publishable intermediate result — connectome-on-grid is directly
comparable to Phase 5's grid-world baseline. The env-upgrade delta lands at T6.

12 MUST cells (4 families × 3 behaviours). Each cell is its own training run; tick as
it ships.

### T4 — Connectome-constrained (focal architecture)

- [ ] T4.connectome.klinotaxis — PPO weight-search on Cook 2019 connectome, strict-mask
- [ ] T4.connectome.thermotaxis
- [ ] T4.connectome.predator_evasion (consumes T3 corrected ASH/ADL nociception)

### T4 — MLP-PPO

- [ ] T4.mlp_ppo.klinotaxis
- [ ] T4.mlp_ppo.thermotaxis
- [ ] T4.mlp_ppo.predator_evasion (consumes T3)

### T4 — LSTM / GRU-PPO

- [ ] T4.lstm_gru_ppo.klinotaxis
- [ ] T4.lstm_gru_ppo.thermotaxis
- [ ] T4.lstm_gru_ppo.predator_evasion (consumes T3)

### T4 — NEAT-evolved (weights only at T4; topology evolution lives at T7)

- [ ] T4.neat_weights.klinotaxis
- [ ] T4.neat_weights.thermotaxis
- [ ] T4.neat_weights.predator_evasion (consumes T3)

### T4 — connectome ablation

- [ ] T4.connectome_soft_prior.\* — documented ablation per [design.md § Decision 7](design.md): synaptic counts as weight initialisation only; PPO free to grow new connections. Compared head-to-head with strict-mask on at least one behaviour (klinotaxis is the natural first).

### T4 — cross-architecture analysis + logbook

- [ ] T4.analysis.ranking — paired-seed Wilcoxon + bootstrap CIs across the four MUST families × three behaviours on the grid substrate. The first architecture-comparison ranking.
- [ ] T4.analysis.connectome_grid_ranking — explicit answer to "where does the wild-type connectome (strict-mask) rank under PPO weight search on the existing grid substrate?"
- [ ] T4.analysis.strict_vs_soft — strict-mask vs soft-prior delta on at least one behaviour.
- [ ] T4.logbook — publish T4 logbook (suggested: `docs/experiments/logbooks/0XX-l2-first-pass.md`). Feeds Gate 3's evidence base together with T6.

**T4 risk-mitigation pivot (per roadmap)**: if after reasonable hyperparameter search no architecture family reaches Phase 0-3 baselines on any of the three behaviours, run the diagnostic sequence (topology density / reward shaping; continuous-action head is NOT yet on the table — that's T5). If none resolve, the finding is "PPO-on-the-real-connectome requires further substrate work" — itself a publishable negative result and a Phase 7 prerequisite.

## Tranche 5 — Env Upgrades (Rung 2 gradients + continuous-2D physics + continuous-action heads)

**OpenSpec change**: likely split into two — `add-rung2-chemical-gradients` and `add-continuous-2d-physics-and-action-heads` (placeholders; not yet created). Continuous-action heads are a prerequisite for continuous-2D physics; both ship together.
**Status**: 🔲 not started
**Roadmap layer**: env-upgrade (sits between T4 and T6 per [design.md § Decision 1](design.md) so the env-upgrade delta is itself a finding)
**Approx duration**: 6-8 weeks (longest tranche; absorbs scope creep risk)
**Bio fidelity**: HIGH for Rung 2 (matches the computational-chemotaxis field's actual fidelity standard); MEDIUM for continuous-2D (matches plate-arena geometry)
**Dependencies**: T4 closed (so the env-upgrade delta has a baseline to compare against)
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § Continuous environment + sensory physics

T5 closes Gate 2 (the env-upgrades work exercises the L1 plugin interface across
multiple architectures and verifies the "≤ 1 week to add a new architecture" parity
test in practice).

### T5.physics — Continuous 2D coordinates + continuous-action heads

- [ ] T5.physics.1 Continuous 2D coordinate system: realistic spatial scales (~1mm worm body on cm-scale plates). Native sinusoidal undulation, omega turns, and pirouettes are explicitly NOT in scope (per roadmap); the c302 / Sibernetic export path covers those if/when behavioural-fidelity claims later require them.
- [ ] T5.physics.2 Continuous action space (speed 0-to-max + turning angle −π to π). Replaces the discrete 4-action `DEFAULT_ACTIONS` (`packages/quantum-nematode/quantumnematode/brain/actions.py:8-31`) on the upgraded substrate; T4's discrete substrate stays intact for the T4-vs-T6 delta comparison.
- [ ] T5.physics.3 Continuous-action policy heads on every MUST PPO-family brain (Gaussian policy; exact parameterisation — Gaussian vs Beta vs Tanh-squashed — deliberately deferred to this tranche per [design.md § What This Change Explicitly Does Not Decide](design.md)).
- [ ] T5.physics.4 Continuous-output adapter for the connectome-constrained brain (T1 data model + T2 plugin interface deliver topology; continuous-output is added here).
- [ ] T5.physics.5 SHOULD/MAY architectures (quantum, spiking, reservoir, hybrid, transformer) — continuous-output adaptation is opportunistic; not gating Gate 2.

### T5.gradients — Rung 2 dynamic Fick's-law + log-concentration chemosensory adaptation

Rung 2 has **two coupled components** — environment dynamics AND chemosensory
adaptation kinetics. They MUST be designed together (per roadmap: "without
log-concentration adaptation on the sensory side, the gradient realism is wasted").

- [ ] T5.gradients.1 Heat-equation diffusion (∂C/∂t = D∇²C) with signal-type-specific D values (food vs pheromone vs CO₂).
- [ ] T5.gradients.2 Source dynamics — depletion when worms feed; source replenishment; decay terms for short-lived signals.
- [ ] T5.gradients.3 Log-concentration chemosensory adaptation kinetics on AWC/AWA/ASE-style sensors. Coupled component; ships with T5.gradients.1+.2.
- [ ] T5.gradients.4 Cross-tranche dependency: T6 klinotaxis + thermotaxis evaluations use Rung 2 gradients.

### T5 — plugin-parity verification (closes Gate 2)

- [ ] T5.parity.1 Add a hypothetical new architecture family (or revive a Phase 0-3 architecture that didn't make the MUST set) through the L1 plugin interface during T5. Measure wall-time and files touched; verify ≤ 1 week per the parity test.
- [ ] T5.parity.2 Document the parity verification in the T5 logbook; this is the evidence Gate 2 needs.

### T5 — analysis + logbook

- [ ] T5.analysis — quantify the env-upgrade fidelity gain (Rung 2 vs Rung 0 chemical gradients on a smoke task; continuous-2D vs discrete-grid on movement kinematics).
- [ ] T5.logbook — publish T5 logbook (suggested: `docs/experiments/logbooks/0XX-env-upgrades.md`). Required reading material for Gate 2.

**T5 risk-mitigation**: T5 is the longest tranche and the most likely scope-creep target. Stay bounded to Rung 2 + continuous-2D + continuous-action heads. 3D physics, Sibernetic interop, aerotaxis/pheromone restoration, and multi-agent are all explicitly out of scope (Decision 5 + roadmap Future Directions).

## Gate 2 (closes Tranche 5, month ~4-5 cumulative): L1 plugin parity achieved in practice?

**Trigger**: T5 closed with the parity test verified during the env-upgrades work.
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § Mid-phase decision gates § Gate 2.
**Decision must be written in**: the T5 OpenSpec change(s) logbook(s) or `tasks.md`. This tracker links to the decision once it lands.

- [ ] **Gate 2 decision recorded**: adding a new architecture demonstrably ≤ 1 week of work, verified during T5 (GO) — OR — interface accumulates per-architecture branches; L1 refactor pivot triggered (PIVOT) — OR — plugin interface fundamentally incompatible with one or more MUST architecture families (STOP).
- [ ] **Gate 2 decision link**: [add link to the OpenSpec change / logbook where the decision is recorded]

## Tranche 6 — L2 Re-run on Upgraded Substrate + Real-Worm Validation

**OpenSpec change**: `add-l2-final-pass-and-validation` (placeholder; not yet created)
**Status**: 🔲 not started
**Roadmap layer**: L2 (final) + real-worm validation
**Approx duration**: 4-6 weeks
**Bio fidelity**: HIGH (Rung 2 gradients + corrected nociception + continuous-2D + real-worm validation target)
**Dependencies**: T4 closed (grid baseline for env-upgrade delta), T5 closed (upgraded substrate)
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § The layered platform § L2 + § Built-in real-worm validation

T6 re-runs the L2 sweep on the upgraded substrate, ships the real-worm validation
(now defensible because the behavioural numbers come from Rung 2 + continuous-2D +
corrected nociception), and opportunistically evaluates SHOULD/MAY architectures.

T6 closes Gate 3.

### T6 — MUST architectures × three behaviours on upgraded substrate

12 cells, same MUST families × behaviours as T4 but on the upgraded substrate:

- [ ] T6.connectome.klinotaxis — strict-mask, Rung 2 + continuous-2D
- [ ] T6.connectome.thermotaxis
- [ ] T6.connectome.predator_evasion
- [ ] T6.mlp_ppo.klinotaxis
- [ ] T6.mlp_ppo.thermotaxis
- [ ] T6.mlp_ppo.predator_evasion
- [ ] T6.lstm_gru_ppo.klinotaxis
- [ ] T6.lstm_gru_ppo.thermotaxis
- [ ] T6.lstm_gru_ppo.predator_evasion
- [ ] T6.neat_weights.klinotaxis
- [ ] T6.neat_weights.thermotaxis
- [ ] T6.neat_weights.predator_evasion

### T6 — SHOULD architectures (opportunistic, not gating)

- [ ] T6.quantum.\* — one quantum row at continuous-physics complexity (Phase 2 baseline reference; SHOULD per [design.md § Decision 4](design.md)).
- [ ] T6.spiking.\* — opportunistic; if PPO-trained spiking doesn't train cleanly on the connectome, document the failure mode and defer to Phase 7 L4 STDP work.

### T6 — MAY architectures (opportunistic, not gating)

- [ ] T6.reservoir.\* — one row if cheap (QRH / CRH; Phase 2 pursuit-advantage rep).
- [ ] T6.hybrid.\* — one row if cheap (Phase 2 SOTA rep).
- [ ] T6.transformer.\* — only if scope and engineering effort allow.

### T6 — real-worm validation

- [ ] T6.validation.1 Choose validation target — chemotaxis indices à la Bargmann / mechanosensation escape latencies / whole-brain Ca²⁺ correlation matrices à la Kavli/Janelia. Selection deliberately deferred to this tranche per [design.md § What This Change Explicitly Does Not Decide](design.md).
- [ ] T6.validation.2 Implement the comparison pipeline: extract the model's analogue of the chosen real-worm metric; document data source + version; record the comparison procedure.
- [ ] T6.validation.3 Run the comparison; report quantitative agreement with confidence intervals. Required for Phase 6 exit (MUST in the roadmap).

### T6 — cross-architecture analysis + env-upgrade delta + logbook

- [ ] T6.analysis.ranking_upgraded — paired-seed Wilcoxon + bootstrap CIs across the four MUST families × three behaviours on the upgraded substrate.
- [ ] T6.analysis.env_delta — head-to-head T4-grid vs T6-upgraded ranking delta. The load-bearing finding that justifies the deliberate T4-vs-T6 split per [design.md § Decision 1](design.md).
- [ ] T6.analysis.connectome_final_ranking — explicit answer to "where does the wild-type connectome (strict-mask) rank under PPO weight search on the upgraded substrate?"
- [ ] T6.logbook — publish T6 logbook (suggested: `docs/experiments/logbooks/0XX-l2-final-pass.md`). Required reading material for Gate 3.

**T6 risk-mitigation pivot (per roadmap)**: if the L2 re-run on the upgraded substrate fails to learn after diagnostic sequencing (topology density / continuous-action head parameterisation / reward shaping), the finding is publishable as "PPO-on-the-real-connectome under continuous-physics requires further substrate work" — Phase 7 L4 plasticity work inherits the substrate-engineering question.

## Gate 3 (closes Tranche 6, month ~6-7 cumulative): L2 results across architectures?

**Trigger**: T6 closed with the cross-architecture ranking on the upgraded substrate in hand.
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § Mid-phase decision gates § Gate 3.
**Decision must be written in**: the T6 OpenSpec change's logbook (likely the T6 logbook itself). This tracker links to the decision once it lands.

- [ ] **Gate 3 decision recorded**: weight-search results across the four MUST families and all three behaviours in hand at the Phase 5 statistical bar, on the upgraded substrate, with real-worm validation shipped (GO to T7) — OR — partial coverage; sub-phase split into Phase 6a (T1–T6) ships first, Phase 6b (T7 NEAT + T8 synthesis) becomes follow-on (PIVOT-scope) — OR — Phase 6 overshoots 10 months despite T6 being technically complete; sub-phase split triggered for delivery reasons (PIVOT-scope) — OR — L2 fails to learn on the upgraded substrate after the diagnostic sequence (STOP / publishable negative result).
- [ ] **Gate 3 decision link**: [add link to the OpenSpec change / logbook where the decision is recorded]

## Tranche 7 — L3 NEAT Topology Search on Upgraded Substrate

**OpenSpec change**: `add-l3-neat-topology-search` (placeholder; not yet created)
**Status**: 🔲 not started (gated on Gate 3 GO, or scoped into Phase 6b under PIVOT)
**Roadmap layer**: L3
**Approx duration**: 6-10 weeks
**Bio fidelity**: LOW (NEAT is an ML tool; the bio-fidelity contribution lives in T1 + the T6 connectome rank)
**Dependencies**: T6 closed
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § The layered platform § L3

T7 ranks the wild-type connectome's topology against NEAT-evolved alternatives on at
least one behaviour, using the lag-matrix instrument (or equivalent discriminative
gate from Phase 5 M5's methodology contributions). T7 is also the natural follow-up
to Phase 5 M5's architecture-asymmetry hypothesis: matched-capacity NEAT-vs-NEAT
co-evolution vs asymmetric NEAT-vs-MLP is a clean falsification test.

Coarse-grained sub-tasks below; the T7 OpenSpec change elaborates them.

- [ ] T7.1 Integrate TensorNEAT (GPU-accelerated NEAT, JAX/vmap; ~500× speedup over neat-python documented in the field).
- [ ] T7.2 NEAT topology + weight evolution on the L1 plugin interface from T2. Plugin should accommodate NEAT-evolved topologies as natively as it accommodates fixed-topology architectures (the topology/rule factoring from T2 pays off here).
- [ ] T7.3 Topology-vs-connectome head-to-head on at least one behaviour on the upgraded substrate (klinotaxis is the natural first).
- [ ] T7.4 Matched-capacity NEAT-vs-NEAT vs asymmetric NEAT-vs-MLP (Phase 5 M5 follow-up). Uses the lag-matrix instrument from Phase 5 logbook 017.
- [ ] T7.5 Cross-architecture analysis: "is the wild-type connectome a local optimum?" Cross-references the T6 connectome ranking with the T7 NEAT-evolved baseline.
- [ ] T7.6 Update this checklist + `docs/roadmap.md` Phase 6 Tranche Tracker T7 row.
- [ ] T7.7 Publish T7 logbook (suggested: `docs/experiments/logbooks/0XX-l3-neat-topology-search.md`).

**T7 risk-mitigation pivot (per roadmap)**: if NEAT-evolved topologies and the wild-type connectome converge to indistinguishable performance, that *is* the finding — "the connectome is competitive with evolved topologies on these behaviours." The optimal-primary framing weakens; the connectome-primary framing strengthens. Acceptable outcome; pivot the headline framing if it lands.

## Tranche 8 — Phase 6 Synthesis Logbook

**OpenSpec change**: `add-phase6-synthesis-logbook` (placeholder; not yet created)
**Status**: 🔲 not started
**Roadmap layer**: synthesis
**Approx duration**: 1-2 weeks
**Dependencies**: T1–T7 closed (or T7 scoped into Phase 6b under PIVOT)
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § Phase 6 exit criteria

- [ ] T8.1 Walk through each Phase 6 exit criterion (the seven MUSTs in the roadmap) with evidence: L0 substrate operational (T1), L1 plugin parity (T2 + verified at T5), L2 results across MUST architectures × three behaviours at the Phase 5 statistical bar (T6), L3 NEAT topology-search results (T7), Rung 2 chemical gradients operational (T5), corrected ASH/ADL nociception operational (T3), ≥ 1 real-worm validation (T6).
- [ ] T8.2 Document the connectome ranking honestly. The roadmap pre-commits to a framing pivot: connectome-primary if connectome wins decisively; optimal-primary if connectome is competitive-but-not-dominant. Whichever the data supports.
- [ ] T8.3 Document negative findings honestly (Phase 5 precedent). If any L2 / L3 cell came back STOP, the diagnosis is itself a publishable contribution.
- [ ] T8.4 Quantify the env-upgrade delta (T4-grid vs T6-upgraded ranking change) as a standalone result. This was a load-bearing reason for the deliberate T4-vs-T6 split per [design.md § Decision 1](design.md); the synthesis should foreground it.
- [ ] T8.5 Phase 7 trigger recommendation: which Phase 7 priorities (L4 plasticity / *P. pacificus* transfer / publication / collaboration) are best-supported by Phase 6 evidence.
- [ ] T8.6 Publish `docs/experiments/logbooks/0XX-phase6-synthesis.md`.
- [ ] T8.7 Update `docs/roadmap.md` Phase 6 status → ✅ COMPLETE; record exit criterion outcomes. Update Phase 6 Tranche Tracker rows to their terminal verdicts.

> Archiving `phase6-tracking` itself is an operator-side step that intentionally does NOT block task completion: `openspec archive` requires every task ticked, so an "archive me" task here would self-block (same precedent as `phase5-tracking/tasks.md` and `add-tei-prior-on-m3/tasks.md`). The archive happens after the synthesis-logbook PR merges to main via `openspec archive phase6-tracking`; the synthesis change archives separately.

## Phase 6 Research Questions

Open research questions surfaced during Phase 6 planning or during in-flight Phase 6
work that don't fit cleanly under any single tranche but are worth tracking. Each
question has a concrete trigger condition for escalation; nothing here commits the
project to work upfront.

### RQ1: Architecture-asymmetry hypothesis (Phase 5 M5 follow-up)

**Status**: open — gated on T7 NEAT integration
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § The layered platform § L3 + Phase 5 logbook 017
**Trigger**: T7 ships matched-capacity NEAT-vs-NEAT vs asymmetric NEAT-vs-MLP head-to-head. If the matched arm produces own-vs-cross lag delta ≤ −0.05 while the asymmetric arm reproduces M5's +0.017 delta, the architecture-asymmetry hypothesis is confirmed. If both arms produce ~+0.017, the hypothesis is falsified (capacity symmetry is irrelevant).
**Recorded by**: this tracker (open question); will be settled inside T7's OpenSpec change.

### RQ2: M6.x substrate-shape diagnosis under continuous physics

**Status**: open — gated on T5 (Rung 2 gradients + continuous-2D physics) shipping
**Roadmap reference**: Phase 5 logbook 019 + logbook 020 (TEI bias-network logit-prior was the wrong abstraction)
**Trigger**: Once continuous-2D + Rung 2 gradients are operational, the substrate carrying any future TEI-style experiment is materially different from M6.x's discrete-grid logit-bias substrate. Whether the upgraded substrate surfaces a Baldwin or TEI signal serendipitously is the open question. If a signal appears, escalate to a dedicated Phase 6+ change scoped explicitly to characterise it; if no signal appears across the T6 sweep, close RQ2 with the substrate-shape diagnosis carried forward to Phase 7.
**Recorded by**: Phase 5 logbook 020 § What's next; tracked here as a Phase 6 watch item.

### RQ3: Connectome-primary vs optimal-primary headline framing

**Status**: open — settles at T8.2
**Roadmap reference**: `docs/roadmap.md` § Executive Summary § Framing note + § Phase 6 Conclusion
**Trigger**: T6 + T7 results in hand. If the connectome wins decisively on the curated behaviours, the headline shifts toward "connectome-primary" (a neuroscience result). If the connectome is competitive-but-not-dominant, "optimal-primary" remains the natural framing. Both are platform contributions; the scientific framing follows the evidence.
**Recorded by**: this tracker; settled in T8.2 by the Phase 6 synthesis logbook.

### RQ4: c302 / NeuroML export path for OpenWorm Sibernetic interop

**Status**: open — Phase 6 Future Directions, not Phase 6 MUST
**Roadmap reference**: [design.md § Decision 2](design.md) (deferred from L0 primary)
**Trigger**: A Phase 6 sub-task or Phase 7 deliverable requires handing the connectome topology to OpenWorm Sibernetic for body-physics validation. If triggered, scope a small `add-c302-export` change; the import-side `cect` choice does not block the export-side c302 work.
**Recorded by**: this tracker (open question); not gating any Phase 6 exit criterion.

### RQ5: Env-upgrade delta — does the substrate upgrade change the architecture ranking?

**Status**: open — settles at T6 (and is the load-bearing finding for the deliberate T4-vs-T6 split per [design.md § Decision 1](design.md))
**Roadmap reference**: implicit in `docs/roadmap.md` § Phase 6 § Architecture-comparison protocol
**Trigger**: T4 and T6 both shipped. The interesting outcomes are (a) ranking is stable across substrate change — substrate complexity is below the threshold for differential architecture advantage; (b) ranking shifts — some architectures (likely temporal / continuous-action-native) benefit disproportionately from Rung 2 + continuous-2D, others (likely connectome under strict-mask) don't. Either outcome is publishable; (b) reinforces the Phase 6 "platform" claim more strongly.
**Recorded by**: this tracker; settled in T8.4 by the Phase 6 synthesis logbook.
