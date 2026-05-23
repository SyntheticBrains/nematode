# Tasks: Phase 6 (Connectome Substrate & Architecture Comparison) Milestone Tracker

This is the living checklist for all of Phase 6. Each layer (L0–L4) and cross-cutting
deliverable has its own OpenSpec change directory listed below as it is created. Each
Phase 6 milestone PR MUST update this file to mark sub-tasks complete as part of its
diff.

**Status legend**: `[ ]` not started, `[x]` complete. Layer-level "in progress" status
lives in the **Status** header of each layer section (matches the roadmap Phase 6
Milestone Tracker emoji column).

**Layer-to-tranche map** (see [design.md § Decision 1](design.md) for rationale):

- **Tranche 1**: L0
- **Tranche 2**: L1 + first L2 attempt (Gate 1 trigger)
- **Tranche 3**: full L2 sweep (Gate 2 + Gate 3 triggers)
- **Tranche 4**: L3 NEAT
- **Tranche 5**: cross-cutting substrate (continuous-2D + Rung 2 gradients + corrected ASH/ADL nociception + ≥1 real-worm validation) — may interleave with Tranches 2-3
- **Tranche 6**: Phase 6 synthesis logbook

## P6-0: Phase 6 Tracking Scaffold (THIS CHANGE)

**Branch**: `feat/phase6-tracking`
**OpenSpec change**: `phase6-tracking` (this directory)
**Status**: in progress

- [x] P6-0.1 Create `openspec/changes/phase6-tracking/proposal.md`
- [x] P6-0.2 Create `openspec/changes/phase6-tracking/design.md`
- [x] P6-0.3 Create `openspec/changes/phase6-tracking/tasks.md` (this file)
- [x] P6-0.4 Create `openspec/changes/phase6-tracking/specs/phase6-tracking/spec.md`
- [x] P6-0.5 Update `docs/roadmap.md` Phase 6 row in Timeline Overview to `🟡 IN PROGRESS`
- [x] P6-0.6 Add "Phase 6 Milestone Tracker" sub-section to `docs/roadmap.md` Phase 6 block
- [ ] P6-0.7 Validate change: `openspec validate phase6-tracking --strict`
- [ ] P6-0.8 Run targeted `uv run pre-commit run --files <changed>` clean
- [ ] P6-0.9 Open PR

## L0: Connectome Substrate

**OpenSpec change**: `add-connectome-substrate` (not yet created — first Phase 6 milestone change)
**Status**: 🔲 not started
**Tranche**: 1
**Bio fidelity**: HIGH (real wiring is the substrate)
**Dependencies**: P6-0 (this scaffold)
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § The layered platform § L0

L0 imports the *C. elegans* 302-neuron connectome (Cook et al. 2019 hermaphrodite via
OpenWorm `cect` per [design.md § Decision 2](design.md)) and validates it against an
independent reference (Witvliet 2021). No plugin work, no training, no behaviour
evaluation in this tranche — the deliverable is "a real connectome loaded, validated,
and forward-passable" plus the data model that L1 will conform to.

- [ ] L0.1 Choose and document the import library (decision pre-recorded: `cect` / ConnectomeToolbox; see [design.md § Decision 2](design.md)). Pin the dependency version; record the upstream dataset identifiers.
- [ ] L0.2 Build the connectome data model (neurons, synapses, gap junctions, neurotransmitter labels, anatomical roles — sensory / interneuron / motor). The data model is the topology interface every architecture plugin will conform to in L1.
- [ ] L0.3 Import the Cook 2019 hermaphrodite connectome. Verify neuron count (302), synapse count (within published bounds), gap-junction count, and neuron-name conformance to standard *C. elegans* naming.
- [ ] L0.4 Cross-validate against Witvliet et al. 2021 (the EM-reconstructed L1-larva-through-adult time series). Document where the two datasets agree, where they diverge (developmental stage + lineage-tracing differences are expected), and which divergences would materially affect downstream learning.
- [ ] L0.5 Document the import provenance: upstream dataset DOI / commit, `cect` version, any transformations applied during import, any neurons / synapses dropped or merged, any metadata gaps.
- [ ] L0.6 Forward-pass smoke test: instantiate a trivial PPO weight set on the connectome topology, run a single forward pass, verify shapes / no NaNs. Not a training run — a "does the substrate respond to inputs" sanity check.
- [ ] L0.7 Unit + smoke tests for the data model and import pipeline; CI integration.
- [ ] L0.8 Update this checklist + `docs/roadmap.md` Phase 6 Milestone Tracker L0 row.
- [ ] L0.9 Publish L0 logbook (suggested: `docs/experiments/logbooks/0XX-connectome-substrate.md`). Required reading material for Gate 1.

**L0 risk-mitigation pivot (per roadmap)**: if `cect` import / dataset access proves harder than expected (format incompatibility, missing metadata, unclear synaptic-weight provenance), drop to a hand-curated subset of the Cook 2019 connectome — sensory-interneuron-motor subgraph for the three target behaviours, ~50-100 neurons. The pivot decision is itself a written gate-style decision; the L0 OpenSpec change documents it.

## Gate 1 (month ~2): L0 import working?

**Trigger**: L0 closed (whether full 302-neuron or hand-curated subset pivot).
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § Mid-phase decision gates § Gate 1.
**Decision must be written in**: the L0 OpenSpec change's logbook or `tasks.md`. This tracker links to the decision once it lands.

- [ ] **Gate 1 decision recorded**: connectome substrate loaded, validated, basic-MLP-PPO baseline trainable on it (GO) — OR — full 302-neuron import infeasible; hand-curated subset pivot triggered with documented subset choice (PIVOT) — OR — both paths infeasible at substrate level (STOP).
- [ ] **Gate 1 decision link**: [add link to the OpenSpec change / logbook where the decision is recorded]

## L1: Architecture-Plugin Interface

**OpenSpec change**: `add-architecture-plugin-interface` (not yet created)
**Status**: 🔲 not started (gated on Gate 1 GO or PIVOT)
**Tranche**: 2
**Bio fidelity**: n/a (interface refactor)
**Dependencies**: L0 closed
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § The layered platform § L1

L1 introduces the clean `Brain` interface every architecture family conforms to. The
parity test — adding a new architecture is ≤ 1 week of work — is the contract. The
first PPO-on-connectome training run lands in this tranche (Tranche 2) because the L1
interface only proves itself by carrying at least one architecture (connectome) and
the strongest classical baseline (MLP-PPO) through end-to-end training.

These sub-tasks are coarse-grained; they will be elaborated when the L1 OpenSpec
change is scoped.

- [ ] L1.1 Audit the existing `Brain` protocol(s) across the 19 Phase 0-3 architectures. Identify per-architecture branches that need to be moved behind a clean plugin boundary.
- [ ] L1.2 Refactor or extend the `Brain` interface so the connectome-constrained architecture (L0 output) and the eight MUST families (see [proposal.md § Decision 4](proposal.md)) all conform without bespoke per-architecture plumbing in the simulation loop.
- [ ] L1.3 Plugin registry — a single place where architecture families are registered + discovered. CLI / config plumbing flows through the registry, not through per-family imports.
- [ ] L1.4 Plugin-parity test: a documented procedure (or CI integration test) that demonstrates adding a hypothetical 9th architecture is ≤ 1 week of work — counting LOC, files touched, tests required.
- [ ] L1.5 First PPO-on-connectome training run: instantiate the connectome architecture through the new plugin interface, train weights with PPO on one target behaviour (likely klinotaxis, since Phase 4's substrate carries forward). This is the first closed-loop learning on the real *C. elegans* connectome — the platform claim.
- [ ] L1.6 Documentation: plugin-developer guide; "how to add a new architecture family in ≤ 1 week" walkthrough.
- [ ] L1.7 Update this checklist + `docs/roadmap.md` Phase 6 Milestone Tracker L1 row.
- [ ] L1.8 Publish L1 logbook (suggested: `docs/experiments/logbooks/0XX-architecture-plugin-interface.md`). Required reading material for Gate 2.

**L1 risk-mitigation pivot (per roadmap)**: if multiple architecture families need bespoke plumbing and the interface accumulates per-architecture branches, pause architecture-sweep work and spend 2-4 weeks refactoring L1 toward genuine plugin parity. Better to delay L2/L3 results than to ship a "platform" that isn't one.

## Gate 2 (month ~4-5): L1 plugin parity achieved?

**Trigger**: L1 closed; plugin-parity test documented.
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § Mid-phase decision gates § Gate 2.
**Decision must be written in**: the L1 OpenSpec change's logbook or `tasks.md`. This tracker links to the decision once it lands.

- [ ] **Gate 2 decision recorded**: adding a new architecture demonstrably ≤ 1 week of work (GO) — OR — interface accumulates per-architecture branches; L1 refactor pivot triggered (PIVOT) — OR — plugin interface fundamentally incompatible with one or more MUST architecture families (STOP).
- [ ] **Gate 2 decision link**: [add link to the OpenSpec change / logbook where the decision is recorded]

## L2: Weight Search (PPO et al.) on Connectome Substrate

**OpenSpec change**: `add-l2-weight-search-sweep` (placeholder name; not yet created)
**Status**: 🔲 not started (gated on Gate 2 GO or PIVOT)
**Tranche**: 3
**Bio fidelity**: depends on architecture family (HIGH for connectome-constrained; lower for others)
**Dependencies**: L0 closed, L1 closed, at least one behaviour from Tranche 5 substrate work ready (klinotaxis is the natural first behaviour; thermotaxis + predator evasion follow as Rung 2 + corrected nociception land)
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § The layered platform § L2 + § Architecture-comparison protocol

L2 is the full weight-search sweep across the MUST architecture-family set (eight
families) × the three Phase 6 behaviours, with paired-seed Wilcoxon tests + bootstrap
CIs + n ≥ 4 seeds per condition (Phase 5 statistical bar carries forward). The
deliverable is the architecture-comparison ranking on weight search alone, before L3
topology search.

Per-architecture-family + per-behaviour rows below are placeholders — each cell is
its own training run; tick as it ships. The eight MUST families and three behaviours
are fixed by [proposal.md § Decisions 4 + 5](proposal.md).

### L2 — Connectome-constrained (focal architecture)

- [ ] L2.connectome.klinotaxis — PPO weight-search on Cook 2019 connectome
- [ ] L2.connectome.thermotaxis
- [ ] L2.connectome.predator_evasion (requires corrected ASH/ADL nociception from Tranche 5)

### L2 — MLP-PPO

- [ ] L2.mlp_ppo.klinotaxis
- [ ] L2.mlp_ppo.thermotaxis
- [ ] L2.mlp_ppo.predator_evasion

### L2 — LSTM / GRU-PPO

- [ ] L2.lstm_gru_ppo.klinotaxis
- [ ] L2.lstm_gru_ppo.thermotaxis
- [ ] L2.lstm_gru_ppo.predator_evasion

### L2 — Spiking (PPO-trained)

- [ ] L2.spiking.klinotaxis
- [ ] L2.spiking.thermotaxis
- [ ] L2.spiking.predator_evasion

### L2 — Reservoir

- [ ] L2.reservoir.klinotaxis
- [ ] L2.reservoir.thermotaxis
- [ ] L2.reservoir.predator_evasion

### L2 — Quantum (Phase 2 representatives)

- [ ] L2.quantum.klinotaxis
- [ ] L2.quantum.thermotaxis
- [ ] L2.quantum.predator_evasion

### L2 — Hybrid quantum-classical

- [ ] L2.hybrid.klinotaxis
- [ ] L2.hybrid.thermotaxis
- [ ] L2.hybrid.predator_evasion

### L2 — NEAT-evolved (weights only at L2; topology evolution lives at L3)

- [ ] L2.neat_weights.klinotaxis
- [ ] L2.neat_weights.thermotaxis
- [ ] L2.neat_weights.predator_evasion

### L2 — MAY: Transformer / attention-based

- [ ] L2.transformer.\* — OPTIONAL; only if scope and engineering effort allow. Not a Phase 6 exit criterion.

### L2 — cross-architecture analysis + logbook

- [ ] L2.analysis.ranking — paired-seed Wilcoxon + bootstrap CIs across all eight MUST families × three behaviours. The architecture-comparison ranking.
- [ ] L2.analysis.connectome_ranking — explicit answer to "where does the wild-type connectome rank under weight search alone?"
- [ ] L2.logbook — publish L2 logbook (suggested: `docs/experiments/logbooks/0XX-l2-weight-search-sweep.md`). Required reading material for Gate 3.

**L2 risk-mitigation pivot (per roadmap)**: if after reasonable hyperparameter search no architecture family reaches Phase 0-3 baselines on any of the three behaviours, run the diagnostic sequence (topology density / continuous action head / reward shaping). If none resolve, the finding is "PPO-on-the-real-connectome requires further substrate work" — itself a publishable negative result and a Phase 7 prerequisite.

## Gate 3 (month ~7-8): L2 results across architectures?

**Trigger**: L2 sweep closed with the cross-architecture ranking in hand.
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § Mid-phase decision gates § Gate 3.
**Decision must be written in**: the L2 OpenSpec change's logbook (likely the L2 logbook itself). This tracker links to the decision once it lands.

- [ ] **Gate 3 decision recorded**: weight-search results across the eight MUST families and all three behaviours in hand at the Phase 5 statistical bar (GO to L3) — OR — partial coverage; sub-phase split into Phase 6a (L0+L1+L2) ships first, Phase 6b (L3) becomes follow-on (PIVOT-scope) — OR — Phase 6 overshoots 10 months despite L2 being technically complete; sub-phase split triggered for delivery reasons (PIVOT-scope) — OR — L2 fails to learn on the connectome after the diagnostic sequence (STOP / publishable negative result).
- [ ] **Gate 3 decision link**: [add link to the OpenSpec change / logbook where the decision is recorded]

## L3: Topology Search (NEAT) on Connectome Substrate

**OpenSpec change**: `add-l3-neat-topology-search` (placeholder name; not yet created)
**Status**: 🔲 not started (gated on Gate 3 GO, or scoped into Phase 6b under PIVOT)
**Tranche**: 4
**Bio fidelity**: LOW (NEAT is an ML tool; the bio-fidelity contribution lives in L0 + the L2 connectome rank)
**Dependencies**: L0 closed, L1 closed, L2 closed
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § The layered platform § L3

L3 ranks the wild-type connectome's topology against NEAT-evolved alternatives on at
least one behaviour, using the lag-matrix instrument (or equivalent discriminative
gate from Phase 5 M5's methodology contributions). L3 is also the natural follow-up
to Phase 5 M5's architecture-asymmetry hypothesis: matched-capacity NEAT-vs-NEAT
co-evolution vs asymmetric NEAT-vs-MLP is a clean falsification test.

Coarse-grained sub-tasks below; will be elaborated in the L3 OpenSpec change.

- [ ] L3.1 Integrate TensorNEAT (GPU-accelerated NEAT, JAX/vmap; ~500× speedup over neat-python documented in the field).
- [ ] L3.2 NEAT topology + weight evolution on the architecture-plugin interface from L1. Plugin should accommodate NEAT-evolved topologies as natively as it accommodates fixed-topology architectures.
- [ ] L3.3 Topology-vs-connectome head-to-head on at least one behaviour (klinotaxis is the natural first).
- [ ] L3.4 Matched-capacity NEAT-vs-NEAT vs asymmetric NEAT-vs-MLP (Phase 5 M5 follow-up). Uses the lag-matrix instrument from logbook 017.
- [ ] L3.5 Cross-architecture analysis: "is the wild-type connectome a local optimum?" Cross-references the L2 connectome ranking with the L3 NEAT-evolved baseline.
- [ ] L3.6 Update this checklist + `docs/roadmap.md` Phase 6 Milestone Tracker L3 row.
- [ ] L3.7 Publish L3 logbook (suggested: `docs/experiments/logbooks/0XX-l3-neat-topology-search.md`).

**L3 risk-mitigation pivot (per roadmap)**: if NEAT-evolved topologies and the wild-type connectome converge to indistinguishable performance, that *is* the finding — "the connectome is competitive with evolved topologies on these behaviours." The optimal-primary framing weakens; the connectome-primary framing strengthens. Acceptable outcome; pivot the headline framing if it lands.

## Tranche 5: Cross-cutting substrate work

Sub-tasks below may interleave with Tranches 2-3. Each is a roadmap MUST exit
criterion. They are listed together here for visibility, not because they ship as one
change — most will be standalone OpenSpec changes scoped to their own scope.

### Continuous 2D physics + continuous action space

**OpenSpec change**: `add-continuous-2d-physics` (placeholder; not yet created)
**Status**: 🔲 not started
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § Continuous environment + sensory physics

- [ ] T5-physics.1 Continuous 2D coordinate system + continuous action space (speed 0-to-max + turning angle −π to π). Realistic spatial scales: ~1mm worm body on cm-scale plates.
- [ ] T5-physics.2 Extend existing PPO-family brains with continuous action heads (Gaussian policy).
- [ ] T5-physics.3 Adapt quantum architectures with continuous-output circuits.
- [ ] T5-physics.4 Cross-tranche dependency: L2 weight-search uses the continuous-2D substrate; L3 NEAT topology search uses the continuous-2D substrate.

### Rung 2 chemical gradients (dynamic Fick's-law + log-concentration adaptation)

**OpenSpec change**: `add-rung2-chemical-gradients` (placeholder; not yet created)
**Status**: 🔲 not started
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § Continuous environment + sensory physics § Chemical-gradient fidelity

Rung 2 has **two coupled components** — environment dynamics AND chemosensory
adaptation kinetics. They MUST be designed together (per roadmap: "without
log-concentration adaptation on the sensory side, the gradient realism is wasted").

- [ ] T5-gradients.1 Heat-equation diffusion (∂C/∂t = D∇²C) with signal-type-specific D values (food vs pheromone vs CO₂).
- [ ] T5-gradients.2 Source dynamics — depletion when worms feed; source replenishment; decay terms for short-lived signals.
- [ ] T5-gradients.3 Log-concentration chemosensory adaptation kinetics on AWC/AWA/ASE-style sensors. Coupled component; ships with T5-gradients.1+.2.
- [ ] T5-gradients.4 Cross-tranche dependency: L2 klinotaxis + thermotaxis evaluations use Rung 2 gradients.

### Corrected ASH/ADL contact-based nociception

**OpenSpec change**: `fix-ash-adl-contact-nociception` (placeholder; not yet created)
**Status**: 🔲 not started
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § Continuous environment + sensory physics § Mechanosensation

Real *C. elegans* nociception is contact-based mechanosensation (ASH/ADL neurons),
not chemosensory at distance. Phase 4's Logbook 011 flagged this; Phase 6 fixes it.

- [ ] T5-nociception.1 Implement ASH/ADL contact-based mechanosensation with realistic sensory ranges scaled to worm body length.
- [ ] T5-nociception.2 Cross-tranche dependency: L2 predator-evasion evaluations require this. Without it, the L2 predator-evasion row is gated.
- [ ] T5-nociception.3 Document the corrected model against the existing (incorrect) Phase 4 model; quantify the behavioural difference under matched conditions.

### Built-in real-worm validation

**OpenSpec change**: `add-real-worm-validation` (placeholder; not yet created — may live inside L2's or a specific behaviour's change rather than standalone)
**Status**: 🔲 not started
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § Built-in real-worm validation

Phase 6 validates *at least one* model output quantitatively against published
real-worm data, as a Phase 6 exit criterion. Three concrete candidates:

- [ ] T5-validation.1 Choose validation target (chemotaxis indices à la Bargmann / escape latencies / whole-brain Ca²⁺ correlation matrices) and record the choice with rationale.
- [ ] T5-validation.2 Implement the comparison pipeline: extract the model's analogue of the chosen real-worm metric; document data source + version; record the comparison procedure.
- [ ] T5-validation.3 Run the comparison; report the quantitative agreement with confidence intervals. Required for Phase 6 exit (MUST in the roadmap).

## Tranche 6: Phase 6 Synthesis Logbook

**OpenSpec change**: `add-phase6-synthesis-logbook` (placeholder; not yet created)
**Status**: 🔲 not started
**Dependencies**: L0 + L1 + L2 closed; Tranche 5 substrate work closed; at least one Phase 6 exit criterion (real-worm validation) closed; L3 closed or scoped into Phase 6b under PIVOT.
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § Phase 6 exit criteria

- [ ] T6.1 Walk through each Phase 6 exit criterion (the seven MUSTs in the roadmap) with evidence: L0 substrate operational, L1 plugin parity, L2 results across MUST architectures × three behaviours at the Phase 5 statistical bar, L3 NEAT topology search results, Rung 2 chemical gradients operational, corrected ASH/ADL nociception operational, ≥ 1 real-worm validation.
- [ ] T6.2 Document the connectome ranking honestly. The roadmap pre-commits to a framing pivot: connectome-primary if connectome wins decisively; optimal-primary if connectome is competitive-but-not-dominant. Whichever the data supports.
- [ ] T6.3 Document negative findings honestly (Phase 5 precedent). If any L2 / L3 cell came back STOP, the diagnosis is itself a publishable contribution.
- [ ] T6.4 Phase 7 trigger recommendation: which Phase 7 priorities (L4 plasticity / *P. pacificus* transfer / publication / collaboration) are best-supported by Phase 6 evidence.
- [ ] T6.5 Publish `docs/experiments/logbooks/0XX-phase6-synthesis.md`.
- [ ] T6.6 Update `docs/roadmap.md` Phase 6 status → ✅ COMPLETE; record exit criterion outcomes. Update Phase 6 Milestone Tracker rows to their terminal verdicts.

> Archiving `phase6-tracking` itself is an operator-side step that intentionally does NOT block task completion: `openspec archive` requires every task ticked, so an "archive me" task here would self-block (same precedent as `phase5-tracking/tasks.md` and `add-tei-prior-on-m3/tasks.md`). The archive happens after the synthesis-logbook PR merges to main via `openspec archive phase6-tracking`; the synthesis change archives separately.

## Phase 6 Research Questions

Open research questions surfaced during Phase 6 planning or during in-flight Phase 6
work that don't fit cleanly under any single layer but are worth tracking. Each
question has a concrete trigger condition for escalation; nothing here commits the
project to work upfront.

### RQ1: Architecture-asymmetry hypothesis (Phase 5 M5 follow-up)

**Status**: open — gated on L3 NEAT integration
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § The layered platform § L3 + Phase 5 logbook 017
**Trigger**: L3 ships matched-capacity NEAT-vs-NEAT vs asymmetric NEAT-vs-MLP head-to-head. If the matched arm produces own-vs-cross lag delta ≤ −0.05 while the asymmetric arm reproduces M5's +0.017 delta, the architecture-asymmetry hypothesis is confirmed. If both arms produce ~+0.017, the hypothesis is falsified (capacity symmetry is irrelevant).
**Recorded by**: this tracker (open question); will be settled inside L3's OpenSpec change.

### RQ2: M6.x substrate-shape diagnosis under continuous physics

**Status**: open — gated on Rung 2 gradients + continuous-2D physics shipping
**Roadmap reference**: Phase 5 logbook 019 + logbook 020 (TEI bias-network logit-prior was the wrong abstraction)
**Trigger**: Once continuous-2D + Rung 2 gradients are operational, the substrate carrying any future TEI-style experiment is materially different from M6.x's discrete-grid logit-bias substrate. Whether a Phase 6 quantum re-evaluation arc surfaces a Baldwin or TEI signal serendipitously is the open question. If a signal appears, escalate to a dedicated Phase 6+ change scoped explicitly to characterise it; if no signal appears across the L2 sweep, close RQ2 with the substrate-shape diagnosis carried forward to Phase 7.
**Recorded by**: Phase 5 logbook 020 § What's next; tracked here as a Phase 6 watch item.

### RQ3: Connectome-primary vs optimal-primary headline framing

**Status**: open — settles at T6.2
**Roadmap reference**: `docs/roadmap.md` § Executive Summary § Framing note + § Phase 6 Conclusion
**Trigger**: L2 + L3 results in hand. If the connectome wins decisively on the curated behaviours, the headline shifts toward "connectome-primary" (a neuroscience result). If the connectome is competitive-but-not-dominant, "optimal-primary" remains the natural framing. Both are platform contributions; the scientific framing follows the evidence.
**Recorded by**: this tracker; settled in T6.2 by the Phase 6 synthesis logbook.

### RQ4: c302 / NeuroML export path for OpenWorm Sibernetic interop

**Status**: open — Phase 6 Future Directions, not Phase 6 MUST
**Roadmap reference**: [design.md § Decision 2](design.md) (deferred from L0 primary)
**Trigger**: A Phase 6 sub-task or Phase 7 deliverable requires handing the connectome topology to OpenWorm Sibernetic for body-physics validation. If triggered, scope a small `add-c302-export` change; the import-side `cect` choice does not block the export-side c302 work.
**Recorded by**: this tracker (open question); not gating any Phase 6 exit criterion.
