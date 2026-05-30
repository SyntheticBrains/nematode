# Tasks: Phase 6 (Connectome Substrate & Architecture Comparison) Tranche Tracker

This is the living checklist for all of Phase 6. Each tranche (T1–T9) has its own
OpenSpec change directory listed below as it is created. Each Phase 6 milestone PR
MUST update this file to mark sub-tasks complete as part of its diff.

**Status legend**: `[ ]` not started, `[x]` complete. Tranche-level "in progress"
status lives in the **Status** header of each tranche section (matches the roadmap
Phase 6 Tranche Tracker emoji column).

## Phase 6 Tranche Map

Phase 6 has nine tranches with deliberate ordering. See
[design.md § Decision 1](design.md) for the load-bearing rationale (why T3 precedes
T4; why T5 + T6 split the env-upgrade work into platform refactor vs. env fidelity;
why T7 holds real-worm validation; why T8 runs against the upgraded substrate).

| Tranche | Scope | Roadmap layer | Approx duration | Gate trigger |
|---|---|---|---|---|
| 1 | L0 connectome ingest — Cook 2019 via OpenWorm `cect`, vendored data, cross-validated against Witvliet 2021, smoke-test forward pass, no env wiring | L0 | 2-3 weeks | — |
| 2 | L1 plugin refactor + connectome-as-brain wired through existing grid env | L1 | 3-5 weeks | **Gate 1** — basic PPO-on-connectome trainable on existing grid (see Gate 1 § below for quantitative criteria) |
| 3 | Corrected ASH/ADL contact-based nociception (owed correctness work per Logbook 011) | env-correctness | 1-2 weeks | — |
| 4 | L2 initial pass — MUST architectures × 3 behaviours, grid-world substrate, corrected nociception | L2 (first pass) | 4-6 weeks | — |
| 5 | Platform refactor — continuous-2D coordinates + continuous-action heads on existing MUST brains; plugin-parity verification | env-upgrade (platform) | 3-4 weeks | **Gate 2** — L1 plugin parity primary checks: ≤ 6 files touched + no per-architecture branches when adding a new architecture family during this work; engineer-hours documented but not load-bearing |
| 6 | Env fidelity — Rung 2 dynamic Fick's-law diffusion + log-concentration chemosensory adaptation kinetics | env-upgrade (fidelity) | 3-4 weeks | — |
| 7 | L2 re-run on fully-upgraded substrate; real-worm validation; SHOULD/MAY architectures evaluated opportunistically | L2 (final) | 4-6 weeks | **Gate 3** — L2 results across MUST set in hand (see Gate 3 § below for quantitative criteria) |
| 8 | L3 NEAT topology search on upgraded substrate | L3 | 6-10 weeks | — |
| 9 | Phase 6 synthesis logbook | — | 1-2 weeks | — |

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
- [x] P6-0.8 Annotate `docs/roadmap.md` Phase 6 layered-platform table with the env-upgrade tranche note
- [x] P6-0.9 Validate change: `openspec validate phase6-tracking --strict`
- [x] P6-0.10 Run targeted `uv run pre-commit run --files <changed>` clean
- [x] P6-0.11 Open PR

## Tranche 1 — L0 Connectome Substrate

**OpenSpec change**: `add-connectome-substrate` (complete; ready for archive post-merge)
**Status**: ✅ complete
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
to. Per [design.md § Decision 7](design.md), the data model MUST expose chemical
synapses (directed, weighted) and gap junctions (undirected, electrical) as
separately-typed connections.

- [x] T1.1 Choose and document the import library (revised from pre-recorded `cect`/ConnectomeToolbox to direct *Nature* SI parsing with pandas + openpyxl; cect investigation surfaced licence inconsistency and pre-1.0 maturity risks — see `add-connectome-substrate/design.md` § Decision T1.1 for rationale). Cook 2019 SI 5 + Witvliet 2020/2021 dataset 8 chosen as the canonical sub-files; vendored via cect's MIT-licensed mirror.
- [x] T1.2 Build the connectome data model. MUST expose: chemical synapses (directed, weighted by synapse count); gap junctions (undirected, electrical, fixed-weight per Decision 7); neurons (with anatomical roles — sensory / interneuron / motor — and neurotransmitter labels). Extra-synaptic/peptidergic signalling is explicitly out of scope for Phase 6 (Decision 7 reserves that for Phase 7 L4 plasticity). The data model is the topology interface every architecture plugin will conform to in T2.
- [x] T1.3 Import the Cook 2019 hermaphrodite connectome. Verify neuron count (302 ✓), chemical-synapse count 3709 (loose lower-bound > 3000; project-docs cite ~7000 but include muscles/glia/end-organs that the loader filters out), gap-junction count 1093 after merging Cook 2019's symmetric + asymmetric sheets, and neuron-name conformance via canonical-name unpadding (VC01 → VC1, etc.).
- [x] T1.4 Cross-validate against Witvliet et al. 2021 nerve-ring subset. Observed: 180 shared neurons, 1271 agreement / 1942 disagreement on chemical-synapse presence, weight ratio mean 0.86 (median 0.57). The disagreement is dominated by Cook-only pairs (Cook covers whole-animal; Witvliet covers only nerve-ring); the divergence map is captured in `DivergenceReport` and surfaced in the T1 logbook.
- [x] T1.5 Vendor the connectome data (no network access at training time). Cook 2019 SI 5 + Witvliet 2020/2021 dataset 8 vendored under `data/connectome/` with LFS-tracking. `PROVENANCE.md` records per-file SHA256, source URLs, DOIs, paper citations, and the cect-mirror redistribution rationale.
- [x] T1.6 Forward-pass smoke test: instantiate a trivial PPO-shaped weight set on the connectome topology (chemical synapses strict-masked + fixed gap-junction weights with fan-in normalisation), run a single forward pass, verify finite output + non-degenerate variance. Verified end-to-end via `uv run python -m quantumnematode.connectome.smoke`: 116 motor outputs, variance 0.69, PASS.
- [x] T1.7 Unit + smoke tests for the data model and import pipeline; CI integration. 71 tests across 5 files (`test_loader.py`, `test_model.py`, `test_neurons.py`, `test_validate.py`, `test_smoke.py`); all pass in the default `uv run pytest -m "not nightly"` tier.
- [x] T1.8 T1↔T2 handshake: T1 publishes a signature-level data-model API sketch as part of the T1 logbook. Published in [logbook 022 § T1↔T2 API sketch](../../../docs/experiments/logbooks/022-connectome-substrate.md). T2's design review will validate against this sketch.
- [x] T1.9 Update this checklist + `docs/roadmap.md` Phase 6 Tranche Tracker T1 row. Done.
- [x] T1.10 Publish T1 logbook at [docs/experiments/logbooks/022-connectome-substrate.md](../../../docs/experiments/logbooks/022-connectome-substrate.md). Feeds into Gate 1's evidence base (full Gate 1 decision lands at T2 close).

**T1 risk-mitigation pivot (per roadmap)**: if `cect` import / dataset access proves harder than expected (format incompatibility, missing metadata, unclear synaptic-weight provenance), drop to a hand-curated subset of the Cook 2019 connectome — sensory-interneuron-motor subgraph for the three target behaviours, ~50-100 neurons. The pivot decision is itself a written gate-style decision; the T1 OpenSpec change documents it.

## Tranche 2 — L1 Plugin Refactor + Connectome-as-Brain on Existing Grid

**OpenSpec change**: `add-architecture-plugin-interface`
**Status**: ✅ complete — Gate 1 GO (logbook 023)
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

- [x] T2.1 Audit the existing `setup_brain_model()` dispatcher at `packages/quantum-nematode/quantumnematode/utils/brain_factory.py` (459 LOC, 19 elif branches) and the `BrainType` enum. Identify every per-architecture branch and every file touched when a new architecture is added (currently includes the enum, dispatcher, per-family config class, YAML loader, and tests).
- [x] T2.2 Refactor the dispatcher into a registry pattern. The exact pattern (decorator-registration vs entry-points vs config-driven) is deliberately deferred to this tranche per [design.md § What This Change Explicitly Does Not Decide](design.md). *Shipped: decorator-registration in `brain/arch/_registry.py`.*
- [x] T2.3 Factor *topology* out from *learning rule* in the registry-registered architectures. The L0 connectome data model is the new topology API; PPO, spiking, and NEAT-evolved-topology+PPO must be able to consume it without bespoke wiring. *Shipped as forward-compat scaffolding (`BrainTopology` + `LearningRule` Protocols) consumed by `ConnectomePPOBrain`. The legacy 19 brains keep their fused `(topology, rule)` `__init__` per the implementation-time scope decision (deferred to a follow-up change).*
- [x] T2.4 Migrate the existing 19 architectures behind the new registry. The migration is mechanical for the 17 already-built MUST/SHOULD/MAY families; LSTMPPO + MLPPPO go first as the architectures that will carry the first L2 runs.
- [x] T2.5 Migration regression bar (per [design.md § Decision 3](design.md)). Phase 5 M1's PredatorBrain refactor precedent: documented byte-equivalence (or seeded-RNG-noise-tolerance, declared up-front) on at least one smoke config per migrated architecture, pre- and post-refactor. MLPPPO + LSTMPPO MUST hit byte-equivalence (these are the gate-1 + L2-first-pass workhorses); the other 17 may declare a noise tolerance if byte-equivalence is impractical. *Shipped: in-process two-construct equivalence tests for MLPPPO + LSTMPPO; registration-only migration on the other 17 (no executing code changes — byte-equivalence trivially preserved).*
- [x] T2.6 Wire the connectome-as-brain through the existing grid env via the new plugin interface. Train weights with PPO on klinotaxis (Phase 4's substrate carries forward; chemical-synapse strict-mask per [design.md § Decision 7](design.md)). This is the first closed-loop learning on the real *C. elegans* connectome — the platform claim, on the existing grid (the upgraded-substrate version lands in T7).
- [x] T2.7 Plugin-parity test (methodology). Document the procedure (or CI integration test) for measuring "how long to add a new architecture family" — primary metrics are files-touched count + no-per-architecture-branches code-review verdict (Gate 2 G2.b + G2.c). Engineer-hours are recorded for future reference but are not a load-bearing pass criterion per Gate 2 G2.a. T2 establishes the methodology and runs the test once on a hypothetical addition; T5 re-runs it against the platform-refactor work as the real-world verification. *Methodology documented in `docs/architecture/plugin-developer-guide.md`; baseline run against the `ConnectomePPOBrain` migration touched 5 files (new module + dtypes + `__init__` + config_loader + brain_factory) — under the ≤ 6 budget.*
- [x] T2.8 Documentation: plugin-developer guide; "how to add a new architecture family — files to touch, no-branches discipline" walkthrough. *Shipped: `docs/architecture/plugin-developer-guide.md`.*
- [x] T2.9 Update this checklist + `docs/roadmap.md` Phase 6 Tranche Tracker T2 row.
- [x] T2.10 Publish T2 logbook at [docs/experiments/logbooks/023-architecture-plugin-interface.md](../../../docs/experiments/logbooks/023-architecture-plugin-interface.md). Required reading material for Gate 1.

**T2 risk-mitigation pivot (per roadmap)**: if multiple architecture families need bespoke plumbing and the interface accumulates per-architecture branches, pause architecture-sweep work and spend 2-4 weeks refactoring L1 toward genuine plugin parity. Better to delay L2/L3 results than to ship a "platform" that isn't one.

## Gate 1 (closes Tranche 2, month ~2-3 cumulative): L0 import working on the L1 plugin interface?

**Trigger**: T2 closed; connectome-as-brain trainable via the new plugin interface on the existing grid env.
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § Mid-phase decision gates § Gate 1.
**Quantitative pass criteria**: see [design.md § Decision 6 § Gate 1](design.md) for the full criterion set (G1.a connectome loaded + cross-validation shipped; G1.b plugin registry instantiates both MLP-PPO and connectome through the same code path; G1.c PPO-on-connectome trains ≥ 100 episodes without NaNs, mean return over last 25 episodes ≥ frozen-random-weights forward-pass control by ≥ 10% AND monotonic improvement first-25 → last-25; G1.d migration regression byte-equivalent for MLPPPO + LSTMPPO only — connectome brain establishes own baseline in T2.6 and is evaluated against G1.c).
**Decision must be written in**: the T2 OpenSpec change's published logbook (NOT `tasks.md`, which becomes hard to amend post-archive per Decision 6). This tracker links to the decision once it lands.

- [x] **Gate 1 decision recorded**: **GO** (2026-05-24). G1.a connectome substrate loaded + cross-validated ✓ (logbook 022). G1.b L1 plugin registry instantiates connectome + MLPPPO via the same code path ✓ (no per-arch branches in `scripts/run_simulation.py`). G1.c PPO-on-connectome on klinotaxis trains 500 episodes without NaN/Inf AND beats frozen-random-weights forward-pass control by 16.1× on last-25 mean reward AND monotonic improvement first-25 → last-25 (76% → 100%) ✓ (R2b reference run, `entropy_coef=0.005`). G1.d migration regression byte-equivalent for MLPPPO + LSTMPPO ✓ (in-process two-construct equivalence tests). Connectome architecture within 6 points of MLPPPO + LSTMPPO klinotaxis baselines on same task / env / seed. Caveat: constant `entropy_coef=0.02` triggers late-training drift; an entropy decay schedule or `entropy_coef=0.005` is the documented default for any production use.
- [x] **Gate 1 decision link**: [docs/experiments/logbooks/023-architecture-plugin-interface.md § Gate 1 decision](../../../docs/experiments/logbooks/023-architecture-plugin-interface.md#gate-1-decision)

## Tranche 3 — Corrected ASH/ADL Contact-Based Nociception

**OpenSpec change**: `fix-predator-sensing-biology` (scope expanded from the original "fix-ash-adl-contact-nociception" placeholder after biology research surfaced four substantive corrections — ADL is not a touch sensor; Liu et al. 2018 distal chemosensation is real; anterior/posterior matters; ASH is graded + adapts. See change's design.md for full decision history.)
**Status**: 🟡 in progress
**Roadmap layer**: env-correctness (precedes L2 first pass per [design.md § Decision 1](design.md))
**Approx duration**: 1-2 weeks
**Bio fidelity**: HIGH (corrects an existing biologically-wrong model)
**Dependencies**: T2 closed (Gate 1 GO or PIVOT)
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § Continuous environment + sensory physics § Mechanosensation

Real *C. elegans* predator detection is a two-channel signal: contact-mechanosensory
(ASH / ALM / AVM / PVD / PLM, anterior vs posterior receptive fields) plus distal-
chemosensory (ASH + ASI sulfolipid signal per Liu et al. 2018 *Nat. Commun.*).
Phase 4's Logbook 011 flagged the chemosensory-at-distance bug originally; T3
ships the full corrected two-channel biology. T3 precedes the L2 first pass (T4)
so the L2 predator-evasion cells run against the corrected model from the start.

- [x] T3.1 Implement contact-based mechanosensation + distal-chemosensation as separate channels with biologically faithful zone discrimination (anterior/posterior/lateral) and graded contact intensity. *Shipped via the fix-predator-sensing-biology OpenSpec change — see the change's tasks.md for the 10 sub-task breakdown.*
- [x] T3.2 Document the corrected model against the existing (incorrect) Phase 4 model; quantify the behavioural difference under matched conditions on the existing grid env. *100-episode head-to-head smoke evaluation results in `tmp/evaluations/predator-sensing-biology-smoke/` and design.md § Modelling caveat 6: new biology learns substantially slower than legacy `nociception_klinotaxis` at matched 100-ep budget (MLPPPO 3% vs 51% success; LSTMPPO 0% vs 7%). Pipeline correctness confirmed; convergence-rate investigation carried forward to T4 as T4.0g.*
- [x] T3.3 Update existing predator-evasion configs / smoke tests to consume the corrected sensor; deprecate the chemosensory-at-distance code path or feature-flag it for archaeology only. *Two new sample configs ship under `configs/scenarios/pursuit/` (mlpppo + lstmppo with the new modules). The legacy `nociception_*` modules are kept frozen-in-place per design.md § Decision T3.2 (22 archived Phase 5 evolution configs continue to load byte-identical — guarded by `test_legacy_nociception_configs_load.py` regression).*
- [x] T3.4 Unit + smoke tests for the corrected sensor; CI integration. *Shipped: 38 ContactZone-discrimination tests + 24 sensor-module extraction tests + 45 legacy-config regression tests. Full test suite runs via the existing pytest discovery.*
- [x] T3.5 Update this checklist + `docs/roadmap.md` Phase 6 Tranche Tracker T3 row. *This commit + the roadmap row flip.*
- [x] T3.6 Brief T3 logbook entry (may live as a section in `docs/experiments/logbooks/0XX-l2-first-pass.md` rather than its own logbook — the fix is small, the verification belongs alongside the L2 cells that depend on it). *Stub logbook section authored within the fix-predator-sensing-biology change's task §10 — full section moves into the T4 L2 first-pass logbook when T4 lands.*

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
corrected nociception (T3), using **strict-mask on chemical synapses with fixed gap
junctions** for the connectome rows ([design.md § Decision 7](design.md)). Phase 5
statistical bar carries forward (paired-seed Wilcoxon + bootstrap CIs + n ≥ 4 seeds
per condition; per-cell n may rise to 8 per [design.md § What This Change Does Not
Decide](design.md) if T4 planning sees the M4.5-style variance argument).

T4 produces a publishable intermediate result — connectome-on-grid is directly
comparable to Phase 5's grid-world baseline. The env-upgrade delta lands at T7.

12 MUST cells (4 families × 3 behaviours). Each cell is its own training run; tick as
it ships.

### T4 — planning sub-tasks

**Carry-forward from T2 + T3** (logbooks [023](../../../docs/experiments/logbooks/023-architecture-plugin-interface.md) + T3's section in [the L2 logbook stub](../../../docs/experiments/logbooks/)): the T2 + T3 evaluation runs surfaced four practical findings that T4 must consume — see T4.0d, T4.0e, T4.0f, T4.0g below.

- [ ] T4.0a Per-cell wall-time estimate. After T2's first PPO-on-connectome training run, extrapolate to a wall-time estimate per cell; revise Decision 1's "4-6 weeks" if the projection diverges by > 2× and amend `design.md` per the Decision-1 amendment mechanism.
- [ ] T4.0b Per-cell seed-count decision. Phase 5 inherited floor is n ≥ 4; rationale per cell if n > 4 (M4.5 precedent — n = 8 if SE on the chosen primary metric is > 0.5× the gate threshold for that metric at n=4).
- [ ] T4.0c Sensor-projection + motor-readout ablation choice for the connectome-constrained family (per Decision 7: these are T4-scope ablations inside the connectome row, not separate MUST families). Document the chosen sensor→sensory-neuron mapping and motor-output readout in the T4 OpenSpec change's design.md.
- [ ] T4.0d **Connectome PPO entropy schedule.** T2's R2 → R2b empirical evidence (logbook [023](../../../docs/experiments/logbooks/023-architecture-plugin-interface.md)): constant `entropy_coef=0.02` triggers late-training drift on the connectome architecture (success collapses from sustained 100% to 52% by ep 475-499); `entropy_coef=0.005` eliminates the drift. T4's `T4.connectome.*` cells MUST pick either a documented constant `entropy_coef ≤ 0.005` or an entropy decay schedule. Document the choice in the T4 OpenSpec change's design.md alongside T4.0c. The same scrutiny may apply to other architectures running klinotaxis-mode sensing — note in T4's per-architecture configs whether an entropy decay is needed.
- [ ] T4.0e **Promote connectome klinotaxis low-entropy variant to canonical.** Three klinotaxis configs ship from T2: `connectomeppo_small_klinotaxis.yml` (`entropy_coef=0.02`, has the drift), `connectomeppo_small_low_entropy_klinotaxis.yml` (`entropy_coef=0.005`, the R2b reference run), `connectomeppo_small_frozen_control_klinotaxis.yml`. The low-entropy variant strictly outperforms; before the T4 sweep starts, swap the canonical config's `entropy_coef` to 0.005 and drop the `_low_entropy_` variant (or rename the low-entropy variant in place and drop the high-entropy one). The frozen-control config keeps `entropy_coef=0.02` (irrelevant under `freeze_updates: true`) but should be updated to match canonical for diff cleanliness.
- [ ] T4.0f **Acknowledge `mlpppo_small_klinotaxis.yml` as a new canonical baseline.** T2 inferred the first MLPPPO klinotaxis foraging config (`entropy_coef=0.03`, `max_steps=1000`, `sensory_modules: [food_chemotaxis, proprioception]`); validated at 98.4% success first-try. T4's `T4.mlp_ppo.klinotaxis` row consumes this directly. The config header documents its provenance ("inferred at T2 from `mlpppo_small_oracle.yml` PPO hyperparams + `lstmppo_small_klinotaxis.yml` env settings"); no further changes needed unless T4.0b's seed-count decision raises n.
- [ ] T4.0g **Investigate new-biology predator-sensing convergence-rate gap.** T3 Section 8 smoke evaluation surfaced that the corrected two-channel predator sensors (`predator_mechanosensation_klinotaxis` + `predator_chemosensation_klinotaxis`) learn substantially slower than the legacy single-channel `nociception_klinotaxis` at matched 100-episode training budgets on pursuit predators: MLPPPO 3% vs 51% success; LSTMPPO 0% vs 7%. Two candidate root causes flagged in [fix-predator-sensing-biology design.md § Modelling caveat 6](../fix-predator-sensing-biology/design.md): (a) sparse contact-mechano signal (`predator_contact_intensity` is 0.0 outside damage radius — most steps), and (b) information redundancy across the two channels (6 input dims for the predator-position information the legacy encodes in 3). T4 must: (i) run new-biology predator-evasion cells at the same compute budget as legacy and quantify the gap on the canonical T4 evaluation metrics; (ii) ablate the sparse-signal hypothesis by trying a variant that puts the distal sulfolipid concentration into the mechanosensation strength field when not in contact (acts as a "predator-proximity-with-zone" signal); (iii) decide whether the convergence-rate gap is acceptable (the corrected biology *is* a harder learning problem on the same env, and that's itself a substrate finding worth recording) or whether the sensor encoding needs revisiting before T7. Coupled with the existing reward-shape ablation (gradient_proximity vs distal-chemo + contact-damage) inside each `T4.*.predator_evasion` cell. Out-of-scope clarification: this is *not* a T3 bug; T3's job was to ship the corrected biology surface, and it does — the surface is well-tested, the env-side semantics are correct, and the brain pipeline consumes the new channels cleanly. T4 owns the empirical evaluation under matched compute. *Resolved by the `weight-search-architecture-ranking` change's Phase 0 (merged 2026-05-28): root cause was a silent `predator_lateral_gradient` gating bug (single-knob check on `nociception_mode` only), not a substrate finding. Post-fix the canonical two-channel sensors + new `distal_chemo_contact_trigger` reward beat legacy by +14pp at n=4 × 500ep on MLPPPO. Forensics: [supporting/025-weight-search-architecture-ranking/phase-0/](../../../docs/experiments/logbooks/supporting/025-weight-search-architecture-ranking/phase-0/). Checkbox tick batched with the other T4.0 ticks at Phase 5 closeout per the change's Task 6.8.*

### T4 — cell structure (integrated-C3 pattern; amended 2026-05-30 by `weight-search-architecture-ranking`)

**Amended from the original 4-families × 3-behaviours (12-cell) pattern to the integrated-C3 pattern** by the [`weight-search-architecture-ranking`](../weight-search-architecture-ranking/design.md) change (per its design.md Decision 8 + the user's "all behaviours in one config" decision). Each architecture runs ONE integrated C3 cell — food + predator + thermotaxis active **simultaneously** in one simulation, n ≥ 4 seeds — preceded by cheap C1 (foraging-only) + C2 (foraging+predator) curriculum smokes (n=1) for per-architecture de-risking. Per-behaviour-component sub-metrics (foraging success / predator survival / thermotaxis isotherm-tracking) are **extracted** from the integrated C3 runs per the `architecture-comparison-protocol` capability — NOT run as separate per-behaviour cells. The n ≥ 4 seed floor (Phase 5 inheritance) is preserved verbatim; only the cell SHAPE changes (12 per-behaviour → 4 integrated).

The four MUST architecture families: connectome-constrained PPO (focal), MLP-PPO, LSTM/GRU-PPO, FeedforwardGA (GA-evolved weights on a fixed feed-forward topology via `run_evolution.py`; genuine NEAT topology evolution lives at T8).

### T4 — integrated-C3 primary cells (one per architecture, n ≥ 4 seeds)

- [ ] T4.connectome.c3_integrated — Cook 2019 connectome (chemical-synapse strict-mask, fixed gap junctions) on the combined food+predator+thermotaxis cell. Focal architecture. Consumes T3 corrected two-channel predator biology + the connectome predator/thermotaxis projections.
- [ ] T4.mlp_ppo.c3_integrated
- [ ] T4.lstm_gru_ppo.c3_integrated
- [ ] T4.feedforward_ga.c3_integrated — GA-evolved weights on a fixed feed-forward topology.

### T4 — curriculum smokes (per architecture, n=1; de-risk before the n ≥ 4 C3 cells)

- [ ] T4.\*.c1_foraging — foraging-only smoke per architecture (may reuse existing `configs/scenarios/foraging/` klinotaxis configs).
- [ ] T4.\*.c2_foraging_predator — foraging+predator smoke per architecture (may reuse the `configs/scenarios/pursuit/` predator-biology configs).

### T4 — C3 ablations

- [ ] T4.\*.c3_reward_ablation — per-family reward-shape ablation ON the integrated C3 substrate: existing `gradient_proximity` vs `distal_chemo_penalty + binary_contact_damage_trigger`. Carry-forward from [fix-predator-sensing-biology design.md § Decision T3.7](../fix-predator-sensing-biology/design.md) + Phase 0's 2×2 outcome.
- [ ] T4.connectome_soft_prior.c3 — connectome strict-mask vs soft-prior ablation per [design.md § Decision 7](design.md) (chemical-synapse counts as weight initialisation only; PPO free to grow new connections; gap junctions still fixed). Run on the integrated C3 substrate.

### T4 — cross-architecture analysis + logbook

- [ ] T4.analysis.ranking — paired-seed Wilcoxon + bootstrap CIs across the four MUST families' integrated-C3 cells on the grid substrate, with per-behaviour-component sub-metrics extracted per the `architecture-comparison-protocol` capability. BH-FDR multiple-comparisons correction within-pass (per the `weight-search-architecture-ranking` MCC commitment). The first architecture-comparison ranking.
- [ ] T4.analysis.connectome_grid_ranking — explicit answer to "where does the wild-type connectome (chemical-synapse strict-mask) rank under PPO weight search on the existing grid substrate?"
- [ ] T4.analysis.strict_vs_soft — strict-mask vs soft-prior delta on at least one behaviour.
- [ ] T4.logbook — publish T4 logbook (suggested: `docs/experiments/logbooks/0XX-l2-first-pass.md`). Feeds Gate 3's evidence base together with T7.

**T4 risk-mitigation pivot (per roadmap)**: if after reasonable hyperparameter search no architecture family reaches Phase 0-3 baselines on any of the three behaviours, run the diagnostic sequence (topology density / reward shaping; continuous-action head is NOT yet on the table — that's T5; gap-junction-as-learnable is a Decision-7 amendment trigger per design risk 8). If none resolve, the finding is "PPO-on-the-real-connectome requires further substrate work" — itself a publishable negative result and a Phase 7 prerequisite.

## Tranche 5 — Platform Refactor (continuous-2D coordinates + continuous-action heads + parity verification)

**OpenSpec change**: `add-continuous-2d-and-action-heads` (placeholder; not yet created)
**Status**: 🔲 not started
**Roadmap layer**: env-upgrade (platform) (sits between T4 and T7 per [design.md § Decision 1](design.md) so the env-upgrade delta is itself a finding; split from the env-fidelity work in T6)
**Approx duration**: 3-4 weeks
**Bio fidelity**: MEDIUM (matches plate-arena geometry; chemical-gradient fidelity ships in T6)
**Dependencies**: T4 closed (so the env-upgrade delta has a baseline to compare against)
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § Continuous environment + sensory physics

T5 is the platform-level refactor: continuous 2D coordinates, continuous-action heads
on every MUST brain, continuous-output adapter for the connectome. Single verifiable
outcome (parity test + continuous-substrate smoke training). The Rung 2 env-fidelity
work is deliberately separated into T6 per [design.md § Decision 1](design.md) so T5's
parity-test outcome (Gate 2) isn't bundled with chemosensory-adaptation iteration.

T5 closes Gate 2.

### T5.physics — Continuous 2D coordinates

- [ ] T5.physics.1 Continuous 2D coordinate system: realistic spatial scales (~1mm worm body on cm-scale plates). Native sinusoidal undulation, omega turns, and pirouettes are explicitly NOT in scope (per roadmap); the c302 / Sibernetic export path covers those if/when behavioural-fidelity claims later require them.
- [ ] T5.physics.2 Env-level adapters: existing grid-based food / predator / thermal sources translate cleanly to continuous coordinates; document any sources that don't (and propose their fix or defer to T6).

### T5.action — Continuous-action heads

- [ ] T5.action.1 Continuous action space (speed 0-to-max + turning angle −π to π). Replaces the discrete 4-action `DEFAULT_ACTIONS` (`packages/quantum-nematode/quantumnematode/brain/actions.py:8-31`) on the upgraded substrate; T4's discrete substrate stays intact for the T4-vs-T7 delta comparison.
- [ ] T5.action.2 Continuous-action policy heads on every MUST PPO-family brain (Gaussian policy; exact parameterisation — Gaussian vs Beta vs Tanh-squashed — deliberately deferred to this tranche per [design.md § What This Change Explicitly Does Not Decide](design.md)).
- [ ] T5.action.3 Continuous-output adapter for the connectome-constrained brain (T1 data model + T2 plugin interface deliver topology; continuous-output is added here).
- [ ] T5.action.4 SHOULD/MAY architectures (quantum, spiking, reservoir, hybrid, transformer) — continuous-output adaptation is opportunistic; not gating Gate 2.

### T5.parity — Plugin-parity verification (closes Gate 2)

- [ ] T5.parity.1 Add a hypothetical new architecture family (or revive a Phase 0-3 architecture that didn't make the MUST set) through the L1 plugin interface during T5. Record engineer-hours and files-touched count in the T5 logbook. Primary parity checks are G2.b (files touched ≤ 6) and G2.c (no per-architecture branches); engineer-hours are documented for future reference but the "≤ 5 working days" target is not load-bearing per Gate 2 G2.a (no pre-refactor baseline measurement exists to compare against).
- [ ] T5.parity.2 File-count check: addition touches ≤ 6 files (registry registration + brain implementation + config class + config example + smoke test + docs) per Gate 2 G2.b.
- [ ] T5.parity.3 Code-review check: no per-architecture branches in the simulation loop or training loop after the addition (per Gate 2 G2.c).
- [ ] T5.parity.4 Continuous-substrate floor check (per Gate 2 G2.d): at least the connectome and MLP-PPO MUST families train cleanly on klinotaxis on the new continuous-2D substrate with continuous-action heads; **mean episode return ≥ 50% of T4's per-architecture grid-substrate baseline mean episode return for the same architecture**. The 50% floor is deliberately wide because grid-discrete-action and continuous-2D-continuous-action return scales are not directly comparable; this verifies the substrate change didn't break training, not an apples-to-apples ranking. The actual upgraded-substrate ranking lands in T7.
- [ ] T5.parity.5 Document the parity verification in the T5 logbook; this is the evidence Gate 2 needs.

### T5 — analysis + logbook

- [ ] T5.analysis — quantify the platform-refactor delta (continuous-2D vs discrete-grid on movement kinematics; continuous-action vs discrete-action on training stability).
- [ ] T5.logbook — publish T5 logbook (suggested: `docs/experiments/logbooks/0XX-platform-refactor.md`). Required reading material for Gate 2.

**T5 risk-mitigation pivot (per roadmap)**: if continuous-action heads prove harder to integrate than estimated (most likely failure mode: continuous-output adapter for the connectome-constrained brain is incompatible with the chemical-synapse strict-mask), pause and re-scope; either revise Decision 7 to relax the strict-mask claim on continuous substrates, or document the connectome family as "discrete-action only" and proceed with the other three MUST families on the continuous substrate. Either outcome amends this tracker.

## Gate 2 (closes Tranche 5, month ~4-5 cumulative): L1 plugin parity in practice?

**Trigger**: T5 closed with the parity test verified during the continuous-action work.
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § Mid-phase decision gates § Gate 2.
**Quantitative pass criteria**: see [design.md § Decision 6 § Gate 2](design.md) for the full criterion set. Primary (load-bearing): G2.b files-touched ≤ 6; G2.c no per-architecture branches in simulation/training loops. Documented but not load-bearing: G2.a engineer-hours for the addition (no pre-refactor baseline measurement exists). Floor check: G2.d continuous-substrate smoke training mean episode return ≥ 50% of T4 per-architecture grid baseline on connectome + MLPPPO.
**Decision must be written in**: the T5 OpenSpec change's published logbook. This tracker links to the decision once it lands.

- [ ] **Gate 2 decision recorded**: ≤ 6 files touched (G2.b PRIMARY) AND no per-architecture branches (G2.c PRIMARY) AND continuous-substrate floor check passes (G2.d ≥ 50% of T4 baseline) → GO. G2.a engineer-hours recorded for future reference. G2.b OR G2.c fails → L1 refactor PIVOT (T5 amends to add 2-4 weeks of L1 refactor work before re-evaluating Gate 2). Post-pivot, plugin interface still fundamentally incompatible with one or more MUST families → STOP (connectome-specific interface scoped; parity test not met).
- [ ] **Gate 2 decision link**: [add link to the T5 logbook where the decision is recorded]

## Tranche 6 — Env Fidelity (Rung 2 dynamic Fick's-law + log-concentration chemosensory adaptation)

**OpenSpec change**: `add-rung2-chemical-gradients` (placeholder; not yet created)
**Status**: 🔲 not started
**Roadmap layer**: env-upgrade (fidelity) (separated from T5 platform refactor per [design.md § Decision 1](design.md) so T6 can iterate without re-opening Gate 2)
**Approx duration**: 3-4 weeks (allowing for M4-style iteration if Rung 2.5 or 2.6 ends up needed)
**Bio fidelity**: HIGH (matches the computational-chemotaxis field's actual fidelity standard)
**Dependencies**: T5 closed (continuous-2D substrate is the platform Rung 2 deposits gradients onto)
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § Continuous environment + sensory physics § Chemical-gradient fidelity

Rung 2 has **two coupled components** — environment dynamics AND chemosensory
adaptation kinetics. They MUST be designed together (per roadmap: "without
log-concentration adaptation on the sensory side, the gradient realism is wasted").

- [ ] T6.gradients.1 Heat-equation diffusion (∂C/∂t = D∇²C) on the continuous-2D substrate with signal-type-specific D values (food vs pheromone vs CO₂).
- [ ] T6.gradients.2 Source dynamics — depletion when worms feed; source replenishment; decay terms for short-lived signals.
- [ ] T6.gradients.3 Log-concentration chemosensory adaptation kinetics on AWC/AWA/ASE-style sensors. Coupled component; ships with T6.gradients.1+.2.
- [ ] T6.gradients.4 Cross-tranche dependency: T7 klinotaxis + thermotaxis evaluations use Rung 2 gradients.
- [ ] T6.analysis — quantify the env-fidelity gain (Rung 2 vs Rung 0 chemical-gradient quality on a smoke task; chemosensory adaptation transient on a step-input test).
- [ ] T6.logbook — publish T6 logbook (suggested: `docs/experiments/logbooks/0XX-rung2-gradients.md`). Required reading material before T7.

**T6 risk-mitigation**: if Rung 2 needs M4-style iteration (Rung 2.5, Rung 2.6 — most likely the chemosensory adaptation kinetics need parameter sweeps), T6 amends to add iteration sub-tranches without re-opening Gate 2. The L2 re-run (T7) just measures a slightly narrower env-upgrade scope and the env-upgrade delta (T4→T7) reports that scope explicitly. Stay bounded to Rung 2 + log-concentration adaptation; 3D physics, Sibernetic interop, aerotaxis/pheromone restoration, and multi-agent are all explicitly out of scope (Decision 5 + roadmap Future Directions).

## Tranche 7 — L2 Re-run on Fully-Upgraded Substrate + Real-Worm Validation

**OpenSpec change**: `add-l2-final-pass-and-validation` (placeholder; not yet created)
**Status**: 🔲 not started
**Roadmap layer**: L2 (final) + real-worm validation
**Approx duration**: 4-6 weeks
**Bio fidelity**: HIGH (Rung 2 gradients + corrected nociception + continuous-2D + continuous-action + real-worm validation target)
**Dependencies**: T4 closed (grid baseline for env-upgrade delta), T5 closed (platform refactor), T6 closed (env fidelity)
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § The layered platform § L2 + § Built-in real-worm validation

T7 re-runs the L2 sweep on the fully-upgraded substrate (continuous-2D + continuous-
action + Rung 2 gradients + log-concentration adaptation + corrected ASH/ADL), ships
the real-worm validation (now defensible because the behavioural numbers come from
the full upgrade stack), and opportunistically evaluates SHOULD/MAY architectures.

T7 closes Gate 3.

### T7 — MUST architectures × three behaviours on upgraded substrate

12 cells, same MUST families × behaviours as T4 but on the upgraded substrate:

- [ ] T7.connectome.klinotaxis — chemical-synapse strict-mask, fixed gap junctions, continuous-2D + continuous-action + Rung 2 gradients
- [ ] T7.connectome.thermotaxis
- [ ] T7.connectome.predator_evasion
- [ ] T7.mlp_ppo.klinotaxis
- [ ] T7.mlp_ppo.thermotaxis
- [ ] T7.mlp_ppo.predator_evasion
- [ ] T7.lstm_gru_ppo.klinotaxis
- [ ] T7.lstm_gru_ppo.thermotaxis
- [ ] T7.lstm_gru_ppo.predator_evasion
- [ ] T7.neat_weights.klinotaxis
- [ ] T7.neat_weights.thermotaxis
- [ ] T7.neat_weights.predator_evasion

### T7 — SHOULD architectures (opportunistic, not gating)

- [ ] T7.quantum.\* — one quantum row at continuous-physics complexity (Phase 2 baseline reference; SHOULD per [design.md § Decision 4](design.md)).
- [ ] T7.spiking.\* — opportunistic; if PPO-trained spiking doesn't train cleanly on the connectome, document the failure mode and defer to Phase 7 L4 STDP work.

### T7 — MAY architectures (opportunistic, not gating)

- [ ] T7.reservoir.\* — one row if cheap (QRH / CRH; Phase 2 pursuit-advantage rep).
- [ ] T7.hybrid.\* — one row if cheap (Phase 2 SOTA rep).
- [ ] T7.transformer.\* — only if scope and engineering effort allow.

### T7 — real-worm validation

- [ ] T7.validation.1 Choose validation target — chemotaxis indices à la Bargmann / mechanosensation escape latencies / whole-brain Ca²⁺ correlation matrices à la Kavli/Janelia. Selection deliberately deferred to this tranche per [design.md § What This Change Explicitly Does Not Decide](design.md).
- [ ] T7.validation.2 Implement the comparison pipeline: extract the model's analogue of the chosen real-worm metric; document data source + version; record the comparison procedure.
- [ ] T7.validation.3 Run the comparison; report quantitative agreement with confidence intervals. Required for Phase 6 exit (MUST in the roadmap).

### T7 — cross-architecture analysis + env-upgrade delta + logbook

- [ ] T7.analysis.ranking_upgraded — paired-seed Wilcoxon + bootstrap 95% CIs across the 12 MUST cells (per Gate 3 G3.a). Multiple-comparisons strategy applied consistently with T4 per the documented choice in the T4 OpenSpec change's design.md (default Holm-Bonferroni within-pass, but the T4 design may argue for BH-FDR or experiment-wide correction; whichever T4 commits to is the strategy T7 reuses).
- [ ] T7.analysis.env_delta — head-to-head T4-grid vs T7-upgraded ranking delta (RQ5). The load-bearing finding that justifies the deliberate T4-vs-T7 split per [design.md § Decision 1](design.md).
- [ ] T7.analysis.connectome_final_ranking — explicit answer to "where does the wild-type connectome (chemical-synapse strict-mask) rank under PPO weight search on the upgraded substrate?" (per Gate 3 G3.b — "failed to train" is STOP, not tie).
- [ ] T7.logbook — publish T7 logbook (suggested: `docs/experiments/logbooks/0XX-l2-final-pass.md`). Required reading material for Gate 3.

**T7 risk-mitigation pivot (per roadmap)**: if the L2 re-run on the upgraded substrate fails to learn after diagnostic sequencing (topology density / continuous-action head parameterisation / reward shaping / gap-junction-as-learnable per Decision 7 amendment trigger), the finding is publishable as "PPO-on-the-real-connectome under continuous-physics requires further substrate work" — Phase 7 L4 plasticity work inherits the substrate-engineering question.

## Gate 3 (closes Tranche 7, month ~7-8 cumulative): L2 results across architectures + real-worm validation in hand?

**Trigger**: T7 closed with the cross-architecture ranking on the upgraded substrate + real-worm validation in hand.
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § Mid-phase decision gates § Gate 3.
**Quantitative pass criteria**: see [design.md § Decision 6 § Gate 3](design.md) for the full criterion set (G3.a all 12 MUST cells at Phase 5 statistical bar with the documented multiple-comparisons strategy declared in the T4 OpenSpec change's design.md and applied consistently to T4 and T7; G3.b connectome lands in the ranking with a clear wins/ties/loses verdict per behaviour; G3.c env-upgrade delta analysis shipped; G3.d real-worm validation shipped with quantitative agreement + CIs).
**Decision must be written in**: the T7 OpenSpec change's published logbook (likely the T7 logbook itself). This tracker links to the decision once it lands.

- [ ] **Gate 3 decision recorded**: all 12 MUST cells (G3.a) AND connectome ranking clear (G3.b) AND env-upgrade delta (G3.c) AND real-worm validation (G3.d) → GO to T8. Partial MUST coverage (1-2 cells missing) → PIVOT-scope: Phase 6a (T1–T7) ships, Phase 6b (T8 NEAT + T9 synthesis) becomes follow-on. Phase 6 overshoots 10 months cumulative → PIVOT-scope for delivery reasons (same 6a/6b split). Fewer than half MUST cells reach the statistical bar after T7 risk-mitigation pivot → STOP (publishable negative result; Phase 7 L4 inherits substrate-engineering question).
- [ ] **Gate 3 decision link**: [add link to the T7 logbook where the decision is recorded]

## Tranche 8 — L3 NEAT Topology Search on Upgraded Substrate

**OpenSpec change**: `add-l3-neat-topology-search` (placeholder; not yet created)
**Status**: 🔲 not started (gated on Gate 3 GO, or scoped into Phase 6b under PIVOT-scope)
**Roadmap layer**: L3
**Approx duration**: 6-10 weeks
**Bio fidelity**: LOW (NEAT is an ML tool; the bio-fidelity contribution lives in T1 + the T7 connectome rank)
**Dependencies**: T7 closed
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § The layered platform § L3

T8 ranks the wild-type connectome's topology against NEAT-evolved alternatives on at
least one behaviour, using the lag-matrix instrument (or equivalent discriminative
gate from Phase 5 M5's methodology contributions). T8 is also the natural follow-up
to Phase 5 M5's architecture-asymmetry hypothesis: matched-capacity NEAT-vs-NEAT
co-evolution vs asymmetric NEAT-vs-MLP is a clean falsification test.

Coarse-grained sub-tasks below; the T8 OpenSpec change elaborates them.

- [ ] T8.1 Integrate TensorNEAT (GPU-accelerated NEAT, JAX/vmap; ~500× speedup over neat-python documented in the field).
- [ ] T8.2 NEAT topology + weight evolution on the L1 plugin interface from T2. Plugin should accommodate NEAT-evolved topologies as natively as it accommodates fixed-topology architectures (the topology/rule factoring from T2 pays off here).
- [ ] T8.3 Topology-vs-connectome head-to-head on at least one behaviour on the upgraded substrate (klinotaxis is the natural first).
- [ ] T8.4 Matched-capacity NEAT-vs-NEAT vs asymmetric NEAT-vs-MLP (Phase 5 M5 follow-up). Uses the lag-matrix instrument from Phase 5 logbook 017.
- [ ] T8.5 Cross-architecture analysis: "is the wild-type connectome a local optimum?" Cross-references the T7 connectome ranking with the T8 NEAT-evolved baseline.
- [ ] T8.6 Update this checklist + `docs/roadmap.md` Phase 6 Tranche Tracker T8 row.
- [ ] T8.7 Publish T8 logbook (suggested: `docs/experiments/logbooks/0XX-l3-neat-topology-search.md`).

**T8 risk-mitigation pivot (per roadmap)**: if NEAT-evolved topologies and the wild-type connectome converge to indistinguishable performance, that *is* the finding — "the connectome is competitive with evolved topologies on these behaviours." The optimal-primary framing weakens; the connectome-primary framing strengthens. Acceptable outcome; pivot the headline framing if it lands.

## Tranche 9 — Phase 6 Synthesis Logbook

**OpenSpec change**: `add-phase6-synthesis-logbook` (placeholder; not yet created)
**Status**: 🔲 not started
**Roadmap layer**: synthesis
**Approx duration**: 1-2 weeks
**Dependencies**: T1–T8 closed (or T8 scoped into Phase 6b under PIVOT-scope)
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § Phase 6 exit criteria

- [ ] T9.1 Walk through each Phase 6 exit criterion (the seven MUSTs in the roadmap) with evidence: L0 substrate operational (T1), L1 plugin parity (T2 + verified at T5), L2 results across MUST architectures × three behaviours at the Phase 5 statistical bar (T7), L3 NEAT topology-search results (T8), Rung 2 chemical gradients operational (T6), corrected ASH/ADL nociception operational (T3), ≥ 1 real-worm validation (T7).
- [ ] T9.2 Document the connectome ranking honestly. The roadmap pre-commits to a framing pivot: connectome-primary if connectome wins decisively; optimal-primary if connectome is competitive-but-not-dominant. Whichever the data supports.
- [ ] T9.3 Document negative findings honestly (Phase 5 precedent). If any L2 / L3 cell came back STOP, the diagnosis is itself a publishable contribution.
- [ ] T9.4 Quantify the env-upgrade delta (T4-grid vs T7-upgraded ranking change) as a standalone result. This was a load-bearing reason for the deliberate T4-vs-T7 split per [design.md § Decision 1](design.md); the synthesis should foreground it.
- [ ] T9.5 Phase 7 trigger recommendation: which Phase 7 priorities (L4 plasticity / *P. pacificus* transfer / publication / collaboration) are best-supported by Phase 6 evidence.
- [ ] T9.6 Publish `docs/experiments/logbooks/0XX-phase6-synthesis.md`.
- [ ] T9.7 Update `docs/roadmap.md` Phase 6 status → ✅ COMPLETE; record exit criterion outcomes. Update Phase 6 Tranche Tracker rows to their terminal verdicts.

> Archiving `phase6-tracking` itself is an operator-side step that intentionally does NOT block task completion: `openspec archive` requires every task ticked, so an "archive me" task here would self-block (same precedent as `phase5-tracking/tasks.md` and `add-tei-prior-on-m3/tasks.md`). The archive happens after the synthesis-logbook PR merges to main via `openspec archive phase6-tracking`; the synthesis change archives separately.

## Phase 6 Research Questions

Open research questions surfaced during Phase 6 planning or during in-flight Phase 6
work that don't fit cleanly under any single tranche but are worth tracking. Each
question has a concrete trigger condition for escalation; nothing here commits the
project to work upfront.

### RQ1: Architecture-asymmetry hypothesis (Phase 5 M5 follow-up)

**Status**: open — gated on T8 NEAT integration
**Roadmap reference**: `docs/roadmap.md` § Phase 6 § The layered platform § L3 + Phase 5 logbook 017
**Trigger**: T8 ships matched-capacity NEAT-vs-NEAT vs asymmetric NEAT-vs-MLP head-to-head. If the matched arm produces own-vs-cross lag delta ≤ −0.05 while the asymmetric arm reproduces M5's +0.017 delta, the architecture-asymmetry hypothesis is confirmed. If both arms produce ~+0.017, the hypothesis is falsified (capacity symmetry is irrelevant).
**Recorded by**: this tracker (open question); will be settled inside T8's OpenSpec change.

### RQ2: M6.x substrate-shape diagnosis under continuous physics

**Status**: open — gated on T6 (Rung 2 gradients) + T5 (continuous-2D physics) shipping
**Roadmap reference**: Phase 5 logbook 019 + logbook 020 (TEI bias-network logit-prior was the wrong abstraction)
**Trigger**: Once continuous-2D + Rung 2 gradients are operational, the substrate carrying any future TEI-style experiment is materially different from M6.x's discrete-grid logit-bias substrate. Whether the upgraded substrate surfaces a Baldwin or TEI signal serendipitously is the open question. If a signal appears, escalate to a dedicated Phase 6+ change scoped explicitly to characterise it; if no signal appears across the T7 sweep, close RQ2 with the substrate-shape diagnosis carried forward to Phase 7.
**Recorded by**: Phase 5 logbook 020 § What's next; tracked here as a Phase 6 watch item.

### RQ3: Connectome-primary vs optimal-primary headline framing

**Status**: open — settles at T9.2
**Roadmap reference**: `docs/roadmap.md` § Executive Summary § Framing note + § Phase 6 Conclusion
**Trigger**: T7 + T8 results in hand. If the connectome wins decisively on the curated behaviours, the headline shifts toward "connectome-primary" (a neuroscience result). If the connectome is competitive-but-not-dominant, "optimal-primary" remains the natural framing. Both are platform contributions; the scientific framing follows the evidence.
**Recorded by**: this tracker; settled in T9.2 by the Phase 6 synthesis logbook.

### RQ4: c302 / NeuroML export path for OpenWorm Sibernetic interop

**Status**: open — Phase 6 Future Directions, not Phase 6 MUST
**Roadmap reference**: [design.md § Decision 2](design.md) (deferred from L0 primary)
**Trigger**: A Phase 6 sub-task or Phase 7 deliverable requires handing the connectome topology to OpenWorm Sibernetic for body-physics validation. If triggered, scope a small `add-c302-export` change; the import-side `cect` choice does not block the export-side c302 work.
**Recorded by**: this tracker (open question); not gating any Phase 6 exit criterion.

### RQ5: Env-upgrade delta — does the substrate upgrade change the architecture ranking?

**Status**: open — produced as a T7 deliverable (T7.analysis.env_delta) and foregrounded at T9.4
**Roadmap reference**: implicit in `docs/roadmap.md` § Phase 6 § Architecture-comparison protocol
**Trigger**: T4 and T7 both shipped. The interesting outcomes are (a) ranking is stable across substrate change — substrate complexity is below the threshold for differential architecture advantage; (b) ranking shifts — some architectures (likely temporal / continuous-action-native) benefit disproportionately from continuous-2D + Rung 2, others (likely connectome under strict-mask) don't. Either outcome is publishable; (b) reinforces the Phase 6 "platform" claim more strongly.
**Recorded by**: T7 produces the analysis (T7.analysis.env_delta); T9.4 foregrounds it in the synthesis logbook.

### RQ6: Compute budget per L2 cell

**Status**: open — settles during T4 planning
**Roadmap reference**: roadmap § Phase 6 § Compute / infrastructure planning
**Trigger**: T2's first PPO-on-connectome training run produces one real wall-time data point. T4 planning (T4.0a) extrapolates to a per-cell estimate × 12 cells × 2 L2 passes (T4 + T7), checks against available GPU budget, and either confirms Decision 1's "4-6 weeks per L2 pass" estimate or amends Decision 1 + this tracker.
**Recorded by**: T4 OpenSpec change's design.md; amends this tracker if Decision 1 estimates diverge by > 2×.
