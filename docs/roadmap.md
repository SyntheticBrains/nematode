# Quantum Nematode Project Roadmap

**Vision**: Determine the most efficient brain architecture for nematode-like embodied tasks, using the *Caenorhabditis elegans* connectome as the focal comparison point against unconstrained and evolved alternatives. The platform brings learning, evolution, and a curated subset of biologically-faithful sensing into one closed sensory-motor loop, so that architecture comparisons answer scientific questions rather than rank benchmarks.

**Version**: 4.3

**Last Updated**: 2026-09-20

**Horizon**: Milestone-based (aspirational timeline ~2025-2028+, phases advance when exit criteria are met)

______________________________________________________________________

> **Disclaimer**: This roadmap describes a research program with hypotheses to be tested, not established results. Many claims about quantum advantages, biological insights, and scaling properties are research questions requiring empirical validation. Outcomes may differ significantly from projections as evidence accumulates. The adaptive decision gates throughout this document reflect our commitment to evidence-driven pivots when hypotheses are not supported by experimental results.

______________________________________________________________________

**Note**: This roadmap aims for scientifically rigorous contributions at the intersection of computational neuroscience, embodied learning, and comparative architecture analysis. Quantum architectures remain one family in the comparison, not the project's organising principle.

______________________________________________________________________

## Table of Contents

01. [Timeline Overview](#timeline-overview)
02. [Executive Summary](#executive-summary)
03. [Current State](#current-state)
04. [Phase Roadmap](#phase-roadmap)
    - [Phase 0: Foundation & Baselines](#phase-0-foundation--baselines-complete) (COMPLETE)
    - [Phase 1: Sensory & Threat Complexity](#phase-1-sensory--threat-complexity-complete) (COMPLETE)
    - [Phase 2: Architecture Analysis & Standardization](#phase-2-architecture-analysis--standardization-complete) (COMPLETE)
    - [Phase 3: Temporal Sensing & Memory](#phase-3-temporal-sensing--memory)
    - [Phase 4: Multi-Agent Complexity](#phase-4-multi-agent-complexity)
    - [Phase 5: Evolution & Adaptation](#phase-5-evolution--adaptation)
    - [Phase 6: Connectome Substrate & Architecture Comparison](#phase-6-connectome-substrate--architecture-comparison)
    - [Phase 7: Deepen — Plasticity & Cross-Species Transfer](#phase-7-deepen--plasticity--cross-species-transfer)
    - [Phase 8: Ground, then Embody — Measured Substrate & Body](#phase-8-ground-then-embody--measured-substrate--body)
05. [Architecture-Comparison Protocol](#architecture-comparison-protocol)
06. [Complexity Dashboard](#complexity-dashboard)
07. [Biological Fidelity](#biological-fidelity)
08. [Adaptive Roadmap Philosophy](#adaptive-roadmap-philosophy)
09. [Ongoing Validation Milestones](#ongoing-validation-milestones)
10. [Success Metrics Framework](#success-metrics-framework)
11. [Success Levels](#success-levels)
12. [Relationship to External Projects](#relationship-to-external-projects)
13. [Future Directions](#future-directions)
14. [Technical Debt & Maintenance](#technical-debt--maintenance)
15. [Scoping Changes from v3](#scoping-changes-from-v3)
16. [Conclusion](#conclusion)

______________________________________________________________________

## Timeline Overview

> This roadmap is milestone-based: phases advance when exit criteria are met, not when calendar dates arrive. Aspirational timelines for forward-looking phases are estimates; completed phases show "—" because the dates that matter are in commit history and per-milestone logbooks.

| Phase | Aspirational Timeline | Focus | Status | Key Deliverable |
|-------|----------------------|-------|--------|-----------------|
| **0** | — | Foundation & Baselines | ✅ COMPLETE | Validated optimization methods, SOTA baselines, first QPU run |
| **1** | — | Sensory & Threat Complexity | ✅ COMPLETE | Thermotaxis, enhanced predators, mechanosensation, HP system |
| **2** | — | Architecture Analysis | ✅ COMPLETE | 300-session quantum architecture campaign across 15 variants; established that grid-world complexity is below the threshold for quantum advantage |
| **3** | — | Temporal Sensing & Memory | ✅ COMPLETE | Temporal/derivative sensing, STAM, LSTM/GRU PPO brain (19th architecture). Temporal Mode A reaches 94% L500 on the hardest environment. Aerotaxis with 5-zone oxygen system |
| **4** | — | Multi-Agent Complexity | ✅ COMPLETE | Pheromones, social dynamics, klinotaxis sensing. Temporal collective exploration +14.3%; social feeding +35% food under scarcity. Coordination did not produce genuine multi-agent complexity at the scales tested |
| **5** | — | Evolution & Adaptation | ✅ COMPLETE (2026-05-23) | M3 Lamarckian inheritance is the headline-positive result. M4 Baldwin / M5 co-evolution / M6.x transgenerational memory closed with substrate-grounded STOP verdicts (architectural diagnoses, not implementation failures) |
| **6** | — | Connectome substrate + architecture comparison | 🟡 **6a COMPLETE / 6b deferred unscheduled** *(2026-09-19: `phase6b-tracking` closed off with every item marked deferred-with-destination; the L3 exit criterion stays unmet)* (delivered in two shipments — see § Phase 6a/6b split) | First closed-loop learning on the real *C. elegans* connectome with a pluggable architecture interface, and a full architecture ranking across six families on three behaviours (klinotaxis, thermotaxis, predator evasion). **6a — COMPLETE, Gate 3 GO** ([Logbook 037](experiments/logbooks/037-phase6a-synthesis.md); T1–T7 + connectome-structure controls + validation): the platform + the ranking (Logbook 029 — MLP dominant, wild-type connectome 5th of 6 under PPO weight search, a *degree-statistics* result per the 034 rewired-null) + real-worm behavioural validation (035 chemotaxis both strategies; 036 thermotaxis partial). **6b** (T8 NEAT topology search): deferred completion, gated on GPU + env-vectorisation. Phase 6 is marked ✅ COMPLETE only when the 6b synthesis lands |
| **7** | — | Deepen — plasticity + cross-species transfer | ✅ **COMPLETE (SPLIT, 2026-09-19)** — see [Logbook 069](experiments/logbooks/069-phase7-synthesis.md) and § Phase 7's resolved work. **7a shipped, 7b deferred to the phase after 7 (D14, 2026-09-15)**; the D10 primary is unmet and was unmeetable in the closing scope, since every learner that reaches competence leaves `w_chem` frozen (the resolved history is in § Phase 7 progress record) | Biologically-plausible plasticity on the connectome — rate-based three-factor rules, resolved as the 2×2 plastic wild-type vs plastic rewired-null (spiking-STDP is MAY). Cross-species **head-circuit** transfer using the Cook et al. 2025 *P. pacificus* data (head-only, chemical-synapse-only), with the dauer connectome (Yim et al. 2024) as a scope-matched within-species condition. Optional biological-validation collaboration and paper drafts |
| **8** | — | Ground, then embody — measured substrate + body | 🟡 **8a COMPLETE / 8b PENDING** *(2026-09-26, [Logbook 076](experiments/logbooks/076-8a-synthesis.md): every 8a criterion assigned a status; the D20 gate written as **GO**; block V holds with its conditions named, about half of its `auc_success` lead having come from the null's rewired gap junctions; D21 makes the chemical-only null 8b's primary)* — planned as v4.3, 2026-09-20; tracker [`phase8-tracking`](../openspec/changes/phase8-tracking/tasks.md); pre-structured as two shipments (D20) | **8a**: the init-vs-rewiring control on block V (D15), the operating-point robustness surface (D16), measured synaptic signs and strengths on the Cook edges (Creamer–Leifer–Pillow; D17), the dynamics rung with plastic gap junctions. **8b**: a step–time calibration, reversal and proprioception, the anatomical motor-to-muscle readout into a 2D body (D18/D19), patchy lawns + internal state, body-level validation, the six-family ranking re-run through the body. Cross-species, multi-agent, evolution, 3D and whole-organism fidelity are out of scope — see § Phase 8 |

______________________________________________________________________

## Executive Summary

The Quantum Nematode project asks one primary research question expressed along two comparison dimensions: *what is the most efficient brain architecture for nematode-like embodied tasks, and how does the C. elegans connectome rank against unconstrained and evolved alternatives — when learning and evolution operate on it in a closed sensory-motor loop?*

The project's central contribution is a **platform that makes this question answerable**. It integrates four capabilities that, separately, exist across the computational neuroscience and embodied-AI fields but have not yet been combined on a single substrate:

- Biologically-grounded sensing (klinotaxis, thermosensation, mechanosensation, pheromone-mediated signalling) shipped through Phases 1-4.
- Multiple learning and evolutionary regimes (PPO, CMA-ES, Lamarckian inheritance) shipped through Phase 5, with neuromodulated plasticity (three-factor family) targeted at Phase 7.
- A pluggable architecture interface that admits MLP, recurrent, spiking, reservoir, quantum, hybrid, NEAT-evolved, and connectome-constrained brains as comparable rows in one experimental sweep (Phase 6).
- The real *C. elegans* connectome (302 neurons, Cook et al. 2019) imported as the focal architecture to rank against the others, with the *P. pacificus* head connectome (Cook et al. 2025) as the planned cross-species comparator at Phase 7.

Phase 5 results sharpened the framing in a load-bearing way. M3 Lamarckian inheritance shipped as the headline-positive Phase 5 result. M4 (Baldwin), M5 (co-evolution arms race), and M6.x (transgenerational memory) closed with substrate-grounded STOP verdicts that were architectural diagnoses, not implementation failures: each pointed at the substrate or architecture rather than at the experimental protocol. Of those diagnoses, M6.x's wrong-abstraction-for-plasticity carries forward directly into Phase 7's plasticity work. M5's architecture-asymmetry diagnosis would be tested by matched-capacity co-evolution (NEAT-vs-NEAT vs asymmetric NEAT-vs-MLP); co-evolution is **deferred out of Phase 6 entirely with no scheduled destination** — it is too compute-intensive for the phase, and the M5 diagnosis was already independently corroborated (Resendez Prado 2026), so it is a revisit-if-a-concrete-need-reappears item rather than a committed deliverable of any phase.

### Key Differentiators

- **vs. OpenWorm**: OpenWorm has the connectome (c302) and body physics (Sibernetic) but lacks closed-loop learning, evolution, and modern RL integration. The platform interoperates with OpenWorm at the c302 boundary rather than competing on body-physics fidelity.
- **vs. Izquierdo & Beer's klinotaxis arc**: their work has evolution and learning on minimal evolved circuits, but not the real connectome. The platform places both architectures in the same comparison rather than picking one.
- **vs. Boyle / Bryden / Cohen (Leeds)**: best-in-class undulatory locomotion modelling, but no learning or evolution. Complementary; not a competitor.
- **vs. standard RL benchmarks**: tasks are derived from documented *C. elegans* behaviours with quantitative biological validation targets (Bargmann-style chemotaxis indices, Ca²⁺ recording correlation matrices), not synthetic gridworlds tuned for benchmark difficulty.
- **vs. quantum ML research**: the project's 300-session quantum architecture campaign (Phase 2) remains the most comprehensive comparative evaluation of quantum architectures on biologically-grounded RL tasks to date. Phase 6 carries it forward as a baseline reference in the architecture-comparison protocol — not as an organising principle.

### North Star

Be the platform on which learning and evolution operate on the real *C. elegans* connectome in a closed sensory-motor loop, and use it to rank the wild-type connectome against unconstrained, evolved, and quantum architectures on a curated set of nematode behaviours.

*Framing note: the platform contribution and the scientific contribution are mutually reinforcing — building the platform answers the architecture-comparison question; the architecture-comparison question motivates each platform layer. If post-Phase-6 evidence shows the connectome wins decisively on the curated behaviours, the headline framing may shift toward "connectome-primary" (a neuroscience result). If the evidence shows the connectome is competitive but not dominant, "optimal-primary" (an architecture result) remains the natural framing. Both readings are platform contributions; the scientific framing follows the evidence.*

______________________________________________________________________

## Current State

Phases 0–5, Phase 6a and Phase 7 are complete (Phase 6b — NEAT topology search — is deferred unscheduled; Phase 8 is planned, v4.3). The platform now supports: 26 brain architectures spanning quantum, classical, recurrent, spiking, reservoir, hybrid, GA-evolved, and connectome-constrained families; thermotaxis, mechanosensation, aerotaxis, klinotaxis, and pheromone-based sensing; multi-agent dynamics at 5-10 agent scales; CMA-ES and TPE hyperparameter evolution; Lamarckian weight inheritance across generations. The connectome layer, the pluggable architecture interface, and continuous-2D physics are the work of Phase 6; the persistent-trace substrate, the three-factor and e-prop rule seams, the per-neuron motor readout, the degree-preserving rewiring control and the parallel campaign runner are the work of Phase 7.

### Phase 0 — Foundation & Baselines

- 6 brain architectures shipped: QVarCircuitBrain, QQLearningBrain, MLPReinforceBrain, MLPDQNBrain, MLPPPOBrain, SpikingReinforceBrain.
- PPO validated as classical SOTA (94-98% across thermotaxis configurations).
- CMA-ES validated for quantum circuits (88% success, 4x better than gradient-based).
- Spiking neural networks rewritten with surrogate gradient descent (73.3% success).
- First IBM QPU deployment.

### Phase 1 — Sensory & Threat Complexity

- Thermotaxis implemented with 9 configurations (3 sizes × 3 task variants); see [Logbook 007](experiments/logbooks/007-ppo-thermotaxis-baselines.md).
- Mechanosensation (boundary + predator contact detection).
- Stationary + pursuit predator types with configurable behaviour.
- Health/HP system with damage, healing, and strategic trade-offs.
- Oxygen sensing deferred to Phase 3 (paired with temporal sensing infrastructure).

### Phase 2 — Architecture Analysis

The 300-session quantum architecture campaign across 15 variants is the project's most comprehensive comparative evaluation of quantum architectures on biologically-grounded RL tasks to date (see [Logbook 008](experiments/logbooks/008-quantum-brain-evaluation.md)). It established two load-bearing findings:

- **Grid-world complexity (2-9D observations, 4 discrete actions, ~10K effective states) is below the threshold at which any of the 15 quantum variants tested produced a genuine advantage over matched-capacity classical baselines.** HybridQuantum achieved 96.9% on pursuit, but the HybridClassical ablation matched at 96.3% — the curriculum and fusion drove performance, not the quantum component. QRH showed a +9.4pp pursuit advantage but at low absolute performance (41.2%).
- **Statistical framework operational**; brain naming migrated to paradigm-prefix scheme; novel architectures evaluated include QRH, QEF, HybridQuantum, HybridClassical, QSNN, QRC, QSNN-PPO, HybridQuantumCortex, CRH, and variants.

The campaign carries forward as a baseline reference in Phase 6's architecture-comparison protocol. The 15-architecture results table is available in [Logbook 008](experiments/logbooks/008-quantum-brain-evaluation.md); see [research/quantum-architectures.md](research/quantum-architectures.md#strategic-assessment-environment-complexity--quantum-advantage) for the full strategic assessment.

### Phase 3 — Temporal Sensing & Memory

- Temporal Mode A (raw scalar + STAM memory buffer) reaches 94% L500 on the hardest environment, matching oracle at convergence.
- LSTM/GRU PPO shipped as the 19th architecture; GRU outperforms LSTM by 3-40pp on temporal tasks.
- Aerotaxis (5-zone oxygen field) with combined thermal+oxygen environments.
- See [Logbook 009](experiments/logbooks/009-temporal-sensing-evaluation.md) and [Logbook 010](experiments/logbooks/010-aerotaxis-baselines.md).

### Phase 4 — Multi-Agent Complexity

- 5-10 agent scaling operational. Pheromones (aggregation, alarm, food-marking) and social dynamics shipped.
- Temporal collective exploration: +14.3% advantage. Social feeding: +35% food under scarcity. Coordination overhead: zero with proportional resources.
- Klinotaxis sensing (head-sweep mode) shipped; pheromone signals were neutral on the campaign's tasks.
- The campaign found no genuine multi-agent complexity at the scales tested (coordination resolved to resource allocation rather than game-theoretic interaction); see [Logbook 011](experiments/logbooks/011-multi-agent-evaluation.md).

### Phase 5 — Evolution & Adaptation

Phase 5 closed 2026-05-23 with one headline-positive result and three substrate-grounded STOP verdicts. All five Phase 5 exit criteria are met with evidence; the STOP results are scientifically informative architectural diagnoses, not implementation failures, and the methodological yield from them carries directly into Phase 6 and Phase 7. See [Logbook 021](experiments/logbooks/021-phase5-synthesis.md) for the full synthesis.

- **M2 Hyperparameter Evolution** — GO. Four-arm CMA-ES then TPE campaign closed RQ1 on optimiser choice; +47pp / +79pp predator-arm acceleration on the M3 inheritance config. See [Logbook 012](experiments/logbooks/012-hyperparam-evolution-mlpppo-pilot.md).
- **M3 Lamarckian Inheritance** — GO, headline-positive. Speed gate passes at +5.25 generations; +17.5pp F1-F3 mean retention on the M6.10 environment; n=8 paired-seed rerun confirms. The "learned behaviour becomes innate" exit criterion is satisfied; see [Logbook 013](experiments/logbooks/013-lamarckian-inheritance-pilot.md).

### What we tried and stopped (Phase 5)

Three Phase 5 themes — Baldwin effect, co-evolution arms races, transgenerational memory — produced **STOP verdicts as field-consistent substrate diagnoses**, not implementation failures. Each diagnosis pointed at the substrate or architecture rather than at the experimental protocol; each shipped reusable methodology. The M6.x plasticity-abstraction question carries directly forward into Phase 7's L4 work; the M5 co-evolution / architecture-asymmetry question was originally slated for Phase 6 but is now **deferred with no scheduled destination** (co-evolution is too compute-intensive for Phase 6, and the M5 diagnosis was independently corroborated).

- **M4 Baldwin Effect** — STOP after three iterations (M4 → M4.5 → M4.6). Diagnosis: **single-task K=50 PPO has no Baldwin axis** because the optimal strategy on a single task is innate good behaviour for that task — the opposite of what Baldwin canalisation selects for. The substrate constraint, not the algorithm, blocks the result. Multi-task aggregation infrastructure is the prerequisite for a clean Baldwin demonstration and is deferred. Methodology shipped: F1 evaluator, 4-way aggregator, 8-field config schemas, n=8 Lamarckian rerun extending M3. See [Logbook 014](experiments/logbooks/014-baldwin-inheritance-pilot.md) and [Logbook 015](experiments/logbooks/015-baldwin-iterative-evaluation.md).
- **M5 Co-evolution Arms Race** — STOP after 13 single-seed lever ablations. Diagnosis: **LSTMPPO-prey-vs-MLPPPO-predator architecture asymmetry suppresses Red Queen entanglement** — own-vs-cross fitness lag delta stayed in the +0.017 to +0.024 range across all ablations, against a target of ≤−0.05. Independent corroboration arrived from Resendez Prado's [Personality Requires Struggle](https://arxiv.org/abs/2604.03565) (April 2026): "transparent regime" same-architecture self-play suppresses the heterogeneity needed for measurable Red Queen / Baldwin signal — the same hypothesis from a different group. The matched-capacity NEAT-vs-NEAT co-evolution test of this question was originally slated for Phase 6 (T8.4) but is **deferred out of Phase 6 with no scheduled destination** — co-evolution is too compute-intensive for the phase, and the external corroboration reduces the marginal value of re-confirming the diagnosis in-house. Methodology shipped: lag-matrix cross-pairing instrument and cell-grid fair-test methodology, both reusable, and available whenever a future phase commits to the co-evolution question. See [Logbook 017](experiments/logbooks/017-coevolution-arms-race.md).
- **M6 / M6.9+ / M6.13 Transgenerational Memory** — STOP across three pilot rounds. The `TransgenerationalInheritance` framework + `TransgenerationalMemory` dataclass + LSTMPPO `tei_prior` actor-logit hook ship as functional infrastructure, but no K value or substrate variant produced a positive memory effect. Diagnosis: **the bias-network logit-prior is the wrong abstraction for the wet-lab single-circuit excitability shift** documented in Kaletsky 2025 and the 2025 mammalian-TEI literature. This is a *substrate* finding — different from a hyperparameter or training failure — and points at Phase 6's connectome-substrate work as the natural next step. Pure-TEI K=0 was substrate-inert (cross-arm delta −49pp); substrate-on-top-of-Lamarckian at K=1000 showed zero acceleration; at K=200 showed −9.33pp active interference under fair-F0. See [Logbook 018](experiments/logbooks/018-transgenerational-memory.md), [Logbook 019](experiments/logbooks/019-transgenerational-memory-redesign.md), and [Logbook 020](experiments/logbooks/020-tei-prior-on-lamarckian.md).

Two reusable methodology contributions ship unscooped: the **lag-matrix cross-pairing instrument** and the **cell-grid fair-test methodology**, both from [Logbook 017](experiments/logbooks/017-coevolution-arms-race.md). Independent corroboration of the M5 diagnosis arrived from outside the project during Phase 5 close-out.

### Phase 6a — Connectome Substrate & Architecture Comparison

Phase 6a closed 2026-07-07 with a Gate 3 GO ([Logbook 037](experiments/logbooks/037-phase6a-synthesis.md)): the Cook 2019 connectome imported and validated (L0), the plugin registry (L1), and the six-family ranking on the continuous substrate — `MLP 89.0 ≫ {CfC 75.8 ~ Transformer 74.0} > LSTM 60.1 > connectome 52.2 ≫ GA 15.0` ([Logbook 029](experiments/logbooks/029-continuous-architecture-ranking.md)) — with the degree-preserving rewired null showing that standing to be a degree-statistics property ([Logbook 034](experiments/logbooks/034-connectome-structure-controls.md)), and real-worm validation of both chemotaxis strategies ([Logbook 035](experiments/logbooks/035-realworm-chemotaxis-validation.md)). Phase 6b (NEAT) is deferred unscheduled, so the L3 criterion stays unmet.

### Phase 7 — Deepen: Plasticity

Phase 7 closed 2026-09-19 as **SPLIT** ([Logbook 069](experiments/logbooks/069-phase7-synthesis.md)): the registered 2×2 (rule × wiring) could not be answered because no biologically plausible rule that *writes* the connectome learns the substrate to any benefit — a diagnosed negative (the three-factor rule failed its positive control; node perturbation is near its theoretical worst case at 302 units; e-prop reaches competence only with the chemical matrix frozen, [Logbook 063](experiments/logbooks/063-l4-eprop.md)). What shipped instead is three citable results: **under PPO the wild-type wiring reaches competence 23–55% sooner than its degree-preserving rewired null** on two cells and fresh rewirings (block V, [057](experiments/logbooks/057-wiring-premise-contrast.md)/[058](experiments/logbooks/058-wiring-premise-difficulty.md)/[065](experiments/logbooks/065-wiring-fresh-rewiring.md)) — carrying the standing condition that rewiring and initialisation vary together *(**Restated 2026-09-21**, [Logbook 070](experiments/logbooks/070-init-sharing-control.md): A.1 partially discharges the pairing half of this condition — no dissolution was detected under either definition of a shared initialisation, survival was **established on five of eight readings**, and three remain **unresolved at that panel's sensitivity**. The across-seed half is untested. And the magnitude is less stable than the direction: on fresh seeds the thermal effect came in at **35%** of its committed size while hard350 came in at 105%.)* *(**Conditioned 2026-09-23**, [Logbook 071](experiments/logbooks/071-operating-point-surface.md): the advantage holds at a settling depth of at least four hops with a pooled motor readout under PPO — abolished at depth 3, reversed at depth 2 — and the reading learner puts the rewired null ahead at the same point. A.1's shared-initialisation survival is consistent at depth 6 on 32 seeds.)*; the wiring is **not** legible as fixed features through a four-class readout, rate-robustly ([064](experiments/logbooks/064-l4-frozen-features.md)); and widening the readout to one weight per motor neuron makes it legible **at one learning rate only** ([066](experiments/logbooks/066-l4-readout-width.md)/[068](experiments/logbooks/068-l1b-rate-calibration.md)). 7b (cross-species) was deferred (D14). The phase paid for the [phase protocol](research/phase-protocol.md).

### Known Gaps Carried into Phase 6+

- **No connectome-constrained architecture** — Phase 6's focal deliverable.
- **No biologically-plausible plasticity rules** — neuromodulator-gated three-factor rules (rate-based primary arm per D1; spiking-STDP optional) are Phase 7's focal deliverable, resolved as the D10 2×2 (rule × wiring).
- **No energy/metabolic model** (satiety is abstract, not ATP-based) — blocks dauer-state and dwelling-vs-roaming behaviours. Not on the Phase 6 critical path, but promoted from a pure Future-Directions item to a **soft Phase 7 dependency**: the L4 diffusible-signal layer grounds neuromodulator concentrations in internal state ("satiety where available"), and the faithful naturalistic-memory work both lean on a real metabolic-state signal. A *minimal* metabolic-state model (not full ATP biophysics) should be scoped alongside L4 when Phase 7 is planned; see [Phase 7](#phase-7-deepen--plasticity--cross-species-transfer). *(2026-08-27 review: this gap is smaller than stated — `BrainParams` already carries satiety/health to the brain boundary; the missing pieces are an internal-state sensory module and the neuromodulator concentration field, not agent plumbing.)*
- **Discrete grid-world (not continuous physics)** — addressed in Phase 6 alongside Rung 2 chemical gradients and corrected ASH/ADL contact-based nociception (the latter is owed correctness work flagged in [Logbook 011](experiments/logbooks/011-multi-agent-evaluation.md)).
- **No native body mechanics** (sinusoidal undulation, omega turns, pirouettes) — interop with OpenWorm/Sibernetic at the c302 boundary if needed; native implementation is not on the Phase 6 critical path.
- **Multi-task aggregation infrastructure** — Baldwin prerequisite; revisits if a future phase commits to the Baldwin question.
- **Dynamic-diffusion / source-dynamics chemical fields** — Phase 6's Fick gradients (T6) are *static* (frozen at assay time). Carried forward, decomposed: **(a) source dynamics** (food/chemical depletion → within-episode field change) is the biological route to area-restricted search / within-episode memory — held as the conditional `T7.separation.ars_depletion` task, gated on the bit-memory positive control; **(b) the full `∂C/∂t = D∇²C` PDE** is phase-7-depth, lowest priority (no behavioural model uses a live PDE — a point sensor can't perceive global field dynamics). Plus **per-signal `D` literature calibration** (food / predator sulfolipid via Liu et al. 2018 / CO₂): Phase 6 (`extend-fick-chemical-fields`) ships the Fick *mechanism* with tuned-scale defaults; biologically-grounded `D` values are phase-7 fidelity polish.
- **Naturalistic working-memory fidelity** — Phase 6 established that the architecture comparison *can* resolve working memory (the artificial `bit_memory_control`, [Logbook 030](experiments/logbooks/030-bit-memory-positive-control.md)), but the *biological* twins under-deliver at current fidelity: depletion-driven ARS demands only **short-horizon** integration and returns null ([Logbook 032](experiments/logbooks/032-ars-source-depletion.md)) — biologically correct, since real *C. elegans* memory is slow adaptation/integration, associative plasticity, and slow reference set-points, **not** delay-bridging working memory (the worm is reactive-dominated). The **Phase-6-testable** versions are *engineered within-episode* DMTS tasks — bit-memory and the chemosensory associative-memory probe, which **both separate the memory arms** from the memoryless MLP ([030](experiments/logbooks/030-bit-memory-positive-control.md), [033](experiments/logbooks/033-associative-memory-probe.md)); the associative probe's probabilistic-reversal variant also splits genuine working-memory *update* from hold-only retention (thermal cultivation-temperature DMTS remains an unused backup; [design sketch](research/associative-memory-probe.md)). The **faithful** versions — memory that *forms slowly* over minutes-to-hours / repeated trials / across episodes via neuromodulator-gated plasticity — are **phase-7**: the same computation as the L4 modulated-STDP deliverable applied to a *learning* task, needing long / multi-episode training + a neuromodulatory/metabolic-state substrate (couples to the no-energy-model gap above). The strongest new memory-arm candidate, **modified-S5**, is deferred here too: the within-episode probes are solved at ceiling by the existing arms, so S5 needs one of these harder / longer-horizon memory tasks to show separation headroom (see the deferred-arms gap below). Recorded so the naturalistic-memory question survives Phase 6 closure.
- **Deferred T7 architecture arms (non-gating)** — the candidate policy arms not brought into the T7 ranking are carried to Phase 7; none is gating for Gate 3. **modified-S5** — the strongest new memory candidate, deferred with the naturalistic-memory gap above (the within-episode probes are solved at ceiling, so it needs a harder / longer-horizon memory task for headroom). **NCP** (tap-withdrawal worm-circuit wiring; [arXiv:1803.08554](https://arxiv.org/abs/1803.08554)) — a biological-fidelity / interpretability arm that overlaps CfC + the connectome (would land in their reactive-cell band), so it fits Phase 7's L0-connectome / plasticity / cross-species work rather than the ranking. The **SHOULD/MAY opportunistic arms** — spiking (folds into the L4 STDP deliverable), quantum (RQ4 already settled negative at T4 under controlled attribution, [Logbook 025](experiments/logbooks/025-weight-search-architecture-ranking.md) — closed, not carried), and reservoir/hybrid (opportunistic only, no row earned per the 2026-06-29 architecture-candidate research). Per-arm rationale in `openspec/changes/phase6-tracking/tasks.md` § Tranche 7.

### Research Questions for Phase 6+

1. **Connectome ranking.** How does the wild-type *C. elegans* connectome rank against unconstrained MLP/LSTM, NEAT-evolved topologies, and quantum architectures on klinotaxis, thermotaxis, and predator evasion when learning and evolution operate on a common substrate? *(Grid answer in [Logbook 025](experiments/logbooks/025-weight-search-architecture-ranking.md): the connectome ranks mid-pack, below a tied quantum/CfC/spiking/LSTM top cluster, with no quantum advantage under controlled attribution. **Continuous-substrate answer now in hand — [Logbook 029](experiments/logbooks/029-continuous-architecture-ranking.md): `MLP 89.0 ≫ {CfC 75.8 ~ Transformer 74.0} > LSTM 60.1 > connectome 52.2 ≫ GA 15.0` (n=8, three significant tiers). The wild-type connectome ranks 5th of 6 under PPO weight search — beaten on all three behaviours by a plain MLP.** The rewired-null control (6a) tests whether the connectome's specific wiring is the cause; the NEAT-evolved-optimum comparison is deferred to Phase 6b.)*
2. **Connectome fitness landscape.** Is the wild-type connectome a local optimum on these behaviours, a basin, or a saddle? What synaptic-weight changes does evolution find when permitted to modify it?
3. **Plasticity and the connectome.** Does biologically-plausible plasticity (STDP, neuromodulator-modulated STDP) on the real connectome reproduce dynamics that match published *C. elegans* learning data (chemotaxis indices, Ca²⁺ correlation matrices)? *(Validation-method caveat for whoever scopes this — surfaced at the 2026-06-04 checkpoint, recorded here because the full methodology note otherwise lives only in the Phase 6 `tasks.md` T7.validation.1, which archives at Phase 6 close: a raw Ca²⁺-trace correlation-matrix match is a **category mismatch** for a behavioural/policy model and is only applicable to the connectome arm. The defensible form is comparing **behaviour-encoding tuning** — neuron-vs-behaviour tuning à la Atanas et al. 2023 / CePNEM — after mapping model units to named neurons (NeuroPAL/WormID). L4 plasticity makes this materially more defensible than under Phase 6's PPO weights, which is why it belongs here. Pin the concrete deliverable when Phase 7 is planned. This caveat is the *dynamics-claim* branch of the [Phase 7 § Claim discipline](#phase-7-deepen--plasticity--cross-species-transfer) gate — matching model tuning to named-neuron data is exactly bar (b), and it must clear bar (a) ensemble-invariance too.)*
4. **Architecture asymmetry under matched capacity.** Phase 5 M5 diagnosed architecture asymmetry as the blocker for Red Queen entanglement. Does matched-capacity NEAT-vs-NEAT or connectome-vs-connectome co-evolution produce the dynamics that LSTMPPO-vs-MLPPPO suppressed? *(**Deferred out of Phase 6 with no scheduled destination** — this is the co-evolution question, too compute-intensive for the phase and already independently corroborated (Resendez Prado 2026). The lag-matrix instrument is retained for whenever a future phase commits to it.)*
5. **Cross-species transfer.** Do learned/evolved architectures transfer from *C. elegans* to *P. pacificus* (Cook et al. 2025) on the shared behaviours? Where do they break, and what does that say about the connectome's role? *(Rescoped 2026-08-27: the Cook 2025 dataset is a head-only, chemical-synapse-only connectome from two animals — the question is answered at head-circuit scope with a matched-truncation *C. elegans* baseline; see Phase 7 Deliverable 2.)*

______________________________________________________________________

## Phase Roadmap

### Phase 0: Foundation & Baselines (COMPLETE)

**Status**: ✅ All required and stretch exit criteria met.

See [Current State — Phase 0](#phase-0--foundation--baselines) for achievements. Key breakthroughs:

- Evolutionary optimization (CMA-ES) achieving 4x better performance than gradient-based on quantum circuits
- Spiking neural network rewrite to surrogate gradient descent enabling viable learning
- PPO established as classical SOTA across all thermotaxis configurations

______________________________________________________________________

### Phase 1: Sensory & Threat Complexity (COMPLETE)

**Status**: ✅ All core exit criteria met. Oxygen sensing deferred to Phase 3.

See [Current State — Phase 1](#phase-1--sensory--threat-complexity) for achievements. Key deliverables:

- Thermotaxis system with 9 validated configurations (Logbook 007)
- Mechanosensation with boundary and predator contact detection
- Stationary and pursuit predator types
- HP-based health system with strategic damage/healing trade-offs

______________________________________________________________________

### Phase 2: Architecture Analysis & Standardization (COMPLETE)

**Status**: ✅ 300-session quantum architecture campaign complete. Carries forward as baseline reference in Phase 6's architecture-comparison protocol.

See [Current State — Phase 2](#phase-2--architecture-analysis) for the campaign summary; full results in [Logbook 008](experiments/logbooks/008-quantum-brain-evaluation.md). Key outcomes:

- 15 architecture variants systematically evaluated against matched-capacity classical baselines.
- Established that grid-world complexity is below the threshold for quantum advantage on every variant tested.
- Brain naming migration complete; paradigm-prefix scheme operational.
- Statistical framework operational (paired-seed Wilcoxon, bootstrap CIs).

Quantum architecture interpretability, mechanism discovery, and external biological-prediction validation are not separate Phase 2 deliverables — they fold into Phase 6's architecture-comparison protocol (with the 300-session campaign as baseline reference) and Phase 7's optional biological-validation collaboration.

______________________________________________________________________

### Phase 3: Temporal Sensing & Memory

**Goal**: Transform the simulation from stateless reflex to temporal integration. Make *C. elegans* sense the way it actually senses — through temporal derivatives, not spatial gradient lookups. This is the single most impactful biological-fidelity upgrade in the early phases; it also raises the substrate's complexity along two dimensions (non-Markovian dependencies and partial observability) that matter for architecture-comparison interpretation.

**Aspirational timeline**: —

#### Background

Real C. elegans uses temporal sensing for most modalities:

- **Thermotaxis**: AFD neurons detect temperature changes (dT/dt) with extraordinary sensitivity (0.01°C changes over a >10°C range). The worm compares current temperature to recent history, not spatial sampling.
- **Chemotaxis**: ASE neurons perform temporal concentration comparisons during head sweeps — the worm moves forward, senses concentration change over time, then adjusts.
- **Oxygen sensing**: URX/BAG neurons integrate oxygen changes over time.

Our current implementation provides spatial gradient information directly (gradient magnitude + direction), which is computationally convenient but **constitutes environmental cheating**. The environment computes central differences by sampling adjacent cells (T(x+1,y) - T(x-1,y))/2 and superposition of exponential decay functions from all food sources — information a ~1mm worm at position (x,y) cannot access. Switching to biologically honest sensing fundamentally changes the computational problem: agents must maintain memory, integrate signals over time, and infer gradient direction from their own movement history.

#### Anti-Cheating Principle

Phase 3 enforces **biological honesty**: the agent must only receive information available through its actual sensory neurons. For gradient-based modalities (chemotaxis, thermotaxis, aerotaxis), this means:

- **The agent receives only the scalar value at its current position** (concentration, temperature, O2 level)
- **The agent receives its own proprioceptive state** (heading, recent movement)
- **The agent must infer gradient direction** by correlating how scalar values change with its own movement over time

This is how real C. elegans navigates: a "biased random walk" where the worm moves forward, detects whether concentration is increasing or decreasing (temporal comparison), then modulates its turning probability. It does not follow a pre-computed gradient vector — it learns to turn less when things improve and turn more when they worsen.

**Mechanosensation** (boundary_contact, predator_contact) is already biologically honest — binary contact signals the agent actually experiences. These remain unchanged.

**Nociception** currently provides `predator_gradient_strength` and `predator_gradient_direction` — the same spatial gradient oracle as chemotaxis. A real C. elegans cannot sense predator direction at distance; it detects predator-secreted chemicals (sulfolipids) via the same temporal comparison mechanism as chemotaxis. Nociception must receive the same honest-sensing treatment: scalar chemical concentration at current position, with the agent inferring predator direction from temporal changes. This is included in deliverable 1 alongside chemotaxis and thermotaxis.

#### Deliverables

1. **Biologically Honest Sensory Inputs** [CRITICAL]

   Two sensing modes, both replacing the current spatial gradient oracle:

   - **Mode A — Raw scalar + memory (most biologically honest)**: Agent receives only the scalar reading at its current position (temperature in °C, chemical concentration, O2 level). No gradient information of any kind. The brain must use STAM memory buffers to store recent readings and learn temporal integration entirely on its own — discovering that "I moved forward and concentration increased, so food is probably ahead" from raw experience. This is the hardest mode and the most scientifically interesting.

   - **Mode B — Pre-computed temporal derivative (biologically plausible)**: Agent receives the scalar reading + dC/dt or dT/dt (rate of change over recent steps). This models what sensory neurons actually output — AFD neurons signal "warming" or "cooling", not "gradient points north-east". Still much harder than spatial gradients because there is no directional information — only "things are getting better/worse". The agent must correlate its movement direction (from proprioception) with whether values improved to infer where to go.

   - **Legacy mode**: Spatial gradients remain available for backward compatibility and as a comparison baseline, but are explicitly labelled as "oracle sensing" in configs and documentation.

   - Biologically calibrated: AFD sensitivity ~0.01°C changes, ASE concentration comparisons over ~1-second head sweep timescales.

   - Configurable per modality: each sensory module can independently use Mode A, B, or legacy.

2. **Short-Term Associative Memory (STAM)** [CRITICAL — prerequisite for Mode A sensing]

   - Exponential-decay memory buffers for recent sensory history (biological timescale: minutes to ~30 minutes)
   - Stores recent scalar readings, recent positions, recent actions — the raw material for temporal integration
   - No protein synthesis required (immediate formation, matches biological STAM)
   - Molecular basis: cAMP and calcium signaling pathways
   - Use cases: Remember recent sensory readings (for temporal derivative computation in Mode A), recent food/predator encounters, build spatial map from temporal experience
   - Integration with all brain architectures: memory state appended to observation vector

3. **Oxygen Sensing** [Pairs with temporal infrastructure]

   - O2 concentration gradient fields (5-12% optimal range, matching real C. elegans preference)
   - URX/AQR/PQR neuron simulation (detect hyperoxia >12%)
   - BAG neuron simulation (detect hypoxia \<5%)
   - Temporal O2 sensing using STAM buffers (dO2/dt)
   - Multi-objective: balance food quality vs. oxygen comfort vs. predator avoidance

4. **ITAM/LTAM** [Conditional on STAM success]

   - Intermediate-Term Associative Memory (30 min to hours): Two-pathway decay model inspired by cAMP + CaMKII signaling. Requires simulated protein synthesis gate.
   - Long-Term Associative Memory (hours to days): Persistent across simulation sessions. Spaced vs. massed training distinction matching biology.
   - **Validation gate**: Implement only if STAM improves foraging efficiency by ≥10% over baseline

5. **Associative Learning Paradigms**

   - **Classical conditioning**: Odor (CS) + food (US) → approach odor
   - **Aversive learning**: Pathogen exposure → avoid pathogen
   - **Context conditioning**: Temperature + food → prefer that temperature (NMDA receptor-dependent, RIM interneuron integration)

#### Metrics Focus

- **Oracle vs. honest comparison**: Quantify the performance gap between spatial gradient (oracle) and biologically honest sensing modes. This gap IS the measure of how much we were cheating.
- **Mode A vs. Mode B**: Does pre-computing dT/dt (Mode B) substantially help versus raw scalars (Mode A)? If so, the temporal derivative is a key computational primitive.
- **Temporal integration**: Do agents learn to correlate movement direction with value changes?
- **Memory utilisation**: Does STAM improve performance over stateless policies?
- **Classical ceiling change**: Does honest sensing lower classical success rates (creating headroom for quantum)?

#### Phase 3 Results (March 2026)

**Implementation completed:**

- ✅ Biologically honest sensing (Mode A and Mode B) operational for chemotaxis, thermotaxis, and nociception
- ✅ STAM implemented with biologically-calibrated exponential decay rates (buffer_size=30, decay_rate=0.1)
- ✅ New brain architecture: LSTMPPOBrain (`lstmppo`) — 19th architecture — with LSTM/GRU + chunk-based truncated BPTT
- ✅ GRU variant identified as superior to LSTM across all tasks
- ✅ 6 new lstmppo config files covering foraging, pursuit predators, and stationary predators
- ✅ Comprehensive evaluation across 4 environments × 3 sensing modes × 2 RNN types (see [logbook 009](experiments/logbooks/009-temporal-sensing-evaluation.md))

**Key findings:**

| Environment | Oracle L500 | GRU Derivative L500 | GRU Temporal L500 |
|---|---|---|---|
| Pursuit predators (large+thermo) | 97% | 88% | **94%** |
| Stationary predators (large+thermo) | 79% | 74% | **74%** |

- **Temporal Mode A achieves 94% L500 on the hardest environment** — within 3pp of oracle. Scalar-only sensing with GRU memory matches oracle at convergence.
- **GRU outperforms LSTM** by 3-40pp across all tasks. Fewer parameters, faster training.
- **BPTT chunk length** is the most critical hyperparameter — must match the temporal scale of the behavioral sequence.
- **Training efficiency** is the main gap: temporal needs 6000-12000 episodes vs oracle's ~300-1000. Capability is equivalent at convergence.

**Completed after initial Phase 3 evaluation:**

- ✅ Oxygen sensing (aerotaxis) — OxygenField with asymmetric 5-zone system, combined thermal+oxygen environments, full oracle/temporal/derivative support, STAM expanded to 4 channels

**Deferred to later phases:**

- ITAM/LTAM (STAM was sufficient — the GRU's internal memory makes explicit ITAM/LTAM less critical)
- Associative learning paradigms (deferred to Phase 5)

#### Phase 3 Exit Criteria

**Required (must complete before Phase 4):**

- ✅ Biologically honest sensing (Mode A or B) operational for thermotaxis, chemotaxis, and nociception
- ✅ STAM implemented with biologically-calibrated exponential decay rates
- ✅ Oracle vs. honest performance gap quantified: **converged gap is 3-7pp** (much smaller than expected)
- ✅ Classical approaches show measurable difficulty increase vs. oracle baseline: **training time increases 4-12x, but converged performance matches oracle**

**Stretch (can continue into Phase 4):**

- ✅ Oxygen sensing — implemented with asymmetric 5-zone system (URX/BAG neuron-inspired), combined thermal+oxygen environments, full temporal/derivative sensing support, and experiment tracking pipeline
- 🔲 Associative learning paradigms — deferred to Phase 5

#### Quantum architecture-comparison assessment (historical, Phase 3)

The v3 roadmap planned a Phase 3 quantum checkpoint to re-evaluate QRH and QEF on non-Markovian temporal tasks if classical ceiling dropped under temporal sensing.

**Result**: classical GRU PPO achieves oracle-level converged performance on temporal sensing. The classical ceiling did *not* drop — GRU temporal reaches 94% L500 on the hardest environment. The checkpoint did not trigger a quantum-campaign re-run; QRH's Phase 2 temporal-pursuit advantage was noted as worth revisiting if substrate changes plausibly amplified it. Training-efficiency comparisons (classical temporal agents need ~10× more episodes than oracle to converge) were flagged as a candidate axis for future quantum evaluation.

This assessment is preserved as historical record. Going forward, quantum architectures sit in the Phase 6 architecture-comparison sweep as one row among many — see [Architecture-Comparison Protocol](#architecture-comparison-protocol).

#### Go/No-Go Decision

**GO to Phase 4**: temporal sensing is operational and validated. Classical approaches handle temporal derivatives effectively with GRU PPO.

______________________________________________________________________

### Phase 4: Multi-Agent Complexity

**Goal**: Create multi-agent state spaces and study *C. elegans* social behaviours (aggregation, pheromone communication, alarm signalling) — all well-documented in the literature. Multi-agent dynamics expand the substrate's complexity surface for architecture comparison.

**Aspirational timeline**: —

**Prerequisites**: Phase 3 memory infrastructure (agents need to remember past interactions)

#### Background

C. elegans, while often considered solitary, exhibits sophisticated social behaviors:

- **Social feeding**: Feeding rate increases near conspecifics (social facilitation)
- **Aggregation**: Clustering on bacterial lawns mediated by ascaroside pheromones
- **npr-1 variation**: Natural genetic variation determines solitary vs. social feeding behavior
- **Alarm pheromones**: Injured worms release signals that repel nearby individuals
- **Cooperative-like behaviors**: Worms following pheromone trails benefit from others' foraging discoveries
- **Competition**: Limited food creates resource competition and dominance dynamics

Multi-agent scenarios create exponential state spaces (state × number of agents), partial observability (each agent has local view), and strategic interactions. These are complexity dimensions that shape what the architecture comparison can interpret — not separate quantum-advantage gates.

#### Deliverables

1. **Multi-Agent Infrastructure** [CRITICAL]

   - 2-10 independent agents in same environment
   - Each agent has its own brain instance (can be different architectures)
   - Agent-agent interaction tracking (proximity, collisions, food competition)
   - Scalable: performance linear in agent count, not quadratic

2. **Pheromone Communication**

   - **Aggregation pheromones**: Ascaroside-inspired chemical trails that attract nearby agents
   - **Alarm pheromones**: Released on predator contact or HP loss, repel conspecifics
   - **Food-marking trails**: Agents deposit chemical markers near food sources
   - Diffusion dynamics: pheromones spread and decay over time (uses Phase 3 temporal infrastructure)

3. **Social Feeding**

   - Feeding rate enhancement when near other agents (social facilitation)
   - Aggregation behavior: agents cluster on food patches
   - npr-1 behavioral variation: configurable solitary vs. social phenotypes

4. **Competitive Foraging**

   - Zero-sum resource competition: limited food, agents compete for access
   - Territorial behavior: agents defend food-rich zones
   - Game-theoretic analysis: Nash equilibria, evolutionarily stable strategies

5. **Collective Predator Response**

   - Coordinated evasion when one agent detects predator (via alarm pheromones)
   - Information sharing about predator locations
   - Collective aggregation as defense strategy

6. **Food Spatial Persistence** [IMPLEMENTED — PR #124]

   - ✅ **Food patches/hotspots**: Configurable regions where food spawns preferentially with exponential decay sampling
   - ✅ **Satiety-dependent foraging**: Agents cannot eat above satiety threshold (environmental gate)
   - Evaluation showed pheromones still neutral due to temporal sensing limitation (klinokinesis only — see D7)

7. **Klinotaxis Sensing** [IMPLEMENTED — issue #125]

   - ✅ **Head-sweep sensing mode**: Samples concentration at left/right offsets from heading direction, providing local spatial gradient + temporal derivative
   - Biologically most accurate mode — models ASE neuron bilateral comparison during head sweeps
   - Applies to all 7 gradient modalities (food, predator, temperature, oxygen, 3 pheromones)
   - Evaluation pending — expected to enable pheromone trail-following that was impossible with temporal-only sensing

#### Metrics Focus

- **Emergent phenomena**: Identify behaviors not explicitly programmed (spontaneous aggregation, division of labor, communication strategies)
- **Cooperation quantification**: Cooperation intensity, stability, efficiency gains over individual foraging
- **State space explosion**: Quantify effective state space growth with agent count
- **Classical ceiling**: Do classical approaches struggle with multi-agent coordination?

#### Phase 4 Exit Criteria

- ✅ ≥5 agents running stably with independent brains — *5 and 10-agent configurations run reliably across all evaluation campaigns (Logbook 011)*
- ✅ **≥1 emergent behavior documented (now met after Klinotaxis Era K1-K7)** — *K3 mixed-phenotype evaluation produced a robust within-episode frequency-dependent fitness gap of 40-45pp between followers (with `pheromone_food` perception) and loners (without), constant across all tested mixing ratios. The gap emerges entirely from sensory channel access, not programmed reward asymmetry — qualifying as emergent learned behaviour. Plus K1 collective cluster discovery (89% all-fed L100 vs 2% control) and K6 social feeding clustering.*
- ✅ Pheromone communication functional (at least alarm + food-marking) — *Infrastructure works correctly. Klinotaxis Era K1 demonstrates food-marking pheromones provide +77pp on agents-fed and +36× on all-fed under proper conditions (single persistent food cluster, klinotaxis sensing, strong pheromone parameters). Aggregation and alarm channels are informationally inert under all tested conditions — these limitations are documented for future code work.*
- ⚠️ Classical approaches show measurable strain on coordination tasks — *Partially met. 19.7% degradation at 10 agents with scarce resources (B1-v2), but zero coordination overhead with proportional resources (Campaign G). Strain is from resource scarcity, not coordination complexity. Klinotaxis Era did not re-test classical strain (out of scope; pheromone evaluation focus).*

#### Quantum architecture-comparison assessment (historical, Phase 4)

The v3 roadmap planned a Phase 4 quantum checkpoint to evaluate quantum entangled strategy spaces and quantum-enhanced Nash equilibrium solvers against multi-agent coordination tasks if classical ceiling dropped below ~85%.

**Assessment (Logbook 011, post Klinotaxis Era)**: the 80.2% ceiling at 10 agents (B1-v2) meets the numerical threshold, but Campaign G proved this is resource-allocation difficulty, not computational complexity. With proportional resources, classical MLP PPO achieves 100% of ceiling at all scales. The Klinotaxis Era K1 finding (89% L100 on 5-agent collective discovery with pheromones) represents successful learned use of an extra observation channel, not a search-space explosion that would favour quantum approaches. **The checkpoint did not trigger.** The environment does not create genuine coordination complexity at the scales tested. Conditions that could: harder partial observability, multi-cluster foraging requiring trail specialisation (K7 weakening suggests a candidate area), or co-evolutionary phenotype dynamics (Phase 5 territory).

This assessment is preserved here as historical record. Going forward, quantum architectures sit in the Phase 6 architecture-comparison sweep as one row among many — see [Architecture-Comparison Protocol](#architecture-comparison-protocol).

#### Go/No-Go Decision

**GO**: Multi-agent infrastructure is complete, functional, and produces emergent behaviour under proper conditions (K3 frequency-dependent phenotype fitness, K1 collective discovery, K6 social feeding). Pheromone communication mechanism is precisely characterised (six conditions for collective benefit). Negative findings (alarm channel inert, aggregation channel inert, multi-cluster generalisation partial) are scientifically informative. **Phase 4 is complete; proceed to Phase 5.** Deferred work for Phase 4: alarm pheromone emission semantics code changes (issue to be opened), multi-cluster pheromone scaling at higher agent counts, and per-step trajectory analysis for behavioural verification of trail-following.

______________________________________________________________________

### Phase 5: Evolution & Adaptation

**Goal**: Evolve optimal learning strategies and study how learning guides evolution, including biologically-documented transgenerational memory.

**Aspirational timeline**: —

**Prerequisites**: Phase 3 (memory infrastructure for transgenerational memory). Phase 4 multi-agent infrastructure required only for co-evolution (deliverable 4) — other deliverables can begin in parallel with Phase 4.

**Pilot-Then-Focus Approach**: Start with lightweight pilots of 2-3 evolutionary approaches using small populations and few generations. Based on pilot results, select 1-2 approaches for deep investigation.

#### Phase 5 Milestone Tracker

Phase 5 is broken into milestones M0–M8 plus a tracking scaffold (M-1). The living sub-task checklist lives in [openspec/changes/archive/2026-05-23-phase5-tracking/tasks.md](../openspec/changes/archive/2026-05-23-phase5-tracking/tasks.md); design decisions (pilot-first, no QVarCircuit backwards-compat, LSTMPPO+klinotaxis as first-class brain for M4/M5/M6) are recorded in that change's [proposal.md](../openspec/changes/archive/2026-05-23-phase5-tracking/proposal.md).

| # | Milestone | Bio fidelity | Status |
|---|-----------|--------------|--------|
| M-1 | Phase 5 tracking scaffold | — | ✅ complete |
| M0 | Brain-agnostic evolution framework | LOW | ✅ complete |
| M1 | Predator-as-brain refactor | MEDIUM | ✅ complete — zero-behavioural-change refactor (23 byte-equivalence tests + 80/80 metric-cell delta = 0.0). See [logbook 016](experiments/logbooks/016-predator-brain-refactor.md) |
| M2 | Hyperparameter evolution pilot | LOW | ✅ GO — predator arm CMA-ES +47pp, TPE +79pp (seed 43 rescued). RQ1 closed: TPE is M3's default. See [logbook 012](experiments/logbooks/012-hyperparam-evolution-mlpppo-pilot.md) |
| M3 | Lamarckian evolution pilot | MEDIUM | ✅ GO — predator + TPE + Lamarckian accelerates convergence +5.25 gens; all 4 seeds reach best fitness 1.00 (vs control 0.88-0.96); inheritance rescues TPE-unlucky seed 42. **Strongest concrete Phase 5 result**. See [logbook 013](experiments/logbooks/013-lamarckian-inheritance-pilot.md) |
| M4 | Baldwin effect demonstration | MEDIUM | ⚠️ STOP after 3 iterations — substrate-constraint diagnosis. Single-task K=50 PPO has no Baldwin axis; published demos require task distributions (Fernando 2018, Chiu 2024). Reusable infrastructure (F1 evaluator, 4-way aggregator, n=8 Lamarckian extension) shipped. See [logbook 014](experiments/logbooks/014-baldwin-inheritance-pilot.md) + [logbook 015](experiments/logbooks/015-baldwin-iterative-evaluation.md) |
| M5 | Co-evolution arms race | HIGH | ❌ STOP — 13 single-seed lever ablations all produced own-vs-cross fitness lag delta +0.017 to +0.024 (target ≤−0.05). Architecture-asymmetry diagnosis (LSTMPPO-prey vs MLPPPO-predator capacity gap). Methodology contributions (lag-matrix + cell-grid + per-gen re-aggregation) ship as standalone outputs. See [logbook 017](experiments/logbooks/017-coevolution-arms-race.md) |
| M6 | Transgenerational memory (gated on M3 ✅) | HIGH | ⚠️ Framework shipped, INCONCLUSIVE on the science — post-pilot audit identified four blocking design issues (substrate shape, training reward, F0 probe context, F1+ compute asymmetry). Re-evaluated in M6.9+ / M6.13. See [logbook 018](experiments/logbooks/018-transgenerational-memory.md) |
| M6.9+ | TEI re-evaluation (audits A/B/C) | HIGH | ⛔ STOP on pure-TEI K=0 — cross-arm `tei_on − control` mean delta −49pp; all four tripwires pass at calibration. Mechanism: a fresh-init brain at K=0 cannot use a substrate to substitute for trained PPO weights + LSTM hidden state. Empirical + theoretical + biological evidence converges. See [logbook 019](experiments/logbooks/019-transgenerational-memory-redesign.md) |
| M6.13 | TEI-as-prior-on-Lamarckian | HIGH | ⛔ STOP — fair-F0 comparison shows `tei_weights − weights_only` F1-F3 delta +0.00pp at K=1000 (inert) and −9.33pp at K=200 (interferes). Bias-network logit-prior is the wrong abstraction for the wet-lab single-circuit excitability shift; M6 closes coherently. See [logbook 020](experiments/logbooks/020-tei-prior-on-lamarckian.md) |
| M7 | NEAT topology evolution | LOW | 🔲 not started. Single-population NEAT topology search ships as **Phase 6b** (ex-T8). The co-evolution reframe (matched-capacity NEAT-vs-NEAT vs asymmetric NEAT-vs-MLP, testing M5's architecture-asymmetry) is **deferred out of Phase 6 with no scheduled destination** (2026-07). Independent corroboration: Resendez Prado arXiv 2604.03565 identifies the same "transparent regime" suppression effect |
| M4.7 | Multi-task Baldwin retry | HIGH | 🔲 deferred — gated on M5 producing multi-task aggregation infrastructure or M5's secondary-Baldwin instrumentation coming back null with rising priority |
| M8 | Phase 5 synthesis logbook | — | ✅ complete — all five Phase 5 exit criteria verified MET (two with substrate-grounded STOP caveats). See [logbook 021](experiments/logbooks/021-phase5-synthesis.md) |

**How to orient**: read this tracker for the current milestone, then [tasks.md](../openspec/changes/archive/2026-05-23-phase5-tracking/tasks.md) for sub-task detail, then any active per-milestone OpenSpec change under `openspec/changes/` (archived changes live under `openspec/changes/archive/`). Open Phase 5 research questions are tracked in `tasks.md` under "Phase 5 Research Questions"; check there before assuming a Phase 5 design choice is settled.

#### Phase 5 Results (May 2026)

**Implementation completed:**

- ✅ Brain-agnostic evolution framework (M0): CMA-ES, TPE, GA optimisers + Lamarckian / Baldwin / transgenerational inheritance strategies
- ✅ Predator-as-brain refactor (M1): `PredatorBrain` Protocol + per-predator metrics with zero behavioural change
- ✅ Hyperparameter + Lamarckian pilots (M2 + M3) — both GO verdicts
- ✅ Transgenerational-memory framework (M6 + M6.9+ + M6.13): substrate dataclass, LSTMPPO `tei_prior` actor-logit hook, F0 substrate-extraction telemetry, per-gen `lawn_schedule` consumer, paired-arm aggregator
- ✅ Methodology contributions: lag-matrix + cell-grid fair-test + per-gen re-aggregation instruments (M5)

**Key findings:**

- **Lamarckian (M3) is the strongest concrete result**: predator + TPE + inheritance produces +47pp / +79pp lift over hand-tuned and rescues TPE-unlucky seeds.
- **Three negative findings** with substrate-grounded diagnoses, all field-corroborated: Baldwin (M4) needs task distributions; co-evolution (M5) needs architecture-symmetric capacity; TEI (M6) needs an upstream sensory-excitability transform rather than an action-distribution bias.
- **All five Phase 5 exit criteria MET** (two with substrate-grounded STOP caveats on M5/M6) — see [logbook 021 § M8.2](experiments/logbooks/021-phase5-synthesis.md).

**Deferred to later phases:**

- M6 substrate-redesign carries forward as an open direction addressable through connectome-constrained + continuous-physics architectures. M5 architecture-asymmetry (co-evolution) is **deferred with no scheduled destination** (2026-07) — removed from Phase 6, not reassigned; the lag-matrix instrument is retained for whenever a future phase commits to it.

#### Deliverables

1. **Hyperparameter Evolution** [Priority: Pilot First]

   - Genome = learning rates, layer sizes, circuit depths, reward weights
   - Tournament selection, fitness = final performance after fixed training episodes
   - Use case: Find optimal hyperparameter sets for each architecture

2. **Lamarckian Evolution** [Pilot alongside Hyperparameter]

   - Offspring inherit learned weights (not biologically accurate but fast convergence)
   - Fitness = final performance after learning
   - Use case: Rapidly evolve high-performing initial conditions

3. **Baldwin Effect** [Conditional: if Lamarckian shows promise]

   - Offspring inherit *ability to learn*, not learned weights
   - Over generations: learned behaviors become innate (genetic assimilation)
   - Biologically significant: study how learning guides evolution

4. **Co-Evolution (Predators + Prey)** [Benefits from Phase 4 multi-agent]

   - Predators evolve hunting strategies while prey evolve evasion
   - Red Queen dynamics: arms race between predator and prey
   - Fitness: Prey = survival rate, Predators = kill rate

5. **Transgenerational Memory** [NEW — biologically documented]

   - Based on Posner et al. (2023): associative memories can be inherited across generations in C. elegans
   - Epigenetic mechanisms: small RNAs and chromatin modifications
   - Implementation: selected memory traces transfer to offspring (configurable heritability)
   - Use case: Study how learned pathogen avoidance or temperature preferences persist across generations

6. **Architecture Evolution (NEAT-style)** [Optional]

   - Genome = network topology + weights
   - Speciation: protect novel architectures during early stages
   - Use case: Discover novel hybrid quantum-classical architectures

#### Phase 5 Exit Criteria

- ✅ ≥2 evolution approaches piloted with documented results
- ✅ Baldwin Effect or Lamarckian inheritance demonstrated — Lamarckian path (M3); Baldwin attempted but substrate-grounded STOP
- ⚠️ Co-evolution produces arms race dynamics — MET WITH CAVEAT: M5 ran exhaustive screen sweep; verdict STOP (architecture-asymmetry diagnosis)
- ⚠️ Transgenerational memory functional — MET WITH CAVEAT: framework fully functional + tested; three rounds of pilot all STOP (substrate-shape diagnosis)
- ✅ Generational fitness tracking shows continuous improvement over ≥50 generations

Caveats walkthrough at [logbook 021 § M8.2](experiments/logbooks/021-phase5-synthesis.md).

#### Quantum Note

Phase 5 does not include a formal quantum checkpoint — evolution does not directly create new computational complexity in the way temporal sensing or multi-agent dynamics do. However, NEAT-style architecture evolution (deliverable 6) could discover novel quantum-classical hybrid topologies worth evaluating. If architecture evolution produces interesting quantum circuit structures, these should be flagged for evaluation at the Phase 6 checkpoint.

#### Go/No-Go Decision

**GO if**: Evolution produces novel, high-performing behaviors OR demonstrates Baldwin Effect.
**PIVOT if**: Evolution plateaus quickly → Focus on hand-designed architectures. Document evolutionary limitations.
**STOP if**: Evolutionary algorithms fail to converge → Revisit fitness functions or population parameters.

______________________________________________________________________

### Phase 6: Connectome Substrate & Architecture Comparison

**Goal**: Build the platform on which learning and evolution operate on the real *C. elegans* 302-neuron connectome in a closed sensory-motor loop, and use it to rank the wild-type connectome against MLP, recurrent, spiking, reservoir, quantum, hybrid, and NEAT-evolved architectures on three nematode behaviours (klinotaxis, thermotaxis, predator evasion). The headline platform claim is *first closed-loop learning + evolution on the real C. elegans connectome with a pluggable architecture interface*.

**Aspirational timeline**: ~6-10 months from Phase 5 close.

#### Phase 6 Tranche Tracker

Phase 6 is broken into nine tranches with deliberate ordering — L0 ingest (T1) → L1 plugin refactor (T2) → corrected ASH/ADL nociception (T3) → L2 first pass on grid substrate (T4) → platform refactor (T5: continuous-2D + continuous-action heads + plugin-parity verification) → env fidelity (T6: Rung 2 gradients + log-concentration adaptation) → L2 re-run on fully-upgraded substrate + real-worm validation (T7) → L3 NEAT (T8) → synthesis logbook (T9). Three mid-phase decision gates close T2 (Gate 1), T5 (Gate 2), and T7 (Gate 3), each with quantitative pre-registered pass criteria. The ordering is load-bearing: T3 precedes T4 so predator-evasion L2 cells run against corrected nociception from the start; T5 is split from T6 so Gate 2 closes against a single verifiable platform-refactor outcome rather than a bundled env-upgrade tranche; T5 + T6 sit between T4 and T7 so a (qualitative, cross-regime — the substrates are non-commensurable; reframed 2026-06-14) grid-vs-continuous comparison is itself a Phase 6 finding. L4 (biologically-plausible plasticity) is deferred to Phase 7. The living sub-task checklist lives in [openspec/changes/phase6-tracking/tasks.md](../openspec/changes/archive/2026-07-06-phase6-tracking/tasks.md); seven design decisions (tranching, Cook 2019 via `cect` as L0 primary, L1 plugin-parity is real refactor work, four-MUST architecture-family scope, three-behaviour scope, mid-phase gate discipline with quantitative criteria, L2 connectome semantics with explicit connection-type taxonomy) are recorded in that change's [proposal.md](../openspec/changes/archive/2026-07-06-phase6-tracking/proposal.md) and [design.md](../openspec/changes/archive/2026-07-06-phase6-tracking/design.md).

| Tranche | Scope | Roadmap layer | Approx duration | Gate trigger | Status |
|---|---|---|---|---|---|
| P6-0 | Phase 6 tracking scaffold (this change) | — | — | — | ✅ done (archived at 6a close) |
| 1 | L0 connectome ingest — Cook 2019 hermaphrodite (302 neurons, 3709 chemical synapses, 1093 gap junctions) via direct *Nature* SI parsing; cross-validated against Witvliet 2021 nerve-ring (180 shared); forward-pass smoke + 71-test suite | L0 | 2-3 weeks | — | ✅ complete (logbook [022](experiments/logbooks/022-connectome-substrate.md)) |
| 2 | L1 plugin refactor (dispatcher → registry + topology/rule factoring + 19-architecture migration with regression bar) + connectome-as-brain wired through existing grid env | L1 | 3-5 weeks | **Gate 1 — GO** | ✅ complete (logbook [023](experiments/logbooks/023-architecture-plugin-interface.md)) |
| 3 | Corrected biology-driven predator sensing — two-channel split (contact-mechanosensory ASH/ALM/AVM/PLM + distal-chemosensory ASH/ASI sulfolipid per Liu et al. 2018) replacing the single chemosensory-at-distance `nociception` model flagged in Logbook 011 | env-correctness | 1-2 weeks | — | ✅ complete (logbook [024](experiments/logbooks/024-predator-sensing-biology.md)) |
| 4 | L2 initial pass — one integrated grid-substrate C3 cell (food + predator + thermotaxis active simultaneously, n=8 paired seeds), chemical-synapse strict-mask connectome; 4 MUST families + 3 Phase-4.5 promotions (quantum, spiking, CfC) = 7 families, per-behaviour sub-metrics extracted | L2 (first pass) | 4-6 weeks | — | ✅ complete (logbook [025](experiments/logbooks/025-weight-search-architecture-ranking.md)) |
| 5 | Platform refactor — continuous-2D coordinates + continuous-action heads on existing MUST brains; plugin-parity verified in practice | env-upgrade (platform) | 3-4 weeks | **Gate 2** | ✅ complete — **Gate 2 GO** ([logbook 027](experiments/logbooks/027-platform-refactor-continuous-2d.md)) |
| 6 | Env fidelity — signal-type-specific **static** Fick-shaped gradients + adaptive/biphasic chemosensory sensor (dynamic-diffusion PDE descoped to a stretch 2026-06-04) | env-upgrade (fidelity) | 3-4 weeks | — | ✅ complete (gating scope) — [logbook 028](experiments/logbooks/028-rung2-gradients-adaptive-sensor.md); fidelity renderer landed (`add-continuous-fidelity-renderer`, `--theme pixel_continuous`; non-gating) |
| 7 | L2 re-run on fully-upgraded substrate + real-worm validation + connectome-structure controls (rewired-null + learnable-gap-junction, decoupled from T8) | L2 (final) | 4-6 weeks | **Gate 3 → GO** | ✅ **CLOSED** — ranking [029](experiments/logbooks/029-continuous-architecture-ranking.md); validation [035](experiments/logbooks/035-realworm-chemotaxis-validation.md)/[036](experiments/logbooks/036-realworm-thermotaxis-validation.md); rewired-null [034](experiments/logbooks/034-connectome-structure-controls.md) (learnable-gj deferred) |
| 8 | L3 NEAT topology search on upgraded substrate (co-evolution removed) | L3 | 6-10 weeks | — | ⏸️ **deferred to Phase 6b, and 6b is now deferred unscheduled** *(2026-09-19)* — gated on GPU + the env-vectorisation decision, which has not been taken |
| 9a | Phase 6a synthesis logbook (T1–T7 + controls) | synthesis | 1-2 weeks | — | ✅ done — [Logbook 037](experiments/logbooks/037-phase6a-synthesis.md) (Gate 3 GO) |
| 9b | Phase 6b synthesis addendum (NEAT results) + Phase 6 COMPLETE marker | synthesis | 1-2 weeks | — | ⏭️ Phase 6b |
| — | Co-evolution (matched-capacity NEAT-vs-NEAT; ex-T8.4) | L3 | — | — | ⏭️ **deferred, no scheduled destination** |
| — | L4 biologically-plausible plasticity (STDP + neuromodulator-modulated) | L4 | — | — | ⏭️ deferred to Phase 7 |

**How to orient**: read this tracker for the current tranche, then [tasks.md](../openspec/changes/archive/2026-07-06-phase6-tracking/tasks.md) for sub-task detail, then any active per-tranche OpenSpec change under `openspec/changes/` (archived changes live under `openspec/changes/archive/`). Open Phase 6 research questions are tracked in `tasks.md` under "Phase 6 Research Questions"; check there before assuming a Phase 6 design choice is settled. The three mid-phase gates each produce a written go/no-go decision in the triggering tranche's published *logbook* (not in `tasks.md`, which is hard to amend post-archive) — the tracker links to each logbook decision once it lands. Each gate has quantitative pass criteria pre-registered in [openspec/changes/phase6-tracking/design.md § Decision 6](../openspec/changes/archive/2026-07-06-phase6-tracking/design.md).

#### Phase 6a / 6b split

Phase 6 is **delivered in two shipments** — one phase, two cuts, not two phases. The pre-registered 6a/6b sub-phase split (see § Risk-mitigation) is invoked here **by success rather than by overrun**: the platform, the full architecture ranking, and the memory-axis findings are done, so T1–T7 forms a complete, self-contained, citable result, while the L3 NEAT topology search is a compute-heavy tail whose headline is partly pre-answered by Logbook 029 (a plain MLP already beats the connectome, so "is the connectome a topological optimum?" is directionally *no* before NEAT runs).

- **Phase 6a = T1–T7 + connectome-structure controls + a 6a synthesis (T9a).** The platform (L0/L1/L2), all sensory physics, the six-family MUST architecture ranking (Logbook 029), the memory-axis programme (Logbooks 030–033), real-worm validation, **plus the rewired-null and learnable-gap-junction controls decoupled from T8** (these are PPO-based connectome ablations on the existing pipeline — they extend the T7 connectome ranking's credibility and need no NEAT infrastructure). 6a is the milestone / paper-ready cut. **6a is COMPLETE — Gate 3 GO** ([Logbook 037](experiments/logbooks/037-phase6a-synthesis.md)): real-worm validation landed as [035](experiments/logbooks/035-realworm-chemotaxis-validation.md) (chemotaxis, both strategies) and [036](experiments/logbooks/036-realworm-thermotaxis-validation.md) (thermotaxis, PARTIAL), and the rewired-null control as [034](experiments/logbooks/034-connectome-structure-controls.md). The learnable-gap-junction variant is deferred and is **not** gating.
- **Phase 6b = T8 NEAT topology search + a 6b synthesis addendum (T9b).** The full TensorNEAT topology search, gated on GPU availability **and** an environment-vectorisation decision (the ~500× TensorNEAT speedup assumes a vmappable env; `Continuous2DEnvironment` is not one yet — env throughput, not the GPU, is the binding constraint). 6b gets its **own lightweight tracker** (`phase6b-tracking`, authored when 6b starts) so `phase6-tracking` can close and archive at 6a completion rather than staying open on the deferred tail. Co-evolution (ex-T8.4) is **not** part of 6b — it is deferred with no scheduled destination.

**Phase 6's exit criteria are met across 6a with one exception**: the L3 criterion (NEAT topology-search results) is satisfied only by 6b, and *(updated 2026-09-19 at the Phase 7 close)* **6b is deferred unscheduled with its tracker closed off**, so that criterion **stays unmet** and Phase 6 stays *6a COMPLETE / 6b pending*. It was originally scoped as satisfied by 6b. 6b was initially slotted into Phase 7's early L4-software window; the 2026-08-27 Phase 7 review **decoupled it** (D13 — opportunistic, no scheduled window, pending a dated GPU/cloud decision). Phase 7 never gates on it. Phase 6 is marked COMPLETE only when 6b's synthesis lands.

#### The layered platform

Phase 6 is built as four layers (L0-L3) that together materialise the architecture-comparison sweep. L4 (biologically-plausible plasticity) is deferred to Phase 7 — Phase 6 stays on PPO-family learning rules so the layer stack remains tractable.

| Layer | What it is | Phase 6 commitment |
|---|---|---|
| **L0 — Connectome substrate** | Import *C. elegans* 302-neuron wiring (Cook et al. 2019 / OpenWorm c302 in NeuroML 2 format). Real synaptic adjacency. Defines the topology interface that pluggable brains conform to. | **MUST.** The headline claim doesn't exist without this. |
| **L1 — Architecture-as-plugin** | A clean `Brain` interface where every architecture family conforms. The comparison is one experimental sweep, not a per-architecture re-implementation. Plugin parity test: adding a new architecture is bounded by an informal "≤ 1 week" target; the **load-bearing** parity checks are files-touched count + no per-architecture branches in the simulation/training loops (see [openspec/changes/phase6-tracking/design.md § Decision 6 § Gate 2](../openspec/changes/archive/2026-07-06-phase6-tracking/design.md) for the criterion details and the explicit demotion of the wall-clock target). Developer-facing how-to: [docs/architecture/plugin-developer-guide.md](architecture/plugin-developer-guide.md). | **MUST.** Without this, "swap in another brain" is words, not code. |
| **L2 — Weight search (PPO et al.)** | Train weights on the connectome topology and on every comparison architecture. This is the *first closed-loop learning on the C. elegans connectome*. | **MUST.** Cheapest scientifically meaningful Phase 6 result. |
| **L3 — Topology search (NEAT-style)** | Search topologies unconstrained, compare to the real connectome's topology. Tests "is the wild-type connectome a local optimum?" *(The Phase 5 M5 architecture-asymmetry / co-evolution follow-up that once rode along here is deferred out of Phase 6 — L3 keeps only the single-population topology search.)* | **MUST** *(ships as Phase 6b)*. Without this, the optimal-vs-connectome comparison has no "optimum" to compare against. |
| **L4 — Plasticity / learning rules** | Biologically-plausible plasticity (STDP, neuromodulator-modulated three-factor STDP) on the connectome. The Nature-Neuroscience-tier claim. | **DEFERRED to Phase 7.** Substantial new code; clean L1 is a prerequisite. |

> **Tranche note**: L2 ships as two passes per the Phase 6 Tranche Tracker above — a first pass (Tranche 4) on the existing grid substrate with corrected ASH/ADL nociception (Tranche 3), followed by the env-upgrade work split across two tranches (Tranche 5 platform refactor — continuous-2D coordinates + continuous-action heads on the existing PPO-family brains + plugin-parity verification; Tranche 6 env fidelity — Rung 2 chemical gradients + log-concentration chemosensory adaptation), then an L2 re-run on the fully-upgraded substrate plus real-worm validation (Tranche 7). The comparison between the two L2 passes is a Phase 6 finding — but a **qualitative cross-regime** one: the grid and continuous substrates are **non-commensurable** (continuous float `(speed, turn)` kinematics, Euclidean-disc geometry, Fick/adaptive fidelity all shift together, so carried-over parameters do not define a "matched" difficulty), so it is reported as "do the same architectures learn the same repertoire, and does the ranking's character survive the fidelity jump?", not as a controlled single-variable delta. The clean primary result is the **within-T7 ranking** (all arms on the same continuous substrate); the continuous substrate is calibrated on its own terms (biological-where-valid + multi-objective-learnable) and anchored externally by real-worm behavioural validation. *(Reframed 2026-06-14 — see the tracker's two-regimes checkpoint.)* L3 (Tranche 8) runs against the upgraded substrate. See [openspec/changes/phase6-tracking/design.md § Decision 1](../openspec/changes/archive/2026-07-06-phase6-tracking/design.md) for the load-bearing rationale (T5/T6 split places Gate 2 against a single verifiable platform outcome rather than a bundled env-upgrade tranche).

#### Behavioural scope: three behaviours on a common substrate

Phase 6 commits to the same three behaviours across every architecture in the comparison sweep:

- **Klinotaxis** — chemical gradient ascent via head-sweep modulation. Phase 4's klinotaxis sensing is the substrate.
- **Thermotaxis** — thermal gradient navigation. Phase 1's thermotaxis configurations carry forward.
- **Predator evasion** — escape from pursuit predators, integrating the corrected ASH/ADL contact-based nociception (see Realistic Sensory Physics below).

Aerotaxis (oxygen sensing), pheromone signalling, and multi-agent dynamics are *deferred* — including them pushes Phase 6 past 10 months and weakens the focused architecture-comparison framing. They re-enter scope in a future phase if the connectome story holds and additional behaviours become scientifically warranted.

#### Architecture families in the comparison sweep

The L1 plugin interface accommodates this curated set. The list is not "all 19 existing architectures" — Phase 6 picks the representatives that test the load-bearing questions and leaves historical variants in their Phase 0-3 logbooks. The MUST / SHOULD / MAY classification below was tightened from the roadmap's initial eight-MUST framing during Phase 6 scoping, then CfC was promoted SHOULD→MUST at the 2026-06-04 mid-phase checkpoint, and Transformer MAY→MUST at the 2026-06-07 post-T5 checkpoint (now six MUST families); rationale lives in [openspec/changes/phase6-tracking/design.md § Decision 4](../openspec/changes/archive/2026-07-06-phase6-tracking/design.md). Under the realised integrated-C3 cell shape (one cell per family, n=8, per-behaviour sub-metrics extracted), that is six MUST integrated-C3 cells at T7 (T4 ran five — Transformer was built at T5, so it has no T4 grid cell unless one is back-filled). SHOULD/MAY rows are evaluated opportunistically in Tranche 7 and do not gate any Phase 6 exit criterion.

| Family | Existing impl | Scope | Phase 6 role |
|---|---|---|---|
| **Connectome-constrained (302-neuron, Cook 2019)** | Not yet | **MUST** | **Focal architecture.** The wild-type topology with PPO-learned weights. The headline rank. |
| **MLP-PPO** | `MLPPPOBrain` | **MUST** | Strongest classical baseline (Phase 2 SOTA on foraging); cheapest run; sanity anchor. |
| **LSTM / GRU-PPO** | `LSTMPPOBrain` | **MUST** | Strongest temporal baseline (Phase 3 reached 94% L500); matched-capacity comparator for connectome (both have recurrent state). |
| **NEAT-evolved (topology + weights)** | Not yet (L3 deliverable) | **MUST** | The unconstrained-optimal baseline against which the connectome is ranked. L3's whole point — answers "is the connectome a local optimum?" |
| **CfC (liquid / closed-form continuous-time)** | `CfCPPOBrain` | **MUST** *(promoted 2026-06-04)* | Co-top T4 performer (Logbook 025); the NCP/LTC/CfC lineage is C. elegans-derived (Lechner et al. *Nat. MI* 2020; Hasani et al. *Nat. MI* 2022) — the most worm-relevant non-connectome comparator. Promoted from the Phase-4.5 set to a first-class row in T7 + T8. |
| **Quantum** | `QVarCircuitBrain`, `QEF`, others | **SHOULD** | Phase 2's 300-session campaign carries forward as baseline reference per the Architecture-Comparison Protocol. RQ4 settled negative at T4 under controlled attribution (Logbook 025); one quantum row at T7 continuous-physics complexity tests only whether higher complexity crosses the quantum-advantage threshold. |
| **Spiking (PPO-trained)** | `SpikingReinforceBrain` | **SHOULD** | Bridge to L4. But Phase 0's 73.3% on much easier tasks isn't a strong precedent; demoted so Phase 6 doesn't gate on spiking-on-connectome training. Phase 7 L4 (STDP — spiking's native plasticity rule) is where spiking-on-connectome actually belongs. |
| **Reservoir** | `QRHBrain`, `CRHBrain` | **MAY** | Phase 2 preserved QRH's +9.4pp pursuit advantage at low absolute performance. One row if cheap; not worth blocking on. |
| **Hybrid quantum-classical** | `HybridQuantum`, `HybridClassical` | **MAY** | Phase 2 SOTA finding (96.9% / 96.3%) survives as baseline reference. One row to confirm at higher complexity if cheap. |
| **Transformer / attention-based** | `TransformerPPOBrain` (built at T5) | **MUST** *(promoted 2026-06-07)* | Temporal-window self-attention, built + validated at T5 as the Gate-2 parity vehicle (5-file addition; clean discrete-klinotaxis learning; continuous head already wired). Promoted from MAY because it is the strongest working-memory / long-context comparator — the axis Logbook 025 found the top cluster ties on — and the marginal engineering is near-zero. First-class T7 integrated-C3 row. See [openspec/changes/phase6-tracking/design.md § Decision 4](../openspec/changes/archive/2026-07-06-phase6-tracking/design.md). |

> **Anti-scope-creep**: promoting a SHOULD or MAY family to MUST (so that family's results gate a Phase 6 exit criterion) requires amending [openspec/changes/phase6-tracking/design.md § Decision 4](../openspec/changes/archive/2026-07-06-phase6-tracking/design.md) before the promoting tranche merges. Same for adding an eleventh family beyond the ten listed here. The amendment must document the cross-tranche budget impact.

#### Continuous environment + sensory physics

The platform leaves the discrete grid behind in Phase 6, but the body-mechanics fidelity story is deliberately scoped down from v3.

- **Continuous 2D coordinates + continuous action space** (speed 0-to-max + turning angle −π to π). Realistic spatial scales: ~1mm worm body on cm-scale plates. Existing PPO-family brains extend with continuous action heads (Gaussian policy); quantum architectures adapt with continuous-output circuits.
- **Body mechanics**: native sinusoidal undulation, omega turns, and pirouettes are **not** implemented in Phase 6. If behavioural-fidelity claims later require them, the platform interoperates with OpenWorm Sibernetic at the c302 boundary. The platform claim ("first closed-loop learning on the connectome") survives without native body physics; the behavioural-fidelity claim ("matches real-worm movement statistics") is a separate, optional add.
- **3D environment is deferred to Future Directions.** Wild *C. elegans* lives on agar plates (functionally 2D); the experimental reference data Phase 6 validates against (Bargmann chemotaxis indices, Kavli Ca²⁺ recordings, BAAIWorm correlation matrices, Witvliet connectomes) is all 2D-plate data. Going 3D widens the model-vs-data gap; it does not strengthen the platform claim.

**Chemical-gradient fidelity — Rung 2 commitment.** The v3 phrase "Fick's-law gradients" is too vague. Phase 6 commits explicitly to Rung 2 of four possible rungs of fidelity:

| Rung | What it adds | Field-level realism |
|---|---|---|
| 0 (current) | Superposition of static 1/r or exponential-decay terms; no time evolution | Not biologically realistic |
| 1 (minimal Fick's-law) | Heat-equation diffusion (∂C/∂t = D∇²C); single diffusion coefficient; static sources | Refactor, not a fidelity upgrade — the brain still sees a normalised scalar |
| **2 (Fick-shaped gradients + adaptive sensor) — Phase 6 target** *(rebalanced 2026-06-04)* | Signal-type-specific D values (food vs pheromone vs CO₂) setting **static** Fick-shaped gradient geometry; paired with an **adaptive-threshold / biphasic chemosensory sensor** on AWC/AWA/ASE-style sensors. **Stretch goal (descoped from the gating target):** time-evolving diffusion (∂C/∂t = D∇²C), source depletion/replenishment, decay terms — pursued only where a concrete behavioural need exists (e.g. depletion-driven area-restricted search). | The sensor half is what the computational-chemotaxis field actually invests in |
| 3 (multi-species + substrate) | Vector of chemical signals with cross-modal receptor overlap; substrate-varying diffusion; bacterial biofilm boundary layers | OpenWorm-level fidelity; specialist territory |

Rung 2 has two components — gradient geometry AND chemosensory adaptation kinetics — but the 2026-06-04 checkpoint rebalanced where the fidelity budget goes. **The sensor half is the high-leverage investment; the environment-dynamics half was over-built relative to the field.** No published C. elegans chemotaxis model (Pierce-Shimomura 1999 → Iino & Yoshida 2009 → Izquierdo & Beer 2013 → Hironaka & Sumi 2025) uses a live time-evolving diffusion PDE — they use the *frozen analytic* Fick solution at one assay time, because a single-point sensor on assay timescales cannot perceive global field dynamics. So Phase 6 keeps signal-type-specific D for *static* gradient geometry and reallocates the PDE effort into a properly adaptive sensor: the field standard is an adaptive-threshold / biphasic LN model (Kato et al. 2014 *Neuron*; Levy & Bargmann 2020 *Neuron* — "hyper-Weber"), of which plain log-concentration is an under-powered special case. Dynamic diffusion + source depletion stays available as a stretch goal, justified case-by-case (its one strong use is creating depletion-driven area-restricted search — the only biologically-plausible route to the within-episode-memory demand that T4 found the reactive regime lacks; see Logbook 025 Limitations).

**Mechanosensation: corrected ASH/ADL contact-based nociception.** Phase 4's Logbook 011 surfaced that the current nociception model is biologically wrong — real *C. elegans* nociception is contact-based mechanosensation (ASH/ADL neurons), not chemosensory at distance. The corrected model is owed correctness work and lands in Phase 6's sensory-physics stack.

**Other sensors:** physical temperature fields with conduction; contact mechanics for mechanosensation with continuous bodies; realistic sensory ranges scaled to worm body length.

#### Architecture-comparison protocol

The architecture-comparison protocol is one experimental sweep across three dimensions: architecture family × behaviour × seed. The Phase 5 statistical bar carries forward — paired-seed Wilcoxon tests, bootstrap confidence intervals, n ≥ 4 seeds per condition with explicit power analysis when smaller. Phase 5's lag-matrix instrument and TEI fair-F0 paired comparison are the methodological templates.

The protocol explicitly answers the four primary Phase 6 research questions (connectome ranking; fitness landscape; architecture asymmetry under matched capacity; first closed-loop learning on the connectome) via the same sweep, not via separate experiments.

#### Built-in real-worm validation

Phase 6 validates *at least one* model output quantitatively against published real-worm data, as a Phase 6 exit criterion rather than a Phase 7 collaboration deliverable. Three candidate targets, **ordered by defensibility-per-effort for a behavioural (non-biophysical) model** *(reordered 2026-06-04)*:

- **Behavioural chemotaxis metrics — PRIMARY, lead with this.** Turn-rate (pirouette initiation) vs. dC/dt (Pierce-Shimomura, Morse & Lockery 1999, *J. Neurosci.*) and curving-rate vs. bearing (Iino & Yoshida 2009, *J. Neurosci.*). These are exactly the outputs an RL worm produces — clean public scalar/curve targets, no neuron-identity mapping needed. Highest defensibility per effort.
- **Mechanosensation escape — tertiary.** Force-graded reversal probability (Goodman lab microfluidic touch, 2017, *Lab on a Chip*); the corrected ASH/ADL nociception is the natural pair. Viable but the escape-*latency* numbers are fragmented across assays.
- **Whole-brain Ca²⁺ imaging correlation matrices — secondary, higher effort + category caveat.** Public data: Kato et al. 2015 (OSF), Atanas et al. 2023 *Cell* (DANDI/NWB, behaviour-linked). The field-standard metric is inter-neuron correlation-matrix similarity. **Note:** raw Ca²⁺ matching is a category mismatch for a behavioural (non-biophysical) model and requires mapping model units → real neuron classes (NeuroPAL/WormID); prefer comparing *behaviour-encoding tuning* (Atanas/CePNEM) if pursued. **Factual correction:** the prior "BAAIWorm 92.4% fidelity" figure is not verifiable — BAAIWorm/MetaWorm (Zhao et al. 2024, *Nat. Comput. Sci.*) reports **MSE 0.076** on a 65-neuron correlation matrix vs. Uzel et al. 2022; cite that, not 92.4%.

**Positioning / precedent (framing for the T9 synthesis).** The connectome-constrained approach has a strong recent precedent: Lappalainen et al. 2024 (*Nature*) showed a fly-visual-system network constrained to the connectome topology and task-optimised recovers single-neuron biological responses — this project is the *C. elegans / RL-behaviour* analogue of that lineage and should be framed as such. The closest *learning-in-the-loop* precedent is closer still: a whole-brain connectomic graph model (Jin et al., arXiv:2602.17997, 2026) trains the adult *Drosophila* whole-brain connectome as a graph-structured policy for whole-body locomotion via deep reinforcement learning — so "closed-loop RL on a real connectome" already exists cross-organism. The project's defensible novelty is therefore the *narrower, dated* claim — first closed-loop *reward-driven RL and neuromodulated plasticity* on the real *C. elegans* connectome, in a controlled architecture comparison — cited as **convergent with** the fly work rather than pre-empted by it. The interpretive counter-weight is Beiran & Litwin-Kumar 2025 (*Nat. Neurosci.*): a connectome often under-constrains dynamics, so many functionally-distinct weight solutions fit the same wiring. The project's n=8 seeds already provide the material to report solution **degeneracy** rather than treating one PPO weight set as "the" biological solution (see [Phase 7 § Claim discipline](#phase-7-deepen--plasticity--cross-species-transfer) for how this bounds structure-function claims).

Internal validation against public data is required at Phase 6 close; external lab collaboration (Phase 7) is additive, not a precondition.

#### Phase 6 exit criteria

**Required (MUST):**

- ✅ L0 connectome substrate operational: Cook 2019 hermaphrodite connectome (302 neurons, 3709 chemical synapses, 1093 gap junctions) imported via direct *Nature* SI parsing; cross-validated against Witvliet 2021; vendored under `data/connectome/` with provenance documented. Logbook [022](experiments/logbooks/022-connectome-substrate.md).
- ✅ L1 architecture-plugin interface ships as a decorator-registration registry; first 20-arch consumer (`ConnectomePPOBrain`) added in 5 files (≤ 6 budget) with no per-arch branches outside the new module + `_build_infra_kwargs`. Logbook [023](experiments/logbooks/023-architecture-plugin-interface.md). Gate 2 G2.b plugin-parity verification re-runs against the platform refactor in Tranche 5.
- ✅ **(6a)** L2 weight-search results across all MUST architectures on all three behaviours, at the Phase 5 statistical bar (paired-seed, bootstrap CIs, n ≥ 8 seeds per condition). Shipped at T7 — Logbook [029](experiments/logbooks/029-continuous-architecture-ranking.md): six MUST families, n=8, three significant tiers (`MLP 89.0 ≫ {CfC 75.8 ~ Transformer 74.0} > LSTM 60.1 > connectome 52.2 ≫ GA 15.0`). *(Real-worm validation and the rewired-null control have since landed — see below.)*
- ✅ **(6a)** Degree-preserving rewired-null control on the connectome, under matched initialisation + training budget — the control that makes the connectome ranking credible (Dhiman 2026): does the wild-type wiring matter, or is its 5th-place standing a generic property of its degree/sparsity? Shipped at T7 — [Logbook 034](experiments/logbooks/034-connectome-structure-controls.md): the ranking is a *degree-statistics* result, not a wild-type-wiring one. The **learnable-gap-junction variant is deferred** (carried to Phase 7) and was not gating for Gate 3.
- ⏸️ **(6b) DEFERRED-WITH-DESTINATION, UNSCHEDULED** *(status assigned 2026-09-19 at the Phase 7 close, in the vocabulary [Logbook 069](experiments/logbooks/069-phase7-synthesis.md) introduced; `phase6b-tracking` is closed off with every item marked)* — **this criterion stays unmet**, and Phase 6 therefore stays *6a COMPLETE / 6b pending* rather than flipping to COMPLETE. The binding precondition, whether to port `Continuous2DEnvironment` to a vmappable form, has not been decided, and nothing in Phase 7's results or Phase 8's opening scope depends on it. Revisited if and when that decision is worth taking, at which point a fresh change is authored. L3 NEAT topology-search results comparing the wild-type connectome to NEAT-evolved topologies on at least one behaviour, with the lag-matrix or equivalent discriminative instrument. Deferred to Phase 6b (gated on GPU + env-vectorisation); satisfies the L3 exit criterion at 6b close.
- ✅ Rung 2 env fidelity operational: signal-type-specific **static Fick-shaped** chemical gradients paired with an **adaptive-threshold / biphasic chemosensory sensor** (dynamic-diffusion PDE + source dynamics is a stretch goal, not a gating criterion — rebalanced 2026-06-04). Shipped at T6 ([logbook 028](experiments/logbooks/028-rung2-gradients-adaptive-sensor.md)): `gradient_field_mode: fick` Gaussian geometry + the adaptive sensor (fold-change / contrast / log readouts), passing the step-input adaptation-transient gate (Weber spread 0.0056 adaptive vs 1.639 log baseline).
- ✅ Corrected ASH/ADL contact-based nociception operational. Shipped at T3 — Logbook [024](experiments/logbooks/024-predator-sensing-biology.md) (two-channel contact-mechanosensory + distal-chemosensory split per Liu et al. 2018).
- ✅ **(6a)** At least one model output quantitatively validated against published real-worm data. Shipped at T7 — [Logbook 035](experiments/logbooks/035-realworm-chemotaxis-validation.md) grades klinokinesis + weathervane bias curves against the *C. elegans* chemotaxis literature (both strategies PRESENT), extended to thermotaxis in [036](experiments/logbooks/036-realworm-thermotaxis-validation.md) (**PARTIAL** — klinokinesis present, weathervane collapsed).

**Optional (MAY) — not phase exit criteria:**

- ⭐ Connectome-learning platform paper drafted.
- ⭐ Connectome fitness-landscape science paper drafted.
- ⭐ Reproducibility artefacts (Docker, evaluation scripts) updated to current Phase 6 state.

Papers and external collaboration are explicitly optional — the project may pursue them when evidence and context justify, but pursuing them is not a precondition for closing Phase 6 or advancing to Phase 7.

#### Mid-phase decision gates

Phase 6 is long enough (~6-10 months) that mid-phase gates matter. Each gate produces a written go/no-go decision in the relevant OpenSpec change, not just an implicit continuation — the same discipline Phase 5 used.

- **Gate 1 (month ~2): L0 import working?** Connectome substrate loaded, validated, and basic-MLP-PPO baseline trainable on it. If not, trigger the L0 hand-curated-subset pivot (see Risk-mitigation below). *Outcome: **GO** (2026-05-24, logbook [023](experiments/logbooks/023-architecture-plugin-interface.md)). All four sub-criteria (G1.a–G1.d) pass: PPO-on-wild-type-Cook-2019-connectome reaches 100% sustained success on the last 100 episodes of klinotaxis foraging (R2b reference run); within 6 points of MLPPPO + LSTMPPO baselines on same task / env / seed; migration regression byte-equivalent for MLPPPO + LSTMPPO; no PIVOT to subset triggered.*
- **Gate 2 (month ~4-5): L1 plugin parity achieved? — ✅ GO (2026-06-07).** Adding a new architecture (Transformer) cleared the files-touched (5 ≤ 6) + no-per-architecture-branches checks documented in [openspec/changes/phase6-tracking/design.md § Decision 6 § Gate 2](../openspec/changes/archive/2026-07-06-phase6-tracking/design.md) (informal "≤ 1 week" target carries forward but is not load-bearing); the continuous-substrate floor check passed (connectome + MLP train without collapse, with a continuous-PPO entropy-0.10 calibration). No L1 refactor pivot triggered. See [logbook 027](experiments/logbooks/027-platform-refactor-continuous-2d.md).
- **Gate 3 (month ~7-8): L2 results across architectures?** Weight-search results across MUST architectures and all three behaviours in hand — **✅ delivered** (Logbook [029](experiments/logbooks/029-continuous-architecture-ranking.md), six MUST families n=8). The **Phase 6a/6b split is now taken** (by success, not overrun — see § Phase 6a/6b split), and Gate 3 has **closed Phase 6a with a GO** — real-worm validation ([035](experiments/logbooks/035-realworm-chemotaxis-validation.md)/[036](experiments/logbooks/036-realworm-thermotaxis-validation.md)) and the rewired-null control ([034](experiments/logbooks/034-connectome-structure-controls.md)) have landed, and the L3 NEAT work moves to Phase 6b. The written Gate 3 go/no-go decision is recorded in [Logbook 037](experiments/logbooks/037-phase6a-synthesis.md).

The hard phase boundary between Phase 5 and Phase 6 protects the narrative arc (no Phase 6 work begins until Phase 5 is synthesised); the mid-phase gates protect the execution (fail-fast at the architecture and substrate level, not at the phase level). Both are intentional.

#### Risk-mitigation: failure modes and pivots

| Failure mode | Trigger | Pivot |
|---|---|---|
| **L0 c302 import takes > 2 months** | OpenWorm c302 / NeuroML integration proves harder than expected (format incompatibility, missing metadata, unclear synaptic-weight provenance) | Drop to a hand-curated subset of the Cook 2019 connectome — sensory-interneuron-motor subgraph for the three target behaviours, ~50-100 neurons. Document the subset choice; defer full 302-neuron import to Phase 7. The platform claim survives ("learning on a real *C. elegans* subgraph"); the comparative-completeness claim weakens. |
| **L1 architecture-plugin interface proves messy** | Multiple architecture families need bespoke plumbing; the interface accumulates per-architecture branches; "swap in another brain" stops being one-line | Pause architecture-sweep work; spend 2-4 weeks refactoring L1 toward genuine plugin parity. Better to delay L2/L3 results than to ship a "platform" that isn't one. |
| **L2 PPO-on-connectome fails to learn** | After reasonable hyperparameter search, no architecture family reaches the Phase 0-3 baselines on any of the three behaviours | Diagnostic sequence: (a) is the connectome topology dense enough to support gradient flow? (b) is the continuous action head the bottleneck? (c) is reward shaping the issue? If none resolve: the finding is "learning on the real connectome with PPO requires further substrate work" — itself a publishable negative result and a Phase 7 prerequisite. |
| **L3 NEAT produces no separation from connectome** | NEAT-evolved topologies and the wild-type connectome converge to indistinguishable performance | This is itself a finding: "the connectome is competitive with evolved topologies on these behaviours." The optimal-primary framing weakens; the connectome-primary framing strengthens. Acceptable outcome — pivot the headline framing if it lands. |
| **Phase 6 overshoots 10 months** | Scope creep, L1 refactor, multi-architecture debugging push timeline past 12 months | Trigger a sub-phase split: Phase 6a (L0+L1+L2) ships first as a standalone result; Phase 6b (L3 + NEAT topology search) becomes a follow-on. Pre-commit the split criterion: if L2 isn't producing publishable architecture comparisons by month 6, split the phase. **⏭️ Split taken (2026-07) — invoked by success, not overrun: L2 produced a publishable ranking (Logbook 029), so 6a ships that + validation + controls, and the compute-heavy L3 NEAT becomes 6b (gated on GPU + env-vectorisation). See § Phase 6a/6b split.** |

The general principle: **fail-fast at the architecture and substrate level, not at the phase level.** A Phase 6 that fails a mid-phase gate produces a documented pivot, not a silent slide past the gate.

#### Compute / infrastructure planning

Phase 5 ran on CPU. Phase 6's L3 (NEAT topology search across many candidates × multiple architectures × multiple behaviours × multiple seeds) is substantially more compute-intensive. Three tiers should be considered explicitly, in order of preference:

1. **CPU + targeted parallelism (no new infra).** Sufficient for L0, L1, L2 on small-to-medium architecture sweeps. Probably not sufficient for full L3 NEAT search at population sizes the field uses (~1000+ genomes × generations).
2. **GPU (single or small cluster, including consumer-class cards).** The realistic baseline. Sufficient for L3 with TensorNEAT vectorisation (~500× speedup over neat-python via JAX/vmap on GPU is documented in the field); sufficient for full L2 sweeps across all architectures.
3. **HPC allocation (NERSC, JURECA, or similar).** Optional, not required for Phase 6. Pursue an allocation if a specific Phase 6 or Phase 7 need justifies it (e.g., Phase 7 pacificus comparative work, neuromorphic-hardware deployment). National HPC centres regularly fund computational-neuroscience time on this class of work — realistic, not aspirational, but not a precondition.

The roadmap encodes GPU as the realistic baseline. The L3 implementation choices (TensorNEAT, JAX vmap, batched fitness evaluation) flow from that.

#### Go/No-Go Decision

- **GO if**: L0+L1+L2 ship and the three mid-phase gates pass with results in hand. **L3 now ships as Phase 6b under the sub-phase split (taken 2026-07 — see § Phase 6a/6b split); Phase 6a GO is L0+L1+L2 + validation + connectome-structure controls.**
- **PIVOT-narrative if**: L3 shows the connectome competitive-but-not-dominant with evolved alternatives. The headline framing shifts toward connectome-primary; the platform claim is unchanged.
- **PIVOT-scope if**: a mid-phase gate fails — execute the relevant Risk-mitigation row above and document the pivot in the OpenSpec change.
- **STOP if**: L0 connectome import is fundamentally infeasible after the hand-curated-subset pivot has also failed — at which point the diagnosis itself is the Phase 6 deliverable, and Phase 7 inherits a substrate-engineering question rather than a plasticity question.

______________________________________________________________________

### Phase 7: Deepen — Plasticity & Cross-Species Transfer

**Goal**: Add the L4 plasticity layer to the architecture-comparison platform (biologically-plausible learning rules on the connectome) and extend the comparison across nematode nervous systems — the *P. pacificus* head connectome (Cook et al. 2025) and, as a scope-matched within-species condition, the dauer connectome (Yim et al. 2024, *Nat. Commun.* 15:1546). The headline framing is *deepen, don't broaden* — Phase 7 doesn't add new behaviours or new sensory modalities; it deepens the platform's biological-plausibility and species-coverage along the same three Phase 6 behaviours.

*(Revised at the 2026-08-27 pre-start review. Two load-bearing changes from v4.1: the cross-species deliverable is rescoped to the **head circuit**, because the Cook 2025 dataset is head-only and chemical-synapse-only — not a whole-animal counterpart to Cook 2019; and the plasticity deliverable is re-staged around a rate-based three-factor primary arm, because the connectome substrate is a rate-code with no spike times and no cross-step state — see § Pre-registered design decisions.)*

**Sharpened central hypothesis (post-Logbook-029; restated as a 2×2 after the 2026-08-27 adversarial review).** Phase 6 established two facts that together fix the question: under **PPO gradient learning** the wild-type connectome ranks 5th of 6 ([Logbook 029](experiments/logbooks/029-continuous-architecture-ranking.md)), and that standing is a *degree-statistics* property — a degree-preserving rewired null matches it ([Logbook 034](experiments/logbooks/034-connectome-structure-controls.md)), so under PPO the specific wiring is **inert**. "Does plasticity recover the ranking?" is therefore not the flagship — a recovery against that ranking would license no wiring claim, since a degree-matched scramble might recover identically. The flagship is a pre-registered **2×2 (learning rule × wiring)**: *under a biologically-plausible three-factor rule — a member of the rule family available to the real animal — does the wild-type wiring become load-bearing? Does plastic wild-type beat its plastic degree-preserving rewired null?* The four cells, named: **PPO × wild-type** and **PPO × rewired-null** are the *completed Phase 6 cells* (Logbooks 029/034 — pre-D7 substrate, so they enter the design as qualitative context, not as panel rows); the new quantitative panel is the plastic row — **three-factor × wild-type** vs **three-factor × rewired-null** is the **primary contrast** (paired across the same n ≥ 8 seed list), and three-factor × **matched-rule MLP** is the ranking contrast (D10). Ranking recovery against the reference arms is the secondary axis. Either primary outcome is strong: wild-type ≫ rewired-null under the native rule family is a first-order structure-function result ("the wiring is legible only to its own learning regime"), and it stays available **even if the plastic connectome still trails the MLP**; a null replicates the degree-statistics verdict across learning regimes and hardens the architecture finding. One substrate argument runs *for* the rate-based primary arm here and is stated up front: *C. elegans* neurons are predominantly graded/non-spiking, so a rate-code three-factor rule is arguably higher-fidelity for this animal than spike-timing STDP — not an approximation of it.

The convergent fly result sharpens the stakes. FlyGM (Jin et al., arXiv:2602.17997 — as of 2026-08 still a workshop poster + preprint, unreplicated) reports the *Drosophila* whole-brain connectome *beating* an MLP under an imitation-then-PPO pipeline — the opposite of Logbook 029's ranking. Whether the learning regime (and warm start) flips the connectome's standing is therefore now an open cross-organism question, not a private one. Accordingly, the imitation-warm-start lever is **scheduled, not held in reserve** (D13): it requires no L4 code, it is the one intervention with cross-organism evidence of flipping the connectome-vs-MLP standing, and its result is needed to *interpret* the L4 outcome — if warm-start + PPO already recovers the ranking, the 5th place was an optimisation artefact, and a plasticity "recovery" read without that control would conflate two mechanisms.

**Claim discipline — pre-register the claim *type* before L4 runs.** The sharpened hypothesis has a failure mode the Beiran & Litwin-Kumar 2025 (*Nat. Neurosci.*) degeneracy result makes concrete: "the connectome's own rule recovers its standing" can come out positive via a plasticity/weight solution that bears no relation to real *C. elegans* neural dynamics, because a connectome plus task output does not uniquely determine single-neuron dynamics — a degenerate space of solutions fits the same wiring and behaviour. Every Phase 7 result is therefore pre-registered as one of two claim types *before* the run:

- **Performance claim** — "architecture/rule X reaches performance Y on behaviour Z." Defensible from behaviour alone; this is the default and the safe headline. The learning-rule-recovery result itself is, by default, a performance claim.
- **Dynamics / biology claim** — "the connectome's wiring (or a specific edge, rule, or neuromodulator) is *responsible* for behaviour." Admissible only if it clears two bars: **(a) ensemble-invariance** — the effect (a ranking, an ablation delta, a rewired-null delta, a neuromodulator-toggle delta) is shown invariant across the degenerate seed-ensemble, not read off a single fit; a delta measured on one PPO/plasticity fit is a property of *that fit*, not the wiring, until this bar is met. **(b) named-neuron grounding** — model units are mapped to named neurons (NeuroPAL/WormID) and compared on *behaviour-encoding tuning* (Atanas et al. 2023 / CePNEM), not raw activity. *C. elegans*'s low activity dimensionality (~tens of components) is what makes bar (b) reachable here where it is not in larger animals — an option that unlocks *only* when a dynamics claim is committed, not a standing requirement.

This is the discipline that keeps the negative-result and any structure-function claim defensible to a degeneracy-literate referee; it is a pre-registration habit, not new infrastructure.

**Evidence base hardened at the 2026-08-27 review — in both directions.** *For* caution: Dvali, Seguin, Betzel & Leifer (*PRX Life* 3:033021, 2025) show — in *C. elegans*, on the Randi/Leifer signal-propagation data — that functional-network modules **diverge** from anatomical-connectome modules; and Currier et al. (*Cell* 188, 2025) show infrequent strong synapses make connectome-based predictions of physiology unreliable, which bounds how far Cook synapse counts may be trusted as weight priors (state the weight-initialisation choice explicitly in the L4 design). *Against* over-caution: Creamer, Leifer & Pillow (bioRxiv 2024.09.22.614271) find weights fitted on connectome edges alone suffice to explain measured causal (optogenetic) signal propagation, and non-connectome edges don't help — reweighting the fixed topology, which is exactly what plasticity does, is the right lever. Bar (a) now has an **in-house precedent**: Logbook [034](experiments/logbooks/034-connectome-structure-controls.md)'s single-seed smoke read the *opposite sign* (+10.7pp "wiring matters") from the n=8 null — cite it whenever tempted to read a connectome effect off one fit. For bar (b), a decoder-free **representational-geometry comparison** (RSA/CKA against the harmonized WormID corpus, *Cell Reports Methods* Jan 2025) is admitted as an additional instrument alongside behaviour-encoding tuning. Beiran & Litwin-Kumar's final citation is *Nat. Neurosci.* 28:2561–2574 (2025); its constructive half — a few recorded neurons break the degeneracy — is the argument for checking the plastic model against recordings, not behaviour alone, whenever a dynamics claim is attempted.

**Aspirational timeline**: ~8-12 months from Phase 6**a** close (2026-07-07) at the outside; the demonstrated cadence suggests substantially less active work (see the Deliverable 1 estimate restatement). Default expectation, stated plainly: **7a lands within the window; 7b lands beyond it** — the pre-structured shipment shape absorbs either. *(Re-anchored 2026-08-27: the previous anchor, "Phase 6 close", required the 6b synthesis — an event Phase 7 explicitly does not gate on.)* **Phase 6b is decoupled** (D13, superseding Logbook 037 § T9.5's early-window suggestion): opportunistic, no scheduled window, pending a dated GPU/cloud decision — the early 7a window goes to the 6a preprint and the imitation-warm-start arm instead. Phase 7 never gates on 6b.

**How to orient**: the living sub-task checklist lives in [openspec/changes/phase7-tracking/tasks.md](../openspec/changes/archive/2026-09-19-phase7-tracking/tasks.md) (shipments 7a-i / 7a-ii / 7b + synthesis), with tracker-level decisions and inherited execution-protocol standards in that change's design.md. This roadmap section remains the authoritative plan; D-decision amendments are dated in both places.

**Reframed after Logbooks 040–043 and the clone-destruction diagnostic (2026-09-08).** Four registered panels and one probe gave a single answer from every direction: as modelled, the wild-type wiring carries almost nothing — inert under gradient learning ([034](experiments/logbooks/034-connectome-structure-controls.md)), inert or destroyed under the minimal local rule from random weights ([040](experiments/logbooks/040-l4-panel.md)–[042](experiments/logbooks/042-l4-panel3.md)) and from a cloned competent policy ([043](experiments/logbooks/043-l4-warm-start.md)), with reward-free Hebbian alignment on the real wiring the one repeated, descriptive, wiring-specific signal. The probe ([supporting/043-l4-warm-start/destruction-diagnostic.md](experiments/logbooks/supporting/043-l4-warm-start/destruction-diagnostic.md)) found the mechanism: the minimal rule is a biased, near-constant-speed drift on the norm sphere that passes through good policies and never consolidates, whatever the rate. Two consequences are now the frame for the rest of Phase 7 and for the phase after it. **(i) The substrate fidelity ladder is the programme.** The wiring as modelled is a sparse graph with random synapse signs, no receptor classes, no neuromodulation, rate units without intrinsic dynamics, fixed gap junctions, and a two-number readout in place of a body — close to a degree distribution, which is what 034 measured. Each restored piece of the wiring's biology is a registered rung of the same question, "does the wiring start to matter once this is real?": transmitter identities and receptor classes (7a-ii's atlas, a substrate deliverable in its own right with its own test), routed neuromodulation, gap-junction and intrinsic dynamics, and a body via interoperation. **The first rung ran 2026-09-09** ([Logbook 044](experiments/logbooks/044-l4-atlas-signs.md)) and answered negatively in the informative direction: grounding 3,176 of 3,709 synapse signs in the Wang 2024 atlas leaves the untrained prior unchanged and makes reward-free Hebbian learning substantially worse, because a purely potentiating rule on the animal's 80%-excitatory network has no inhibitory brake. The substrate's arbitrary signs were not what limited the rule, so the ladder continues but the rule work is where the next result has to come from. Organism-level fidelity (biophysical neurons, muscle, development) is interoperation with OpenWorm, not a build — see § Future Directions. **(ii) Consolidation before instruction.** The rule's failure is the absence of a mechanism that slows or stops updating once a policy is good, not a credit-assignment sign error; structured, pathway-specific instruction (B.4b) is still required but is not sufficient. 7a-ii's rule work starts with consolidation, and every rule variant clears the one-hour clone assay (load a competent policy, run the rule, does it hold or improve it) before a registered panel is re-run. **(iii) 7b's comparative runs are gated** on a rung turning positive: transferring a wiring-indifferent learner between species measures nothing about wiring; the pipeline items (scaffold, truncation, *P. pacificus* ingest, homology table) proceed regardless.

**Reframed again after Logbooks 044–046 (2026-09-09), and updated 2026-09-10 once the
structured-instruction test and the rule's positive control had run: the instrument before the
substrate.** Two rule-level interventions followed the ladder's first rung — consolidation (three
mechanisms, [045](experiments/logbooks/045-l4-consolidation.md)) and decorrelation (two terms,
[046](experiments/logbooks/046-l4-decorrelation.md)) — and neither moved the wild-type connectome
off its floors; a third, the routed third factor (B.4b), completed 2026-09-10 as registered and
resolved `no_routing_effect` ([Logbook 047](experiments/logbooks/047-l4-structured-instruction.md)),
which under a one-sided family means **no confirmed benefit** and in fact came with observed
degradation — routing made both arms worse and removed their good seeds — and is read against I.0 as
a measurement of what a non-learning rule does under two routing regimes. Stepping back from the
ladder exposes what every rung has assumed. **The rule has no positive control.** Logbook 040
recorded that the matched-rule MLP yardstick sits at chance and that the rule destroys a 96% policy
on a dense feedforward network within three episodes: the three-factor rule as implemented has never
learned this task on *any* substrate, including the one where it should be easiest. Every registered
result since has asked whether the wiring is legible to that rule while the rule's ability to learn
anything here was never demonstrated — and "the wiring is not legible" is indistinguishable from
"this rule cannot learn" until it is. There is a specific reason to expect the second reading. The
eligibility trace is Hebbian, `pre × post`, with exploration noise applied only at the action
output; the three-factor rules that are policy-gradient estimators put the *noise* in the
eligibility — the deviation of a unit's activity from its mean, times pre (Williams 1992; node
perturbation, Fiete & Seung 2006; the Frémaux & Gerstner 2016 review). With deterministic units and
output-only noise, an internal synapse's Hebbian trace carries no information about which way to
move to make the sampled action more likely, which predicts exactly what every panel measured: a
constant-speed drift toward correlation structure, indifferent to reward, on every substrate. The
B.4 recon is the same fact from the reward side — the per-step prediction error is the same size for
a competent policy and a dead one. Three setup choices unexamined since A.3 compound it: a ~10-step
eligibility horizon (`trace_decay 0.9`) in 2400-step episodes, homeostasis pinning every unit to its
construction norm, and a cliff metric (full clear or nothing) that is where the bimodality every
panel has failed to test comes from. **Consequences.** *(i)* The fidelity ladder is paused, not
abandoned: no further substrate rung runs until the rule has a positive control, because a rung
cannot be read against an instrument that has not been shown to work. **That control ran 2026-09-10
and the rule failed it** ([Logbook 048](experiments/logbooks/048-l4-rule-positive-control.md)): on a
one-step association whose analytic reference closes 99.9% of the available gap on every seed, the
three-factor rule ends *below* the cue-blind floor at every rate over two orders of magnitude, with
a live trace, a well-behaved modulator and a gradient alignment of +0.031 mean / +0.009 median — its
updates are near-orthogonal to the policy gradient. The instrument is confirmed broken in the way
the theory predicted, so the reframing is no longer a hypothesis about the rule but a measurement of
it. **The repair works.** An eligibility carrying each unit's own perturbation rather than its
activity — I.1, run 2026-09-10 — passes the same control at σ = 0.2, reaching 88% of the
floor-to-optimum gap on 8 of 8 seeds with a gradient alignment of +0.263 against the old rule's
+0.009, with nothing but the eligibility changed
([records](experiments/logbooks/supporting/049-l4-node-perturbation/details.md)). That is the first
mechanism in this sequence that works, and it makes Logbooks 040–047 re-runnable questions rather
than closed ones. It cleared the control; the clone assay then **failed** it — though the
frozen-perturbation control run beside it shows why that is not a verdict on the rule: perturbation
alone takes a competent policy from 38.7 to 8.9, and the learning arm's 12.0 sits above that. The σ
that makes this rule learn is the σ that makes a competent policy unrunnable, which is the tension
the next mechanism has to resolve. **σ-annealing was that mechanism and it failed** (I.1b, run
2026-09-10, [records](experiments/logbooks/supporting/051-l4-sigma-annealing/details.md)): a
geometric decay from 0.2 to 0.02 over half the budget closes 38.1% of the gap where the constant σ =
0.2 arm closes 89.0%, landing on the constant σ = 0.05 arm, because the decay puts σ below 0.05
after 30% of the run — it annealed through the learning phase rather than after it. The registered
alignment split shows the estimator was still aimed while the scale was large (+0.176 over the decay
against +0.051 at the floor), so this constrains the schedule's length against the task's horizon
rather than refuting annealing; the clone assay did not run, as registered. The tension is therefore
unresolved: no setting has yet both learned and left a runnable policy. **Evaluating the I.1
endpoints with the perturbation off** (I.1c step 0, 2026-09-11,
[records](experiments/logbooks/supporting/052-l4-endpoint-evaluation/details.md)) asked what the
rule had actually left behind, since every earlier score was taken while the arm's own perturbation
ran. It **fails** the registered rule at a mean of 20.6 against 38.7 — removing the noise is worth
+8.6 over the same weights' 12.0 and does not close the shortfall — but the result is **bimodal**:
six seeds degraded, two improved, and **seed 2 reached 73.4 from a clone of 44.0, the highest
full-clear rate this substrate has produced**, stable across a frozen 2000-episode run. Mean 20.6
against a median of 9.2, with a paired Wilcoxon at p = 0.074. So the rule does not merely fail to
hold policies — on a quarter of the panel it transforms them, and nothing measured predicts which.
That makes I.2's mixture-aware statistic a demonstrated need rather than an argued one, and it is
the first evidence in this phase of the rule *improving* a real policy on the real substrate. **I.2
followed on 2026-09-11**
([records](experiments/logbooks/supporting/053-l4-mixture-statistic/details.md)): a family with one
member per component of the mixture — how often an arm reaches competence, and how good it is when
it does — beside the test the record was scored with, plus the graded metric every committed panel
table already carried. Its re-read of seventeen contrasts **promotes none of them** — fourteen
`no_effect`, three `degrades`. The descriptive level differences 042 described are large and
consistent on the Hebbian wiring comparison (+25.0, +22.7, +12.9) and not one is significant against
a pooled-label null: those contrasts carry two to five competent seeds an arm, below the
combinatorial floor at which any distribution-free test can resolve a difference after correction.
So the wiring hypothesis stays closed, and the committed record's negative reading survives a
statistic matched to the outcome's shape rather than being an artefact of the wrong test. Three
criteria were corrected during the build, two of them in review: the registered split criterion,
which simulation showed firing on 81–100% of draws from a null where both arms come from one bimodal
law and which is now descriptive rather than a verdict; and the level contrast's p-value, which an
earlier build took from draws centred on the observed difference and which reported two `level_only`
results that a correct null removes. Every committed verdict stands as registered. **I.3 followed on
2026-09-11** ([records](experiments/logbooks/supporting/054-l4-instrument-knobs/details.md)) and is
the first positive mechanism the phase has produced. Its registered platform failed — the MLP
yardstick under the repaired rule ends below its own frozen control — and the eligibility horizon
was unmeasurable on the one-step control, so the control gained a delay between the scored action
and the reward, with zero delay bit-identical to the committed arm. At the pinned `trace_decay 0.9`
the rule closes 89% of the floor-to-optimum gap undelayed, 44.8% at ten steps of delay, and **−6.8%
at twenty — below the cue-blind floor**; at `0.99` that same cell closes 45.3%. One setting, a
52-point swing. Episodes on the real task run 244 to 2400 steps, orders beyond where the pinned
decay already fails, so the horizon is the first account that *predicts* the rule's one-step success
and multi-step failures instead of restating them. Homeostasis and the exploration noise are
near-neutral by comparison. **I.3b then tested that mechanism where it would matter** (2026-09-11,
[records](experiments/logbooks/supporting/055-l4-horizon-multistep/details.md)), because every
result in the phase had run at the pinned 0.9 on episodes of 244 to 2400 steps and none had varied
it. **It does not transfer.** On the MLP yardstick the learning arm is worse than its own frozen
control at every horizon — 0.393, 0.148 and 0.144 foods against 2.233 — and *worse as the horizon
lengthens*, the opposite of the control's prediction. The registered drift measurement says why:
weights move 1.28–1.31 times their own norm at every setting, so this is a rule writing a great deal
in a wrong direction rather than one starved of credit, and the horizon barely changes how much. The
suspicion is closed, the connectome was deliberately not spent on it at ten hours a run, and Logbook
040's account of the yardstick — a local rule collapsing a dense stack without decorrelation —
survives the repaired eligibility and every horizon setting. The panel stays gated, I.2 still gates
any statistic read from one, and **no panel arm has been run or authorised under this rule** — the
connectome panels already recorded in Logbooks 040–047 ran under the original one. *(ii)* 7a-ii's
remaining work is reordered around the instrument — a positive control on a minimal task, an
eligibility formulation with the noise inside it cleared on the yardstick, a statistic and a graded
metric matched to the outcome's shape, and the three unexamined knobs examined — tracked as block I
in the 7a-ii tracker. *(iii)* The 7a shipment decision (B.8) is taken after block I, on evidence
about the rule rather than about the substrate. **I.4 closed block I on 2026-09-12 by doing that
re-read contrast by contrast, and it does not support the blanket version of this reframing**
([Logbook 056](experiments/logbooks/056-l4-ladder-reread.md)). Of 32 registered contrasts across
040–047, **ten** are instrument findings; **sixteen** are substrate findings the instrument block
does not reach — four ran under no rule at all and twelve under the *unmodulated, reward-free
Hebbian* rule, which the positive control never tested and which Logbook 040 recorded settling at
78.3%, 64.4% and 67.3% on individual seeds with no reward; **five** are about neither, because the
premise they rest on has no demonstration under any optimiser —
[034](experiments/logbooks/034-connectome-structure-controls.md) ran the same wiring contrast under
PPO weight search and found the wirings indistinguishable (−3.28, q = 0.770, the null nominally
higher), and 043's low-noise PPO arms put the null ahead by 12.6 on 0 of 8, so neither tested regime
found a confirmed wild-type advantage and those nulls are uninformative about both the wiring and
the instrument, a classification a repaired rule would not have changed; and **one**, W6, is about
warm-starting PPO and carries a correction: the earlier reading that PPO also destroys competent
policies is withdrawn, since the two warm starts are different starts and what is comparable is that
PPO reaches 68.5 from random weights where the rule reaches 17.8. No committed verdict changed. The
citable claim is therefore narrower and more defensible than either "the wiring does not matter" or
"the instrument was broken": on this task and this substrate **no optimiser tested — PPO included —
finds an advantage for the wild-type wiring**, and the local three-factor rule additionally fails a
positive control that names exactly why. The low-σ programme licensed by 052 is **deferred behind a
task the repaired rule can be shown to learn**, not retired. *(iv)* B.3's receptor layer is queued
behind block I and paid for only if a rung after it can conclude. *(v)* 7b's gate is unchanged.
**(vi) Block V then tested the premise itself, 2026-09-12, and it holds** ([Logbook
057](experiments/logbooks/057-wiring-premise-contrast.md)). Every wiring contrast in this project
had run on the integrated C3 cell — food chemotaxis *plus predator evasion plus thermotaxis* — a
limitation [034](experiments/logbooks/034-connectome-structure-controls.md) recorded in its own
words, with both architecture rankings placing the connectome's deficit in the predator component.
Run under PPO on a klinotaxis foraging cell under lethal thermal pressure, **the wild-type wiring
reaches competence about 35% sooner than its degree-preserving rewired null** — 396 episodes against
613, pooled over 64 paired seeds, all four efficiency metrics at q ≤ 0.001 — replicated on an
independent 32-seed panel after the first panel drew high (+46.4% → +32.6% → +31.8%). Both controls
hold at 64 seeds: both wirings learn, and the untrained prior is indistinguishable (−0.17, q =
0.735), so the advantage is **created by learning rather than inherited from the graph**. Two bounds
are part of the claim: it is **speed, not performance** — the endpoint saturates and the null is
nominally ahead there, which is what 034, 043 and this panel's own food-only cell were measuring —
and it is **PPO, not a local rule**, so 7b's gate as written stays unmet while the premise beneath
it is now supported. **(vii) V.3 then removed temperature and the advantage survives** ([Logbook
058](experiments/logbooks/058-wiring-premise-difficulty.md)): the two cells 057 compared differ in
three respects and only two bind — the thermal cell's episodes end 14.7% `health_depleted`, 12.4%
`max_steps` and 0.6% `starved` — so the separable non-temperature factor is the episode budget, and
on a hard food-only cell with no temperature and no thermosensory projection the wild type still
reaches competence **+23.5% sooner over 32 paired seeds**, three of four efficiency metrics
significant, both gates 32/32, prior indistinguishable. **Difficulty is sufficient; the projection
is not necessary.** The citable claim for the phase is therefore no longer a pure negative: **the
specific wiring is worth roughly a quarter to a third off the time to learn a foraging behaviour the
animal performs, and nothing at the endpoint** — demonstrated on two cells of one hard-foraging
family (thermal-pressure and time-limited), so whether it extends to task families beyond that one
needs further cells and is not claimed — alongside a local rule that fails a positive control for a
reason the record names. Two qualifications stand with it: the effect is on learning speed rather
than final performance, and every wiring panel so far draws its rewired graphs from run seeds 1–64,
so a fresh-rewiring panel (V.4) is what would make one independent in rewiring as well as in task.
**V.4 then ran that panel 2026-09-16 and the advantage replicates on both cells**
([Logbook 065](experiments/logbooks/065-wiring-fresh-rewiring.md)). Seeds **65–96**, fresh to both
prior panels, 256 of 256 runs, no new configs and both committed harnesses unmodified: **+55.3%** off
time-to-competence on the thermal cell (309.22 against 691.50 episodes, 24/32) against V.1's +35.4%,
and **+40.1%** on the hard food-only cell (827.75 against 1380.88, 28/32) against V.3's +23.5% — all
four efficiency metrics at q = 0.000 on both cells, nothing censored, both gates 32/32, and neither
untrained prior detecting a pre-update difference between the wirings (a failure to detect at 32
pairs, not a demonstration that none exists). **The shared-nulls caveat closes**: the two positives no longer rest on the
same 32 shuffles, and the estimates came in above their comparators rather than shrinking toward the
bar. **The second qualification does not close and is not weakened**: `rewire_seed` stays unset, as
V.1 and V.3 ran it, so a fresh seed moves the rewiring, the task draw and the initial weights
together. What is excluded is that the committed figures rode on a particular set of shuffles; that
initialisation contributes is **not** excluded, and isolating the graph needs `rewire_seed` pinned
across seeds — an experiment nothing has registered. And no mechanism is available for any of it:
V.2 scored 64 rewirings on four graph properties fixed in advance and none predicts learning time.
*(Noted 2026-09-16.)* **Two fly preprints now bracket block V from outside.** Dhiman 2026
(arXiv:2604.04033) found that the apparent connectome advantage in a `flyvis` network **dissolves
entirely under a shared from-scratch initialisation and a degree-preserving null** (loss 0.5155
against 0.5172), the earlier advantage having come from checkpoint-initialisation and naive-random-null
confounds. Those are exactly block V's controls — matched init, degree-preserving null — and block V's
effect **survives them**, on a different axis: time-to-competence under PPO rather than early loss under
backprop. Read together with [034](experiments/logbooks/034-connectome-structure-controls.md) (endpoint-
inert) and [L.0](experiments/logbooks/064-l4-frozen-features.md) (inert as fixed features), the two
projects agree wherever they measure the same thing, and the surviving effect is narrow in exactly the
way this record states it. FlyGM (Jin et al., arXiv:2602.17997) reports the degree-preserving null
"comparable on simple tasks" and "substantially worse on complex turning" under imitation + PPO —
[V.3](experiments/logbooks/058-wiring-premise-difficulty.md)'s "difficulty is sufficient" in another
species. Both are preprints; both are cited as convergent, not as support.
**(viii) R.1 then found the dimension nobody had varied, 2026-09-13, and the rule works at a low
one** ([Logbook 060](experiments/logbooks/060-l4-perturbation-scale.md)). The rule's one success and
every failure differed in the number of perturbed units — **8** on the one-step control it passes,
**128** on every failing MLP yardstick arm, **302 neurons at each of four settling steps** on the
connectome — and no experiment had varied it. On the one-step control the dimension costs almost
nothing: every width 8→128 passes, the yardstick's exact two-layer arrangement passes and reaches
criterion fastest, and time-to-criterion *falls* as the dimension grows (slope −0.216, CI \[−0.331,
−0.092\] against a prediction of +1.0). **On the calibrated hard-food cell the same dimension is
decisive**: at eight perturbed units the rule reaches **19.64 foods of 20 and 90.7% full clear
against PPO's 19.69 and 87.6%**, from a frozen control at 1.69, and collapses monotonically to
**3.0% full clear at 128 units** (rho −1.000), with drift from each width's own frozen control
rising 0.91 → 1.39. So I.3b's 1.28–1.31 drift was an **over-dimensioned estimator's noise rather
than a signal pointing the wrong way** — the same magnitude of writing is productive at low N and
destructive at high N — and the binding constraint is an **interaction between dimension and
horizon** rather than either alone. Dilution of per-unit credit across units and decisions is the
candidate mechanism; a product law would need a matched factorial sweep, which this is not, since
the two axes were varied on different platforms. The registered verdict is `mixed` with both halves
stated, since neither pre-registered row fits. **This changes what is missing.** Consequence (vii)
closed with "what is missing is a plausible rule that reads it"; there is now a biologically
plausible local rule that **learns** a block-V cell to PPO's level, and what is missing is the
**wiring contrast under it**, which has not been run — so **7b's gate still stands as written**, and
059's condition for re-registering B.5, B.1, B.4 and B.4b on the block-V cells is **met**. What does
*not* follow is anything about the connectome: at 302 units × 4 settling steps its dimension sits
far beyond this grid's failing end, so a connectome arm needs a way to reduce the perturbation
dimension that the substrate does not have, and read plainly the result predicts failure there at
present. **No committed verdict changed** — 059's negative recorded what had been measured, every
arm it summarised ran at 128 or 302 units, and any re-read is a new registration. **(ix) R.1c then
built the mechanism, and the dimension turned out not to be the connectome's
problem, 2026-09-14** ([Logbook 061](experiments/logbooks/061-l4-reduced-perturbation.md)). The
per-unit perturbation is now restrictable to a declared set, and recon found the readout mean-pools
**only the 39 motor neurons**, so at settling step `s` a unit can reach the action only from within
`depth − s` hops — leaving **672 of 1208 draws per decision causally connected and 536 (44.4%)
credited against an outcome they cannot influence**. Masking exactly those, and restricting to 109
and then 39 units, **does not make the rule learn**: no set beats its own frozen control by the
registered minima and none reaches competence, with full clear never exceeding 0.08% against PPO's
19.31 foods on the same cell. The durable result is a mechanism — **credited synapses drift
1.37–1.38× their own norm at every set from 302 units down to 39**, matching the MLP's 1.28–1.31
across a 31-fold change in draws per decision, while excluded synapses drift 0.013–0.016 — so the
rule is **not starved of signal at any dimension**; it writes a great deal in a direction that does
not help. With σ and the action noise also calibrated on this substrate for the first time and both
null, **three independent axes leave the learning arm at 2.4–4.4 foods of 20**, which is a
structural limit rather than a hyperparameter one. **R.1b's blocker changed shape rather than
lifting**: the mechanism exists, so what blocks the wiring contrast is no longer a missing
capability but a rule that does not reach competence on this substrate, and **7b's gate is
untouched**. The one structural asymmetry now *measured* rather than guessed is that the rule may
write the chemical synapses alone, leaving the motor readout frozen while it shows the **largest relative
norm change of the measured tensors** — 0.783 in 300 episodes against the sensory projection's 0.177
and the chemical synapses' 0.486. Relative change across an eight-entry matrix and 3709 synapses is not
a clean importance measure, so this is registered as a hypothesis: R.1d. **(x) R.1d then tested the
readout, 2026-09-14, and it is part of the limit without being
sufficient** ([Logbook 062](experiments/logbooks/062-l4-frozen-readout.md)). Substituting it more
than doubles what the rule reaches — **3.751 → 9.639 foods of 20** — and all three substituted arms
beat their own frozen control (q 0.016–0.026) where the anatomical one does not (q 0.098). But
**none reaches competence**: best 5.75% full clear against the 20% threshold and PPO's 18.945 foods
on the same cell, so **R.1b stays blocked and 7b's gate is untouched**. **The scale does the work,
not the direction** — +4.51 foods from the norm alone against −2.14 from PPO's direction — which
points at the readout-scale / action-noise interaction rather than at reading the motor classes
better. A fourth arm added before any arm ran changed the conclusion: PPO's readout is 5.5× the
anatomical norm at cosine −0.178, so it replaces the prior rather than refining it, and at matched
norm **random (9.639) beats anatomical (8.265) beats PPO (6.123)** — PPO's own direction is the
worst of the three, the co-adaptation caveat measured, with `ppo` bimodal across seeds. **The drift
invariance survives**: 1.38–1.42× across all four readouts against R.1c's 1.37–1.38 across every
dimension, so it now spans two independent structural axes and is the strongest remaining argument
that **credit assignment** is what is wrong. All three tensors frozen under the rule are therefore
accounted for, and **R.2 (e-prop) is the live path**. Neither R.1c nor R.1d could have satisfied
deliverable 1: a rule taking a tensor from a gradient-trained run is not a plausible local learner,
which each registered before it ran.

**Reframed after Logbook 063 (2026-09-15): the rule programme closed, and the ladder has an instrument.** R.2 — e-prop, D1's named fallback — resolved `learns_without_the_substrate` ([Logbook 063](experiments/logbooks/063-l4-eprop.md)). Two arms reach competence on the connectome, the first in this programme: **17.570 foods of 20 at 52.61% full clear** and 13.675 at 40.22%, against a shared frozen floor of 2.353 and PPO's matched 18.945, on 16/16 seeds at q = 0.000. And the control registered before either ran says the wiring did not do it: the better arm is the one whose chemical matrix is **frozen**, letting the rule write it **costs 3.895 foods**, and every substrate-writing arm sits below the readout-only control by 3.9 to 15.9 foods. What learned the cell is an **8-parameter linear readout over four pooled motor-class means, on frozen recurrent features**. The 1.37–1.42× credited drift invariant across R.1c, R.1d and R.2 is therefore what the cost of writing the wiring looks like, and the 059 programme's two eligibility families are both closed. Three consequences. **(i) The plausibility claim narrows, and is restated (D14).** No rule that writes the wiring learns this substrate to any benefit; what survives is a narrower claim that this animal's biology would recognise — the connectome is essentially invariant across individuals and much behavioural flexibility is modulatory — that *a small local readout reads the wiring's fixed features*. Every result under it is a performance claim. **(ii) 7b is deferred to the phase after 7 (D14).** Its comparative sweep names a learner that does not exist; its PPO fallback would measure the wiring as a learning-speed prior under gradient descent — real, per block V, but not the question 7b was registered to ask; and its D9 scaffold puts a more artificial readout at the centre than the one R.2 just showed carries the learning; it belongs after the ladder, on a substrate where the wiring has been shown to matter. Phase 7 closes on 7a, the frozen-features R.1b, V.4 and the synthesis. **(iii) The ladder resumes with an instrument.** Every rung so far — atlas signs, consolidation, decorrelation, routing — was tested with a rule that could not learn, so "the wiring did not start to matter" was never separable from "nothing could have shown it". `readout_only` is the first plausible learner on this substrate that reaches competence. The frozen-features R.1b runs first — the 2×2's primary contrast under the learner that works; a null extends 034's degree-statistics verdict to a second regime and every rung below runs against that baseline. Then two cheap rungs R.2 motivates, ahead of the ladder's own order and as SHOULD within the window: **readout width** (the 2×4 pooling may be the bottleneck through which the wiring's features are invisible) and **intrinsic dynamics** (rung 3 — a reservoir with no temporal dynamics is a poor reservoir, and it is the direct follow-up to R.2's finding that the rule writes uniformly over graph distance). The body stays in the phase after 7. R.2 is partly a measurement of that gap — a 2×4 map standing in for the neuromuscular system did the learning — but there is nothing to embody until the fixed features are shown to carry anything a richer motor output could exploit.

#### Phase 7 progress record

The running history of Phase 7's resolved work, moved out of the Timeline Overview's status cell
on 2026-09-13: a table cell cannot hold a line break in GFM, so an append-only history kept there
grows as one unreviewable line. The cell now carries the status and delegates here, mirroring how
Phase 6's row delegates to § Phase 6a / 6b split.

7a-i panel resolved 2026-09-06 to the sanity-floor branch, [Logbook
040](experiments/logbooks/040-l4-panel.md); panels 2 and 3 resolved 2026-09-07 `inconclusive`,
[Logbook 041](experiments/logbooks/041-l4-panel2.md), [Logbook
042](experiments/logbooks/042-l4-panel3.md) — the Hebbian wiring contrast closed as a registered
question; S.2 warm start resolved 2026-09-08 `sanity_floor_fail` + `rule_destroys_clone`, [Logbook
043](experiments/logbooks/043-l4-warm-start.md) — the connectome holds a competent policy, the rule
takes it apart; 7a-ii's first fidelity rung resolved 2026-09-09 `degree_statistics`, [Logbook
044](experiments/logbooks/044-l4-atlas-signs.md) — grounded synapse signs leave the prior alone and
make Hebbian learning worse, so the substrate's signs were not the limit; three consolidation
mechanisms screened 2026-09-09 and none held a cloned competent policy, [Logbook
045](experiments/logbooks/045-l4-consolidation.md) — slowing the update is not consolidating a
policy; the decorrelating term ran 2026-09-09 and resolved `no_recovery`, [Logbook
046](experiments/logbooks/046-l4-decorrelation.md) — the atlas grounds too little inhibition to
build a brake from; structured instruction resolved 2026-09-10 `no_routing_effect`, [Logbook
047](experiments/logbooks/047-l4-structured-instruction.md); block I then found the rule was not a
policy-gradient estimator and repaired it without making it work on any multi-step task, [Logbooks
048](experiments/logbooks/048-l4-rule-positive-control.md)–[056](experiments/logbooks/056-l4-ladder-reread.md);
and **block V found the wiring advantage the phase was looking for** — +35.4% off time-to-competence
on a foraging cell under thermal pressure and +23.5% with temperature removed, **each figure carrying
the standing condition that rewiring varies with initialisation** *(2026-09-19; **pairing half discharged 2026-09-21**, [Logbook 070](experiments/logbooks/070-init-sharing-control.md) — no dissolution detected under a shared initialisation, survival established on five of eight readings with three unresolved, and the thermal magnitude replicating at 35% of its committed size)*, [Logbooks
057](experiments/logbooks/057-wiring-premise-contrast.md) and
[058](experiments/logbooks/058-wiring-premise-difficulty.md). **7a complete 2026-09-13 on the SPLIT
clause** ([Logbook 059](experiments/logbooks/059-7a-shipment.md)): two citable results — a
systematic negative with a diagnosed cause, and a wiring advantage on learning speed — with GO
unreachable on its own clause and STOP overstating. **Status 7a complete / 7b pending**; 7b's gate
stands as written and the forward programme is rule families, bounded. **R.1, that programme's first
item, resolved 2026-09-13 `mixed`** ([Logbook
060](experiments/logbooks/060-l4-perturbation-scale.md)): the rule **solves a multi-step foraging
cell at eight perturbed units** — 19.64 foods of 20, 90.7% full clear, level with PPO — and
collapses monotonically to 3.0% at the 128 units every failing arm ran, while on the one-step
control the same dimension costs almost nothing, so what binds is an interaction between dimension
and horizon rather than either alone, with a product law more than this design supports. The wiring
contrast under that rule is not yet run, so the gate still stands. **R.1c resolved 2026-09-14 `not_reducible`** ([Logbook 061](experiments/logbooks/061-l4-reduced-perturbation.md)) — no perturbation set from 1208 draws down to 39 makes the rule learn, with credited drift invariant at 1.37–1.38×; **R.1d resolved 2026-09-14 `readout_helps_but_not_enough`** ([Logbook 062](experiments/logbooks/062-l4-frozen-readout.md)) — substituting the readout doubles what the rule reaches without any arm becoming competent, scale doing the work; and **R.2 resolved 2026-09-15 `learns_without_the_substrate`** ([Logbook 063](experiments/logbooks/063-l4-eprop.md)) — e-prop reaches competence, 17.570 foods at 52.61% full clear, with the chemical matrix **frozen**, and every arm that writes it does worse. **The bounded rule programme closed 2026-09-15 in a state none of its three outcomes named**, and **D14** records the consequence: 7b deferred to the phase after 7, Phase 7 to close on 7a + the frozen-features R.1b + V.4 + the synthesis, the plausibility claim restated, and the ladder resumed with `readout_only` as its first working instrument

#### Pre-registered design decisions (2026-08-27 pre-start review)

Recorded here so the Phase 7 OpenSpec change inherits explicit resolutions rather than silences. Status: **ratified 2026-08-27**. D1–D8 were ratified at the pre-start review; D2/D3/D7 were then amended and **D9–D13 added the same day** after a three-lens adversarial review of the committed draft (hostile-referee, execution-feasibility, and research-strategy critiques — findings recorded in the pre-start review document). Each decision remains cheap to reverse before implementation starts and expensive after; any further amendment goes through the Phase 7 tracking change with a dated note.

| # | Decision | Resolution |
|---|---|---|
| **D1** | **L4 rule family** — the v4.1 phrase "vanilla STDP on the connectome" is unimplementable as written: `ConnectomeTopology` is a tanh rate-code with no spike times and no cross-step state (its recurrence is within-step settling — why it scored chance, 0.499, on bit-memory in [Logbook 030](experiments/logbooks/030-bit-memory-positive-control.md)). | **Staged.** Primary arm: **rate-based three-factor rules** (reward/neuromodulator-modulated Hebbian with eligibility traces) on the existing rate-code substrate plus new persistent pre/post activity traces — keeps the substrate otherwise frozen and maps directly onto the monoamine-gating deliverable. (v4.1's risk-table fallback, "reward-modulated Hebbian without spike timing", is hereby promoted to the primary path.) Optional arm (**MAY** — demoted from second-arm status 2026-08-27: the LIF-infrastructure reuse claim proved thin on audit — dense layered `nn.Linear` LIF transfers little to a 302-neuron sparse recurrent graph, and gap junctions couple membrane potentials, not spikes): a **spiking connectome variant** for timing-dependent rules, still where spiking-on-connectome belongs if pursued. Fallback family: event-driven e-prop / eligibility propagation. |
| **D2** | **L4 performance bar** — the 2025-26 literature shows no pure STDP/R-STDP result at PPO parity on continuous control; a "reach the L2 PPO baselines" bar would fire the failure pivot by construction. | *(Amended 2026-08-27.)* The frozen-weights and vanilla-rule baselines are **sanity floors** (any rule that learns clears the first); the load-bearing bets are the D10 arms. Success = the **2×2 resolves** — plastic wild-type vs plastic rewired-null — with the tests pre-defined **before any run** and kept within-regime: **(i)** primary — wild-type-plastic beats rewired-null-plastic at q < 0.05, paired seeds; **(ii)** ranking — wild-type-plastic reaches or exceeds the **matched-rule MLP**'s band (same rule, quantitative). Tier placement against the PPO reference arms is reported as descriptive context only, never as a pass/fail metric (the comparison is cross-regime). *(Amended 2026-09-05 per the D7 gate outcome: the substrate froze mode-off, so the **Logbook 029 numbers stand** as that descriptive frame — no re-baseline exists or is needed.)* Comparisons to the PPO-trained arms are **qualitative context only** (the project's own commensurability rule forbids treating cross-regime deltas quantitatively); the quantitative yardstick is the **matched-rule set** (D10). Frozen-weights baseline pinned as: wild-type topology, initial weights, no learning — the same initialisation the plastic arms start from. *(Corrected 2026-09-05, A.4: this row previously read "Cook-2019 synapse-count-derived initial weights", which the implementation has never done. The connectome supplies **which edges exist**; weights along them are drawn `N(0, 1/√(chemical in-degree))`, and `syn.weight` — the EM synapse count — never reaches `w_chem`. The in-degree setting the scale is an edge tally, not a count of synapses. The substrate is therefore **anatomically constrained in topology and randomly initialised in weight** — a defensible modelling choice, but less than the old wording claimed. This satisfies the standing instruction above to state the weight-initialisation choice explicitly in the L4 design. The wording was corrected rather than the initialisation: changing it would move the substrate Amendment A froze and that Logbooks 029/034 are recorded against, and would require re-baselining everything measured on it — a legitimate future change, never a silent one.)* The floors are configured under the **plasticity rule**, not the gradient rule, so they share the plastic arm's anatomical motor readout; a PPO-configured floor would decode differently from the arm it bounds and confound decoding with learning. The vanilla-rule floor is resolved as **unmodulated Hebbian** (`Δw = η·E`): beating the frozen floor shows only that something was learned, whereas beating this one shows something was learned *from reward*. PPO parity is explicitly *not* the bar. |
| **D3** | ***P. pacificus* comparison design** — the Cook 2025 data is head-only (nose → retrovesicular ganglion), chemical-synapse-only, N=2, pharynx excluded. | *(Amended 2026-08-27.)* **Matched head-truncation**: re-cut the Cook 2019 *C. elegans* connectome to the same scope (head circuit, chemical synapses only, the **D9 scaffold** for both arms) so species is the *dominant* independent variable — residual confounds (N=2 shared-core edge-censoring vs single-reconstruction elegans; scaffold choice) are named and sensitivity-checked rather than claimed away. Ingest the **shared-core** matrix, with the homology mapping table an explicit, reviewable artefact. The **MUST comparison covers the two homologous behaviours** (klinotaxis, thermotaxis); the **species-appropriate third behaviour is SHOULD** — *C. elegans* keeps predator evasion (distal-channel-only at head scope, per D9); *P. pacificus* — the predator, not prey — gets predatory approach/bite (serotonin-gated per eLife RP 109557), which requires **new environment machinery** (prey object, capture mechanics, predation reward): a scoped, budgeted exception to "deepen, don't broaden", carried in 7b. **Dauer** (Yim 2024, nerve-ring scope) ships as the SHOULD **pathfinder** condition, run *first* (D11) — a within-species wiring-state transfer with no homology tax. |
| **D4** | **Learnable gap junctions** — twice-deferred with the condition "revisit only if Phase 7 L4 plasticity work calls for it". | **YES for the *C. elegans* L4 work** — the calling condition is met (gap junctions are genuinely plastic in the biology; Bhattacharya & Hobert 2019). Ships as a bounded SHOULD ablation on the L4 substrate, not a MUST. (The pacificus arm has no gap-junction data, so the matched cross-species baseline runs chemical-only regardless.) |
| **D5** | **Config debt #254 (`normalize_advantages` dead keys)** — implementing normalization "would shift all MLP results", i.e. move the Logbook 029 yardstick every L4 claim is measured against. | **Freeze.** Remove the dead keys (config honesty), keep training behaviour unchanged, so the Phase 6 baseline stays a valid comparator. Implementing normalization is deliberately deferred past Phase 7's comparative claims. |
| **D6** | **Env vectorisation** — pre-registered as 6b's binding constraint (T8.0); a vmappable env would also unlock long/multi-episode L4 training for faithful slow-forming memory. | **Scoped-first, decided once, jointly.** Run 6b as a scoped NEAT search without the port; commit the vmappable port only if the faithful-slow-memory L4 training demonstrably needs the throughput. Do not decide it twice. This resolves T8.0's *direction* only: before any NEAT campaign runs, the `phase6b-tracking` change still pins the reduced-search contract — population size, generation count, behaviour scope, seed budget — and the L3 evidence bar stands unchanged (NEAT-vs-connectome results on ≥ 1 behaviour with the lag-matrix or an equivalent discriminative instrument). If a scoped search cannot meet that bar within the pinned budget, env vectorisation returns as the 6b gate. |
| **D7** | **State-dependent action `std`** — root cause of the [036](experiments/logbooks/036-realworm-thermotaxis-validation.md) thermotaxis klinokinesis failure (a state-independent `log_std` can steer but cannot stochastically random-walk); an action-space change, not a plasticity change, and previously covered by no Phase 7 deliverable. | **YES, early-Phase-7 platform change**, landed and validated *before* the L4 panel and then frozen (the substrate-freeze lesson: never change substrate and learning rule in the same comparison). Byte-identical-when-off, per house standard. **Validation gate (added 2026-08-27):** re-run the 036 thermotaxis assay post-D7 — the klinokinesis signature must be *present* before the freeze. **AMENDED 2026-09-05 ([Logbook 038](experiments/logbooks/038-state-dependent-std-gate.md)) — gate FAILED after the one pre-registered entropy-only pass**: attempt 1 hit the per-state clamp-ceiling trap (caught live by the monitor); attempt 2, with healthy std dynamics, left klinokinesis EQUIVOCAL and weathervane regressed — capability without pressure. **Amendment A**: the substrate freezes **mode-off** (`state_independent`); the D7 mechanism ships as a tested, dormant capability; the post-D7 re-baseline is **descoped** (nothing changed under Logbook 029, which remains the reference frame); the Leifer/Chen validation target stays unmeasurable; klinokinesis emergence is recorded future work needing reward-side pressure or longer horizons, retried only via its own pre-registered change. |
| **D8** | **L4 package naming** — `quantumnematode/plasticity/` already exists and is the *quantum-plasticity eval protocol* (catastrophic-forgetting evaluation), not learning rules. | L4 code lands under a **new package** (working name `learning_rules/`); the existing package is not overloaded or silently repurposed. |
| **D9** | **Head-circuit sensor/motor scaffold** *(added 2026-08-27)* — the existing motor readout pools ventral-cord classes VB/DB/VA/DA, of which only **6 of 39 neurons survive** a nose→RVG truncation, and predator evasion's contact channel (ALM/AVM/PLM) is mid-body/tail — outside head scope entirely. v4.2's "identical motor-scaffold policy" named a policy that did not exist. | Both head-truncated arms read out via **command interneurons (AVA/AVB)** — the biologically-defended head-scope motor proxy — pre-registered before any training run, with a **scaffold-sensitivity check** (a second readout choice on one behaviour). A **per-behaviour sensor-coverage audit** of the truncated scope precedes the sweep. Head-scope predator evasion is declared **distal-chemosensory-only** (ASH/ASI) — a different task than the full-body 029/034 cell, and reported as such. |
| **D10** | **L4 panel arms** *(added 2026-08-27)* — "recovery of standing" against the 029 ranking is uninterpretable by the project's own Logbook 034 (the standing is a degree-statistics artefact), and comparing a plastic connectome to PPO-trained arms quantitatively violates the project's own commensurability rule. v4.2's claim that MLP has "no biologically-plausible-plasticity analogue" was **wrong** — the rate-based three-factor rule is substrate-generic — and is withdrawn. | MUST arms of the L4 panel: plastic **wild-type**; plastic **degree-preserving rewired-null** (mechanism already shipped byte-identical-when-off, Logbook 034); frozen-weights and vanilla-rule **sanity floors**; and a **matched-rule MLP** — the yardstick that makes "recovery" well-defined. CfC under the matched rule if it ports cheaply (MAY). The PPO cells of the 2×2 are **not** panel rows — they are the completed Phase 6 cells (Logbooks 029/034), entering as qualitative context per the hypothesis statement. n ≥ 8 paired-seed throughout; effects claimed only if ensemble-invariant (bar (a)). |
| **D11** | **Cross-species protocol** *(added 2026-08-27)* — v4.2 never defined "transfer": the exit criterion described a from-scratch comparative sweep while the novelty map claimed "transfer of trained agents"; no learning regime or success metric was stated. | MUST = **comparative cross-connectome learning** (from-scratch matched sweep, honestly named), run under the L4 rule with PPO as secondary context — under PPO alone, Logbook 034 predicts a wiring-null, and the plan says so in advance. SHOULD = **homology-mapped weight-transplant transfer** (zero-shot + fine-tune; non-homologous-edge policy stated in the transplant spec) — the arm that actually earns the "transfer of trained agents" claim. One pre-registered quantitative transfer metric per behaviour. **Dauer runs first** as the pathfinder. A single-animal-vs-shared-core **sensitivity run** on one behaviour guards the N=2 issue. |
| **D12** | **Diffusible-layer v1 representation** *(added 2026-08-27)* — the connectome model carries no neuron positions, so "concentration field" had no defined geometry; and the Biological Fidelity table placed the field env-side while the deliverable text placed it brain-internal. | v1 = **per-modulator global scalar concentrations** (serotonin, dopamine), brain-internal, driven by release-neuron activity (Wang 2024 atlas identities) + internal state, with **receptor-class gating** doing the targeting. Graph-local or spatial diffusion is an upgrade path gated on ingesting soma coordinates (unscoped). The novelty claim is restated to match (receptor-gated global signals, not a spatial field). **Head-scope source policy (deterministic, pinned):** all head-truncated arms (elegans-head, pacificus, dauer) run **internal-state-driven modulation only** — the global scalars are driven by internal state (food events, satiety); the release-neuron-activity drive term is active only on the full-connectome *C. elegans* arms, where the source neurons (NSM — pharynx; HSN/PDE — body) actually exist. Ghost source nodes and env-side external sources are **rejected** (unmeasured degrees of freedom). The policy is identical across all truncated arms, so every mandatory head arm in the D11 sweep has a matched, deterministic third-factor input; the full-vs-head difference in modulation drive is reported as a limitation. |
| **D13** | **7a sequencing** *(added 2026-08-27)* — the v4.2 bundle put the headline behind the full grounded stack, left the complete 6a result unstaked for a year in a preprint-speed field, kept the one cross-organism-validated lever (imitation warm start) as a post-failure contingency, and slotted 6b into an early window that does not exist (no GPU in the dev environment; the window is the most decision-dense stretch of 7a). | 7a splits: **7a-i** = trace substrate + rule-seam decision + minimal three-factor rule + the **D10 2×2 panel** — the headline lands first and is preprintable; **7a-ii** = the receptor-grounded neuromodulator stack (D12 representation + atlas curation + modulated rules), with the panel re-run under the grounded rule. A **6a preprint** (arXiv/bioRxiv) is SHOULD, drafted in the 7a-i window — the claim-stake, with the pre-registered 2×2 in its discussion; journal strategy stays unpromised. The **imitation-warm-start arm** (behavioural-clone connectome + rewired-null on the MLP champion's rollouts, then PPO fine-tune; n ≥ 8) is SHOULD and runs **early**. **6b is decoupled**: opportunistic, no scheduled window, pending a dated GPU/cloud decision. |
| **D14** | **7b deferral and the Phase 7 close** *(added 2026-09-15, after [Logbook 063](experiments/logbooks/063-l4-eprop.md))* — the bounded rule programme 059 registered ended in a state none of its three outcomes named: e-prop reaches competence on the connectome (17.570 foods of 20, 52.61% full clear, 16/16 seeds), and the arm that does so has its chemical matrix **frozen** — every arm that writes the wiring does worse than the readout-only control, by 3.9 to 15.9 foods. So the learner 7b's comparative sweep names (D11/D12's grounded modulated rule) does not exist and has no substitute; under the PPO fallback the wiring is endpoint-inert ([034](experiments/logbooks/034-connectome-structure-controls.md)) but learning-speed-relevant (block V, +35.4%), so a PPO sweep on time-to-competence would be a legitimate experiment — one that measures the wiring as a *learning-speed prior under gradient descent*, which is not the question 7b was registered to ask; and 7b's D9 scaffold puts a *more* artificial readout at the centre than the 2×4 map R.2 just showed does the learning. | **7b's comparative core (C.3, C.5, C.6, C.7) moves to the phase after 7**, gated on the substrate ladder producing a rung on which the wiring measurably matters; C.1/C.2/C.4 become unscheduled infrastructure, built on data-driven need. **Phase 7 closes on 7a + the frozen-features R.1b + V.4 + the synthesis**, with the unmet MUSTs — the diffusible layer, the modulated rules, 7b — recorded as SPLIT per the shape 059 took. **The plausibility claim narrows**, and is restated: not "a local rule reads the wiring" but "a small local readout reads the wiring's fixed features" — performance claims only, under the claim discipline. **The ladder resumes with an instrument**: `readout_only` is the first plausible learner on this substrate that reaches competence, so a rung can now show whether the wiring starts to matter; two cheap rungs R.2 motivates are pulled forward ahead of the ladder's own order, as **SHOULD** within the Phase 7 window (carried to the phase after 7 if the close arrives first; only L.0 is MUST for the close) — **readout width** (is the four-class pooling the bottleneck? a readout over all 39 motor neurons) and **intrinsic dynamics** (rung 3: units with state across steps, since a reservoir without temporal dynamics is a poor reservoir). **The body rung stays in the phase after 7**: it is Sibernetic/c302 interop, Phase-8-sized, and there is nothing to embody until the fixed features are shown to carry anything. |

#### Required deliverables (MUST)

1. **L4 Plasticity Layer**

   Phase 6's L2 weight search uses PPO-family gradient learning. Phase 7's L4 adds biologically-plausible plasticity on the same connectome substrate, materialised in six sub-deliverables that must be designed together (structured instruction was added 2026-09-06 as its own item and is distinct from the modulated rule: the modulator's *routing* through instructive pathways, not its concentration) — a synaptic rule alone reproduces synaptic-level plasticity but misses the circuit-level behavioural plasticity the *C. elegans* literature is built around. Rule-family staging per **D1**; performance bar per **D2**; panel arms per **D10**; shipment staging per **D13** (7a-i minimal rule + 2×2 panel; 7a-ii grounded stack).

   - **Persistent activity-trace substrate** on `ConnectomeTopology` — cross-step pre/post traces (eligibility traces). The connectome brain currently re-initialises from sensor injection every step; any trace-based rule needs state that survives environment steps. This is an architectural addition, not a rule swap, and it lands first, byte-identical-when-off. *Budget note (2026-08-27 review):* the `LearningRule` Protocol currently has **zero consumers** and the connectome brain's PPO update is inlined — so the trace substrate plus the rule-seam decision (Protocol refactor vs bespoke brain) is the explicit **first 7a OpenSpec change** with its own budget, not a hidden line item. The in-repo precedent for cross-step state in training is `lstmppo`'s per-step-hidden buffer + chunked truncated BPTT.
   - **Rate-based three-factor rules** (primary arm, D1) on the connectome topology (~2-4 active weeks; see the estimate restatement below). The **spiking-STDP arm is MAY** (demoted 2026-08-27 — see D1): the rate-based arm alone carries the biological-plausibility claim for this graded-transmission animal. Nearest in-repo precedent: the legacy 3-factor Hebbian eligibility-trace mode in `qsnnreinforce.py`.
   - **Diffusible-signal layer** modelling at least serotonin and dopamine concentrations as a function of internal state (food detection, sensory input, satiety). Representation pinned per **D12**: per-modulator global scalars with receptor-class gating, brain-internal; no spatial field without soma coordinates. Direct methodological precedent to cite and compare against: diffusing-neuromodulator temporal credit assignment (Barretto-Bittar, Levina, Giannakakis & Zeraati, arXiv:2603.08949) — which does model spatial diffusion, a distinction the writeup states honestly. **Soft dependency — a minimal metabolic-state model** (not full ATP biophysics; the no-energy-model gap in § Known Gaps). *2026-08-27 sizing:* smaller than v4.1 implied — `BrainParams` already carries satiety/health to the brain boundary; the build is an internal-state sensory module plus the **concentration field** itself, which is the genuinely new substrate.
   - **Receptor-class metadata** on the connectome neurons — which neurons express which receptor classes, and which release which transmitters. Sources *(revised 2026-08-27)*: release identities from the CRISPR knock-in **neurotransmitter atlas** (Wang et al., eLife 95402, 2024 — corrects antibody-era assignments; every `neurotransmitter` field in `connectome/neurons.py` is currently unpopulated); receptor expression from the **bulk-integrated CeNGEN profiles** (eLife 106183, 2025) rather than thresholded scRNA-seq alone, which under-detects low-abundance GPCRs — exactly the receptor class at stake. *C. elegans* has documented behavioural roles for 5 serotonin, 4 dopamine, 4 tyramine, and 3 octopamine receptor classes per WormBook. Optional refinement: the multiplex extrasynaptic-signalling framework (arXiv:2604.02057) as a principled selector of *which* neurons the diffusible layer modulates. Ships as an explicit **vendored-data sub-deliverable** with its own provenance doc (the `data/connectome/PROVENANCE.md` house standard), scheduled before the modulated-rule work — ~1-2 focused weeks of two-atlas curation, previously unbudgeted.
   - **Modulated three-factor rules** — the third factor is neuromodulator concentration; receptor-class metadata determines which synapses see which modulators.
   - **Structured instruction** *(added 2026-09-06 after Logbook 040 and Perks et al. 2026)* — the third factor is **pathway-specific**, not a global scalar: the receptor-class metadata determines not only which synapses see a modulator but which *instructive pathways* carry credit to them, the property Perks et al. show the ELL's wiring provides and the property the 7a-i panel's global modulator lacked. 7a-ii's re-run of the panel is therefore the first test of the wiring hypothesis under a rule that can, in principle, use the wiring; the 7a-i null is its baseline.

   **Estimate restatement (2026-08-27).** The repo's demonstrated cadence contradicts month-denominated padding — Phase 6a's planned ~7-10 months of tranches ran in ~6.5 calendar weeks — so estimates are stated in **active-work weeks** with an observed ~2-3× calendar multiplier: 7a-i ≈ 3-5 active weeks (trace substrate + seam + minimal rule + 2×2 panel); 7a-ii ≈ 4-6 active weeks (atlas curation + diffusible layer + modulated rules + panel re-run). Split and gate decisions trigger on **pre-registered criteria, never month counts**. Loihi 2 / SpiNNaker 2 neuromorphic hardware deployment remains a credible MAY target (the spiking-MAY arm is its natural vehicle); the software-only path is fully sufficient for the headline claim.

   **Signalling-layer scope — a primary limitation, stated up front.** The three-factor rule gates plasticity on *monoamine* concentration (dopamine as the canonical reward-related third factor, plus serotonin / tyramine / octopamine). This is the tractable, mechanistically-grounded vehicle — monoamines have documented receptor-class behavioural roles and a clean three-factor mapping — and it is deliberately *not* replaced by the neuropeptide layer. But two facts must be recorded as the largest deferred fidelity gap in the neuromodulatory model: (1) the monoamine network is *sparser* than the synaptic connectome, whereas the dominant "wireless" signalling layer in *C. elegans* is **peptidergic** — the neuropeptide connectome (Ripoll-Sánchez et al. 2023, *Neuron*; CeNGEN-grounded) is >10× denser than the synaptic wiring; and (2) peptidergic transmission is slow, diffuse *volume* signalling, not a spike-timing plasticity gate, so it is the wrong substrate for the *learning rule* even though it dominates the *signalling graph*. Neuropeptide-mediated neuromodulation is therefore a candidate Phase-7+ layer in its own right, not a substitute for monoamine three-factor STDP.

2. **Cross-Species Head-Circuit Transfer** *(rescoped 2026-08-27)*

   Cook et al. 2025 (*Science* 389:eadx2143, 31 Jul 2025) published the *P. pacificus* **head connectome** — two adult hermaphrodite heads, nose tip → retrovesicular ganglion including the nerve ring; **chemical synapses only** (gap junctions identified but excluded as ultrastructurally ambiguous); pharynx excluded (covered by Bumbarger 2013); ~88% of neuron classes in a shared core across samples, with homology mostly one-to-one (AVH absent in pacificus). Data ships as MIT-licensed CSVs (`stevenjcook/cook_et_al_2025_pristionchus`) plus *Science* Supplementary Data S1–S6 — **an easy CSV parse, not a NeuroML ingest**; the hard part is homology, not format. Phase 7 uses it to ask: *do learned architectures transfer across real nematode head circuits, and what does the wiring difference contribute?*

   - **Matched head-truncation baseline** (D3): re-cut the Cook 2019 *C. elegans* connectome to the same scope (head circuit, chemical-only) and re-run it, so species is the dominant independent variable. Comparing full-elegans vs pacificus-head would confound species with network size and coverage. The head-scope sensor/motor scaffold is fixed by **D9**: AVA/AVB command-interneuron readout for *both* arms (the ventral-cord VB/DB/VA/DA readout does not survive truncation — 6 of 39 neurons remain), a per-behaviour sensor-coverage audit before the sweep, and predator evasion declared distal-chemosensory-only at head scope.
   - Import the **shared-core** pacificus matrix through the L0 / L1 pipeline: new CSV loader, species-keyed neuron-classification table, and — the real risk hotspot — a **species-keyed sensor/motor projection map** with an explicit per-homolog biological argument (the current projections are hard-coded *C. elegans* named-neuron tuples in `connectome_ppo.py`). Protocol per **D11**: the MUST sweep is honestly named *comparative cross-connectome learning* and runs under the L4 rule (PPO as secondary context — under PPO alone, Logbook 034 predicts a wiring-null, stated in advance); the **homology-mapped weight-transplant transfer** (zero-shot + fine-tune, non-homologous-edge policy stated) is the SHOULD that earns the "transfer of trained agents" claim; one pre-registered quantitative transfer metric per behaviour; a single-animal-vs-shared-core sensitivity run guards the N=2 issue.
   - Behaviours *(amended 2026-08-27 per D3)*: the **MUST comparison is the two homologous behaviours** — klinotaxis and thermotaxis (amphid homologies conserved — Han et al., eLife 47155, 2019; thermotaxis ground truth is thinner for pacificus, noted as a limitation). The **species-appropriate third behaviour is SHOULD**: *C. elegans* keeps predator evasion (distal-channel-only at head scope, per D9); *P. pacificus* is the *predator*, not prey (Quach & Chalasani 2022) — its third behaviour is predatory approach/bite, serotonin-gated per the 2025 monoaminergic map (eLife RP 109557). The pacificus half is a **new environment task** (prey object, capture mechanics, predation reward) — a scoped, budgeted exception to "deepen, don't broaden", carried in 7b. *Naming note (settled 2026-08-28):* **klinotaxis** is the roadmap-canonical name for the chemical-gradient-navigation behaviour cell (after the head-sweep sensing mode it uses); the validation tooling and `data/chemotaxis/` keep the literature term *chemotaxis* (chemotaxis indices — Logbook 035); and *klinokinesis* / weathervane name the two real-worm strategies **within** the cell. One behaviour, three vocabularies — 7b implementation and validation target the klinotaxis cell and validate it with the chemotaxis-index machinery.
   - **SHOULD**: the **dauer connectome** (Yim et al. 2024, nerve-ring scope) as a third condition — a within-species wiring-state transfer, better scope-matched to the pacificus head data than full Cook 2019 is, and already packaged in OpenWorm's ConnectomeToolbox. Dauer runs **first** (D11) — it builds the multi-connectome pipeline with no homology tax before pacificus contact.
   - **MAY**: extend with one *pacificus*-distinctive behaviour beyond the bite decision (e.g. mouth-form-linked foraging strategy) if scope supports it. Note the mouth-form *switch* itself is a developmental polyphenism — the simulable behaviour is the serotonin-gated feeding decision, not the wiring switch.

   Transferring to *C. briggsae* is **not** a Phase 7 deliverable — re-verified 2026-08: still no published *C. briggsae* connectome. No direct precedent for transferring trained agents across species connectomes was identified in the 2026-08 literature scan; to the best of that scan's coverage, this deliverable is first-in-field even at head-circuit scope.

#### Optional deliverables (MAY)

These are not Phase 7 exit criteria. The project may pursue any combination when evidence and context justify, but Phase 7 closes whether or not they ship.

- **Naturalistic working-memory fidelity (faithful slow-forming memory).** The Phase-6 memory-separation work — bit-memory positive control ([Logbook 030](experiments/logbooks/030-bit-memory-positive-control.md)) and the ARS/short-horizon null ([Logbook 032](experiments/logbooks/032-ars-source-depletion.md)) — showed that *engineered within-episode* memory tasks are Phase-6 territory (the architecture comparison resolves working memory), but the **faithful** biological memories are not delay-bridging working memory: real *C. elegans* memory is chemosensory associative learning and thermotactic set-point memory that **form slowly** over repeated trials / hours via neuromodulator-gated plasticity. That makes faithful naturalistic memory an **L4 application** — modulated-STDP on a *learning* task — needing long / multi-episode training and a neuromodulatory / metabolic-state substrate. Scope it alongside the L4 modulated-STDP deliverable (and the no-energy-model gap) when Phase 7 is planned; the Phase-6 engineered probes (`T7.separation.associative_memory` design sketch: [docs/research/associative-memory-probe.md](research/associative-memory-probe.md)) are the bridge. See § Known Gaps Carried into Phase 6+.
- **Polymodal sensory-neuron integration.** The Phase 6 connectome sensor projections (food → ASE/AWC/AWA, predator → ASH/ASI/ALM/PLM, thermotaxis → AFD) follow a deliberate *primary-role-only* convention: each behaviour routes to its dominant sensory neurons, and secondary polymodal roles are not modelled. But several of these neurons are genuinely polymodal in the biology — AWC carries both odorant and temperature signals (Kuhara et al. 2008); ASH is a polymodal nociceptor (osmotic + mechanical + chemical); AWA has minor nociceptive roles. A cellular-realism refinement would model these dual roles consistently across all projections (e.g. route the thermotaxis signal onto AWC alongside AFD, so AWC integrates odor + temperature the way the real neuron does). Deferred here rather than done piecemeal in Phase 6 because (a) it is not load-bearing for the architecture comparison — primary-role-only is a defensible and internally-consistent Phase 6 modelling choice — and (b) doing it for one projection in isolation would create an inconsistent precedent. Belongs with Phase 7's other cellular-realism work (receptor-class metadata, neuromodulation); scope it properly when Phase 7 is planned.
- **Paper drafts and submissions.** A platform paper, a connectome-learning paper, and a fitness-landscape paper are all plausibly publishable from Phase 6+7 results; the project may write them when evidence justifies. No specific venues are promised. *(2026-08-27: the **6a preprint** specifically is promoted out of this bucket to SHOULD — see D13 and the exit criteria; journal submissions remain MAY.)*
- **Spiking-STDP arm** *(demoted from SHOULD 2026-08-27 — see D1)*: a spiking connectome variant for timing-dependent rules. New architecture with minimal reuse from `_spiking_layers.py`; the neuromorphic MAY's natural vehicle if pursued.
- **Neuromorphic deployment.** Loihi 2 / SpiNNaker 2 implementation of the L4 plasticity layer — credible exotic-hardware angle for STDP-on-connectome work.
- **Reproducibility artefacts updated** to the Phase 7 platform state.

#### Biological validation targets (refreshed 2026-08-27)

The L4 validation question — "does modulated plasticity reproduce documented *C. elegans* learning dynamics?" — now has better-matched published targets than the ones v4.1 gestured at, plus one trap to avoid:

- **Dopamine-gated forgetting** of butanone associative memory (bioRxiv 2025.02.20.639379): dopamine-dependent, **extrasynaptic** (8 DA neurons acting on >100 targets), plasticity-linked — almost exactly the diffusible-dopamine machinery L4 builds. **Co-primary** with the Leifer target below; its evidence base is an unreplicated preprint (~18 months unpublished at plan date), so it never stands alone, and full reproduction depends on the MAY slow-memory deliverable chain — stated, not hidden.
- **Learning-altered navigation-strategy weighting** (Chen, Sharma, Pillow & Leifer, *PLOS Biology* 23(3):e3003005, 2025; state-switching follow-up, *PNAS* 123(25), 2026): quantitative signatures — how learning re-weights biased-random-walk vs klinotaxis strategies — that a modulated-plasticity worm should reproduce, and that plug directly into the Logbook 035/036 behavioural-curve machinery. **Co-primary** and peer-reviewed; gated on the D7 validation gate — klinokinesis must be present post-D7 for this target to be measurable at all. Absent a model-step↔biological-time calibration (none exists), all dynamic-reproduction claims are pre-registered as sign/shape-level.
- **Escape-circuit redundancy / lesion robustness** (He et al., *PNAS* 123(22), 2026): grounds the predator-evasion task and supplies a testable ablation/degeneracy phenomenon for the Robustness metrics.
- **Wiring-solved credit assignment with multi-site anti-Hebbian plasticity** (Perks, Petkova, Muller, Sawtell et al., *Nature*, 2026-09-02; electric fish ELL, EM connectome of ~650 neurons): inhibitory and disinhibitory instructive pathways, structured connectivity that solves credit assignment, fast and slow plasticity sites that cooperate, and recurrence that stabilises learning. Not *C. elegans* and not RL, but the closest sibling to the L4 programme and the biological answer to three things Logbook 040 found: Hebbian collapse without decorrelation, a global modulator that cannot instruct, and a single plastic site whose outcome is a fixed point set by the initial weights.
- ⚠️ **Thermotaxis set-point plasticity is the trap**: its biological mechanism is **receptor-level and intrinsic to AFD** (transcriptional thermoreceptor reconfiguration — *Current Biology*, 2025), not synaptic. It is the wrong validation target for a synaptic rule unless modelled as node-level adaptation; do not pre-register it as an STDP success criterion.

#### Phase 7 exit criteria

**Required (MUST):**

- ✅ **7a-i** — persistent-trace substrate + rule seam + minimal rate-based three-factor rule operational on the connectome; the **D10 2×2 panel** (plastic wild-type, plastic degree-preserving rewired-null, frozen-weights + vanilla-rule sanity floors, matched-rule MLP; n ≥ 8, paired-seed, BH-FDR) resolves the primary hypothesis under the **D2 success tests** (within-regime; PPO-arm tier placement descriptive only). *(Amended 2026-09-05: the descriptive reference frame is **Logbook 029** — the D7 gate failed, the substrate froze mode-off, and the re-baseline was descoped per Amendment A, Logbook 038.)* **Resolved 2026-09-06 ([Logbook 040](experiments/logbooks/040-l4-panel.md)): verdict `sanity_floor_fail` — the plastic wild-type arm beats none of its registered contrasts at n = 8 (T1 +6.6, T2 +9.6, T3 −12.4, T4 +6.2; all q ≥ 0.50); the matched-rule MLP yardstick sits at chance so test (ii) is vacuous. Substantive finding: outcomes are fixed points set by the random initial weights — the unmodulated Hebbian floor reaches 64–78% on three seeds with no reward — and the only wiring signal is descriptive, in the floors (wild-type Hebbian over rewired Hebbian +16.5, 5/8 seeds; frozen floors tie). Four rule/arm defects were found and fixed by pre-registered changes on the way (rate scale, modulator mean, frozen action noise + runaway + unbounded yardstick units, plastic readout).**
- ⚠️ **7a-ii — UNMET-WITH-REASON** *(status assigned at the close, 2026-09-19, [Logbook 069](experiments/logbooks/069-phase7-synthesis.md): no rule that writes the wiring learns the cell to any benefit, and the ladder resumed with a learner that **reads** it instead; the diagnosis is [Logbook 063](experiments/logbooks/063-l4-eprop.md))* *(**SPLIT 2026-09-13**, [Logbook 059](experiments/logbooks/059-7a-shipment.md); **closing scope under D14, 2026-09-15: L.0 the frozen-features R.1b, and V.4** — the remaining rule and fidelity items below are resolved, deferred, or carried as SHOULD)* *(restructured 2026-09-08 after Logbook 043 and the clone-destruction diagnostic; **reordered 2026-09-09 around the instrument** — see the second reframing paragraph above: the substrate ladder is paused behind a positive control for the rule, an eligibility with the noise inside it, and a statistic and metric matched to the outcome)* — **B.1 the atlas first** ✅ *(run 2026-09-09, [Logbook 044](experiments/logbooks/044-l4-atlas-signs.md): **`degree_statistics`**, the registered `substrate_fail` gate did not fire — the prior is indifferent to signs (G1 +0.8 ns), the wiring contrast is not rescued (G2 +4.9 ns), Hebbian learning gets worse (wild-type 31.5 → 14.0) and Dale's law worse again (G4 −4.8 reverse, enforcement silencing the synapses the rule was inverting); receptor classes deferred to B.3)*, as a substrate deliverable with its own registered test (transmitter identities on the Cook 2019 wiring; does grounding the signs change the frozen prior sweep or the Hebbian wiring contrast of Logbooks 041–042?); then **the rule work**, which B.1's result moved ahead of the remaining substrate items (grounded signs left the prior alone and made the purely potentiating rule worse, so the next result has to come from the rule): a **consolidation mechanism** (an update magnitude that scales monotonically with the prediction error's magnitude, so a small `|δ|` produces a small weight change — today's RMS normalisation gives a full-size step however small the raw error is — or a slow protective variable; dopamine-gated consolidation rather than dopamine-gated change), then **structured pathway-specific instruction** (B.4b) — each variant cleared through the [clone assay](experiments/logbooks/supporting/043-l4-warm-start/destruction-diagnostic.md#the-clone-assay), the single protocol defined with the diagnostic, before **B.5**, the 2×2 re-run under the grounded rule, which that assay gates. *([Logbook 045](experiments/logbooks/045-l4-consolidation.md): the first three consolidation mechanisms were screened 2026-09-09 and **none passed**: an elastic anchor, a per-synapse protective variable and an oracle gate on episode success. Rigidity came closest — six of eight seeds inside the per-seed band and the best policy preservation this substrate has shown, at a 91% per-synapse rate cut — and still lost 9.4 points on the mean. The panel stays gated; the oracle's own launch record predicted its failure mode, a brake whose signal arrives a hundred episodes after the damage, so a fast quality-gated brake is untested rather than refuted.)* The internal-state module (B.2) and the diffusible-signal layer with its receptor classes (B.3, D12) stay queued as fidelity work either side of the rule variants, with B.1's finding that the receptor layer should not be expected to rescue learning on its own.
- ⏸️ **7b** — *(**deferred to the phase after 7, D14, 2026-09-15**: the comparative core moves behind the substrate ladder; C.1/C.2/C.4 are unscheduled infrastructure; Phase 7 closes without it, recorded as SPLIT)* *P. pacificus* shared-core head connectome imported through L0 / L1 with an explicit homology table; matched head-truncated *C. elegans* baseline (D9 scaffold) re-run; **comparative cross-connectome learning** sweep (D11) on the two homologous behaviours with a quantitative cross-species comparison.

**Recommended (SHOULD):**

- ⛔ **SUPERSEDED-BY-RESULT** — 6a preprint (arXiv/bioRxiv) submitted in the 7a-i window, with the pre-registered 2×2 in its discussion (D13). *(Recorded **cancelled** 2026-09-15 as S.1 in the tracker, and the status is assigned here at the close, 2026-09-19: not deferred, because D13's case for staking 6a early was a fast-moving field and Phase 7's own results now make a stronger combined package than 6a alone. The publication decision is relooked after this close — see § What Phase 8 opens on.)*
- ✅ **Panel 2 — the Hebbian wiring contrast** *(added 2026-09-06; run 2026-09-07, [Logbook 041](experiments/logbooks/041-l4-panel2.md): **`inconclusive`** — the wild-type Hebbian advantage held its size on fresh seeds, +14.1 over all 16 seeds on 10/16 with the interval clear of zero (+11.9 on the fresh seeds 9–16, +16.2 on panel 1's seeds 1–8), but a bimodal outcome leaves q = 0.28; the 64-seed prior differs by only +3.3, so the advantage is created by alignment, not present in the prior; linear synapse-count-scaled initialisation halves the wild-type Hebbian fixed point and erases the contrast, with no detectable difference on any frozen prior — count structure returns only with signs, in 7a-ii). **Panel 3** (2026-09-07, [Logbook 042](experiments/logbooks/042-l4-panel3.md)) replicated the contrast on 48 fresh seeds: `inconclusive` at +8.1 with a 24-against-22 sign split, pooled 64 seeds +9.6; the advantage lives in the level of the wild-type's good fixed points, not their frequency, and the question is closed as registered)*. As registered: wild-type Hebbian vs rewired Hebbian as the registered primary at n ≥ 16 paired seeds (1000-episode budget; the Hebbian arms settle within a few hundred), with a frozen-floor **prior sweep** over many seeds (the fraction of random initialisations on each wiring that forage and evade with no learning) and **initialisation** as a factor (random vs Cook-2019 synapse-count-scaled weights, an axis that is itself a wiring property and had not been explored before panel 2). It tested the one signal [Logbook 040](experiments/logbooks/040-l4-panel.md) produced and characterised the initialisation landscape every plasticity result on this cell is read against. Both panels are complete; with S.2 also run, the current next lever is 7a-ii with structured instruction.
- ✅ Imitation-warm-start arm *(run 2026-09-08, [Logbook 043](experiments/logbooks/043-l4-warm-start.md): **`sanity_floor_fail`** + **`rule_destroys_clone`** — the clone gate passes on 8/8 seeds (39% via the chemical weights alone, 74% with every PPO parameter; the rewired null carries it at least as well, so the 029 rank is not an inability to hold a policy of that quality — one cloned policy, no broader representability claim), the three-factor rule takes the clone apart, below its frozen clone on the wild-type (registered) and on the rewired null (−22.6, descriptive), the Hebbian rule holds it on the wild-type only, and PPO fine-tuning from the clone loses to low-noise PPO from scratch, which reaches 69–81% in 3000 episodes against 029's 52%)*: connectome + rewired-null, behavioural-clone → PPO fine-tune, n ≥ 8, run early (D13). *(2026-09-06: also supplies the good initial policy the 7a-i panel lacked, turning "the rule destroys good policies" from one seed into a test.)*
- ⏸️ **DEFERRED-WITH-DESTINATION: the phase after 7** *(D14, 2026-09-15; status assigned at the close, 2026-09-19)* — Dauer-connectome pathfinder condition — runs before pacificus (D11).
- ⏸️ **DEFERRED-WITH-DESTINATION: the phase after 7** *(D14, 2026-09-15; status assigned at the close, 2026-09-19)* — Homology-mapped weight-transplant transfer, zero-shot + fine-tune — the arm that earns the "transfer of trained agents" claim (D11).
- ⏸️ **DEFERRED-WITH-DESTINATION: the phase after 7** *(D14, 2026-09-15; status assigned at the close, 2026-09-19)* — Species-appropriate third behaviour (D3 amended; the pacificus predation-task env is scoped, budgeted 7b work).
- ⏸️ **DEFERRED-WITH-DESTINATION, NARROWED: a PPO arm on the block-V cells** *(status assigned at the close, 2026-09-19, [Logbook 069](experiments/logbooks/069-phase7-synthesis.md))* — Learnable-gap-junction ablation on the *C. elegans* L4 substrate (D4). B.6's recorded status names **two** destinations and the phase closed one of them: [Logbook 063](experiments/logbooks/063-l4-eprop.md) found every substrate-writing arm doing worse than the readout-only control, so no local rule makes a further plastic tensor promising, and [Logbook 067](experiments/logbooks/067-l4-feature-ablations.md) lowered the priority further by finding that **removing** gap junctions helped both wirings. **The second destination is untouched and live**: under PPO the wiring is learning-speed-relevant, gap junctions are frozen there, and making them plastic is unattempted. Recorded as deferred rather than superseded, which this synthesis's own first draft got wrong.
- ❌ **UNREACHABLE-WITH-REASON** *(status assigned at the close, 2026-09-19, [Logbook 069](experiments/logbooks/069-phase7-synthesis.md))* — Co-primary biological validation (dopamine-gated forgetting + Leifer navigation re-weighting), sign/shape-level. Both are predictions **about a plastic wiring**: they need a rule that writes the connectome to some benefit in order to yield a sign- or shape-level prediction, which [Logbook 063](experiments/logbooks/063-l4-eprop.md) established does not exist in this rule family and which D14 removed from the plausibility claim. Under a frozen substrate the model has no re-weighting to compare against the data. Not deferred: that would imply a schedule it has never had.

**Optional (MAY):**

- ⭐ **UNREACHABLE-WITH-REASON** (not a gate) — Biological-validation collaboration completed; ≥ 1 model prediction tested against published or partner-lab worm data. *(For the co-primary's reason: there is no plastic-wiring prediction to test.)*
- ⭐ **DEFERRED-WITH-DESTINATION** (not a gate): the post-close publication decision — Journal submissions in flight (the 6a preprint itself is SHOULD, per D13).
- ⭐ **DEFERRED-WITH-DESTINATION** (not a gate): Future Directions, unscheduled — Spiking-STDP arm results; neuromorphic deployment demonstrated.
- ⭐ **MET** (not a gate) — Reproducibility artefacts current, with a limit stated rather than a status invented. *(Every headline figure from Logbook 040 onward is re-derivable from the committed per-seed CSVs in git; re-derivation from raw needs the local campaign logs; the step-level exports before the readout-width era and V.3's tracked-experiment records are **gone**. See [Logbook 069](experiments/logbooks/069-phase7-synthesis.md) § Reproducibility.)*

#### Risk-mitigation: failure modes and pivots

| Failure mode | Trigger | Pivot |
|---|---|---|
| **L4 implementation overshoots** | Neuromodulator grounding more complex than estimated; receptor-class metadata harder to integrate; modulated rules harder to debug than vanilla | **Phase 7 is pre-structured as 7a / 7b** *(promoted 2026-08-27 from a contingency to the default shape, since the ~6-9-month full-grounding estimate made the old month-6 trigger fire by construction)*: 7a ships L4 on *C. elegans* as the headline deliverable; 7b carries the cross-species transfer + SHOULD/MAY items. The 6a/6b precedent applies — the split may equally be invoked **by success** (7a forms a self-contained citable result) as by overrun. Phase 7 closes whether it lands as one shipment or two. |
| **Cross-species homology proves ambiguous** | Sensor/motor projection homologs for pacificus can't be defended for one or more behaviours; shared-core matrix leaves a behaviour's circuit under-covered *(re-aimed 2026-08-27: format risk is retired — the data is MIT-licensed CSV; homology and coverage are the real risks)* | Ship the behaviours whose projections are defensible (klinotaxis is the safest — amphid homology is strong), document the gap per behaviour, and report transfer on that subset. The first-in-field claim survives in restricted form. |
| **L4 plasticity fails to beat its baselines** | The plastic connectome does not beat its own frozen-weights / vanilla-rule sanity floors (D2) after reasonable search on the D1 primary arm (and the MAY spiking arm, if run) | The finding itself is publishable — *"biologically-plausible plasticity on the C. elegans connectome requires further substrate work or different rule families"* — and is a strong *robustness* answer to the sharpened hypothesis. Phase 7 closes with the negative result; the e-prop fallback family and FlyGM-style imitation warm start are the documented next levers. **Branch closed 2026-09-13** — the realised history is in § The L4 baseline-failure branch. |
| **Substrate-vs-rule confound** | Platform changes (D5 key removal, D7 state-dependent `std`) land mid-comparison, making L2-vs-L4 deltas uninterpretable — the Phase 6 grid-vs-continuous non-commensurability lesson | Land all substrate changes **before** the L4 panel, validate, then freeze. Any comparison spanning a substrate change is reported as qualitative, per the 2026-06-14 reframing precedent. |
| **Partial D2 outcome** | The rule clears the frozen/vanilla sanity floors, but the 2×2 primary contrast is null (or the matched-rule ranking test fails) | This is the pre-registered **robustness branch**, not a failure and not the sanity-floor row above: confirm ensemble-invariance (bar (a)) and read it against the warm-start arm, then **close with the negative result** — the degree-statistics verdict replicated across learning regimes, a citable finding. One pre-registered sensitivity pass (rule hyperparameters) is permitted before closure; no open-ended reruns. Rule-family pivots (e-prop fallback, spiking MAY arm) are *new* pre-registered runs, never rescues of this one. Distinct from STOP, which requires substrate-level infeasibility. |

#### The L4 baseline-failure branch

What actually happened along the *L4 plasticity fails to beat its baselines* row above, moved out
of its mitigation cell on 2026-09-13 for the same reason: the row's third column had stopped being
a mitigation and become an append-only record, and a GFM table cell cannot be wrapped.

**Realised 2026-09-06** ([Logbook 040](experiments/logbooks/040-l4-panel.md)): `sanity_floor_fail`
at n = 8; **7a-i closes with the negative result** as this row prescribes — Phase 7 itself continues
— and the documented next levers are now ordered — panel 2 (the Hebbian wiring contrast with a prior
sweep and an initialisation factor; **run 2026-09-07, `inconclusive`**, [Logbook
041](experiments/logbooks/041-l4-panel2.md): the effect held, the test could not carry a bimodal
outcome at n = 16, and count-scaled initialisation hurt; **panel 3**, [Logbook
042](experiments/logbooks/042-l4-panel3.md), replicated at +8.1 on 48 fresh seeds, `inconclusive`
again, and closed the question) — **both panels complete**; the imitation warm start (S.2) ran
2026-09-08 ([Logbook 043](experiments/logbooks/043-l4-warm-start.md): the connectome retains a
cloned competent policy on either wiring, the rule destroys it, a warm start hurts PPO); 7a-ii's
atlas rung ran 2026-09-09 ([Logbook 044](experiments/logbooks/044-l4-atlas-signs.md): grounded signs
leave the prior alone and make Hebbian learning worse, so the substrate was not the limit);
consolidation was screened 2026-09-09 ([Logbook 045](experiments/logbooks/045-l4-consolidation.md))
and none of its three mechanisms held a cloned competent policy, so the panel stays gated; the
anti-Hebbian/decorrelating term ran 2026-09-09 and resolved `no_recovery` ([Logbook
046](experiments/logbooks/046-l4-decorrelation.md)) — the arm redirected every grounded inhibitory
synapse there is, 5.8% of the substrate, and moved the outcome by −0.9, so the inhibitory-brake
explanation cannot be built at transmitter-only fidelity; structured instruction ran 2026-09-10 and
resolved `no_routing_effect`. **This row's branch is now closed (2026-09-13, [Logbook
059](experiments/logbooks/059-7a-shipment.md)).** It prescribed that "the finding itself is
publishable" and named the e-prop family and the imitation warm start as the documented next levers.
Both held: 7a ships on the SPLIT clause with the negative *and* a positive the row did not
anticipate — the wild-type wiring is worth **+35.4%** and **+23.5%** off time-to-competence across
two cells under PPO *(standing condition, 2026-09-19, **pairing half discharged 2026-09-21** by [Logbook 070](experiments/logbooks/070-init-sharing-control.md): rewiring varies with initialisation in every
block V panel, and the control that separates them opens Phase 8 — see § What Phase 8 opens on)*, so the phrase "requires further substrate work or different rule families" is
now sharper than the row could state it: **the wiring is legible to learning, and what is missing is
a plausible rule that reads it.** The e-prop lever is taken as a bounded programme with two stopping
conditions fixed in advance, after which 7b proceeds under PPO with the biological-plausibility
claim given up. **Sharpened again 2026-09-13** ([Logbook
060](experiments/logbooks/060-l4-perturbation-scale.md)): "different rule families" turns out to
understate it — the *same* rule family works at a perturbation dimension nobody had tried, reaching
PPO's level on a block-V cell at eight perturbed units, so what this row anticipated as a
rule-family problem is at least partly a **scale** one. e-prop is not retired and remains the
fallback; the live question is the wiring contrast under a rule now known to learn.

#### Where Phase 7 is first-in-field (novelty map)

Verified against the 2026-08 literature scan; "nearest precedent" is the closest published work, cited as convergent per the § Claim discipline framing. This is the section that carries the North Star — each row is a claim no other group holds, and each has a defined positive-result payoff.

| Phase 7 claim | Nearest precedent (2026-08) | Payoff if positive |
|---|---|---|
| **Reward-driven neuromodulated plasticity on the real *C. elegans* connectome, closed loop — resolved as the D10 2×2 (rule × wiring)** | FlyGM (fly, RL but no plasticity, preprint); **Perks, Petkova, Muller, Sawtell et al. 2026 (*Nature* 2026-09-02, electric fish ELL: EM connectome + multi-site anti-Hebbian plasticity + ephys-constrained model, no RL) — the nearest sibling, and the template for wiring-solved credit assignment**; MetaWorm/BAAIWorm (*Nat. Comput. Sci.*, worm, no learning); NeuroSimWorm (2025, no plasticity). **The plasticity × RL × real-connectome cell is empty.** | Wild-type ≫ rewired-null under the native rule family — with the PPO-null (Logbook 034) as built-in contrast — is a first-order structure-function result ("the wiring is legible only to its own learning regime"), available **even if the plastic connectome still trails the MLP**. A 2×2 null replicates the degree-statistics verdict across learning regimes — an equally citable architecture finding. *Outcome 2026-09-06 ([Logbook 040](experiments/logbooks/040-l4-panel.md)): resolved to the sanity-floor branch; the wiring signal is in the reward-free Hebbian floors, not in the reward-modulated arm — the negative is citable and reframes 7a-ii.* |
| **Cross-connectome learning and transfer** (elegans ↔ pacificus head circuits, + dauer state; the weight-transplant SHOULD is the "transfer of trained agents" claim — the MUST is comparative cross-connectome learning, per D11) | **None identified in the 2026-08 scan** — no published work found that transfers trained agents across species (or wiring-state) connectomes. | First measurement of what connectome wiring *contributes* across species under matched learning — the comparative-connectomics analogue of a transfer-learning study, and a template other connectome pairs (dauer, developmental Witvliet series, fly) can reuse. |
| **Receptor-atlas-grounded diffusible third factor** (receptor-gated global modulator signals per D12; a spatial field only if soma coordinates are ingested) | Diffusive credit assignment exists in the abstract (arXiv:2603.08949, which does model spatial diffusion) — not on a real connectome, not receptor-grounded. | A mechanistic bridge from volume transmission to credit assignment in a real nervous system — a computational claim about *why* the monoamine layer is wired the way it is, testable against the Dvali signaling-network modules. |
| **Faithful slow-forming memory as an L4 application** *(conditional on the MAY slow-memory deliverable chain — stated, not hidden)* | No in-silico connectome account of a documented *C. elegans* memory phenomenon exists. | Sign/shape-level reproduction of dopamine-gated forgetting and/or the Leifer navigation re-weighting (co-primary) would be the first mechanistic connectome-level model of a real worm memory — the flagship biological-validation result *if* its chain is pursued. |
| **Degeneracy-disciplined dynamics claims in a low-dimensional animal** (bars (a)+(b) + representational geometry vs WormID) *(conditional on a committed dynamics claim — the machinery unlocks only then)* | The degeneracy critique exists (Beiran & Litwin-Kumar; Dvali; Currier); no behaviourally-trained connectome model has yet been *held* to it with named-neuron grounding. | A validation paradigm contribution independent of which way the science lands — the methods template a referee-proof connectome-model paper needs. Model-accessible covariates are velocity and turning; head curvature and feeding are conceded out of reach (no body). |

#### Go/No-Go Decision

- **GO (7a shipment) if**: 7a-i's 2×2 resolves with D2-bar results in hand and 7a-ii grounds it in the receptor-gated neuromodulator stack. **UNREACHABLE (2026-09-13)** — the receptor-gated stack is B.3, which was never built because its own condition was that a rung after it could conclude, and that rung's clone-assay gate was failed by four mechanisms. Unreachable because a gate never opened, not because work was skipped ([Logbook 059](experiments/logbooks/059-7a-shipment.md)). Phase 7 then sits at **7a complete / 7b pending** — mirroring the 6a/6b pattern — and is marked **COMPLETE only when the 7b comparative cross-connectome sweep ships and the synthesis publishes**; "well underway" never satisfies completion.
- **SPLIT-shipment if**: 7a forms a self-contained citable result before 7b work starts, **or** L4 overshoots its software estimate — either way, execute the pre-structured 7a / 7b shape per the Risk-mitigation table. ✅ **TAKEN 2026-09-13** ([Logbook 059](experiments/logbooks/059-7a-shipment.md)): 7a ships **two** self-contained results — a systematic negative with a diagnosed cause (the rule was not a policy-gradient estimator at +0.009 alignment; the repair that is fails on every multi-step task) and a **positive wiring result on learning speed** (+35.4% and +23.5% off time-to-competence across two cells, both gates on every seed, the untrained prior indistinguishable in three measurements). The case against — that the headline MUST, a local rule reading the wiring, is unmet — is recorded rather than dismissed.
- **STOP if**: Both L4 implementation and the cross-species transfer are infeasible at the substrate level — at which point the diagnosis itself is the Phase 7 deliverable, and follow-on work picks up alternative rule families (e-prop, imitation-warm-start) or alternative connectome data sources. **NOT TAKEN (2026-09-13)**: neither is established. The substrate holds a competent policy (73.7% from a full-parameter clone) and PPO solves the task from random weights (68.5%), so what failed is one rule family; cross-species transfer was never attempted. Taking STOP would have repeated [056](experiments/logbooks/056-l4-ladder-reread.md)'s error with the sign flipped. **The e-prop fallback is nonetheless the forward programme**, bounded by two stopping conditions fixed in advance, after which 7b proceeds under PPO.

______________________________________________________________________

### Phase 8: Ground, then Embody — Measured Substrate & Body

**Goal**: Make the connectome substrate real enough that the wiring question can be asked at the level the animal answers it — measured synaptic weights on the Cook edges, intrinsic and gap-junction dynamics, the anatomical motor-to-muscle map, and a two-dimensional body — and re-ask the same registered question at each rung: *does the wild-type wiring start to matter once this is real?* Every rung runs against the degree-preserving rewired null with the MLP as yardstick, under the two learners Phase 7 showed work on this substrate (PPO, and the small local readout on frozen features), and each new component gets its own positive control before any connectome arm runs.

*(Added 2026-09-20 from the Phase 8 pre-start review, which ratified decisions **D15–D20** below. The review's evidence trail — the literature scan and the repository checks behind each decision — is a gitignored working document; what is load-bearing is recorded here.)*

**Scope decision, stated first.** The question the review opened on was whether Phase 8 should pursue the highest biological fidelity the platform could reach — a whole-organism *C. elegans*. It should not, and the reason is this roadmap's own v4 decision (§ Scoping Changes from v3: "simulating all *C. elegans* behaviours is the OpenWorm-15-year-trap"), which the evidence since then strengthens: BAAIWorm, the best-resourced whole-organism worm model, covers 136 neurons and one behaviour on a CUDA cluster (Zhao et al., *Nat. Comput. Sci.* 2024); single-neuron biophysics exists for six of 118 neuron classes (Nicoletti et al., *PLOS One* 2024); no group has a validated whole-organism model; and the 2026 fly demonstrations (Eon Systems, 2026-03-07; Jin et al., arXiv:2602.17997) show that running a connectome through a body *without learning or controls* is now an announcement rather than a result. **A landmark is available, and it is a narrower one**: the first *C. elegans* model in which a learner operates on the real wiring *with measured synaptic weights*, through the real motor-to-muscle map, into a body, against a degree-preserving null under an initialisation control, with the wiring effect's operating-point sensitivity reported and kinematics validated on lab data. Every clause of that sentence is missing from every other worm model and every clause is a rung this project can build. The framing is therefore **ground, then embody** — fidelity chosen by what makes the wiring question answerable, not fidelity for its own sake — and whole-organism fidelity (302 biophysical neurons, muscle electrophysiology, development, the life cycle, a 3D habitat) is stated as *not attempted*, which is what keeps the achievable claim credible.

**Not in Phase 8** (each kept in § Future Directions with its reason): cross-species work (7b's *P. pacificus* comparison, the dauer pathfinder, weight transplant — not interpretable until a readout that does not carry the learning exists, which is this phase's C.1); the male–hermaphrodite wiring contrast (data confirmed on disk, § Future Directions); multi-agent, pheromones, red-queen and ecological co-evolution; evolution in every form (6b NEAT stays deferred unscheduled — a body makes the env-vectorisation decision harder, not easier — Lamarckian and transgenerational work stays closed); structural plasticity across development; the neuropeptide layer as a rule substrate; spiking-STDP and neuromorphic deployment; 3D environments, the Sibernetic body and ion-channel neurons; and the uniform substrate-writing rule programme, closed with a diagnosed cause in [Logbook 063](experiments/logbooks/063-l4-eprop.md). The plasticity programme is **re-aimed, not re-opened**: three narrower rule questions survive, each behind a substrate rung — placed plasticity (ladder rung 6, MAY) once measured weights exist to place it on; plastic gap junctions under PPO (D4's surviving destination) inside the dynamics rung; and node-level adaptation (L.2) as the biologically faithful mechanism for what the worm literature actually documents (context-dependent olfactory plasticity through a lateralised *sensory* pathway, Pandey et al. *PNAS* 2026; starvation re-encoding thermosensory neurons, bioRxiv 2025.07.17.665269; the AFD set-point trap already recorded under Phase 7's validation targets; and the one well-characterised synaptic learning mechanism in the olfactory circuit weakening *gap junctions* — NMDAR-modulated RMG/AIB coupling, *Nat. Commun.* 2020).

**What the 2026-09-20 literature scan changed.** Five items move the plan; the rest constrain it.

- **A measured weight prior exists for this substrate.** Creamer, Leifer & Pillow (bioRxiv 2024.09.22.614271; latest version PubMed-indexed 2025-09-26, **still a preprint**) fit a linear dynamical system whose weights are non-zero only on connectome edges to optogenetic whole-brain recordings from 110 animals over 156 head neurons; it captures single-neuron perturbation responses at 92% of the data's own reproducibility, a fully-connected model does no better, a *shuffled* connectome abolishes the prediction, and **the fitted signs and magnitudes are released as a supplementary resource** with MIT-licensed fitting code. This roadmap cited the paper only as an argument that reweighting the fixed topology is the right lever (Phase 7 § evidence base). Every Phase 6–7 wiring contrast drew chemical weights — magnitudes and signs — at random; the atlas-sign arm ([Logbook 044](experiments/logbooks/044-l4-atlas-signs.md)) grounded 5.8% of synapses as inhibitory. A rung that replaces the random draw with measured signs and strengths was never on the ladder and is the highest-fidelity-per-effort item available (B.1). The Randi et al. 2023 signal-propagation atlas (*Nature* 623:406 — sign, strength and direction for 23,433 head pairs) is the raw measurement behind it. Caveats carried with the rung: preprint status (it never stands alone, as for the dopamine-forgetting preprint); head-only coverage, which grounds the sensory-to-command-interneuron path and the whole klinotaxis circuit but leaves the command-to-motor and motor layers random; a linear fit at calcium-imaging timescale whose units reach the tanh rate model only through a scale that is itself a pin; and Currier et al. 2025 on infrequent strong synapses.

- **An independent negative on this wiring as a reservoir, two weeks old.** Churchland, de Palma Aristides, Garcia-Ojalvo, Ritz, Anderson & Soriano (arXiv:2609.07355, 2026-09-07): *C. elegans* as an echo-state reservoir does **not** outperform shuffle controls and shows a performance–robustness trade-off — higher baseline performance with greater sensitivity to spectral radius, input scaling, leak and bias. That is [Logbook 068](experiments/logbooks/068-l1b-rate-calibration.md)'s finding — whether the wiring shows depends on the learner's operating point — reached on generic tasks by another group. Consequence: the calibration rung (A.2) is run and reported as a **sensitivity surface**, not a new pin, and it is publication-critical rather than housekeeping, because any reservoir-style claim from this project will be read against that paper. Guragain, Kakalis & Godino-Llorente (arXiv:2606.09902, June 2026) point the other way on weights — biological weight values beat random initialisation on the same topology in connectome reservoirs, weak evidence but the direction B.1 tests.

- **A candidate mechanism for block V, with a training-free test.** Therianos (arXiv:2606.17745, June 2026), a frozen rate operator with synapse counts as weights on the complete larval fly connectome: degree and weight statistics govern the gross dynamical signature, while *exact* wiring governs **input routing** (activity confined to a fifth of the core against two thirds under degree-and-weight-matched rewiring) and which neurons drive the dominant modes. V.2 scored four graph properties and none predicted learning time; it did not try routing confinement or mode drivers. A.3 registers them as predictors on the rewirings block V already generated.

- **Dhiman 2026, re-read for its method.** The degree-preserving null is built by directed double-edge swaps preserving in- and out-degree with self-loops held fixed; the shared initialisation is "a shared random seed so that parameter initialization is aligned across graph types as closely as possible" — not further specified; three optimisation seeds, five rewirings. The critique is right in kind and thin in evidence. A.1 answers it with two *stated* definitions of shared initialisation at n ≥ 16 paired seeds — the stronger form of the same control.

- **A validation target for the dynamics rung.** Morrison & Young (*PLOS Comput. Biol.*, Dec 2025): fifteen premotor neurons with gap-junction weights from the connectome and synaptic weights regressed on calcium imaging reproduce forward/reverse switching and dwell-time statistics — "gap junctions synchronise, synaptic dynamics switch". Registered for B.2 as a *behavioural* target (forward and reverse bout durations), which requires reversals to exist in the environment (C.0).

- **Constraints.** The fly "connectome through a body" space is crowded (Eon; Jin et al.), and for the worm Chung & Kim (bioRxiv 2025.07.21.665845; *Sci. Rep.* 2026) already drive a 2D rod-chain body from connectome weights optimised proportionally to synapse counts, while Kim, Florman, Santos, Alkema & Shlizerman (arXiv:2504.18073) couple connectome, dynamics, muscle, force and proprioception — **no worm work combines a body with closed-loop learning and a wiring control**, which is this phase's first-in-field cell. Wang-Chen & Ramdya's review of neuromechanical models (arXiv:2601.08056, 2026) names actuator gains left as free parameters as the field's standing confound (D18). Al-Asmar & Pérez-Escudero (*Proc. R. Soc. B* 293:20251924, Feb 2026) review foraging from the wild (rotting fruit and compost, 3D, patchy, boom-and-bust) to the lab (2D agar) and find the two bodies of work *not yet connected even experimentally* — so a computational natural habitat has nothing to validate against, and the environment fidelity that pays is what lab assays measure: patchy lawns with edges and depletion, food quality, and the roaming/dwelling states serotonin and PDF gate (D.1). Witvliet et al. 2021's 40–50% between-individual connection variability is why a wild-type-versus-wild-type control is a MAY: the repository already vendors Witvliet dataset 8 beside Cook 2019.

- *(Added 2026-09-20 at tracker authoring, from the re-aimed literature watch's first sweep.)* **Lee, bioRxiv 2026.09.06.749731 (2026-09-09)** — an atlas-fitted c302 in a 2D rod body with an external learned controller and wiring shuffles; see § Worm body and whole-organism models. It **narrows the C.1/C.2 novelty row** (the body + learning + wiring-control cell is not empty), **registers a risk against B.1** (atlas grounding gave no functional sensory-to-command step), and **corroborates B.2's gap-junction focus**; each edit is made at its site, and the paper is cited as convergent, not as support.

**Repository facts the review established** (each closes a "not checked" note elsewhere in this document): the vendored Cook 2019 workbook holds the `male chemical` and `male gap jn` sheets **and** the body-wall muscle columns (`dBWML*`, `vm*`) that `connectome/loader.py` currently drops — so the motor-neuron-to-muscle matrix a muscle readout needs is already on disk; the continuous environment has **no reversal** (forward speed is clamped to `[0, max_step_mm]`) and **no proprioceptive channel**, so today the VA/DA and VB/DB motor classes pooled into the readout carry no distinct meaning and escape is turning without reversing; and the per-step cap is **one body length**, which at the validated 0.2 mm/s crawl makes a full-speed step at least 5 s of worm time against a ~1.6 s undulation period — no model-step-to-biological-time calibration exists, and every body estimate depends on one (C.0).

**Aspirational timeline**: **≈ 19–27 active-work weeks** against Phase 7's planned 7–11, at the observed 2–3× calendar multiplier — block A ≈ 3–4, block B ≈ 6–8, block C ≈ 8–12, block D ≈ 2–3 (coupled with B.3). That size is why the phase is **pre-structured as two shipments** (D20) rather than holding a split in reserve: **8a = A + B.1 + B.2** (a citable package on its own — the init control, the robustness surface, measured weights, dynamics) and **8b = C + D + B.3** (the embodied package). Split by success, as 6a/6b and 7a/7b were.

**How to orient**: the living sub-task checklist lives in [openspec/changes/phase8-tracking/tasks.md](../openspec/changes/phase8-tracking/tasks.md) (authored 2026-09-20; the house pattern: `phase5-tracking`, `phase6-tracking`, `phase7-tracking`), with tracker-level decisions, the inherited execution-protocol standards and the open per-milestone questions in that change's design.md. This roadmap section remains the authoritative plan; D-decision amendments are dated in both places. The [phase protocol](research/phase-protocol.md)'s thirteen principles apply to every rung, and A.4 folds Phase 7's single-use rules into it before the first Phase 8 registration. Per that protocol's thirteenth principle, the [literature watch](research/literature-watch/context.md) was re-aimed at this scope on 2026-09-20 — cross-species demoted, measured weights and embodiment promoted, and seeds added for both.

#### Phase 8 shipment tracker

| Block | Scope | Shipment | Class | Status |
|---|---|---|---|---|
| **A** | Close Phase 7's exposed result: A.1 init-vs-rewiring control (D15); A.2 calibration-and-robustness surface (D16); A.3 frozen-operator structural predictors; A.4 methodology consolidation; A.5 the publication decision | 8a | A.1/A.2/A.4 MUST; A.3/A.5 SHOULD | ✅ A.1, A.2, A.4, A.6 met; A.3 and A.5 deferred to the A.5 step (2026-09-26, [076](experiments/logbooks/076-8a-synthesis.md)) |
| **B.1** | Measured synaptic signs and strengths on the Cook edges — vendored data sub-deliverable with provenance and licence; the wiring × weight-prior 2×3 (D17) under the reading learner and PPO | 8a | MUST | ✅ met, without a positive (2026-09-26, [076](experiments/logbooks/076-8a-synthesis.md)) |
| **B.2** | Dynamics rung — across-step leaky-integrator state with intrinsic time constants; gap junctions as ohmic coupling in that dynamics; a PPO arm with plastic gap junctions (D4's destination); bout-duration validation target | 8a | SHOULD | ⏸️ deferred-with-destination to 8b (2026-09-26, [076](experiments/logbooks/076-8a-synthesis.md)) |
| **8a synthesis** | Every 8a criterion assigned one of the five statuses; the 8b gate (D20) | 8a | MUST | ✅ [076](experiments/logbooks/076-8a-synthesis.md): gate **GO** |
| **C.0** | Body prerequisites — step–time calibration; signed speed (reversal); a proprioceptive channel; **D19 decided and recorded** | 8b | MUST | ⬜ |
| **C.1** | Anatomical motor-to-muscle readout (Cook 2019 NMJ matrix, learnable gains only) into a curvature-kinematic body; MLP positive control; the wiring contrast re-run through it with its own baselines | 8b | MUST | ⬜ |
| **C.2** | 2D rod-chain body (ElegansBot-class, 8–12 rods) under a registered cost budget and D18; its optional second half — the connectome's motor circuit generating the wave — behind B.2 | 8b | SHOULD (second half MAY) | ⬜ |
| **C.3** | Body-level validation — eigenworm posture spectrum, undulation frequency and amplitude, omega-turn geometry, Logbook 035/036 curves from *emergent* kinematics | 8b | SHOULD | ⬜ |
| **C.4** | The six-family architecture ranking re-run through the frozen body substrate, with new baselines | 8b | SHOULD | ⬜ |
| **C.5** | Renderer: segmented body from curvature or rod state in `pixel_continuous`; headless unchanged | 8b | SHOULD | ⬜ |
| **B.3 + D.1** | Internal metabolic state and the modulator field, with patchy bacterial lawns (edges, depletion, quality) and a roaming/dwelling readout | 8b | SHOULD | ⬜ |
| **8b synthesis** | Phase 8 close: every criterion assigned a status | 8b | MUST | ⬜ |

#### Pre-registered design decisions (2026-09-20 pre-start review)

Status: **ratified 2026-09-20**. Each remains cheap to reverse before implementation starts and expensive after; any amendment goes through the Phase 8 tracking change with a dated note.

| # | Decision | Resolution |
|---|---|---|
| **D15** | **What "the same initialisation" means once the mask changes** — `rewire_seed` is unset in every block-V panel, so each seed's rewired graph, task draw and weight draw move together. *(Corrected 2026-09-20 against the implementation, at PR review:* the rewiring is a directed double-edge swap, so **every neuron keeps its own labelled in- and out-degree** — `rewire_degree_preserving` records that per-post fan-in, and with it the `1/sqrt(chemical in-degree)` init **scale**, is preserved by construction. The account this decision inherited, from [Logbook 069](experiments/logbooks/069-phase7-synthesis.md) and § What Phase 8 opens on, had rewiring preserving the degree *sequence* but not which neuron holds which degree; that is not what this implementation does, and the ambiguity is correspondingly smaller. What is **not** matched across the two graphs is which drawn values land on which edges.*)* Dhiman 2026's control is a shared seed, unspecified further. | **Two definitions, both run as arms.** (i) *Dense-draw-then-mask*: one dense 302×302 draw per seed, masked by each graph, then scaled by `1/sqrt(in-degree)` — which is the same factor for a given neuron in both graphs — so every edge present in both carries the identical value. (ii) *Per-neuron fan-in sharing*: since each neuron keeps its own in-degree, its fan-in values are drawn once per seed against its own identity and assigned to that neuron's incoming edges in pre-synaptic-index order, so **every neuron receives the identical multiset of incoming weights in both graphs** and only the pairing of value to pre-synaptic partner differs. n ≥ 16 paired seeds on both block-V cells, `rewire_seed` fixed per pair. Registered outcome: the +35.4/+23.5/+55.3/+40.1 learning-speed effects survive shared initialisation (a wiring result) or dissolve (an initialisation result); both are citable, and either is a stronger control than the critique's own. |
| **D16** | **Operating-point discipline** — [Logbook 068](experiments/logbooks/068-l1b-rate-calibration.md) showed one inherited pin setting the *sign* of a registered primary, and Churchland et al. 2026 report the same sensitivity externally. | **No Phase 8 contrast runs at a pin unswept for the learner it uses.** A.2 sweeps `plasticity_rate`, readout width, `forward_pass_depth`, `initial_log_std` and `trace_decay` on the reading learner — one-factor-at-a-time around the current point first, a full crossing only for pins that move the wiring effect's sign — and reports the wiring effect as a sensitivity surface. Every later rung runs at a swept point and cites it. *(**Amended 2026-09-20**, at this decision's own tracking-change review: B.1 and B.2 both register PPO arms "at A.2's swept point", which the original wording blocked because A.2 covered only the reading learner. **A.2 gains a second half** — PPO across the three pins both learners share (readout width, `forward_pass_depth`, `initial_log_std`; the other two belong to the plasticity rule) at a reduced grid. The obligation is per learner and per pin that learner has. **A.1 is exempt**: it re-runs the committed block-V point by design, carries its pins as a standing condition, and is re-read if A.2's PPO half moves the sign. Logbook 069 lists `forward_pass_depth` as never varied and `initial_log_std` as swept only by hand under the rule, so this closes a gap under the phase's most exposed claim rather than merely tidying the rule.)* |
| **D17** | **The measured-weight rung's design** — a measured prior exists only for 156 head neurons *(125 in the released table, 154 in the fitted model — corrected 2026-09-23 at B.1a)*; a rewired null has no measured weights of its own; and the value distribution must be separable from its placement. | **A 2×3, wiring × weight prior, on the reading learner and again under PPO**: {wild type, rewired null} × {random draw, measured, *measured-shuffled*}. The measured-shuffled arm permutes the measured values among the wild type's own edges and is what makes a positive interpretable. The null's measured arm inherits measured values through D15(ii)'s per-neuron, pre-synaptic-index assignment, stated rather than inherited from whichever order the implementation makes easy. Pilot-scale arms: sign-only versus sign-plus-magnitude, and a sweep of the scale mapping LDS units onto the rate model (a pin). Coverage reported at head scope and full scope separately (command-to-motor and motor layers stay random). **Under PPO a measured prior is an initialisation, so that arm runs under D15's protocol** or it re-creates the confound A.1 exists to remove — hence B.1 runs after A.1 and A.2. Data lands as a vendored sub-deliverable with provenance and a licence check, the house standard. *(**Pilot read 2026-09-24**, [Logbook 072](experiments/logbooks/072-measured-prior-pilot.md): the scale pin is **1.0** on both learners, chosen on each learner's own gate. D15's protocol for the PPO arm is the per-neuron fan-in draw, which needed a definition to share each neuron's multiset once some edges are measured; B.1b added it. The PPO arm therefore sits at a draw A.2 did not sweep, and states that beside its claim.)* *(**2×3 read 2026-09-25**, [Logbook 073](experiments/logbooks/073-measured-prior-contrast.md): PPO `null_placement_unresolved`, reading `unresolved`; the placement-shuffled control is now a protocol requirement for any measured-weight positive.)* |
| **D18** | **Muscle gain is not a learner** — Wang-Chen & Ramdya name actuator gains left as free parameters as the neuromechanical field's standing confound; a gain tuned per arm would make the wiring contrast a contrast of gains. | **Calibrated once on the MLP positive control and frozen across arms**, with a registered sensitivity check (principle 7), never a per-arm free parameter. The learnable part of the muscle readout is the gain *vector* over muscle groups, identical in size across arms, so the trainable-parameter count cannot differ by wiring (the L.1 lesson). |
| **D19** | **Who generates the rhythm** — the current step is ≥ 5 s of worm time against a ~1.6 s undulation; a brain stepped at that rate cannot produce an undulation, a memoryless brain cannot produce one at any step size without a time-varying input, and the connectome brain has no across-step state until B.2. Bringing the step to ~0.2 s so the brain *could* produce it multiplies steps per unit worm time by ~25 on a substrate whose panels already need 16-way parallelism to fit in a day. | **Default: the body carries the rhythm.** A proprioceptive wave generator at body level (a local propagation rule on the biology that wave propagation is proprioceptive — Wen et al. 2012); the brain sets segmental muscle *drive* through the neuromuscular map, and thereby speed, direction and turning, at roughly today's step. **The alternative — the connectome's motor circuit produces the wave itself — is C.2's optional second half**, not C.1's question: it needs B.2 (across-step dynamics), the proprioceptive channel as sensory input, and the shorter step, and it is registered against the rewired null like every other rung. Decided and recorded before C.1 registers. *(**Noted 2026-09-25**, from the Wormlight review, a sister project building a connectome-driven body: **Wen et al. 2012 describe how the wave propagates, not what generates it**, so the default's generator needs a named source. The documented generators are a proprioceptive relaxation switch in the head (Ji et al. 2021, *eLife*, fitted to phase-response data; placed in SMDD by Yeon et al. 2018) and intrinsic oscillators in the ventral cord — B-type motor neurons for forward (Fouad et al. 2018; Xu et al. 2018) and A-type for backward, which run without premotor drive (Gao et al. 2018). The natural default is Ji et al.'s head switch with Wen et al.'s front-to-back propagation. **The alternative, as worded, has a known negative:** a graded network with fixed thresholds settles to a stable fixed point under constant drive (Kunert-Graf et al. 2017), and no precedent obtains a rhythm from proprioception alone. C.2's second half is therefore reframed below.)* |
| **D20** | **Shipment shape** — at 19–27 active weeks the phase is two to three times Phase 7's planned size, and its second half depends on decisions (C.0, D19) whose cost is unknown until 8a's substrate is fixed. | **Pre-structured as 8a / 8b, split by success.** 8a = A + B.1 + B.2 and its synthesis; 8b = C + D + B.3 and the phase synthesis. **8b does not start until 8a's synthesis has assigned every 8a criterion a status** from the five-word vocabulary. Phase 8 is marked COMPLETE only when the 8b synthesis lands; "8a complete / 8b pending" is the honest intermediate state, as 6a/6b and 7a/7b were. |
| **D21** | **8b's primary null** *(added 2026-09-26 at the 8a synthesis, [Logbook 076](experiments/logbooks/076-8a-synthesis.md))* — the degree-preserving null behind every 8a result also moves gap-junction strength with its edges and loses the wild type's autapses, and Logbook 075 found its rewired gap junctions carry about half of block V's `auc_success` lead. | **Every 8b wiring contrast reads against the chemical-only null** (`rewired_chemical_only`: the chemical graph rewired by the same degree-preserving swap, gap junctions and autapses held at the wild type's), **with the current null reported beside it** for continuity with every 8a result. That covers C.1e, C.4 and B.2's arms carried into 8b. It is Wormlight's primary null, so the two projects' wiring tests compare. The body is a new reference frame, so no committed result is re-read, and the choice is made before 8b's first registration, so it cannot have been chosen for a result. |

#### Required deliverables (MUST)

**Shipment 8a.**

1. **A.1 — the init-vs-rewiring control** (D15). The control Phase 7 named as its first act ([Logbook 069](experiments/logbooks/069-phase7-synthesis.md)): both definitions of shared initialisation, n ≥ 16 paired seeds, on the thermal and hard food-only block-V cells, through the committed block-V harnesses. The learning-speed result is restated with its status in the same sentence as the claim, whichever way it reads. *(2026-09-25: A.1 shared the chemical weights between wild type and null; the null still differs in gap-junction strength and in autapses, which **A.6** controls. Until A.6 reads out, block V's learning-speed advantage carries that as a standing condition.)* *(**Resolved into a quantified condition 2026-09-26**, [Logbook 074](experiments/logbooks/074-null-strength-control.md): against a null that also holds the wild type's gap junctions and autapses, block V's PPO advantage on hard350 is **+0.022 `auc_success` [+0.007, +0.036]** (against +0.049 versus the current null) and **+94 episodes [−81, +280]** (against +372) — a chemical-wiring advantage remains on `auc_success`, and the learning-speed form is unresolved at 32 seeds.)* *(**Sharpened 2026-09-26**, [Logbook 075](experiments/logbooks/075-gap-only-split.md): the move is the null's rewired gap junctions — against a null with the wild type's gap junctions (placement and strength), block V's PPO advantage on hard350 is **+0.025 `auc_success` [+0.012, +0.038]** and **+235 episodes to competence [+62, +409]**: about half of its `auc_success` lead over the degree-preserving null (+0.049) came from that null's rewired gap junctions, and a chemical-wiring advantage in learning speed remains.)*

2. **A.2 — the calibration-and-robustness surface** (D16 as amended). Two halves: the reading learner (`readout_only`, the first plausible learner shown to reach competence on this substrate, [Logbook 063](experiments/logbooks/063-l4-eprop.md)) across all five pins, and **PPO across the three shared pins at a reduced grid**, since B.1 and B.2 both run PPO arms and a pin swept on one learner is not swept for another. Reported as the wiring effect's sensitivity across the operating region, per learner; fixes the point every later rung cites.

3. **A.4 — methodology consolidation.** *(Done 2026-09-20, `consolidate-plasticity-methodology`.)* `plasticity-evaluation` held **44** requirements and was two specs stapled together: six describing the forgetting harness its own Purpose statement names, and thirty-eight methodology rules that eighteen Phase 7 milestone changes registered there because it was the nearest plasticity-named capability. The split follows that Purpose statement rather than a judgement call. **Seven stay** (the six, plus the delayed-reward control, which describes live code and was misfiled); **eleven move** to `architecture-comparison-protocol`, the durable home Phase 7 had already used for its wiring rules, because each still binds Phase 8 work; **eighteen fold** into the [phase protocol](research/phase-protocol.md) as clauses under existing principles, with no new principle and no renumbering; **eight retire** with a reason and a migration each, being specific to the rule programme that closed with a diagnosed cause. The reading surface is now **thirteen principles plus the eleven requirements that stayed enforceable** in `architecture-comparison-protocol`, rather than thirty-eight rules in a capability whose Purpose never described them.

4. **B.1 — measured synaptic signs and strengths** (D17). The Creamer–Leifer–Pillow fitted weights, with the Randi 2023 atlas as the raw source, vendored under `data/connectome/` with provenance and licence, the way the transmitter atlas landed; then the 2×3. *(**Corrected 2026-09-23** at B.1a's licence check: the Randi 2023 atlas is **cited, not vendored** — its OSF deposit states no licence and its only licensed copy is a GPL-3.0 file, in an Apache-2.0 repository. The Creamer–Leifer–Pillow table is vendored under MIT. It covers **125 neurons** (the fitted model holds 154; 156 is the paper's recording figure), and on Cook 2019 it reaches 1,049 of 3,709 chemical edges — 77.0% of the 1,363 coverable at head scope, none onto the body motor neurons.)* Payoff either way: a positive is the first "the animal's weights make its wiring legible" result; a null extends [Logbook 034](experiments/logbooks/034-connectome-structure-controls.md)'s degree-statistics verdict to measured weights, and both are performance claims under the Phase 7 claim discipline. **Risk registered 2026-09-20** (Lee, bioRxiv 2026.09.06.749731 — § Worm body and whole-organism models): fitting six *global* conductance scales of c302 to the Randi atlas produced **no functional sensory-to-command step** — direct stimulation of eight sensory neurons stayed at chance — with most inhibitory atlas responses falling between unconnected pairs named as the likely cause. Per-edge fitted weights are a different and richer grounding, but the failure mode is now named: measured weights may leave the klinotaxis pathway unlearnable, and B.1's sign-only pilot and the reading learner's positive control are what would show it, closing the rung *unmet-with-reason* with the pathway named rather than as a null on the wiring. *(**The sign-only pilot did not show it, 2026-09-24**, [Logbook 072](experiments/logbooks/072-measured-prior-pilot.md): on hard350 the pathway was learnable under both learners at every scale tried, and as signs alone. B.1c keeps the risk registered, since its gate is the finer one.)*

5. **A.6 — the null-strength control** *(added 2026-09-25, from the Wormlight review; **read 2026-09-26**, [Logbook 074](experiments/logbooks/074-null-strength-control.md): `below_minimum` on both learners, each a significant move toward the null of about half the effect — under PPO against a null that also holds the wild type's gap junctions and autapses, block V's PPO advantage on hard350 is **+0.022 `auc_success` [+0.007, +0.036]** (against +0.049 versus the current null) and **+94 episodes [−81, +280]** (against +372) — a chemical-wiring advantage remains on `auc_success`, and the learning-speed form is unresolved at 32 seeds; the gap-only split is deferred to the 8a synthesis; **split run 2026-09-26**, [Logbook 075](experiments/logbooks/075-gap-only-split.md): `gap_junctions` on both learners — holding the gap junctions alone reproduces about 87% of A.6's move under PPO and all of it under the reading learner, and against a null with the wild type's gap junctions (placement and strength), block V's PPO advantage on hard350 is **+0.025 `auc_success` [+0.012, +0.038]** and **+235 episodes to competence [+62, +409]**: about half of its `auc_success` lead over the degree-preserving null (+0.049) came from that null's rewired gap junctions, and a chemical-wiring advantage in learning speed remains)*. Every wiring contrast in this project — [Logbook 034](experiments/logbooks/034-connectome-structure-controls.md), block V, and [070](experiments/logbooks/070-init-sharing-control.md)–[073](experiments/logbooks/073-measured-prior-contrast.md) — compared the wild type against a degree-preserving null that differs from it in **two things besides which neurons connect**, and A.1 controlled neither:

   - **Gap-junction strength.** The brain uses each gap junction's EM count as its coupling weight, normalised only by degree, and the undirected swap carries counts with the edges. The wild type concentrates gap coupling in a few hubs: ALA's gap input is **232**, against at most 6.3 of chemical input on any neuron. On a null ALA's falls to about 3, and about half of all neurons' gap totals move by more than 50%.
   - **Autapses.** The wild type has 38. The swap never creates a self-loop but can remove an existing one, so rewiring loses autapses and never regains them; each of the three nulls checked had none.

   [Logbook 067](experiments/logbooks/067-l4-feature-ablations.md) already found that under the reading learner the wild type's advantage consisted in its gap junctions costing it less than the rewired ones cost the null, without tracing why; this is a candidate mechanism. **The control:** a null that rewires the chemical graph only — a **combined** control, holding the wild type's gap-junction placement, gap-junction strength and autapses together — under PPO on hard350 at A.1's size, against the current null; the reading learner optionally. If block V survives, the advantage is in the chemical wiring **given the wild type's gap junctions and autapses held in place**, and the phase's claim is stronger; if it dissolves, it came at least partly from how the current null rewires gap junctions (placement and strength together) or drops autapses — the combined control cannot say which, and a split follows only if it moves. Runs before the 8a synthesis and before A.5, since both would cite block V without it.

6. **The 8a synthesis**, with every 8a criterion assigned one of the five statuses, and the D20 gate written as a go/no-go decision in that logbook.

**Shipment 8b.**

7. **C.0 — body prerequisites.** (a) *Step–time calibration*: what one environment step is in seconds, fixed from the validated crawl speed (0.2 mm/s), the undulation period (~1.6 s) and the arena scale already in the environment, recorded before any body cost estimate or kinematic target is registered. (b) *Signed speed*: reversal as a first-class action, so VA/DA and VB/DB mean different things, escape can be reversal-plus-turn, and bout statistics exist to validate against. (c) *A proprioceptive channel*: posture or stretch fed back as sensory input. (d) **D19 decided and recorded.** *(2026-09-25, from the Wormlight review: the target neurons for (c) have a literature answer — B-type motor neurons driven by the bending of the ~200 µm **in front** of their muscles (Wen et al. 2012, the direction measured, not Boyle, Berri & Cohen's posterior integration), SMDD in the head (Yeon et al. 2018, stretch-sensing through TRPC channels), and A-type motor neurons as a hypothesis (Gao et al. 2018). **Substrate at the freeze:** the vendored Cook 2019 workbook states no licence; Emmons 2024, *PLoS Biol* 22:e3002939, republishes the same matrices under CC BY 4.0 with the lab's corrections (two more gap-junction pairs, two strengthened), loadable since PR #404. 8a stays on 2019 so its results remain reproducible; since the body is a new reference frame anyway, C.0 should consider freezing 8b on Emmons 2024.)*
8. **C.1 — the anatomical motor-to-muscle readout into a kinematic body.** *(D21, 2026-09-26: C.1e's wiring contrast reads against the chemical-only null, the current null beside it.)* Replace the learned 2×39 motor readout with the Cook 2019 motor-neuron-to-body-wall-muscle matrix (on disk, currently dropped by the loader), pooled into four quadrants × N segments, with learnable gains only (D18). Muscle drive → segmental curvature; displacement per step from the change of posture between steps by resistive-force theory, no ODE. Speed and turning now *emerge from motor output* at roughly today's cost. **Positive control first**: MLP-PPO must forage through it before any connectome arm runs (principle 4). Then the wiring contrast, with its own floors and baselines on the new substrate — body-substrate results are a **new reference frame**, never a controlled delta against [Logbook 029](experiments/logbooks/029-continuous-architecture-ranking.md) or block V (the grid-versus-continuous lesson, 2026-06-14). *(2026-09-25: the neuromuscular parse landed in PR #404 — `load_emmons_2024_neuromuscular()` returns the 956 synapses onto the 95 body wall muscles, identical to the 2019 file's. Two facts the readout must carry: those synapses come from **162 cells, sensory neurons and interneurons among them**, so a readout restricted to motor neurons drops some and must say so; and muscle signs follow its receptors (Richmond & Jorgensen 1999) — acetylcholine excites, GABA inhibits, and 32 of the 162 cells release neither.)*
9. **The Phase 8 synthesis**, every criterion assigned a status.

#### Recommended deliverables (SHOULD)

- **A.3 — frozen-operator structural predictors** (from Therianos 2026): routing confinement and mode-driver metrics on Cook 2019 synapse-count weights versus every rewired null block V generated, registered as predictors of time-to-competence. No training; about a week. *(**2026-09-23**: A.2 handed this a concrete candidate. [Logbook 071](experiments/logbooks/071-operating-point-surface.md)'s hop probe found the wild type has **no** motor neuron one hop from a food sensor while a degree-preserving rewiring manufactures about nine, so at a settling budget of 2 the wild type reaches 26 of 39 motor neurons against the null's 39 — which tracks the depth surface exactly. It was computed after the fact and registers nothing, and [Logbook 069](experiments/logbooks/069-phase7-synthesis.md) recorded that no graph property measured there predicted learning time. This one predicts the **operating point at which the effect exists**, a different and testable claim, and A.3 is where a statistic, a direction and a minimum get named before the correlation.)*
- **A.5 — the publication decision**, taken after A.1 reads out (deferred from Phase 7's S.1 cancellation). The package is the replicated wiring advantage under gradient descent *with* its initialisation control, the rule-programme negative with a diagnosed cause, and the operating-point finding with its sensitivity surface. Not a gate for anything after it.
- **B.2 — the dynamics rung** (069 item 3 + L.2). *(**Carried to 8b 2026-09-26**, [Logbook 076](experiments/logbooks/076-8a-synthesis.md): its design absorbs Logbook 075's gap-junction finding, and B.2b's plastic gap junctions are the natural test of placement against strength; its wiring arms read against the chemical-only null per D21.)* Per-neuron leaky-integrator state across environment steps with intrinsic time constants; gap junctions as ohmic coupling inside that dynamics rather than a fixed symmetric matrix; a PPO arm with plastic gap junctions (D4's surviving destination, now with the *Nat. Commun.* 2020 precedent). Positive control per protocol: the dynamical substrate must first learn the cell under PPO at least as well as the settling substrate. Validation target, once C.0 supplies reversals: forward/reverse bout-duration statistics per Morrison & Young, pre-registered as a behavioural sign/shape-level claim. This rung is also the precondition for the connectome ever generating its own rhythm (C.2's second half). *External convergence, noted 2026-09-20:* Lee (bioRxiv 2026.09.06.749731), in an atlas-fitted c302 driving a 2D rod body, finds that shuffling the **gap-junction layer alone collapses chemotaxis** (8–13% source reach against 91% intact) while shuffling the chemical layer alone barely moves it (86%) — the same layer [Logbook 067](experiments/logbooks/067-l4-feature-ablations.md) found carrying this project's per-neuron effect. One trained policy per condition with Wilson intervals over evaluation episodes, so cited as convergent, not as support. *(2026-09-25, from the Wormlight review: **a stiffness warning for B.2a.** Gap-junction weights are raw EM counts today, harmless in a memoryless tanh map; once neurons carry time constants they make the system stiff — Wormlight measured ALA's effective membrane time constant near 0.05 ms under Cook's counts with Kunert et al.'s conductances. B.2a needs either an integrator that is stable there or a registered rescaling of the counts (Wormlight matches Cook's totals to Varshney's, by 0.2055 for gap junctions and 0.3444 for chemical synapses), and A.6's reading decides whether the counts should move with a rewiring at all.)*
- **C.2 — the rod-chain body.** An ElegansBot-class 2D chain (Chung, Chang & Kim, *eLife* 2024: 25 rods, anisotropic Stokes drag, torsional-spring muscles, crawling/swimming/omega/delta turns validated, ~1:1 real time on one core, Python/Numba, CC-BY) reduced to 8–12 rods in `Continuous2DEnvironment`, driven by C.1's muscle drive. Gated by the MLP positive control and by a **cost budget registered in advance**: a 16-seed panel must fit in roughly a day at 16 workers ([Logbook 039](experiments/logbooks/039-runtime-acceleration-audit.md)'s measured ceiling), or the rung stops at C.1. **Its optional second half** asks whether the connectome's wiring gates documented class-level oscillators correctly — AVB driving the B-types, AVA the A-types — against the rewired null; it needs B.2 and a shorter step. *(**Reframed 2026-09-25**: the half was first framed as the connectome's motor circuit plus proprioception generating the wave, which for a graded network has a known negative (D19's note). **That is Wormlight's design** — the connectome as a graded model driving a Boyle, Berri & Cohen body, with the rewired null given the same tuning procedure — so **whether C.2 narrows is decided at C.2a**, when its cost budget is registered, on how far Wormlight has progressed: in particular whether it has cleared its milestone-0c go/no-go (does the connectome-driven body crawl, its checkpoint 1) and the checkpoints after it. If it has, C.2's second half and much of C.3 are better served there and nematode's 8b keeps C.1, learning through a body, which is the part Wormlight does not do. **C.1 is unaffected either way.**)*
- **C.3 — body-level validation**: the eigenworm posture spectrum (Stephens et al. 2008), undulation frequency and amplitude, omega-turn geometry, and the [035](experiments/logbooks/035-realworm-chemotaxis-validation.md)/[036](experiments/logbooks/036-realworm-thermotaxis-validation.md) klinokinesis and weathervane curves re-derived from *emergent* kinematics. Swimming versus crawling gait under liquid drag is a free further target if C.2 ships. *(2026-09-25: Wormlight has already fixed crawling thresholds in advance — frequency, wavelength and speed from Fang-Yen et al. 2010 and Ramot et al. 2008, and a pinned eigenworm basis — so C.3 should adopt them where they apply, keeping the two projects' body results comparable.)*
- **C.4 — the architecture ranking through the body.** *(D21, 2026-09-26: its connectome arm's wiring contrast reads against the chemical-only null, the current null beside it.)* The North Star's deliverable: the six MUST families of [Logbook 029](experiments/logbooks/029-continuous-architecture-ranking.md) re-run on the frozen body substrate, with new baselines, per the § Architecture-Comparison Protocol rule that the sweep re-runs when the substrate plausibly changes the comparison.
- **C.5 — rendering.** The `pixel_continuous` renderer draws the segmented body from curvature or rod state (head, tail, reversals and omega turns visible) with an optional posture overlay; headless unchanged. Modest, and only once C.1 exists.
- **B.3 + D.1 — internal state, the modulator field, and patchy lawns.** A minimal metabolic state (satiety already crosses the brain boundary; the missing pieces are the internal-state sensory module and the concentration field, per the 2026-08-27 sizing), serotonin/PDF gating of roaming versus dwelling as its first behavioural consequence, and lawn geometry with edges, per-patch depletion (the `source_depletion_enabled` mechanism exists, config-gated), and food quality. Validation: Flavell-lab roaming/dwelling fractions and Al-Asmar & Pérez-Escudero's patch-leaving framing. **The 2D agar plate is kept** (D.2): no 3D, no soil, no fluid coupling — all validation data is 2D. **The same three behaviours** (klinotaxis, thermotaxis, predator evasion) remain the comparison set; roaming/dwelling enters as an observable, not a fourth ranked behaviour. *(2026-09-25, from the Wormlight review: **a per-connection sign source for the receptor layer.** This project's transmitter rule is per presynaptic neuron and makes glutamate excitatory, so it signs AWC → AIY positive; physiology says inhibitory, through glutamate-gated chloride channels (Chalasani et al. 2007). That is the food-sensing entry to the klinotaxis circuit, and it conditions the atlas-sign readings of [Logbook 044](experiments/logbooks/044-l4-atlas-signs.md) and [067](experiments/logbooks/067-l4-feature-ablations.md). Fenyves et al. 2020 (*PLoS Comput Biol*, CC BY) predict signs per connection from transmitter and receptor expression, clearly for 47.5% of Cook's chemical edges, and are the natural data source when B.3's receptor classes land. Wormlight's review also reports about 364 sign disagreements between Fenyves's predictions and the Creamer–Leifer–Pillow fitted weights on the 1,049 edges B.1 covered — not verified here, and to be checked before it is cited.)*

#### Optional deliverables (MAY)

- **Placed plasticity** on the klinotaxis circuit (AWC → AIY → AIZ → RIA/RIB → SMB/SMD), on the B.1 substrate, against a degree-stratified random subset of the same size, with the three confounds § Future Directions already specifies (same-site definition under rewiring; per-wiring stratification; identical trainable-synapse count). *(2026-09-25: "the B.1 substrate" needs restating — measured weights left the PPO wiring effect where it was ([Logbook 073](experiments/logbooks/073-measured-prior-contrast.md)), so whether M.1 runs on the measured prior at all is its own decision; and its circuit begins at AWC → AIY, the synapse the per-neuron sign rule gets wrong (B.3's note), so it needs per-connection signs on that path.)*
- **A wild-type-versus-wild-type control**: Cook 2019 against Witvliet dataset 8 (adult, nerve-ring scope, already vendored) — real between-individual variation in place of a shuffle.
- **C.2's second half**: whether the connectome's wiring gates documented class-level oscillators correctly (AVB driving the B-types, AVA the A-types), against the rewired null; needs B.2 and a shorter step. *(Reframed 2026-09-25 from "the motor circuit plus proprioception generating the wave"; see D19's note.)*
- **Swimming/crawling gait transition** as a body validation target.
- **Reproducibility artefacts current** to the Phase 8 platform state, with an artefact-retention rule registered at phase start (which per-campaign artefacts are committed — the parsed per-seed CSVs — and which are archived off-repo), so 8a's step-level exports are not lost the way [Logbook 069](experiments/logbooks/069-phase7-synthesis.md) records the pre-readout-width ones were.

#### Biological validation targets (Phase 8)

Every target below is a **behavioural** claim under the Phase 7 claim discipline unless stated; none needs named-neuron grounding to be measured, and none is offered as a dynamics claim.

- **Kinematics** (C.3, unlocked only by a body): crawling speed ~0.2 mm/s and undulation period ~1.6 s (Chung & Kim 2025/2026 as the modelled reference; Stephens et al. 2008 for the eigenworm spectrum); omega-turn geometry; the 035/036 klinokinesis and weathervane curves from emergent rather than kinematic motion.
- **Forward/reverse bout durations** (B.2, needs C.0's reversal): Morrison & Young 2025's dwell-time statistics, sign/shape-level.
- **Roaming/dwelling fractions on and off food** (B.3 + D.1): Flavell-lab serotonin/PDF phenomenology.
- **Wiring-effect robustness** (A.2): not a biological target but the phase's methodological one — the sensitivity surface Churchland et al. 2026 make a referee expectation.
- ⚠️ **Still the trap**: thermotaxis set-point plasticity is receptor-level and intrinsic to AFD; B.2's intrinsic dynamics are the *only* place it could be modelled, as node-level adaptation, and it is not pre-registered as a synaptic-rule target.

#### Phase 8 exit criteria

**Required (MUST) — 8a:**

- ✅ **A.1** run under both D15 definitions, n ≥ 16 paired seeds, both block-V cells; the learning-speed result restated with its status. *(Met 2026-09-21, [Logbook 070](experiments/logbooks/070-init-sharing-control.md); ticked here 2026-09-25.)*
- ✅ **A.2** sensitivity surface over the five pins on the reading learner **and the three shared pins on PPO**; every later contrast at a point swept for the learner it uses (D16 as amended). *(**Met 2026-09-23**, [Logbook 071](experiments/logbooks/071-operating-point-surface.md): both surfaces exist. PPO is depth-critical — abolished at depth 3, reversed at 2 — and robust to initial noise; the reading learner is flat except in readout width. **One named gap**: readout width under PPO saturates on hard350, so it is unresolved there and B.1's PPO arm runs at the pooled width or buys the reading on a non-saturating cell, per tracker B.1c.)*
- ✅ **A.4** methodology consolidation landed before the first 8a registration. *(Done 2026-09-20, `consolidate-plasticity-methodology`; ticked here 2026-09-25.)*
- ✅ **A.6** the null-strength control: block V read against a null that holds gap junctions and autapses at the wild type. *(Added 2026-09-25; **met 2026-09-26**, [Logbook 074](experiments/logbooks/074-null-strength-control.md): `below_minimum` on both learners, significant and toward the null; block V carries the quantified condition.)*
- ✅ **B.1** data vendored with provenance and licence; the wiring × weight-prior 2×3 under the reading learner and PPO, head scope and full scope reported separately. *(Progress 2026-09-24: data vendored at B.1a; the pilot (B.1b, [Logbook 072](experiments/logbooks/072-measured-prior-pilot.md)) swept the multiplier on both learners and selected **1.0**, the magnitude-matched point, by a rule that never read the wiring gap; the 2×3 is B.1c.)* *(**Met 2026-09-25, without a positive**, [Logbook 073](experiments/logbooks/073-measured-prior-contrast.md): under PPO, on the fan-in draw with the pooled readout and A.2's `edge_order`-swept depth and noise, the measured weights do not move the wiring effect by the registered minimum in either direction (`no_move`), and whether their placement matters is unresolved; under the reading learner the interaction is unresolved at the panel's sensitivity. Neither learner reaches `legible` or `hides`. Reported beside: the wild type's PPO advantage is present at every prior, so block V's effect does not depend on random weights on the covered edges.)*
- ✅ **8a synthesis** with every 8a criterion assigned one of the five statuses and the D20 gate decision written. *(**Met 2026-09-26**, [Logbook 076](experiments/logbooks/076-8a-synthesis.md): the gate is **GO**; D21 registered.)*

**Required (MUST) — 8b:**

- ⬜ **C.0** step–time calibration recorded; signed speed; proprioceptive channel; D19 decided and recorded before C.1 registers.
- ⬜ **C.1** anatomical NMJ readout into the kinematic body; MLP positive control passed; the wiring contrast re-run through it with its own baselines.
- ⬜ **Phase 8 synthesis** with every criterion assigned a status.

**Recommended (SHOULD):** *(8a statuses assigned 2026-09-26, [Logbook 076](experiments/logbooks/076-8a-synthesis.md): A.3 and A.5 **deferred-with-destination** to the A.5 step that follows the synthesis; B.2 **deferred-with-destination** to 8b, for the reason in the overshoot row; the 8b items below are assigned at S8b.)* A.3; A.5, taken after A.6; B.2 with the plastic-gap-junction arm and the bout-duration target; C.2 under its cost budget; C.3; C.4; C.5; B.3 + D.1.

**Optional (MAY):** placed plasticity; the wild-type-versus-wild-type control; C.2's second half; the gait transition; reproducibility artefacts with the retention rule; **the depth finding's two registered follow-ups** *(added 2026-09-23 after A.2, tracker M.6)* — the thermal confirmation at the depths where hard350's sign moved, which is branch 2 of [071's registration](experiments/logbooks/supporting/071-operating-point-surface/launch.md), and the full crossing D16 gives any pin that moves the sign. **The crossing's priority fell once the mechanism was measured**: 071's hop probe explains the depth surface on its own, so a depth-by-pin crossing is now less likely to hold the answer than A.3's registered predictor test. Both are recorded rather than dropped, because they were registered before the readout and dropping them afterwards is the move the protocol forbids.

#### Risk-mitigation: failure modes and pivots

| Failure mode | Trigger | Pivot |
|---|---|---|
| **Block V dissolves under shared initialisation** | A.1 reads no wiring effect under either D15 definition | This is a result, not a failure: the learning-speed claim is restated as an initialisation effect and the phase's citable package becomes the control itself plus the operating-point finding. B.1 still runs — measured weights are a different question from random-weight legibility — and the body rung is unaffected. |
| **The measured prior does not reach the rate model** | B.1's scale sweep finds no setting at which the measured-weight wild type learns at all, or the LDS units cannot be mapped defensibly | Fall back to **sign-only** grounding (the pilot arm), which needs no scale; report magnitude grounding as unreachable-with-reason. *(**Did not fire, 2026-09-24**, [Logbook 072](experiments/logbooks/072-measured-prior-pilot.md): every multiplier from 0.25 to 4 and the sign-only prior learned on both learners and both wirings at pilot scale.)* |
| **The body is too slow** | C.2's registered cost budget (a 16-seed panel in ~a day at 16 workers) is exceeded after the reduced chain, coarser integrator and Numba/JAX have been tried | **Stop at C.1** — the kinematic body already makes behaviour emerge from motor output — and record C.2 as deferred-with-destination behind the env-vectorisation decision (D6), which a body makes more valuable, not less. |
| **No brain locomotes through the body** | The MLP positive control fails to forage through C.1 after the proprioceptive channel and D19's body-level generator are in place | The rung stops and the diagnosis is the deliverable (the phase-protocol lesson: a new component on a validated platform needs its own control). Nothing about the wiring is claimed from a substrate no learner can drive. |
| **Substrate-vs-rung confound** | A platform change (C.0's reversal, proprioception, or the step change) lands mid-comparison | Land every C.0 change **before** C.1 registers, validate, freeze; report anything spanning a substrate change as qualitative. The same rule that D5/D7 enforced in Phase 7. |
| **8a overshoots** | Block A and B.1 exceed ~12 active weeks | D20 already splits the phase; 8a ships on A.1 + A.2 + B.1 + A.6 alone, with B.2 carried to 8b as SHOULD. *(**2026-09-26**: 8a did **not** overshoot — it took about a week against 9–12 active weeks — and B.2 is carried to 8b for a different reason, recorded in [Logbook 076](experiments/logbooks/076-8a-synthesis.md): A.6 and Logbook 075 made gap junctions central to the wiring effect, and B.2 is where they stop being a fixed matrix, so its design should absorb that; B.2c and C.2's second half need 8b's body anyway.)* *(A.6 added 2026-09-25: 8a does not ship without its null-strength readout.)* No month-count trigger — the split fires on the 8a synthesis. |

#### Where Phase 8 is first-in-field (novelty map)

Verified against the 2026-09-20 literature scan; nearest precedents are cited as convergent, per § Claim discipline.

| Phase 8 claim | Nearest precedent (2026-09) | Payoff if positive |
|---|---|---|
| **A learner on the real *C. elegans* wiring with measured synaptic weights, against a degree-preserving null** (B.1) | Creamer, Leifer & Pillow fit the weights (preprint, no task, no learning); Guragain et al. 2026 show biological weights beat random on connectome reservoirs (generic tasks, weak controls); Lee 2026 (bioRxiv, 2026-09-09) fits six *global* conductance scales of c302 to the Randi atlas and tests behaviourally with shuffles — atlas-grounded, but not per-edge weights, not learning on the wiring, one trained policy per condition. **Per-edge measured weights + closed-loop learning on the wiring + a degree-preserving null under an initialisation control is empty.** *(Narrowed 2026-09-20.)* | "The animal's weights make its wiring legible" — the first structure-function result on this substrate that does not depend on a random draw; a null hardens the degree-statistics verdict to measured weights. *(**Read 2026-09-25**, [Logbook 073](experiments/logbooks/073-measured-prior-contrast.md): neither claim was reached. Under PPO the measured weights leave the wiring effect where it was, the wild type still ahead at every prior, with placement unresolved; under the reading learner the result is unresolved. The empty cell is filled with a measured, controlled non-positive rather than the legibility result.)* |
| **The wiring effect's operating-point sensitivity surface** (A.2) | Churchland et al. 2026 report the sensitivity on generic reservoir tasks; no closed-loop behavioural version exists. | The result a referee now expects; and the first statement of *where* in operating space a connectome shows and where it does not. |
| **Closed-loop learning through the anatomical motor-to-muscle map into a body, with a wiring control** (C.1/C.2) | Chung & Kim 2025/2026 (connectome weights → ElegansBot, no learning, no null); Kim–Shlizerman 2025 (integrated, no learning); BAAIWorm (no learning); Eon and Jin et al. (fly); **and Lee 2026 (bioRxiv 2026.09.06.749731, 2026-09-09): an atlas-fitted c302 driving a 24-segment 2D rod body through a Boyle–Berri–Cohen motor layer, an evolution strategy over an external 16-unit policy injecting current into 35 interneurons with the connectome fixed, and wiring shuffles — a body, learning and a wiring control together, so the cell is *not* empty.** *(Narrowed 2026-09-20 at tracker authoring; the earlier claim that it was empty for any organism is withdrawn.)* What remains unoccupied: learning **on** the wiring rather than current injection into a fixed one; the anatomical NMJ map rather than a hand-built motor layer; a null that guarantees each neuron's in- *and* out-degree (Lee's permutes the postsynaptic column and claims out-degree only; whether per-neuron in-degree survives is not stated); an initialisation control; and paired seeds against one trained policy. | Behaviour emerging from motor output rather than a two-number readout, with the wiring's contribution measured against its null — the phase's headline if it lands. |
| **The init-vs-rewiring control with a stated definition at n ≥ 16** (A.1) | Dhiman 2026 (three seeds, five rewirings, shared seed unspecified). | Either a wiring result that survives the one published critique aimed at this design, or a clean retraction that makes the rest of the package trustworthy. |

#### Go/No-Go Decision

- **GO (8a shipment) if**: A.4 has landed, A.1, A.2, B.1 and **A.6** resolve with statuses assigned — every 8a criterion carrying one in the 8a synthesis — and that synthesis writes the D20 gate as GO, meaning the substrate 8b will embody has a known initialisation story, a known operating point, a measured-weight verdict, and **a null-strength readout**. *(A.6 added 2026-09-25.)*
- **SPLIT-shipment if**: 8a forms a self-contained citable result before block C starts, **or** block A + B.1 overshoot — either way the pre-structured 8a / 8b shape applies. This is the default expectation (D20), not a contingency.
- **PIVOT-scope if**: the body cost budget fails (stop at C.1) or no learner drives the body (the diagnosis is the deliverable) — execute the relevant risk row and record the pivot in the tracking change.
- **STOP if**: A.1 dissolves block V **and** B.1 is null at every scale **and** C.1's positive control fails — at which point the phase's deliverable is the statement that on this substrate, under every instrument tried, the wild-type wiring is not distinguishable from its degree statistics, with the controls that make that statement citable. Not expected, and not a reason to withhold any of the three results individually.

______________________________________________________________________

## Architecture-Comparison Protocol

Phase 2's 300-session quantum architecture campaign — covering 15 quantum and hybrid variants against matched-capacity classical baselines — established that grid-world complexity is below the threshold for quantum advantage on every variant tested. Quantum is therefore *one architecture family among many* in the project's comparison sweep, not a separate goal or a separate phase.

The architecture-comparison protocol consolidates this into a single mechanism, applied at Phase 6 and again at Phase 7:

1. Run the full architecture-family sweep (MLP, recurrent, spiking, reservoir, quantum, hybrid, connectome-constrained, NEAT-evolved) on the three target behaviours under the current substrate (Rung 2 chemical gradients, continuous 2D, corrected nociception at Phase 6; same plus L4 plasticity at Phase 7).
2. Report results paired-seed at the Phase 5 statistical bar (Wilcoxon, bootstrap CIs, n ≥ 4 per condition).
3. The 300-session campaign's results carry forward as **baseline reference data** — quantum families are not re-evaluated from scratch unless the substrate change (continuous physics, connectome topology, neuromodulator-modulated STDP) plausibly changes the comparison.
4. If a future phase introduces a complexity dimension that *did* clear a quantum-advantage threshold in the Phase 2 campaign (e.g., long non-Markovian dependencies for QRH), revisit that family's evaluation at that phase. This is opportunistic, not scheduled.

There is no separate "quantum checkpoint" gate, and no "if classical drops below 70% then launch a quantum campaign v2." The optionality of revisiting quantum at higher complexity is preserved through the architecture-family sweep itself.

**T4 grid ranking complete (2026-06-03, [Logbook 025](experiments/logbooks/025-weight-search-architecture-ranking.md)).** The first L2 pass — 7 architecture families on the integrated grid-world C3 cell (food + predator + thermotaxis, n=8 paired seeds, post-convergence full-clear) — is done. A four-way top cluster (equivariant-quantum 86.0, CfC 84.4, spiking 84.2, LSTM 83.6) is statistically tied; the wild-type connectome ranks **mid-pack** (75.6 — competitive on foraging, behind on predator evasion); GA collapses (0.0). A genuinely-quantum (simulated) bilateral-symmetry-equivariant-circuit arm is the numerical leader, but controlled attribution — a matched-capacity fair classical-equivariant control plus a matched-capacity symmetry control — shows **no quantum-circuit advantage and no significant symmetry effect**, a controlled on-task confirmation of the Phase 2 / Logbook 008 baseline (the naive "+24.6 quantum-beats-classical" delta was a weak-baseline artifact). This grid ranking is the reference for the T7 continuous-substrate re-run's **qualitative cross-regime comparison** (the substrates are non-commensurable, so it is not a controlled delta — reframed 2026-06-14; the clean result is the within-T7 ranking).

**T7 continuous ranking complete (2026-06-24, [Logbook 029](experiments/logbooks/029-continuous-architecture-ranking.md)) — the headline architecture-comparison result.** The L2 re-run on the fully-upgraded continuous substrate (continuous-2D + continuous-action + static Fick gradients + adaptive sensor + corrected nociception), six MUST families, n=8, uniform budget: **`MLP 89.0 ≫ {CfC 75.8 ~ Transformer 74.0} > LSTM 60.1 > connectome 52.2 ≫ GA 15.0`** — three statistically significant tiers (BH-FDR). Three load-bearing findings: **(1)** the simplest architecture (plain MLP) wins outright, best on all three behaviours — the higher-fidelity substrate did *not* create headroom that memory/attention architectures could exploit, consistent with the memory-axis finding that these worm behaviours are reactive-dominated (Logbooks 030–033); **(2)** the **wild-type connectome ranks 5th of 6** under PPO weight search — it *learns* the cell (8/8 seeds, so not a train-failure STOP) but its fixed biological wiring is a clear laggard on reactive predator-evasion, a starker result than the grid's mid-pack 75.6; **(3)** GA (gradient-free) floors, reproducing its T4 collapse. The sub-saturation continuous cell **discriminates** where T4's flat top-cluster tied. The connectome's 5th-place standing is what motivates the 6a rewired-null control (is it the *specific* wiring?) and sharpens the Phase 7 L4 hypothesis (is PPO simply the wrong learning rule for the connectome?). Whether this settles RQ3 toward optimal-primary framing is decided at the 6a synthesis (T9a), not pre-committed here.

______________________________________________________________________

## Complexity Dashboard

A snapshot of where the platform sits across five complexity dimensions, tracked across phases. Each dimension matters because the architecture comparison's interpretation depends on it (high-dimensional + non-Markovian observations test different architectural assumptions than low-dimensional Markovian ones), not because any one dimension is a quantum-advantage gate.

| Dimension | Phase 0-2 | Phase 3 | Phase 4 | Phase 5 | Phase 6 target | Phase 7 target | Phase 8 target |
|---|---|---|---|---|---|---|---|
| **Input dimensionality** | 2-9D | ~15-20D | similar | similar | > 50D (continuous + sensory-physics) | similar; cross-species sensors | + proprioceptive posture channel (N segments); + internal-state channel |
| **Partial observability** | Viewport only | STAM temporal memory | Multi-agent fog-of-war | Generational uncertainty | Realistic sensing range + connectome | Adds modulator-state observability | Body state observable only through proprioception; modulator state internal |
| **Multi-agent** | 1 | 1 | 5-10 | Single-agent populations | 1 (multi-agent deferred) | 1 (cross-species, not multi-agent) | 1 |
| **Temporal horizon** | Memoryless | STAM (~minutes) | STAM + social memory | Cross-generational | Full non-Markovian + plasticity-shaped | STDP-modulated long-horizon | Across-step intrinsic dynamics; undulation-scale steps (≤ 0.2 s) only if D19's alternative runs |
| **Classical ceiling on hardest task** | 94-98% (PPO foraging) | 94% (Mode A L500) | partially measured | n/a | TBD on continuous + connectome | TBD with plasticity | TBD through the body — a new reference frame, not a delta against Logbook 029 |

Update protocol: after each phase's results are in, this dashboard records *measured* values for that phase's columns and pencils-in targets for the next. The dashboard documents the substrate's complexity profile, not a quantum-advantage threshold; quantum architectures appear in the architecture-family sweep regardless of where the substrate sits on any one row.

______________________________________________________________________

## Biological Fidelity

### Current snapshot

Where the platform sits across five fidelity dimensions, by phase. This view answers "what does the substrate actually look like today, and what is each forward phase deepening?" — the question the optimal-primary framing makes load-bearing.

| Dimension | Phase 0-4 | Phase 5 | Phase 6 target | Phase 7 target | Phase 8 target | Future |
|---|---|---|---|---|---|---|
| **Connectome topology** | None (MLP/LSTM/etc.) | None | 302-neuron Cook 2019 (vendored SI parsing) | + transmitter identities and receptor classes on the Cook 2019 wiring (7a-ii atlas, a substrate deliverable) + *P. pacificus* head circuit (Cook 2025) + dauer (Yim 2024, SHOULD) | + measured synaptic signs and strengths on the head edges (Creamer–Leifer–Pillow; Randi 2023 atlas), reported at head and full scope (B.1); the init-vs-rewiring control (A.1) | + briggsae (gated on data); male–hermaphrodite contrast (data on disk) |
| **Sensory transduction** | Spatial gradient lookups | + klinotaxis head-sweep | Rung 2 **static** Fick-shaped gradients + adaptive/biphasic sensor *(dynamic PDE descoped 2026-06-04)* | unchanged | + proprioception (posture/stretch feedback, C.0); + internal metabolic state (B.3) | + multi-species receptors; dynamic-diffusion PDE |
| **Plasticity rules** | PPO / DQN / Reinforce | + Lamarckian inheritance, hyperparameter evolution | L2 PPO + L3 NEAT topology search | + L4 three-factor rules (rate-based primary + spiking-STDP arm) | PPO and the frozen-feature readout as *instruments*; plastic gap junctions under PPO (B.2); node-level adaptation via intrinsic dynamics; placed plasticity MAY | + structural plasticity; placed plasticity beyond the klinotaxis circuit |
| **Body mechanics** | Discrete 4-action grid | Discrete | Continuous 2D + spatial scales; OpenWorm Sibernetic interop if needed | + state-dependent action `std` (D7) | Reversal; anatomical motor-to-muscle readout (Cook 2019 NMJ) → curvature-kinematic body (C.1); 2D rod chain SHOULD (C.2); step–time calibration | 3D; fluid-coupled (Sibernetic-class) body; biophysical muscle |
| **Environment** | Grid; static gradients | + multi-agent, pheromones | Rung 2 static Fick-shaped gradients; corrected ASH/ADL contact nociception | + neuromodulator concentration field; internal-state observability | Patchy bacterial lawns with edges, depletion and quality; roaming/dwelling gated by serotonin/PDF from internal state (D.1 + B.3); 2D plate kept | 3D habitat; full ATP/metabolic model; population dynamics |

### Trajectory ladder

The platform progresses through six levels of progressive realism, each building on the previous:

| Level | Phase | Fidelity description |
|---|---|---|
| **1** | 0-2 ✅ | Grid-world, spatial gradient sensing, stateless reflexes, single agent, discrete actions |
| **2** | 3 ✅ | Temporal sensing (dT/dt, dC/dt, dO₂/dt), short-term memory (STAM), non-Markovian decisions |
| **3** | 4 ✅ | Multi-agent, pheromone communication, social dynamics, competitive/cooperative behaviours |
| **4** | 5 ✅ | Evolved hyperparameters, Lamarckian inheritance, methodology for co-evolution and transgenerational memory |
| **5** | 6 | Connectome-grounded learning + evolution + continuous 2D physics + corrected nociception + Rung 2 chemical gradients + chemosensory adaptation kinetics |
| **5+** | 7 | + Biologically-plausible plasticity (STDP, neuromodulator-modulated) + cross-species transfer (*P. pacificus*) |
| **5++** | 8 | + measured synaptic weights, intrinsic and gap-junction dynamics, the anatomical neuromuscular readout into a 2D body with reversal and proprioception, patchy lawns and a minimal internal state — *ground, then embody* |
| **6** | Future | 3D substrate (soil mechanics, fluid dynamics), fluid-coupled body, biophysical neurons and muscle, full energy/metabolic model, population dynamics, life-cycle simulation, briggsae and other species |

Levels 5, 5+ and 5++ together represent the project's target state. Level 6 is aspirational — see Future Directions for the technology selection and gated dependencies.

______________________________________________________________________

## Adaptive Roadmap Philosophy

This roadmap is **adaptive, not linear**. Each phase includes explicit go/no-go decision gates that allow the project to pivot based on empirical findings.

### Decision gate principles

1. **Evidence-driven.** Decisions are based on experimental results recorded in the per-milestone logbooks, not on assumptions or aspirational goals.
2. **Fail-fast at the substrate and architecture level, not at the phase level.** Phase 6's three mid-phase gates (L0 working at month ~2, L1 plugin parity at ~4-5, L2 results at ~7-8) trigger documented pivots — not silent slides past missed milestones.
3. **Hard phase boundaries between completed and in-flight phases.** No Phase N+1 work begins until Phase N is synthesised. Mid-phase gates protect execution; hard phase boundaries protect the narrative arc.
4. **Multiple paths to impact.** Most milestones have alternative success modes: a positive substrate result is the headline; a substrate-grounded STOP diagnosis is a defensible and reusable methodology contribution. Phase 5 demonstrated both.
5. **Scientific rigor over claim inflation.** STOP-with-diagnosis is the correct verdict when the experiment was honest but the substrate or architecture didn't support the question. Publishing "X didn't work and here is the architectural reason why" is a contribution; forcing a marginal positive result is not.

### Potential pivot scenarios

- **Connectome competitive with NEAT-evolved topologies but not dominant** → headline framing shifts from optimal-primary to connectome-primary (a neuroscience finding rather than an architecture one). Platform claim is unchanged. See Phase 6 Risk-mitigation, "L3 produces no separation from connectome."
- **L4 plasticity overshoots its software estimate (or 7a completes as a citable result)** → the pre-structured Phase 7a / 7b shipment shape absorbs it (2026-08-27: promoted from contingency to default, mirroring the 6a/6b split-by-success precedent). See Phase 7 Risk-mitigation.
- **L0 c302 connectome import is harder than expected** → hand-curated subset pivot (sensory-interneuron-motor subgraph). Platform claim survives in restricted form. See Phase 6 Risk-mitigation.
- **Continuous physics doesn't increase task difficulty enough to matter** → keep continuous action space (necessary for the architecture comparison's validity) and accept that the comparison is fundamentally easier than initially modeled. The connectome ranking question doesn't require difficulty escalation to be scientifically interesting.
- **External collaboration unavailable at Phase 7** → optional MAY items shift to internal-only execution. The platform paper and the connectome-learning paper are still draftable from internal Phase 6 + Phase 7 results without external validation.
- **Cross-species transfer reveals fundamental incompatibility** → the *finding* is scientifically informative ("transfer breaks at this layer"); document and ship as a Phase 7 result rather than treating as a failure.

Each pivot maintains scientific value. The Phase 5 STOP pattern is the canonical example: M4 / M5 / M6.x all closed as substrate-grounded diagnoses with reusable methodology, not as silent failures.

### External dependency risk mitigation

| Dependency | Risk | Mitigation |
|---|---|---|
| **Neuroscience lab collaboration** (Phase 7 MAY) | Labs decline or slow response | Optional, not required. Use published behavioural datasets (Bargmann chemotaxis indices, Kavli Ca²⁺ recordings, BAAIWorm correlation matrices) for built-in real-worm validation at Phase 6 close. |
| **OpenWorm c302 integration** (Phase 6 L0) | NeuroML format issues, missing metadata, unclear synaptic-weight provenance | Hand-curated subset of Cook 2019 connectome as L0 fallback (~50-100 neurons). Platform claim survives in restricted form. |
| **Cook et al. 2025 pacificus data scope** (Phase 7) | *(Re-aimed 2026-08-27: format risk retired — MIT-licensed CSVs, easy parse.)* Real risks: head-only / chemical-only scope, N=2 variability, and projection-homology ambiguity for specific behaviours | Matched head-truncation of the *C. elegans* baseline; shared-core matrix; explicit homology mapping table; ship the behaviours whose projections are defensible and document per-behaviour gaps. Cross-species claim survives in restricted form. |
| **GPU / HPC access for L3 NEAT** (Phase 6) | TensorNEAT-scale population search needs GPU; HPC allocation overhead | GPU is realistic baseline (consumer-class cards sufficient with TensorNEAT vectorisation). HPC is optional, pursue only when a specific Phase 7 stretch need justifies. |
| **Neuromorphic deployment** (Phase 7 MAY) | Loihi 2 / SpiNNaker 2 access non-trivial | Software-only L4 path is fully sufficient for the headline claim. Neuromorphic is a stretch / publication enhancement, not a requirement. |

### Adaptive execution

- **Per-milestone logbooks** with audit findings, statistical evidence, and explicit GO/PIVOT/STOP verdicts. Phase 5's logbooks 012-021 are the template.
- **OpenSpec change per non-trivial milestone**, archived on close (proposal → design → tasks → implementation → verification → archive).
- **Mid-phase decision gates** for long phases (Phase 6's three gates; Phase 7's pre-structured 7a/7b shipment shape with split-by-success).
- **Complexity dashboard updates** at each phase close — record measured values; document where the substrate sat.
- **Architecture-comparison protocol** as the single mechanism that places quantum, classical, recurrent, spiking, reservoir, hybrid, NEAT-evolved, and connectome-constrained brains in one experimental sweep.

______________________________________________________________________

## Ongoing Validation Milestones

Throughout all phases, biological validation against published *C. elegans* data is a continuous activity — not a Phase 7 deliverable held in reserve.

### Biological validation (every phase)

**Objective**: ensure model behaviours align with documented real-worm biology, and surface predictions that can be tested against published data without requiring an external lab partnership.

- **Phases 0-2** ✅: chemotaxis, thermotaxis, and predator-evasion behaviours validated against published behavioural data.
- **Phase 3** ✅: temporal sensing validated against published dT/dt / dC/dt sensitivity data.
- **Phase 4** ✅: social-feeding and pheromone behaviours validated against aggregation literature.
- **Phase 5** ✅: evolved-behaviour dynamics framed against natural *C. elegans* adaptation literature; M5 architecture-asymmetry diagnosis independently corroborated by external work (Resendez Prado, arXiv 2604.03565).
- **Phase 6**: locomotion + chemotaxis behaviour quantitatively compared to real worm data as a phase exit criterion. ≥ 1 of: chemotaxis indices (Bargmann lab + others), escape latencies (mechanosensation literature), whole-brain Ca²⁺ correlation matrices (Kavli / Janelia open data). The corrected ASH/ADL nociception is the natural validation pair for escape latencies.
- **Phase 7**: deepen biological validation with the L4 plasticity layer against the refreshed target list (dopamine-gated forgetting; learning-altered navigation-strategy weighting; escape-circuit lesion robustness — see [Phase 7 § Biological validation targets](#phase-7-deepen--plasticity--cross-species-transfer)) and with the cross-species head-circuit transfer. External lab partnership is optional (MAY); internal validation against published data is sufficient for the phase to close. *(Assessed at the close: the plastic-wiring targets were unreachable-with-reason, [Logbook 069](experiments/logbooks/069-phase7-synthesis.md).)*
- **Phase 8**: behavioural targets only, each unlocked by a rung — kinematics from a body (crawl speed, undulation period, eigenworm spectrum, omega-turn geometry; the 035/036 curves from emergent motion), forward/reverse bout durations (Morrison & Young 2025) once reversal exists, roaming/dwelling fractions once internal state exists, and the wiring effect's sensitivity surface as the methodological target a referee now expects (Churchland et al. 2026). See [Phase 8 § Biological validation targets](#phase-8-ground-then-embody--measured-substrate--body).

______________________________________________________________________

## Success Metrics Framework

The project tracks success across five dimensions. Each dimension has metrics and per-phase targets that map to the platform contribution and the scientific contribution, not to a separate quantum-advantage goal.

### 1. Biological Fidelity

**Definition**: how accurately the platform captures real *C. elegans* biology along the dimensions Phase 6+ commits to.

**Metrics**:

- Biological fidelity level achieved (see [Biological Fidelity](#biological-fidelity)).
- Quantitative match to published *C. elegans* behavioural data — chemotaxis indices, escape latencies, whole-brain Ca²⁺ correlation matrices.
- Number of model predictions surfaced and tested (against published data or partner-lab data).

**Targets**:

- **Phase 6**: ≥ 1 model output quantitatively validated against published real-worm data (chemotaxis indices, escape latencies, or Ca²⁺ correlation) as a phase exit criterion.
- **Phase 7**: deepens — L4 plasticity behaviour compared to documented *C. elegans* learning dynamics; cross-species behaviour compared between *C. elegans* and *P. pacificus*.
- **Phase 8**: measured synaptic weights on the substrate (B.1); kinematics validated from emergent motion once a body exists (C.3); bout durations and roaming/dwelling as behavioural targets (B.2, B.3 + D.1).

### 2. Architecture Comparison

**Definition**: how rigorously the platform ranks the connectome against unconstrained, evolved, quantum, and hybrid architectures.

**Metrics**:

- Architecture-family coverage in the Phase 6+ sweep (MUST set per Phase 6's architecture-families table).
- Statistical rigor: paired-seed Wilcoxon, bootstrap CIs, n ≥ 4 per condition.
- Plugin-parity test: adding a new architecture to L1 clears the files-touched + no-per-architecture-branches checks per [openspec/changes/phase6-tracking/design.md § Decision 6 § Gate 2](../openspec/changes/archive/2026-07-06-phase6-tracking/design.md) (informal "≤ 1 week" target; wall-clock not load-bearing).
- Lag-matrix or equivalent discriminative instrument for matched-capacity comparisons.

**Targets**:

- **Phase 6**: L2 weight-search results across the MUST architecture-family set on all three behaviours, at the Phase 5 statistical bar. L3 NEAT topology-search results comparing wild-type connectome to NEAT-evolved on ≥ 1 behaviour.
- **Phase 7**: L4 plasticity results for the connectome against its frozen-weights / vanilla-rule / L2-PPO baselines plus a pre-registered comparison set (D2 bar); cross-species transfer measured at matched head-circuit scope.
- **Phase 8**: every wiring contrast under an initialisation control (D15) at a swept operating point (D16); the six-family ranking re-run through the body with its own baselines (C.4, SHOULD).

### 3. Substrate Coverage

**Definition**: which behaviours, sensors, and substrate dimensions the platform supports at phase close.

**Metrics**:

- Behaviours operational on the connectome substrate (Phase 6 commits to three; Phase 7 retains the same three on a second species).
- Sensory-physics fidelity rung achieved (Phase 6 shipped Rung 2: static Fick-shaped gradients + adaptive/biphasic chemosensory sensor).
- Connectome species/states supported (Phase 6: *C. elegans*; Phase 7: + *P. pacificus* head circuit, + dauer as SHOULD).

**Targets**:

- **Phase 6**: three behaviours × MUST architectures × continuous 2D + Rung 2 + corrected ASH/ADL nociception.
- **Phase 7**: same three behaviours (third behaviour species-appropriate) × L4 + pre-registered comparison arms × *C. elegans* (full + head-truncated baseline) and *P. pacificus* head circuit (+ dauer as SHOULD).
- **Phase 8**: the same three behaviours on *C. elegans* only, through a substrate with measured weights, reversal, proprioception, the anatomical NMJ readout and a 2D body; patchy lawns with internal state; no second species.

### 4. Sample efficiency and convergence

**Definition**: how efficiently the architectures in the comparison reach the Phase 6+ behavioural targets.

**Metrics**:

- Episodes to convergence per architecture family.
- Generational convergence speed (evolutionary regimes).
- Phase 5's Lamarckian-inheritance speed gate (+5.25 generations) carries forward as the methodological template for evolutionary efficiency comparisons.

**Targets**:

- **Phase 6**: report convergence statistics per architecture family on all three behaviours; identify whether the connectome family shows characteristic sample-efficiency differences from unconstrained alternatives.
- **Phase 7**: compare L4 plasticity sample efficiency against L2 PPO on the same connectome substrate.
- **Phase 8**: time-to-competence remains the primary wiring metric (block V's instrument), now under D15's initialisation control and reported across D16's operating surface.

### 5. Robustness

**Definition**: performance under noise, missing sensors, or circuit ablation.

**Metrics**:

- Sensor dropout robustness (10%, 20%, 50% sensors disabled).
- Gaussian observation noise robustness.
- Circuit-ablation graceful degradation (Phase 6+ connectome architectures — matches biological lesion data).

**Targets**:

- **Phase 6**: connectome architecture demonstrates graceful degradation under circuit ablation comparable to documented biological lesion studies (qualitative match acceptable; quantitative match is a stretch).
- **Phase 7**: with L4 plasticity, test whether ablation-then-relearning approximates documented *C. elegans* recovery dynamics.
- **Phase 8**: robustness of the wiring effect itself — its sensitivity to the learner's operating point (A.2) and to the weight prior (B.1's measured-shuffled arm) — rather than sensor-dropout robustness, which is unchanged.

______________________________________________________________________

## Success Levels

Three levels of success, each representing a coherent and publishable scientific contribution. Higher levels do not invalidate lower ones.

### Minimum viable success

The platform exists and produces a defensible architecture-comparison result.

- **L0 connectome substrate operational** (at least the hand-curated subset under the L0 fallback pivot).
- **L1 architecture-plugin interface** at plugin-parity (adding a new architecture meets the parity checks in [openspec/changes/phase6-tracking/design.md § Decision 6 § Gate 2](../openspec/changes/archive/2026-07-06-phase6-tracking/design.md); informal "≤ 1 week" framing).
- **L2 weight-search results** on ≥ 1 behaviour across ≥ 4 architectures of the MUST set.
- **≥ 1 model output validated against published real-worm data.**
- **Phase 5's STOP findings preserved** as documented substrate-grounded diagnoses with reusable methodology.

### Target success

Phase 6 ships cleanly; the first headline-positive result lands.

- All Phase 6 MUST exit criteria met: L0+L1+L2+L3 operational across the full MUST architecture-family set, three behaviours, Rung 2 gradients + adaptation kinetics, corrected ASH/ADL nociception, real-worm validation.
- A defensible answer to "is the wild-type connectome a local optimum?" lands, either as "yes, dominant" (connectome-primary headline) or "competitive but not dominant" (optimal-primary headline). Either framing is a contribution; the data picks the framing.
- Phase 7 has begun and L4 is on a credible trajectory.

*(The Phase 5 M5 architecture-asymmetry / co-evolution question, previously listed here as a Phase 6 stretch, is deferred out of Phase 6 with no scheduled destination — see § Phase 6a/6b split and § Research Questions RQ4.)*

### Stretch success

Phase 7 closes cleanly and external visibility follows.

- L4 plasticity (rate-based three-factor + spiking-STDP arm) operational on the connectome, with D2-bar results against its frozen-weights / vanilla-rule baselines and the pre-registered comparison set.
- Cross-species head-circuit transfer ships (*P. pacificus*, + dauer as SHOULD); behavioural transfer is measurable across wiring; the connectome's role in transfer is characterised.
- At least one of the optional MAY items lands: biological-validation collaboration with a *C. elegans* lab; ≥ 1 paper drafted (platform paper, connectome-learning paper, or fitness-landscape paper); neuromorphic deployment demonstrated.
- Reproducibility artefacts updated to current platform state.

*(Assessed at the Phase 7 close, 2026-09-19, [Logbook 069](experiments/logbooks/069-phase7-synthesis.md).)* **Phase 7 reached target success and did not reach stretch success, and the stretch level's first clause is the one that failed for a reason worth stating.** Minimum viable and target were already met at the Phase 6a close. Of the stretch clauses: **L4 plasticity is operational but not to a D2-bar result** — the rate-based three-factor rule was built and ran against its frozen-weights and vanilla-rule baselines, and the 2×2's primary is unmet because no rule that writes the wiring learns the substrate to any benefit, which is a result rather than an incompletion; **the spiking-STDP arm was never built**; **cross-species transfer did not ship**, deferred with 7b under D14; **no MAY item landed**, the preprint having been cancelled in favour of a stronger combined package after the close; and **reproducibility artefacts are current with a stated limit** on what survives outside git. What Phase 7 adds beyond target success is three citable results — a replicated wiring advantage under gradient descent, a rule-programme negative with a diagnosed cause, and an operating-point finding about when the wiring is legible to a reading learner — none of which the success levels as written anticipated, because they were written expecting the plastic 2×2 to be the deliverable.

### Phase 8 success levels

*(Added 2026-09-20. Written around rungs and controls rather than a single flagship result, because the levels above were written expecting the plastic 2×2 to be the deliverable, [Logbook 069](experiments/logbooks/069-phase7-synthesis.md).)*

- **Minimum viable**: A.1 resolves under both D15 definitions and the block-V learning-speed result is restated with its status; A.2's sensitivity surface exists; B.1's data is vendored and its 2×3 resolves. A citable 8a package whichever way each reads. *(✅ **Met 2026-09-26**, [Logbook 076](experiments/logbooks/076-8a-synthesis.md).)*
- **Target**: 8a ships and 8b reaches C.1 — behaviour emerging from the anatomical motor-to-muscle map through a kinematic body, with an MLP positive control passed and the wiring contrast re-run on the new substrate; B.2 lands with the plastic-gap-junction arm.
- **Stretch**: C.2's rod-chain body under its cost budget with C.3 kinematic validation; C.4's ranking through the body; B.3 + D.1's roaming/dwelling on patchy lawns; the publication decision taken and a preprint out.

Publication, external collaboration, and community-launch metrics are not codified as success-level requirements — the project pursues them when evidence and context justify, not on a phase-locked schedule.

______________________________________________________________________

## Relationship to External Projects

### OpenWorm

**Focus**: cellular biophysics, muscle dynamics, 3D body simulation, *C. elegans* digital twin. Includes c302 (NeuroML-format connectome model) and Sibernetic (SPH-based body physics).

**Relationship**: **substrate dependency** at Phase 6 L0 + **interop boundary** for body-mechanics fidelity if needed.

OpenWorm has the connectome data and the body physics this project does not rebuild. The platform imports c302 as the canonical *C. elegans* topology source at Phase 6 L0 — OpenWorm is upstream infrastructure, not just a complementary project. If behavioural-fidelity claims later require native body mechanics, the platform interoperates with Sibernetic at the c302 boundary rather than re-implementing undulatory locomotion.

The Leeds physics group (Boyle, Bryden, Cohen) is complementary in the same way — best-in-class undulatory locomotion modelling, no learning or evolution. The platform interoperates with this body of work; it does not compete with it.

*(Updated 2026-09-20.)* OpenWorm is maintained, slowly: c302 v0.12.0 (2026-03-31, on the latest `cect`) and a Sibernetic CI workflow (2026-05-28), both by Gleeson. **The body decision changed at the Phase 8 review**: the first native body is a *2D* model (ElegansBot-class rod chain, § Worm body and whole-organism models), not Sibernetic interop, because Sibernetic's 3D SPH fluid coupling runs far above real time and thousands of RL episodes per seed do not fit through it on this project's compute ([Logbook 039](experiments/logbooks/039-runtime-acceleration-audit.md)). Sibernetic and c302 remain the interop reference and the fidelity ceiling; the c302/NeuroML export path is unchanged.

### Izquierdo & Beer (klinotaxis arc, Indiana)

**Focus**: evolved minimal circuits for klinotaxis; ensemble-of-models integrating connectome data; information-flow analysis through evolved circuits.

**Relationship**: **closest conceptual neighbour, deliberate methodological divergence**.

Their evolved minimal circuits are not the real connectome — they are abstract neural networks that behave like worms. The platform's L3 NEAT topology search produces directly comparable "what would evolution find?" results; placing the wild-type connectome (Phase 6 L0) and NEAT-evolved topologies (L3) in the same comparison sweep is the methodological extension. Their information-flow analysis tooling is a natural future-direction interop target.

### Worm body and whole-organism models

*(Added 2026-09-20.)* **Focus**: bodies and whole-organism simulations of *C. elegans* into which a connectome model can be dropped.

**Relationship**: **body reference and fidelity ceiling** for Phase 8's block C; cited as convergent, not competing.

- **ElegansBot** (Chung, Chang & Kim, *eLife* 2024; CC-BY; PyPI `ElegansBot`, GitHub `taegonchung/elegansbot`) — a 25-rod 2D chain with anisotropic Stokes drag and torsional-spring muscles that reproduces crawling, swimming, omega and delta turns from joint-angle inputs at ~1:1 real time on one core, validated at 0.208 mm/s. The reference for C.2's reduced (8–12 rod) chain. The same group then optimised connectome weights *proportionally to synapse counts* to drive it through forward and backward crawling and reproduce SMD ablation (Chung & Kim, bioRxiv 2025.07.21.665845; *Sci. Rep.* 2026) — connectome-to-body without learning or a wiring control.
- **Kim, Florman, Santos, Alkema & Shlizerman** (arXiv:2504.18073, 2025) — a modular connectome + dynamics + muscle calcium + force + proprioception framework reproducing forward/backward locomotion, avoidance and turns. The precedent for proprioceptive feedback as a sensory channel (C.0).
- **Lee** (bioRxiv 2026.09.06.749731, posted 2026-09-09; single author, Yonsei; not peer-reviewed; CC-BY-NC) — the closest thing to Phase 8's block C in print, and found by the re-aimed literature watch on the day its seeds were added: OpenWorm c302 with **six global** conductance scales fitted to the Randi 2023 signal-propagation atlas; a 24-segment 2D viscoelastic rod on agar with a Boyle–Berri–Cohen motor layer supplying oscillation and motor primitives; an evolution strategy training an external 16-unit policy that injects current into 35 whitelisted interneurons, the connectome itself fixed and muscles off limits; open-field chemotaxis and mazes. Five command/steering interneurons (AVB, AVA, SMDD, SMDV, RIV) suffice for chemotaxis at 91% source reach and the policy rediscovers the pirouette rule; sensory-neuron stimulation stays at chance, read as a missing sensory-to-command step in the fitted model; shuffling the gap-junction layer collapses performance (8–13%) where shuffling the chemical layer does not (86%). Its own stated limitations — literature-derived motor gains, one training seed converging to a local optimum, a null described as preserving out-degree only — are respectively D18, the paired-seed bar and D15's null in this plan. Cited as convergent; what it changed is recorded at each site (the C.1/C.2 novelty row, B.1's risk, B.2's convergence).
- **BAAIWorm / MetaWorm** (Zhao et al., *Nat. Comput. Sci.* 2024; Apache 2.0) — 136 multi-compartment neurons, 96 muscles, a 3D soft body; C++/CUDA + NEURON on an RTX 3090. Not runnable on this project's hardware and not needed for the wiring question; the fidelity ceiling to cite.
- **Wang-Chen & Ramdya** (arXiv:2601.08056, 2026) — the review of neuromechanical models whose named open problems (unrealistic controller connectivity, missing sensory organs and muscle models, environment complexity, actuator gains as free parameters) are the confounds D18 and C.0 register.
- **Eon Systems** (2026-03-07; FlyWire + predicted transmitters + a MuJoCo body, no learning, no validation metrics reported) and **Jin et al.** (arXiv:2602.17997) are the fly precedents. Together with the worm work above they fill the "connectome through a body" cell; Lee's is the one with learning and a wiring control in it, and § Phase 8's novelty map states what C.1/C.2 still add.

### Wormlight (sister project)

*(Added 2026-09-25.)* **Focus**: a living *C. elegans* in the browser — the whole connectome as a graded, conductance-based model (Kunert, Shlizerman & Kutz 2014) driving a Boyle, Berri & Cohen 2012 body on an agar plate, on WebGPU, with a fidelity ledger down to each connection's sign and every behavioural checkpoint fixed in advance (`chrisjz/wormlight`, private until ready; same maintainer).

**Relationship**: **sibling, and the biophysical half of the wiring question.** It takes its connectome from this repository through `scripts/export_wormlight.py` (PR #404: the Emmons 2024 CC BY release, the neuromuscular parse, a deterministic export). Its checkpoint 6 asks this project's question in a different model class — the real wiring against rewired nulls, every null given the same tuning procedure and budget — so its answer is an independent test, convergent or not, and the two projects should keep **one definition of the null** so the answers compare.

- **What it has already given this project** (its planning review, 2026-09-24/25): the null-strength confound behind A.6; the rhythm-generation evidence behind D19's note and C.2's reframing; the proprioceptive targets for C.0; the neuromuscular facts for C.1; the per-connection sign source and the AWC → AIY correction for B.3; the stiffness warning for B.2a; and two misattributed entries in `data/chemotaxis/literature_ci_values.json`.
- **Where the projects meet:** C.2's second half and C.3 overlap Wormlight's milestones; whether C.2 narrows is decided at C.2a on Wormlight's progress (C.2's note).
- **Its own open risk** is the one D19 names: whether a connectome-driven body crawls at all. Its milestone-0c go/no-go answers it first.

### Connectome-constrained learning lineage (fly)

**Focus**: connectome-constrained + task-optimised networks, closed-loop connectome-as-policy RL, and rewired-null controls — all in *Drosophila*.

**Relationship**: **method-category precedent + closest living pre-emption — cite as convergent, not competing**.

This fast-moving 2024-2026 lineage bounds what the project may claim, and the project should cite it as convergent evidence rather than get scooped by it. Lappalainen et al. 2024 (*Nature*, `flyvis`) established connectome-constrained + task-optimised modelling as a paradigm (supervised, perceptual). A **whole-brain connectomic graph model (Jin et al., arXiv:2602.17997, 2026)** trains the adult *Drosophila* whole-brain connectome as a graph-structured policy for whole-body locomotion via deep reinforcement learning, reporting better sample efficiency than baselines — the closest precedent to closed-loop RL on a real connectome, in a different organism. **Dhiman 2026 (arXiv:2604.04033)** applied a degree-preserving configuration-model (rewired-null) control to a behaving `flyvis` connectome — the fly precedent that the project's own degree-preserving rewired-null control (Tranche 8 / `add-connectome-structure-controls`) converges with — and *(noted 2026-09-16)* found the connectome's apparent advantage **dissolves under shared initialisation and that null**; see block V's record for how that reads against this project's surviving learning-speed effect. **Watch item *(added 2026-09-16)*: Wang & Christie, `pwang724/fly-circuit-exploration`** — a MaleCNS v1.0 connectome-plus-literature analysis (2026-09-10, audited 2026-09-13) proposing that the fly's home vector is stored in **synaptic weights** at hΔB → hΔH/hΔI, with a velocity-gated dopamine write (FB5H, FB4M) and an octopamine reset at food (OA-VPM3), plus hand-built rate simulations showing the mechanism is *sufficient* (closed-loop return 1.1 vs 6.8 units without memory). Not peer-reviewed, no new experiments, and **no rewired or shuffled-kernel control anywhere in it**, so "the connectome supports this" is untested against "any column-structured kernel supports this". **Cite with its audit**: the public post claims storage in weights "*not* neural activations", but the authors' own audit — published the day before the post — **withdrew that exclusion**, because the recurrence metric used to rule out activity storage returned the same low loop gain on the known EPG ring attractor, a measured persistent-activity network ("cannot tell an integrator from a relay"). Two of the four named types are also already recorded (hΔG a leaky integrator, Janke 2025; hΔA a 7–10 s working memory, Avritzer 2026), both leaking over seconds, which a synaptic store should not. So the finding is *synaptic storage is consistent with the wiring*, and this record must not cite it as a fast-weights result. Its value here is as a **mechanism template** for the placed-plasticity rung (§ substrate fidelity ladder, rung 6) and as external precedent for the phase protocol's positive-control discipline: a headline withdrawn because a positive control failed the metric. Revisit if a preprint appears with a shuffled-kernel control or the cAMP-versus-calcium imaging the authors name as the distinguishing experiment. The project's defensible contribution is the *C. elegans* + closed-loop-learning + neuromodulated-plasticity + controlled-comparison combination, dated and hedged; see [Phase 7 § Claim discipline](#phase-7-deepen--plasticity--cross-species-transfer) for how the Beiran & Litwin-Kumar 2025 degeneracy bound constrains structure-function claims against this lineage.

*(Added 2026-09-20 at the Phase 8 review.)* Three further items bracket the lineage. **Churchland, de Palma Aristides, Garcia-Ojalvo, Ritz, Anderson & Soriano** (arXiv:2609.07355, 2026-09-07) find the *C. elegans* connectome as an echo-state reservoir does **not** beat shuffle controls and trades performance for hyperparameter robustness — [Logbook 068](experiments/logbooks/068-l1b-rate-calibration.md)'s operating-point finding from another group, and the reason Phase 8's A.2 is a sensitivity surface. **Therianos** (arXiv:2606.17745, June 2026), a frozen rate operator on the complete larval fly connectome: degree and weight statistics set the gross dynamics, exact wiring sets *input routing* and mode drivers — Phase 8's A.3 registers those metrics as predictors of block V's learning-speed effect. **Guragain, Kakalis & Godino-Llorente** (arXiv:2606.09902, June 2026): biological weight values beat random initialisation on the same connectome topology in reservoirs, weak but in the direction B.1 tests. And **Creamer, Leifer & Pillow** (bioRxiv 2024.09.22.614271, preprint) is promoted from a caution in Phase 7's evidence base to a **substrate source**: its fitted signs and magnitudes on the Cook edges are B.1's data.

### Cook et al. — *P. pacificus* head connectome (Science, 2025)

**Focus**: the *P. pacificus* **head connectome** (Cook et al., *Science* 389:eadx2143, 31 Jul 2025) — two adult hermaphrodite heads, nose tip → retrovesicular ganglion including the nerve ring; **chemical synapses only** (gap junctions excluded as ultrastructurally ambiguous); pharynx excluded (covered by Bumbarger 2013 at WormWiring); ~88% of neuron classes in a shared core; data as MIT-licensed CSVs (`stevenjcook/cook_et_al_2025_pristionchus`) + *Science* Supplementary Data S1–S6. *(Description corrected 2026-08-27 — v4.1 called this "the full P. pacificus connectome", which overstated its scope; see Phase 7 Deliverable 2 for the rescope.)*

**Relationship**: **substrate dependency** at Phase 7 cross-species transfer (head-circuit scope, matched-truncation design per D3).

The 2025 data makes *C. elegans* → *P. pacificus* head-circuit transfer feasible today with published reference data; the companion 2025 monoaminergic map (eLife RP 109557) supports the species-appropriate predatory behaviour. No third party (cect, NemaNode) packages this dataset yet — the project's loader is itself a small contribution. *C. briggsae*, by contrast, still lacks a published connectome (re-verified 2026-08), so the briggsae direction stays in Future Directions until reference data appears.

### Comparative connectomics community

Witvliet et al. 2021 (developmental connectomes); the dauer-connectome work (Nature Communications 2024); the broader 2024-2026 wave of cross-species connectome papers all provide reference data the platform can interoperate with as Phase 6+7 evidence accumulates. Engagement is data-flow first; partnership is optional.

### Neuroscience labs (Phase 7 MAY)

**Target labs (if collaboration is pursued)**: Bargmann (Rockefeller), Sengupta (Brandeis), Horvitz (MIT), Lockery (Oregon), or smaller groups with relevant Ca²⁺ recording data.

**Collaboration model**: the platform generates model predictions from Phase 6+7 results; the lab designs and executes the targeted experiment; co-authored publication. At most one partnership if pursued — collaboration overhead grows non-linearly with multiple labs.

Phase 6's built-in real-worm validation against published behavioural data is **not contingent on lab partnership** — internal validation against open datasets (Bargmann chemotaxis indices, Kavli / Janelia Ca²⁺ recordings, BAAIWorm correlation matrices) is sufficient for Phase 6 close.

### Quantum computing ecosystem

The 300-session quantum architecture campaign (Phase 2) used IBM Quantum hardware (Qiskit), with Q-CTRL Fire Opal exercised for error suppression. The campaign's results survive as baseline reference data in Phase 6's architecture-comparison protocol. Continued quantum-hardware engagement is opportunistic — if a Phase 6 or Phase 7 substrate change plausibly affects the quantum-family comparison, a targeted re-evaluation is reasonable; otherwise the baseline reference is sufficient.

### Neuromorphic hardware (Phase 7 MAY)

Loihi 2 and SpiNNaker 2 are credible deployment targets for the L4 plasticity layer — they natively implement spiking + STDP at low power, and the *C. elegans* connectome at 302 neurons fits comfortably within their chip-scale. A neuromorphic-hardware deployment would be a tools/methods contribution in its own right, but the software-only L4 path is fully sufficient for Phase 7's headline claim.

### NematodeBench (removed 2026-07-25)

The benchmark infrastructure was **removed**, not merely demoted a second time — the submission workflow, validation, categorisation and leaderboard generation, plus `BENCHMARKS.md` and `docs/nematodebench/` ([`remove-nematodebench`](../openspec/changes/archive/2026-07-25-remove-nematodebench/)).

The v4 demotion above justified keeping it as internal tooling "useful for reproducibility and for the architecture-comparison protocol itself". Two phases of evidence contradicted that. Across Phases 5 and 6 the protocol read per-seed `--track-experiment` output directly via `scripts/analysis/weight_search_architecture_ranking.py` and never once invoked the submission pipeline; the corpus stopped at six submissions from 2025-12-28/29 covering 3 of the eventual 27 architectures; `BENCHMARKS.md` went 19 months without a content commit and still advertised a `static_maze` category deleted from the code in January 2026.

**What survives.** The convergence detector and composite score — the only component the protocol actually depends on — moved to `packages/quantum-nematode/quantumnematode/experiment/convergence.py`. The 72 session experiments behind the six submissions were migrated into `artifacts/experiments/` rather than deleted; nothing was lost. The reproducibility scaffolding that was doing the real work is untouched: `--track-experiment`, `artifacts/experiments/`, git-context capture, and per-run seeding (now specified under `experiment-tracking`).

______________________________________________________________________

## Future Directions

### What Phase 8 opens on

*(Added 2026-09-19 at the Phase 7 close, [Logbook 069](experiments/logbooks/069-phase7-synthesis.md).)*
Phase 7's inheritance, in dependency order. Each item names what it inherits, so this is an
consequence list rather than a wish list. The ladder below is the programme these feed into.

*(2026-09-20: absorbed into [§ Phase 8](#phase-8-ground-then-embody--measured-substrate--body) — item 1 → A.1 (D15), 2 → A.2 (D16), 3 → B.2, 4 → the placed-plasticity MAY, 5 → B.2, 6 → A.4, 7 → A.5. Kept here as the dependency record.)*

1. **The init-vs-rewiring control — first.** Block V's +35.4% / +23.5% / +55.3% / +40.1% are the
   phase's strongest citable result, and `rewire_seed` is unset in every one of those panels, so each
   seed's rewired graph derives from its run seed and **rewiring varies with initialisation**.
   *(**Answered 2026-09-21**, [Logbook 070](experiments/logbooks/070-init-sharing-control.md): the
   pairing half is partially discharged — no dissolution under either definition, survival established
   on five of eight readings and three unresolved — and
   the across-seed half is registered as a follow-up rather than run, because pinning `rewire_seed`
   across seeds reintroduces the shared-nulls caveat V.4 closed. **Conditioned 2026-09-23**,
   [Logbook 071](experiments/logbooks/071-operating-point-surface.md): the effect is depth-critical —
   present at settling depths 4 and 6, abolished at 3, reversed at 2 — and A.1's survival is
   consistent when re-read at depth 6 on 32 seeds.)*
   [Dhiman 2026](https://arxiv.org/abs/2604.04033) reports the fly connectome's advantage dissolving
   under shared initialisation plus a degree-preserving null — the same control, the same kind of
   claim, another organism. **It needs its own design decision before it can be registered**: "the
   same initialisation" has no single meaning once the mask changes, because the init scale is
   `1/sqrt(chemical in-degree)` and a degree-preserving rewiring preserves the degree *sequence* but
   not which neuron holds which degree. Inherited from [065](experiments/logbooks/065-wiring-fresh-rewiring.md).
   *(**Corrected 2026-09-20**, at the Phase 8 PR review: the last clause is wrong about this
   implementation. `rewire_degree_preserving` is a directed double-edge swap, so every neuron keeps
   its **own** in- and out-degree and the init scale is already matched neuron-for-neuron; what
   varies with the seed is which drawn values land on which edges. D15 in § Phase 8 states it
   correctly. [Logbook 069](experiments/logbooks/069-phase7-synthesis.md) carries the same sentence,
   with a dated correction footnote of its own added 2026-09-20 alongside the Phase 8 tracker.)*
2. **A calibration rung, before any new contrast** — over `plasticity_rate`, readout width,
   `forward_pass_depth` and `initial_log_std`. Inherited from
   [068](experiments/logbooks/068-l1b-rate-calibration.md), where one inherited pin **set the sign** of
   a registered primary, and from [062](experiments/logbooks/062-l4-frozen-readout.md), which already
   named the log-std question and called it cheap to ask. `forward_pass_depth` leads it: at depth 4
   only neurons within four hops of a sensory injection reach the motor pool, so the depth partly
   determines *which* wiring a learner can see, and it has never been varied.
3. **The dynamics rung** — gap-junction coupling as a dynamical term rather than a fixed symmetric
   matrix, behind (2). Inherited from [067](experiments/logbooks/067-l4-feature-ablations.md), where
   removing gap junctions helped **both** wirings and the null caught up; it also absorbs **D4's
   surviving destination**, a PPO arm with the electrical synapses plastic on the block-V cells.
4. **The placed-plasticity rung** — the rule at an anatomically identified site against a
   degree-stratified random subset of the same size, with its three confounds already specified below.
5. **L.2, intrinsic dynamics** — Phase 7's carried SHOULD, now with (2) as a precondition rather than
   as free-standing scope.
6. **Methodology consolidation, and one rename.** `plasticity-evaluation` holds **42 requirements** *(44 since this synthesis's own change archived into it; counted 2026-09-20)*,
   most of them single-use rules this phase paid for; they fold into the
   [phase protocol](research/phase-protocol.md) so a rung designer reads a handful of principles
   rather than thirty-three rules. And the committed *byte-identity* requirement is renamed to **parsed-field
   identity**, which is what it actually checks — one seed per arm, every field the analysis reads.
   *(**The rename landed 2026-09-19**, pulled ahead of the consolidation; the consolidation itself stays
   here, to be done with Phase 8's needs in view rather than blind.)*
7. **The publication decision**, deferred to after the close when S.1 was cancelled. The package is
   real: a replicated wiring advantage under gradient descent, a rule-programme negative with a
   diagnosed cause, and an operating-point finding about legibility. A referee reads Dhiman, so item 1
   is the difference between a paper and a rebuttal.

### The substrate fidelity ladder (the phase after 7)

*(Added 2026-09-08.)* Phase 7's frame, carried forward. Each rung restores one piece of the wiring's biology and registers the same question, "does the wild-type wiring start to matter once this is real?", against the degree-preserving rewired null and the frozen and Hebbian floors that panels 1–3 fixed: (1) **synaptic identity** — transmitter identities and receptor classes from the atlases (7a-ii), signs no longer random draws. *(Noted 2026-09-11.)* Our grounding is **per neuron**, from experimental identity; the fly field's current standard is a **predicted transmitter per synapse** across a whole CNS (`male-cns v1.0`). That difference is the substance of B.1's finding — per-neuron identity grounds only 5.8% of synapses as inhibitory, too few to build an inhibitory brake from — so the coarseness is a limitation of the available *C. elegans* data, not a modelling choice, and it is why the receptor layer (B.3) became a prerequisite rather than fidelity work; (2) **routed neuromodulation** — per-modulator signals reaching the synapses the receptor metadata says they reach, through the connectivity that carries them (7a-ii's structured instruction); (3) **dynamics** — gap-junction coupling and intrinsic time constants on the units, so within-step settling becomes across-step state; (4) **a body** — motor-neuron patterns driving undulation, omega turns and pirouettes, by interoperation with OpenWorm's Sibernetic and c302 at the boundary the roadmap already names, so behaviour emerges from motor output instead of a two-number readout; (5) **extrasynaptic signalling** — neuropeptides, the "wireless" connectome, stated as the signalling layer's primary limitation in Phase 7. Whole-organism fidelity — biophysical neurons, muscle, development, the life cycle — stays interoperation, not a build; it is OpenWorm's mission and not this platform's comparative advantage. *(Updated 2026-09-15, D14.)* **The ladder now has an instrument.** Every rung above was tested with a rule that could not learn; R.2's `readout_only` — an 8-parameter readout learned by its own exact gradient on the frozen wiring — reaches competence, so a rung can show whether the wiring starts to matter. Two rungs are pulled ahead of the order above because R.2 motivates them directly: **(0) readout width** — the 2×4 pooling over four motor classes may be the bottleneck through which the wiring's features are invisible; a readout over all 39 motor neurons asks how much of the motor pool's structure a learner must see before the wiring matters. ***(Answered 2026-09-17, [Logbook 066](experiments/logbooks/066-l4-readout-width.md): it was the bottleneck.*** L.1 crossed width with wiring rather than comparing to L.0's committed numbers, because 78 parameters against 8 would confound capacity with legibility. Over 96 paired seeds the **interaction is +0.2818** on `auc_success` at q = 0.000: at the pooled width the rewired null is ahead, at the per-neuron width the wild type is, and **the sign of the wiring effect flips**. **No width main effect was detected** (−0.0112, q = 0.591, CI [−0.0374, +0.0126]) — a failure to detect rather than a demonstration of absence, though the interval bounds any capacity effect far below the interaction, so a capacity-only explanation is not supported. This is **the first positive for the wild-type wiring under a biologically plausible learner in this phase**, and it reopens L.4 and L.5. It is a learning-speed and area-under-curve claim on frozen `w_chem`, so it does **not** convert D2's primary, and **no mechanism is offered** — V.2 still finds no graph property predicting learning time, and nothing explains why the wide null got *worse*.\*) ***(L.4 and L.5 answered 2026-09-19, [Logbook 067](experiments/logbooks/067-l4-feature-ablations.md): both features carry it, and neither reading means what the word suggests.*** Each ablation removed one feature at the per-neuron width and was read as an interaction against L.1's wide baseline, with `carries` requiring abs(Δ) ≥ 0.123 — two-thirds of the removable +0.1852. **Gap junctions off** reads `carries_the_effect` (−0.2035, q = 0.000, 73/96): the wild type's lead is gone because **both wirings learn far better without gap junctions** (null 0.3585 → 0.7506, wild type 0.5437 → 0.7322) — the advantage consists in the wild type's gap junctions costing it less than the rewired ones cost the null. **Atlas signs**, at a rate-matched 0.0001 after a registered rate check found the atlas arms unlearnable at 0.001, reads `carries_the_effect` *qualified carries or unlearnable* (−0.1319, q = 0.000, 66/96; both wirings learn less grounded) — and in its own matched baseline **the null is ahead** (0.7990 vs 0.7013). That baseline is the record's third result, unregistered: **the per-neuron wiring effect is +0.1852 at 0.001 and −0.0977 at 0.0001** on the same 96 seeds, the null gaining +0.44 from the lower rate against the wild type's +0.16. **L.1's positive is therefore a positive at one pinned rate, and may not be cited without it.** The structural probe on L.1's puzzle is null (ρ = +0.03): within-class input correlation does not predict the loss from widening across the 96 rewirings, whose spread (0.014–0.027) is an order of magnitude narrower than the wild type's 0.154, so the null does not reach the contrast that motivated it. **The positive is restated (task 7.4) as living in the electrical synapses' interaction with the learner's operating point**, not in the chemical graph's connectivity alone, and **rung (3) below — gap-junction coupling as a dynamical term — is the rung that manipulates it directly**, behind a registered rate × wiring calibration at the per-neuron width that did not exist before this record. Not a mechanism, not an endpoint claim, no committed verdict changed.\*) ***(And the rate, answered 2026-09-19, [Logbook 068](experiments/logbooks/068-l1b-rate-calibration.md): L.1's interaction does not survive it.*** The 0.001 L.1 ran at was pinned by R.2 on an 8-parameter readout, its registered check waived, and inherited at 78 parameters unswept. L.1b completed the 2×2 at **0.0001** — only the two pooled learning cells were missing — over the same 96 seeds: **the interaction is −0.0657 against +0.2818, with a three-way of +0.3475 at q = 0.000 on 81/96 seeds**, larger than L.1's whole effect, and **the sign reverses**. At the lower rate the dominant effect is **capacity**: the width main effect is **+0.6178 on 96/96**, where L.1 detected none at all — **how much readout capacity matters is itself set by the learning rate**, which no rung had measured. The wild type leads at **neither** width at 0.0001. The reverse direction is reported and **not credited**: 47% of its registered minimum and 23% of L.1's effect, on the primary axis alone, with the registered secondary voided by a 0.604 censoring spread and an unregistered graded axis finding nothing. **So L.1's positive stands as read at `plasticity_rate` 0.001 and is carried with that rate in the same sentence as the claim**; the sign-flip form belongs to width × wiring × rate. The dynamics rung below therefore takes a second precondition: **a rate × width calibration before any contrast**, the ~192-run form L.1b ran.\*) And **(3) dynamics**, since a reservoir with no temporal dynamics is a poor reservoir and the rule was found to write uniformly over graph distance. Rung (4), the body, is where 7b's comparative sweep now sits: R.2 is partly a measurement of that gap, and a cross-species wiring comparison is not interpretable through a readout that carries the learning. The ladder is the programme that turns four negative panels into the first rungs of a systematic answer, and research question 3 (plasticity reproducing published *C. elegans* learning data) becomes reachable only as its rungs land.

*(Added 2026-09-16.)* **(6) Placed plasticity — the rule at an anatomically identified site, against the same rule on a degree-stratified random subset of the same size.** Every plausible-rule arm in Phase 7 applied one rule uniformly across all 3,709 chemical synapses, or to a generic subset (the motor pool in R.1c, the readout in R.2), and R.2 found e-prop writing uniformly over graph distance. **A uniform rule is not blind to the wiring** — each synapse's eligibility is built from its own pre- and post-synaptic activity, which the wiring shapes, and the downstream dynamics the credit flows through are the wiring's too, so the rewired null remains a real control under a uniform rule. What a uniform rule lacks is a **placement prior**: it spreads its updates across every edge rather than concentrating them where a circuit does the work. The rung asks the narrower question that follows — **is an anatomically selected subset required?** — which no Phase 7 panel put. The fly path-integration hypothesis of Wang & Christie 2026 (below, under external projects) is the template for what "placed" means at mechanism level, whatever its own status: **column-matched pre-synaptic input** (a kernel with an offset structure that degree-preserving rewiring destroys), a **write gated on behavioural state** (a dopamine modulator that is on while walking, not a reward-prediction error), and an **event-triggered reset at reward** (octopamine at food) rather than our uniform time-decay `λ_w`. *C. elegans* has the circuit for our own task: the klinotaxis pathway (AWC → AIY → AIZ → RIA/RIB → SMB/SMD, Iino lab) is in Cook 2019 and is the cell every Phase 7 panel ran on. The rung is *the three-factor rule restricted to that circuit's synapses* versus *the same rule on a degree-stratified random subset of the same size*, both against the wild-type and rewired-null wirings. **Three things have to be specified before it is registered, because each is a confound rather than a detail:** (i) **what "the same site" means under rewiring.** A degree-preserving rewiring changes which neurons connect, so the placed set must be defined either by **neuron identity** — plasticity on the edges incident to the named cells in whatever graph they inhabit — or by **following each original edge to its remapped target**; the two ask different questions, edge IDs are not stable across a rewiring, and the choice must be stated rather than inherited from whichever the implementation makes easy. (ii) **The random subset is drawn independently for each wiring**, stratified against *that* wiring's own degree distribution: a subset degree-matched to the wild type is not degree-matched in the null. (iii) **The trainable-synapse count is identical across all four arms**, so that rewiring changes neither the placement interpretation nor the plasticity budget — the lesson L.1 (readout width, in flight at time of writing) is built around, where a wider readout confounds capacity with legibility and only a crossed design separates them. A positive says the wiring's structure is legible to a rule that knows where to look. A null says placement did not help **on this circuit and this task**: stronger than the phase's uniform-rule nulls *for the placement hypothesis specifically*, and **not** a stronger negative in general, since a uniform-rule null and a placement null test different propositions. Sits behind L.1 and L.2, needs its own positive control by the [phase protocol](research/phase-protocol.md), and is not in Phase 7's closing scope.

Beyond Phase 6 and Phase 7, the following research directions are scoped as future work — each is a substantial programme in its own right, and each is gated on either Phase 6/7 evidence or external data availability. The roadmap deliberately does not schedule them.

### Activity-dependent structural plasticity (dynamic wiring)

*(Added 2026-09-06.)* Every wiring in Phases 6–7 is a fixed graph — wild-type, rewired-null, the *P. pacificus* head circuit, the dauer state — and the L3 NEAT search is evolutionary across generations, not activity-dependent within a life. The L4 rule can only redistribute strength among synapses the adult already has, and Logbook 040 showed how much that constraint matters: outcomes are fixed points of a fixed edge set seeded by the initial weights. A structural-plasticity layer — synapse formation and pruning driven by activity and neuromodulatory state, anchored on the Witvliet et al. 2021 developmental connectome series (eight animals across larval stages) and the dauer wiring state — would let a rule create and remove support, test whether the specific adult wiring is a *learnable* endpoint rather than a fixed prior, and connect the plasticity programme to the cross-species one. Substantial: it needs a developmental dataset ingest, a structural rule family, and a validation target of its own.

### Cross-species expansion beyond *P. pacificus*

- **A within-species sex contrast** *(added 2026-09-11)*. The male CNS release's headline is a
  *between-sex wiring contrast at synaptic resolution*: under 5% of male neurons sex-specific or
  dimorphic, concentrated in higher-order centres, yet 12% of male neurons wired differently against
  4% of female. Every wiring contrast this project has run is synthetic — wild type against a
  degree-preserving rewired null. Cook et al. 2019 published **both** *C. elegans* sexes, and the
  loader names `cook_2019_hermaphrodite` specifically, so the same experiment is available on our
  own organism with *real* variation in place of a shuffle. It is only meaningful once a rule is
  shown to learn, so it sits behind block I. *(Checked 2026-09-20: the male wiring **is** in the
  repository's data — the vendored Cook 2019 workbook carries the `male chemical` and `male gap jn`
  sheets. Excluded from Phase 8 by scope; a within-species contrast for a later phase.)*
- ***C. briggsae* transfer**. Phase 7 covers *P. pacificus*; *C. briggsae* lacks a high-quality published connectome as of project planning (chromosome-level genomes only). Becomes a scoped phase once reference connectome data appears.
- **Witvliet developmental connectomes**. The *C. elegans* developmental connectome series (Witvliet et al. 2021) supports a within-species temporal-transfer study: does the platform's L0+L2 setup reproduce documented developmental shifts in behaviour? Optional follow-on to Phase 7.
- **Comparative connectomics community**. Engagement with the broader 2024-2026 connectomics wave — data interop first, scoped partnerships if specific questions emerge.

### Drosophila-scale connectome transfer

- *Drosophila* (~100K neurons) has a connectome dataset (FlyWire) and active connectome-execution-at-scale work (Sandia on Loihi 2 achieving >100× real-time). Transferring the platform to Drosophila scale is fundamentally a **neuromorphic-hardware question**, not a connectome-learning one — the scale-up needs different infrastructure than the *C. elegans* / *P. pacificus* work. Belongs in Future Directions, not on the Phase 7 critical path.
- *(Updated 2026-09-11.)* The scale target has moved. The **complete male CNS** — brain and ventral nerve cord through an intact neck, 166,700 neurons, 125M synapses, 11,710 cell types, CC-BY on neuPrint as `male-cns v1.0` (Janelia + Google Research + Cambridge, *Cell* 189:5504, 2026) — supersedes FlyWire as the natural target, and changes what the transfer *is*. FlyWire is a brain; this is a nervous system with the descending path to the motor periphery mapped end to end, which is the rung the substrate ladder calls "a body". A fly transfer is therefore no longer a pure scale-up question: it is partly the body/motor question, on a substrate where the wiring from sensor to muscle exists. It also ships **per-synapse and per-neuron transmitter predictions**, where our *C. elegans* grounding is per-neuron experimental identity (see the sign-grounding note below). Still Future Directions, still gated on Phase 7's rule question; the target and its framing are what changed, not the priority.
- Worth distinguishing: FlyWire is the connectome dataset + execution-at-scale; NeuroMechFly / FlyWalker are separate 3D-embodied-fly research threads. The platform's plausible next-scale connectome target is the former, not the latter.
- Zebrafish larvae (~100K neurons) is a structurally similar target — visual predator avoidance + schooling — with the same scale-up considerations.

### 3D environment + native body mechanics

3D environment, native sinusoidal undulation, omega turns, and pirouettes are deferred to Future Directions. The platform's claim ("first closed-loop learning + evolution on the real *C. elegans* connectome with pluggable architectures") does not require them, and the *C. elegans* validation data is 2D-plate biology. If a future phase commits to 3D, technology selection should run as follows:

- **MuJoCo MJX** is the default choice — JAX backend, GPU-vectorisable, deterministic, headless-friendly, the de facto standard in embodied-RL-for-biology research.
- **Brax** is the runner-up if raw throughput beats physics fidelity for the question at hand.
- **OpenWorm Sibernetic** is the choice if the 3D need is specifically *C. elegans* fluid-coupled body mechanics — it's SPH-based, 2D + viscous-fluid, validated against real-worm movement.
- **Game engines (Unity, Godot, Unreal) are off-table for this project.** They're optimised for interactive rendering at 60-120 FPS, not for batch RL training at evolution scale (thousands of parallel environments × thousands of generations × deterministic reproducibility). The compute cost-per-step and the reproducibility/headless requirements both work against them.

3D belongs to organisms whose behavioural repertoire is fundamentally 3D (Drosophila flight, fish swimming, mouse navigation), not to *C. elegans* on agar.

*(Updated 2026-09-20.)* **The 2D body moved into Phase 8** (C.1 kinematic, C.2 rod chain — § Phase 8, D19), with ElegansBot as the reference model rather than Sibernetic; **3D, fluid coupling and the Sibernetic body stay here**, and the technology-selection notes above are unchanged for whenever a 3D need appears.

### Energy / metabolic model

The current platform has no energy/metabolic model — satiety is abstract rather than ATP-based. Phase 6 and Phase 7's three behaviours (klinotaxis, thermotaxis, predator evasion) don't depend on internal energy state at the timescales the platform operates on, so this is not on the critical path. But several aspirational behaviours **do** require an energy / metabolic state representation:

- **Dauer transitions** (food-scarcity-induced larval state). Single biggest *C. elegans* behavioural transition not currently representable.
- **Dwelling vs roaming**. Long-timescale foraging strategy modulation by food-detection state. Documented behavioural phenotype with neuromodulator-receptor mapping.
- **Long-timescale foraging**. Resource-depletion + replenishment dynamics on bacterial-lawn substrates.

The "all *C. elegans* behaviours" aspiration is implicitly gated on this gap being filled. Energy/metabolic implementation is itself substantial (~3-6 months software-only) and would warrant a scoped phase if pursued.

*(Updated 2026-09-20.)* The **minimal** metabolic state — an internal-state sensory module and the modulator concentration field, gating roaming versus dwelling — is Phase 8's B.3 (SHOULD, coupled with patchy lawns, D.1). The full ATP model, dauer transitions and long-timescale foraging stay here.

### Applied directions

Each of these is a multi-year research programme in its own right, requiring different funding, partnerships, and expertise than the Phase 6/7 platform work. They are scoped here as future direction headers, not as roadmap commitments.

- **Drug screening assays**. Use the platform for compound screening via behavioural phenotyping. Requires the energy/metabolic model above + pharmacology partners.
- **Neurodegeneration models**. *C. elegans* analogues of Alzheimer's, Parkinson's, ALS at the connectome level. Requires connectome perturbation tooling + neurodegeneration biology partnerships.
- **Brain-computer interfaces**. Neural-decoding insights extracted from connectome-learning dynamics. Requires neural-recording partners.
- **Aging studies**. Age-dependent behavioural changes (*C. elegans* lifespan ~2-3 weeks). Requires the energy/metabolic model + lifespan-modelling expertise.

### Hybrid behavioural-cellular models

Combine behavioural-level RL training with selective cellular-level biophysics — e.g., RL-trained behavioural foraging with detailed AFD-neuron biophysics for thermotaxis. Cross-validation point with OpenWorm's cellular-level predictions. Modest scoped follow-on; sits naturally adjacent to Phase 7's L4 plasticity work.

### Ecological co-evolution

Add state to predators (HP, satiety, death-by-starvation, kill-replenishes-energy) so the Phase 5 frozen-weight Red Queen substrate gains coupled population dynamics. Lotka-Volterra-style oscillations; whether predator-prey populations stabilise, oscillate, or collapse under learned policies. Phase 5's `PredatorBrain` Protocol (M1) and `MLPPPOPredatorBrain` (M5) already supply the policy substrate; the new work is environment-side state machinery + reward shaping. Selection pressure shifts from "maximise kill-rate" to "maintain a viable population against prey escape velocity" — natural follow-up if Phase 5's co-evolution verdict motivates richer eco-dynamics.

### NematodeBench public launch

A public-facing launch (external submissions, public leaderboard, community submission workflow) was moved to Future Directions here: benchmarks crystallise mature communities; they don't bootstrap them. That reasoning stands. What changed on 2026-07-25 is the fallback — the internal tooling this section preserved was itself removed, having gone unused through Phases 5 and 6 (see § NematodeBench above). A future public launch is therefore a from-scratch build against the then-current architecture set, not a reactivation of this one.

### Computational principles emerging from the deep-dive

The original "≥ 3 universal computational principles documented" framing of v3 is dropped — universal-principles extraction is an engineering-the-breakthrough pattern that tends to produce overclaiming. If Phase 6/7 results surface principles that generalise (e.g., specific architectural motifs that beat naive MLPs across all behaviours; specific evolutionary signatures of Baldwin-style canalisation on connectome topologies; specific plasticity-rule families that fail on connectome substrates), those emerge organically in the papers — they are not a scheduled deliverable.

______________________________________________________________________

## Technical Debt & Maintenance

### Resolved through Phase 5

- ~~QQLearningBrain completion~~ — evaluated, not competitive; deprioritised. **Retired 2026-08-23** — the architecture was removed from the codebase ([#282](https://github.com/SyntheticBrains/nematode/issues/282)); git history keeps the implementation.
- ~~MLPReinforceBrain loss bug~~ — investigated and documented.
- ~~Grid size hardcoding~~ — fixed.
- ~~Statistical analysis framework~~ — operational; paired-seed Wilcoxon and bootstrap CIs are the project-wide standard.
- ~~Sensory input refactoring for temporal derivatives~~ — shipped in Phase 3 (STAM, dT/dt, dC/dt, dO₂/dt).
- ~~Memory buffer architecture~~ — STAM buffers operational across all brain architectures.

### Active for Phase 6

1. **L0 connectome substrate import path** — ✅ shipped, but *not* as planned here: direct Cook 2019 *Nature* SI parsing vendored under `data/connectome/` with provenance, **no runtime c302/NeuroML dependency** (c302/NeuroML survives only as a deferred *export* path, RQ4 in the archived phase6-tracking change). Recorded 2026-08-27 so the pacificus import (item 7) inherits the correct framing: it is a CSV parse, not a NeuroML ingest.
2. **L1 architecture-plugin interface** — clean `Brain` interface that admits MLP / recurrent / spiking / reservoir / quantum / hybrid / NEAT-evolved / connectome-constrained without per-architecture branching. Plugin-parity test per [openspec/changes/phase6-tracking/design.md § Decision 6 § Gate 2](../openspec/changes/archive/2026-07-06-phase6-tracking/design.md) (files-touched + no-per-architecture-branches checks; informal "≤ 1 week" framing).
3. **Continuous action heads** — extend the existing PPO-family brains with Gaussian-policy continuous action heads; adapt quantum architectures with continuous-output circuits.
4. **Corrected ASH/ADL contact-based nociception** — owed correctness work flagged in [Logbook 011](experiments/logbooks/011-multi-agent-evaluation.md); lands in Phase 6's sensory-physics stack.
5. **Documentation** — API documentation, tutorials, architecture guides current to Phase 6 state. Required to keep the architecture-plugin interface usable for future contributors and to support reproducibility artefacts (Docker, evaluation scripts) as an optional MAY exit criterion.

### Active for Phase 7

6. **L4 plasticity infrastructure** — persistent pre/post activity traces on `ConnectomeTopology` (cross-step state does not currently exist); rate-based three-factor rules first, spiking-STDP arm second (D1); diffusible-signal concentration field (serotonin, dopamine); receptor-class metadata from the Wang 2024 neurotransmitter atlas + bulk-integrated CeNGEN profiles; modulated three-factor rules. New package (D8: `learning_rules/` working name — `quantumnematode/plasticity/` is the quantum-plasticity *eval* protocol, not learning rules). Substantial new code.
7. **Cross-species head-circuit integration** — Cook et al. 2025 CSV loader + species-keyed neuron-classification table + species-keyed validation pathways + species-keyed sensor/motor projection map (the current projections are hard-coded *C. elegans* named-neuron tuples in `connectome_ppo.py`); head-truncated Cook 2019 baseline; dauer (Yim 2024) loader as SHOULD.
8. **Pre-L4 platform items (D5/D7) — ✅ closed 2026-09-05** ([Logbook 038](experiments/logbooks/038-state-dependent-std-gate.md)): the #254 dead keys were already removed during Phase 6 (verified, no code); the state-dependent action `std` shipped byte-identical-when-off as per-brain std heads (**no `_policy.py` changes** — the shared helpers were already shape-generic, correcting this item's earlier phrasing); the D7 klinokinesis gate **failed** after the pre-registered entropy-only pass, so per Amendment A the substrate froze **mode-off**, the post-D7 re-baseline was **descoped** (Logbook 029 remains the reference frame), and the mechanism ships dormant; load-time validation of the new keys is automatic via the pydantic fields.

### Active for Phase 8

09. **Loader: keep the muscle cells** — `connectome/loader.py` drops the body-wall muscle columns (`dBWML*`, `vm*`) the vendored Cook 2019 workbook carries; C.1 needs the motor-neuron-to-muscle matrix as a first-class tensor with its own smoke tests.
10. **Signed speed** — reversal as a first-class continuous action (speed is clamped to `[0, max_step_mm]` today), byte-identical-when-off, landed and frozen before C.1 registers.
11. **Proprioceptive sensory channel** — posture/stretch feedback into the sensory projection; the connectome-side target neurons stated with a biological argument.
12. **Step–time calibration** — one recorded constant relating an environment step to worm seconds, cited by every kinematic target and cost estimate.
13. **Measured-weight ingest** — Creamer–Leifer–Pillow supplement (licence check first) and the Randi 2023 atlas, vendored under `data/connectome/` with provenance; a unit-scale pin registered for sweep.
14. ~~**Methodology consolidation** — the 44 `plasticity-evaluation` requirements folded into the phase protocol (A.4).~~ **Done 2026-09-20**: 7 stay, 11 moved to `architecture-comparison-protocol`, 18 folded into the phase protocol, 8 retired.
15. **Artefact-retention rule** — which per-campaign artefacts are committed and which archived off-repo, registered at phase start.

### Lower priority (address as needed)

08. **Code quality** — remaining Ruff / Pyright warnings; test coverage gaps.
09. **Configuration system** — hyperparameter search templates; evolution-config schemas current to Phase 6+ work.
10. **Performance profiling** — optimisation for continuous-physics + NEAT-population-search workloads.

______________________________________________________________________

## Scoping Changes from v3

This roadmap is v4. v3 was framed around two co-equal goals — "biological simulation + quantum architecture analysis" — with a Phase 6 major quantum re-evaluation gate and a Phase 7/8 publication-and-community arc. v4 reframes the project around an optimal-primary, connectome-as-control architecture comparison; quantum becomes one architecture family in the comparison sweep rather than a project-organising principle. The Phase 5 evidence base (logbooks 012-021), the post-Phase-5 platform data, and 2024-2026 external research (Cook et al. 2025 pacificus connectome; Resendez Prado architecture-asymmetry corroboration; OpenWorm c302 trajectory) together support the reframe.

For reviewers reading v4 who recall v3 commitments: the full pre-rewrite v3 text is recoverable from git history at commit `3d48e04f` via `git show 3d48e04f:docs/roadmap.md`. This appendix documents which v3 promises v4 does **not** carry forward, with the reframe rationale for each.

| v3 promise | v4 disposition | Reframe rationale |
|---|---|---|
| Two co-equal goals: biological simulation + quantum architecture analysis | One primary research question: optimal architecture comparison with connectome as focal point; quantum is one architecture family in the comparison | The 300-session Phase 2 campaign + Phase 3-4 results established that grid-world complexity is below the threshold for quantum advantage on every variant tested. Carrying quantum as a co-equal goal is no longer evidence-supported. |
| Phase 6 quantum checkpoint (MAJOR), full quantum campaign v2 if classical drops below 70% | Architecture-comparison protocol at Phase 6 includes quantum families as comparison rows; 300-session campaign is baseline reference | Quantum re-evaluation collapses into the architecture-family sweep; no separate gate or escalation rule. |
| Phase 7 "Community, Validation & Publication" with NematodeBench launch as a core deliverable | Phase 7 retitled to "Deepen — Plasticity & Cross-Species Transfer". NematodeBench public launch moves to Future Directions | Benchmarks crystallise mature communities; they don't bootstrap them (OpenWorm, Brian2, NeuroBench all followed this pattern). NematodeBench infrastructure persists as internal tooling. |
| NematodeBench infrastructure persists as internal tooling (the Phase 6/7 disposition set by the row above) | Infrastructure removed entirely 2026-07-25; convergence detector retained under `experiment/`, session experiments migrated to `artifacts/experiments/` | Two phases of use showed the architecture-comparison protocol never routed through the submission pipeline — it read tracked-experiment output directly. ~2.4k lines and 78 LFS JSONs of unmaintained surface for one retained 696-line module. |
| Phase 7 publication campaign with three named papers (quantum evaluation, NematodeBench, connectome) targeting Nature Methods / NeurIPS / ICML / eLife / PNAS | Paper drafts (platform / connectome-learning / fitness-landscape) are Phase 6/7 MAY items; no specific venues promised | Specific venue commitments produce overclaiming; the project may publish when evidence and context justify, not on a phase-locked schedule. |
| Phase 7 external collaboration with ≥1 *C. elegans* lab partnership as exit criterion | Biological-validation collaboration is Phase 7 MAY (optional); Phase 6 commits to ≥1 internal validation against published real-worm data as exit criterion | Internal validation against open datasets (Bargmann chemotaxis indices, Kavli Ca²⁺ recordings, BAAIWorm correlation matrices) is sufficient for Phase 6 close. Lab partnership is additive, not a precondition. |
| Phase 8 "Integration & Comprehensive Evaluation" with definitive quantum vs classical comparison at maximum complexity | Phase 8 deleted entirely | The definitive-quantum-vs-classical question is structurally retired; quantum is one row in the architecture-family comparison at every phase that runs the sweep. |
| Phase 8 "≥3 universal principles documented and validated" | Dropped; principles emerge organically in papers if they emerge at all | The "≥N universal principles" framing is the engineering-the-breakthrough antipattern. Whatever generalisable principles surface, surface organically. |
| Phase 8 applied directions (drug screening, neurodegeneration models, BCI) | Moved to Future Directions | Each is a multi-year programme of its own requiring different funding/partnerships/expertise than the Phase 6/7 platform work. Out of scope for the reframed two-phase post-Phase-6 arc. |
| Phase 8 Drosophila ~100K-neuron transfer proof-of-concept | Moved to Future Directions, reframed as neuromorphic-hardware question | FlyWire + Sandia/Loihi 2 at >100× real-time establishes Drosophila scale-up as a neuromorphic-hardware problem, not a connectome-learning one. Different infrastructure requirements. |
| *C. briggsae* transfer (Future Directions in v3 + Phase 7 wishlist) | Gated on future connectome data availability | No high-quality *C. briggsae* connectome has been published. *P. pacificus* (Cook et al. 2025) supersedes briggsae as the realistic cross-species target. |
| Phase 6 "sinusoidal undulation, omega turns, pirouettes" native body mechanics | Dropped; interop with OpenWorm Sibernetic at c302 boundary if body-physics fidelity needed | High cost (10-100× simulation step cost), low scientific leverage for the architecture-comparison question. The platform claim survives without native body mechanics. |
| Phase 6 "aerotaxis / pheromone signalling / multi-agent in continuous physics" implicit in scope | Deferred to future phases | Three-behaviour Phase 6 scope (klinotaxis + thermotaxis + predator evasion) keeps the architecture-comparison framing focused. Multi-agent in continuous physics is its own research problem. |
| HPC compute access assumed for evolution + connectome work | HPC is explicitly optional, not required | GPU access (consumer-class cards via TensorNEAT vectorisation) is the realistic baseline. HPC is pursued opportunistically when a specific Phase 6/7 stretch need justifies an allocation. |
| Phase 6 "all 19 existing architectures" implicit comparison scope | Curated MUST set of architecture families per the Phase 6 table (transformer promoted MAY→MUST 2026-06-07) | The L1 architecture-plugin interface admits the families that test the load-bearing questions; historical variants from Phases 0-3 stay in their logbooks as reference. |
| Vision: "build the deepest, most complete behavioral simulation of *C. elegans* ever created. Progressively implement all sensory systems, survival behaviors, learning and memory, social dynamics, and realistic physics" | Vision recast around "most efficient brain architecture for nematode-like embodied tasks, using the *C. elegans* connectome as the focal comparison point" | "Simulating all *C. elegans* behaviours" is the OpenWorm-15-year-trap. The platform's contribution is what *learning + evolution + architecture comparison* on a curated subset enables — the all-behaviours framing pushed scope without strengthening the contribution. |
| Phase 7 WormBot ("potential future validation platform") | Removed; no concrete commitment in v3 either | WormBot was an aspirational hardware-embodiment future direction; not a project commitment. If hardware embodiment becomes relevant, it re-emerges in Future Directions. |
| North Star: "demonstrate quantum advantage at high biological fidelity OR characterise complexity thresholds for quantum advantage" | North Star: "be the platform on which learning and evolution operate on the real *C. elegans* connectome in a closed sensory-motor loop, and rank the wild-type connectome against unconstrained, evolved, and quantum architectures" | The platform contribution + scientific contribution are the load-bearing pair; quantum-advantage framing demoted to one architecture family in the comparison. |

The Adaptive Roadmap Philosophy section retains v3's "evidence-driven, fail-fast, scientific rigor over claim inflation" stance; v4 sharpens it with the hard-phase-boundary-vs-mid-phase-gates distinction and with the M3 / M4-M5-M6 STOP pattern from Phase 5 as the canonical demonstration that STOP-with-diagnosis is a scientific contribution.

The roadmap follows the project's standing conventions documented in [AGENTS.md](../AGENTS.md): OpenSpec workflow for non-trivial work; paired-seed statistical rigor (Wilcoxon, bootstrap CIs, n ≥ 4 per condition); no milestone references in implementation code or docstrings; per-completed-milestone logbook with audit findings + decision-gate verdicts; LFS-rules audit before committing artefacts; no autonomous push / PR creation.

______________________________________________________________________

## Conclusion

This roadmap charts a milestone-based adaptive path from a Phase 5 close anchored in M3 Lamarckian inheritance (the positive headline) and three substrate-grounded STOP diagnoses (M4 Baldwin, M5 co-evolution, M6.x transgenerational memory — each surfacing a substrate or architecture finding that carries forward) toward a connectome-grounded architecture-comparison platform at Phase 6 and a biologically-plausible plasticity + cross-species deepening at Phase 7.

The contribution is paired, not dual-goal:

1. **Platform contribution**: first closed-loop learning and evolution on the real *C. elegans* connectome with a pluggable architecture interface that admits MLP, recurrent, spiking, reservoir, quantum, hybrid, NEAT-evolved, and connectome-constrained brains as comparable rows in one experimental sweep.
2. **Scientific contribution**: a defensible answer to "how does the wild-type *C. elegans* connectome rank against unconstrained and evolved alternatives — when learning and evolution operate on it in a closed sensory-motor loop?", with comparative cross-connectome learning on the *P. pacificus* head circuit (+ dauer) — transfer of trained agents contingent on the SHOULD weight-transplant arm — and biologically-plausible plasticity (neuromodulator-gated three-factor rules) extending the answer at Phase 7.

The project's unique position — integrating biologically-grounded sensing, multiple learning and evolutionary regimes, a pluggable architecture interface, and the real *C. elegans* connectome on a single substrate — is the contribution. No competing computational *C. elegans* effort has all four. OpenWorm has the connectome and body physics but not closed-loop learning; Izquierdo & Beer have learning and evolution but on minimal evolved circuits rather than the real connectome; the Leeds physics group has body mechanics but not learning. The platform interoperates with each of these where the boundaries are natural (c302 import, Sibernetic for body mechanics if needed); it does not compete with them on their home turf.

Key principles guiding execution:

1. **Evidence-driven, not aspiration-driven.** Decisions live in per-milestone logbooks; STOP-with-diagnosis is the correct verdict when honest experimentation reveals a substrate or architecture limit.
2. **Fail-fast at the substrate level, not at the phase level.** Mid-phase decision gates (Phase 6's three gates; Phase 7's pre-structured 7a/7b shipment shape) trigger documented pivots — not silent slides past missed milestones.
3. **Hard phase boundaries between completed and in-flight phases.** Phase N+1 work does not begin until Phase N is synthesised — the narrative arc has integrity, even when mid-phase execution is adaptive.
4. **Demote rather than delete.** Quantum demoted to one architecture family. NematodeBench demoted from public-launch deliverable to internal tooling. Optionality preserved; commitments matched to evidence. *Corollary added 2026-07-25:* demotion is a holding position, not a terminal state. Where a demoted component then accrues no use across a full phase, deletion follows — and is recorded as a reversal with its evidence, rather than left to bit-rot in place. NematodeBench is the first component to complete that arc.
5. **One programme, two contributions.** The platform claim and the scientific claim are mutually reinforcing — building the platform answers the architecture-comparison question; the architecture-comparison question motivates each platform layer.

By the Phase 7 close the project had shipped the connectome-grounded architecture-comparison platform, a diagnosed negative on biologically plausible plasticity that writes the wiring, and a replicated wiring advantage on learning speed under gradient descent — with cross-species transfer deferred and the plastic 2×2 unmeetable within the closing scope ([Logbook 069](experiments/logbooks/069-phase7-synthesis.md)). **Phase 8 grounds, then embodies**: the initialisation control the strongest result still needs, the operating-point surface a referee now expects, measured synaptic weights on the real edges, dynamics, and the anatomical motor-to-muscle map into a two-dimensional body — each a registered rung of the same question, each with its own positive control, and none of it a whole-organism emulation. Whether the connectome's wiring turns out to matter once its weights and its body are real, or turns out to be its degree statistics all the way down, the platform contribution stands and the evidence chain is documented at the per-milestone level rather than asserted in the roadmap.
