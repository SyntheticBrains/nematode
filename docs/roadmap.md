# Quantum Nematode Project Roadmap

**Vision**: Determine the most efficient brain architecture for nematode-like embodied tasks, using the *Caenorhabditis elegans* connectome as the focal comparison point against unconstrained and evolved alternatives. The platform brings learning, evolution, and a curated subset of biologically-faithful sensing into one closed sensory-motor loop, so that architecture comparisons answer scientific questions rather than rank benchmarks.

**Version**: 4.0

**Last Updated**: 2026-05-23

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
15. [Conclusion](#conclusion)

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
| **6** | ~6-10 months from Phase 5 close | Connectome substrate + architecture comparison | 🔲 PLANNED | First closed-loop learning + evolution on the real *C. elegans* connectome with a pluggable architecture interface. NEAT topology search ranks the wild-type connectome against evolved alternatives on three behaviours (klinotaxis, thermotaxis, predator evasion) |
| **7** | ~8-12 months from Phase 6 close | Deepen — plasticity + cross-species transfer | 🔲 PLANNED | Biologically-plausible plasticity (STDP + neuromodulator-modulated) on the connectome. *P. pacificus* transfer using Cook et al. 2025 connectome data. Optional biological-validation collaboration and paper drafts |

______________________________________________________________________

## Executive Summary

The Quantum Nematode project asks one primary research question expressed along two comparison dimensions: *what is the most efficient brain architecture for nematode-like embodied tasks, and how does the C. elegans connectome rank against unconstrained and evolved alternatives — when learning and evolution operate on it in a closed sensory-motor loop?*

The project's central contribution is a **platform that makes this question answerable**. It integrates four capabilities that, separately, exist across the computational neuroscience and embodied-AI fields but have not yet been combined on a single substrate:

- Biologically-grounded sensing (klinotaxis, thermosensation, mechanosensation, pheromone-mediated signalling) shipped through Phases 1-4.
- Multiple learning and evolutionary regimes (PPO, CMA-ES, Lamarckian inheritance) shipped through Phase 5, with neuromodulated plasticity (STDP-family) targeted at Phase 7.
- A pluggable architecture interface that admits MLP, recurrent, spiking, reservoir, quantum, hybrid, NEAT-evolved, and connectome-constrained brains as comparable rows in one experimental sweep (Phase 6).
- The real *C. elegans* connectome (302 neurons, Cook et al. 2019) imported as the focal architecture to rank against the others, with *P. pacificus* (Cook et al. 2025) as the planned cross-species comparator at Phase 7.

Phase 5 results sharpened the framing in a load-bearing way. M3 Lamarckian inheritance shipped as the headline-positive Phase 5 result. M4 (Baldwin), M5 (co-evolution arms race), and M6.x (transgenerational memory) closed with substrate-grounded STOP verdicts that were architectural diagnoses, not implementation failures: each pointed at the substrate or architecture rather than at the experimental protocol. Two of those diagnoses (M5's architecture-asymmetry, M6.x's wrong-abstraction-for-plasticity) carry forward directly into Phase 6's architecture-comparison and Phase 7's plasticity work.

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

Phases 0-5 are complete. The platform now supports: 19 brain architectures spanning quantum, classical, recurrent, spiking, reservoir, and hybrid families; thermotaxis, mechanosensation, aerotaxis, klinotaxis, and pheromone-based sensing; multi-agent dynamics at 5-10 agent scales; CMA-ES and TPE hyperparameter evolution; Lamarckian weight inheritance across generations. The connectome layer, the pluggable architecture interface, and continuous-2D physics are the work of Phase 6.

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

Phase 5 closed 2026-05-23 with one headline-positive result and four substrate-grounded STOP verdicts. All five Phase 5 exit criteria are met with evidence; the STOP results are scientifically informative architectural diagnoses, not implementation failures, and the methodological yield from them carries directly into Phase 6 and Phase 7. See [Logbook 021](experiments/logbooks/021-phase5-synthesis.md) for the full synthesis.

- **M2 Hyperparameter Evolution** — GO. Four-arm CMA-ES then TPE campaign closed RQ1 on optimiser choice; +47pp / +79pp predator-arm acceleration on the M3 inheritance config. See [Logbook 012](experiments/logbooks/012-hyperparam-evolution-mlpppo-pilot.md).
- **M3 Lamarckian Inheritance** — GO, headline-positive. Speed gate passes at +5.25 generations; +17.5pp F1-F3 mean retention on the M6.10 environment; n=8 paired-seed rerun confirms. The "learned behaviour becomes innate" exit criterion is satisfied; see [Logbook 013](experiments/logbooks/013-lamarckian-inheritance-pilot.md).

### What we tried and stopped (Phase 5)

Three Phase 5 themes — Baldwin effect, co-evolution arms races, transgenerational memory — produced **STOP verdicts as field-consistent substrate diagnoses**, not implementation failures. Each diagnosis pointed at the substrate or architecture rather than at the experimental protocol; each shipped reusable methodology; and two of the three carry their architectural question directly forward into Phase 6.

- **M4 Baldwin Effect** — STOP after three iterations (M4 → M4.5 → M4.6). Diagnosis: **single-task K=50 PPO has no Baldwin axis** because the optimal strategy on a single task is innate good behaviour for that task — the opposite of what Baldwin canalisation selects for. The substrate constraint, not the algorithm, blocks the result. Multi-task aggregation infrastructure is the prerequisite for a clean Baldwin demonstration and is deferred. Methodology shipped: F1 evaluator, 4-way aggregator, 8-field config schemas, n=8 Lamarckian rerun extending M3. See [Logbook 014](experiments/logbooks/014-baldwin-inheritance-pilot.md) and [Logbook 015](experiments/logbooks/015-baldwin-iterative-evaluation.md).
- **M5 Co-evolution Arms Race** — STOP after 13 single-seed lever ablations. Diagnosis: **LSTMPPO-prey-vs-MLPPPO-predator architecture asymmetry suppresses Red Queen entanglement** — own-vs-cross fitness lag delta stayed in the +0.017 to +0.024 range across all ablations, against a target of ≤−0.05. Independent corroboration arrived from Resendez Prado's [Personality Requires Struggle](https://arxiv.org/abs/2604.03565) (April 2026): "transparent regime" same-architecture self-play suppresses the heterogeneity needed for measurable Red Queen / Baldwin signal — the same hypothesis from a different group. The architecture-asymmetry question carries directly into Phase 6, where matched-capacity NEAT-vs-NEAT and connectome-vs-NEAT comparisons test it on continuous physics. Methodology shipped: lag-matrix cross-pairing instrument and cell-grid fair-test methodology, both reusable. See [Logbook 017](experiments/logbooks/017-coevolution-arms-race.md).
- **M6 / M6.9+ / M6.13 Transgenerational Memory** — STOP across three pilot rounds. The `TransgenerationalInheritance` framework + `TransgenerationalMemory` dataclass + LSTMPPO `tei_prior` actor-logit hook ship as functional infrastructure, but no K value or substrate variant produced a positive memory effect. Diagnosis: **the bias-network logit-prior is the wrong abstraction for the wet-lab single-circuit excitability shift** documented in Kaletsky 2025 and the 2025 mammalian-TEI literature. This is a *substrate* finding — different from a hyperparameter or training failure — and points at Phase 6's connectome-substrate work as the natural next step. Pure-TEI K=0 was substrate-inert (cross-arm delta −49pp); substrate-on-top-of-Lamarckian at K=1000 showed zero acceleration; at K=200 showed −9.33pp active interference under fair-F0. See [Logbook 018](experiments/logbooks/018-transgenerational-memory.md), [Logbook 019](experiments/logbooks/019-transgenerational-memory-redesign.md), and [Logbook 020](experiments/logbooks/020-tei-prior-on-lamarckian.md).

Two reusable methodology contributions ship unscooped: the **lag-matrix cross-pairing instrument** and the **cell-grid fair-test methodology**, both from [Logbook 017](experiments/logbooks/017-coevolution-arms-race.md). Independent corroboration of the M5 diagnosis arrived from outside the project during Phase 5 close-out.

### Known Gaps Carried into Phase 6+

- **No connectome-constrained architecture** — Phase 6's focal deliverable.
- **No biologically-plausible plasticity rules** — STDP and neuromodulator-modulated STDP are Phase 7's focal deliverable.
- **No energy/metabolic model** (satiety is abstract, not ATP-based) — blocks dauer-state and dwelling-vs-roaming behaviours; flagged as a Future Directions prerequisite, not on the Phase 6/7 critical path.
- **Discrete grid-world (not continuous physics)** — addressed in Phase 6 alongside Rung 2 chemical gradients and corrected ASH/ADL contact-based nociception (the latter is owed correctness work flagged in [Logbook 011](experiments/logbooks/011-multi-agent-evaluation.md)).
- **No native body mechanics** (sinusoidal undulation, omega turns, pirouettes) — interop with OpenWorm/Sibernetic at the c302 boundary if needed; native implementation is not on the Phase 6 critical path.
- **Multi-task aggregation infrastructure** — Baldwin prerequisite; revisits if a future phase commits to the Baldwin question.

### Research Questions for Phase 6+

1. **Connectome ranking.** How does the wild-type *C. elegans* connectome rank against unconstrained MLP/LSTM, NEAT-evolved topologies, and quantum architectures on klinotaxis, thermotaxis, and predator evasion when learning and evolution operate on a common substrate?
2. **Connectome fitness landscape.** Is the wild-type connectome a local optimum on these behaviours, a basin, or a saddle? What synaptic-weight changes does evolution find when permitted to modify it?
3. **Plasticity and the connectome.** Does biologically-plausible plasticity (STDP, neuromodulator-modulated STDP) on the real connectome reproduce dynamics that match published *C. elegans* learning data (chemotaxis indices, Ca²⁺ correlation matrices)?
4. **Architecture asymmetry under matched capacity.** Phase 5 M5 diagnosed architecture asymmetry as the blocker for Red Queen entanglement. Does matched-capacity NEAT-vs-NEAT or connectome-vs-connectome co-evolution produce the dynamics that LSTMPPO-vs-MLPPPO suppressed?
5. **Cross-species transfer.** Do learned/evolved architectures transfer from *C. elegans* to *P. pacificus* (Cook et al. 2025) on the shared behaviours? Where do they break, and what does that say about the connectome's role?

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

**Goal**: Transform the simulation from stateless reflex to temporal integration. Make C. elegans sense the way it actually senses — through temporal derivatives, not spatial gradient lookups. This is the single most impactful biological fidelity upgrade and directly addresses two quantum advantage thresholds (non-Markovian dependencies and partial observability).

**Aspirational timeline**: Q2-Q3 2026

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

#### Quantum Checkpoint (Phase 3)

**Trigger**: Temporal sensing operational, classical ceiling measured.

**Result**: Classical GRU PPO achieves oracle-level converged performance on temporal sensing. The classical ceiling did NOT drop — GRU temporal reaches 94% L500 on the hardest environment. This means:

- **QRH re-evaluation on temporal tasks is warranted** — if classical GRU matches oracle, does QRH's temporal advantage (from Phase 2) still hold?
- **The quantum opportunity may be in training efficiency** — temporal agents need 10x more episodes than oracle. If quantum architectures can learn temporal correlations faster, that's a genuine advantage.
- **Phase 4 (multi-agent) remains the primary quantum advantage pathway** — exponential state spaces from agent interactions.

#### Go/No-Go Decision

**GO to Phase 4**: Temporal sensing is operational and validated. Classical approaches handle temporal derivatives effectively with GRU PPO — confirming that multi-agent complexity (Phase 4) is the right next frontier for quantum advantage investigation.

______________________________________________________________________

### Phase 4: Multi-Agent Complexity

**Goal**: Create exponential state spaces through agent-agent interactions. This is where quantum game theory has the strongest theoretical backing for advantage, and where C. elegans social behaviors (aggregation, pheromone communication, alarm signaling) are well-documented.

**Aspirational timeline**: Q3-Q4 2026

**Prerequisites**: Phase 3 memory infrastructure (agents need to remember past interactions)

#### Background

C. elegans, while often considered solitary, exhibits sophisticated social behaviors:

- **Social feeding**: Feeding rate increases near conspecifics (social facilitation)
- **Aggregation**: Clustering on bacterial lawns mediated by ascaroside pheromones
- **npr-1 variation**: Natural genetic variation determines solitary vs. social feeding behavior
- **Alarm pheromones**: Injured worms release signals that repel nearby individuals
- **Cooperative-like behaviors**: Worms following pheromone trails benefit from others' foraging discoveries
- **Competition**: Limited food creates resource competition and dominance dynamics

Multi-agent scenarios create exponential state spaces (state × number of agents), partial observability (each agent has local view), and strategic interactions — all identified as quantum advantage thresholds.

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

#### Quantum Checkpoint (Phase 4)

**Trigger**: Multi-agent operational with ≥5 agents, coordination metrics established.

Evaluate:

- **Quantum entangled strategy spaces**: Can quantum architectures represent correlated multi-agent strategies more efficiently?
- **Quantum game theory approaches**: Do quantum-enhanced Nash equilibrium solvers outperform classical?
- If classical approaches show measurable difficulty on coordination tasks (aspirational: ceiling \<85%), launch targeted quantum evaluation campaign

**Assessment (Logbook 011, post Klinotaxis Era)**: The 80.2% ceiling at 10 agents (B1-v2) meets the numerical threshold, but Campaign G proved this is resource allocation difficulty, not computational complexity. With proportional resources, classical MLP PPO achieves 100% of ceiling at all scales. The Klinotaxis Era K1 finding (89% L100 on 5-agent collective discovery with pheromones) represents successful learned use of an extra observation channel, not a search-space explosion that would favour quantum approaches. **Recommendation: do not trigger quantum multi-agent evaluation.** The environment does not create genuine coordination complexity. Conditions that could: harder partial observability, multi-cluster foraging requiring trail specialisation (K7 weakening suggests a candidate area), or co-evolutionary phenotype dynamics (Phase 5 territory).

#### Go/No-Go Decision

**GO**: Multi-agent infrastructure is complete, functional, and produces emergent behaviour under proper conditions (K3 frequency-dependent phenotype fitness, K1 collective discovery, K6 social feeding). Pheromone communication mechanism is precisely characterised (six conditions for collective benefit). Negative findings (alarm channel inert, aggregation channel inert, multi-cluster generalisation partial) are scientifically informative. **Phase 4 is complete; proceed to Phase 5.** Deferred work for Phase 4: alarm pheromone emission semantics code changes (issue to be opened), multi-cluster pheromone scaling at higher agent counts, and per-step trajectory analysis for behavioural verification of trail-following.

______________________________________________________________________

### Phase 5: Evolution & Adaptation

**Goal**: Evolve optimal learning strategies and study how learning guides evolution, including biologically-documented transgenerational memory.

**Aspirational timeline**: Q4 2026 - Q1 2027

**Prerequisites**: Phase 3 (memory infrastructure for transgenerational memory). Phase 4 multi-agent infrastructure required only for co-evolution (deliverable 4) — other deliverables can begin in parallel with Phase 4.

**Pilot-Then-Focus Approach**: Start with lightweight pilots of 2-3 evolutionary approaches using small populations and few generations. Based on pilot results, select 1-2 approaches for deep investigation.

#### Phase 5 Milestone Tracker

Phase 5 is broken into milestones M0–M8 plus a tracking scaffold (M-1). The living sub-task checklist lives in [openspec/changes/phase5-tracking/tasks.md](../openspec/changes/phase5-tracking/tasks.md); design decisions (pilot-first, no QVarCircuit backwards-compat, LSTMPPO+klinotaxis as first-class brain for M4/M5/M6) are recorded in that change's [proposal.md](../openspec/changes/phase5-tracking/proposal.md).

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
| M7 | NEAT topology evolution | LOW | 🔲 not started — reframed as a direct test of M5's architecture-asymmetry hypothesis (matched-capacity NEAT-vs-NEAT vs asymmetric NEAT-vs-MLP). Independent corroboration: Resendez Prado arXiv 2604.03565 identifies the same "transparent regime" suppression effect |
| M4.7 | Multi-task Baldwin retry | HIGH | 🔲 deferred — gated on M5 producing multi-task aggregation infrastructure or M5's secondary-Baldwin instrumentation coming back null with rising priority |
| M8 | Phase 5 synthesis logbook | — | ✅ complete — all five Phase 5 exit criteria verified MET (two with substrate-grounded STOP caveats). See [logbook 021](experiments/logbooks/021-phase5-synthesis.md) |

**How to orient**: read this tracker for the current milestone, then [tasks.md](../openspec/changes/phase5-tracking/tasks.md) for sub-task detail, then any active per-milestone OpenSpec change under `openspec/changes/` (archived changes live under `openspec/changes/archive/`). Open Phase 5 research questions are tracked in `tasks.md` under "Phase 5 Research Questions"; check there before assuming a Phase 5 design choice is settled.

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

- Phase 6 inherits M5 architecture-asymmetry + M6 substrate-redesign as open future-work directions, addressable through connectome-constrained + continuous-physics + quantum architectures rather than M7 NEAT.

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

#### The layered platform

Phase 6 is built as four layers (L0-L3) that together materialise the architecture-comparison sweep. L4 (biologically-plausible plasticity) is deferred to Phase 7 — Phase 6 stays on PPO-family learning rules so the layer stack remains tractable.

| Layer | What it is | Phase 6 commitment |
|---|---|---|
| **L0 — Connectome substrate** | Import *C. elegans* 302-neuron wiring (Cook et al. 2019 / OpenWorm c302 in NeuroML 2 format). Real synaptic adjacency. Defines the topology interface that pluggable brains conform to. | **MUST.** The headline claim doesn't exist without this. |
| **L1 — Architecture-as-plugin** | A clean `Brain` interface where every architecture family conforms. The comparison is one experimental sweep, not a per-architecture re-implementation. Plugin parity test: adding a new architecture ≤ 1 week of work. | **MUST.** Without this, "swap in another brain" is words, not code. |
| **L2 — Weight search (PPO et al.)** | Train weights on the connectome topology and on every comparison architecture. This is the *first closed-loop learning on the C. elegans connectome*. | **MUST.** Cheapest scientifically meaningful Phase 6 result. |
| **L3 — Topology search (NEAT-style)** | Search topologies unconstrained, compare to the real connectome's topology. Tests "is the wild-type connectome a local optimum?" The architecture-asymmetry question Phase 5 M5 diagnosed re-emerges here under matched capacity. | **MUST.** Without this, the optimal-vs-connectome comparison has no "optimum" to compare against. |
| **L4 — Plasticity / learning rules** | Biologically-plausible plasticity (STDP, neuromodulator-modulated three-factor STDP) on the connectome. The Nature-Neuroscience-tier claim. | **DEFERRED to Phase 7.** Substantial new code; clean L1 is a prerequisite. |

#### Behavioural scope: three behaviours on a common substrate

Phase 6 commits to the same three behaviours across every architecture in the comparison sweep:

- **Klinotaxis** — chemical gradient ascent via head-sweep modulation. Phase 4's klinotaxis sensing is the substrate.
- **Thermotaxis** — thermal gradient navigation. Phase 1's thermotaxis configurations carry forward.
- **Predator evasion** — escape from pursuit predators, integrating the corrected ASH/ADL contact-based nociception (see Realistic Sensory Physics below).

Aerotaxis (oxygen sensing), pheromone signalling, and multi-agent dynamics are *deferred* — including them pushes Phase 6 past 10 months and weakens the focused architecture-comparison framing. They re-enter scope in a future phase if the connectome story holds and additional behaviours become scientifically warranted.

#### Architecture families in the comparison sweep

The L1 plugin interface accommodates this curated set. The list is not "all 19 existing architectures" — Phase 6 picks the representatives that test the load-bearing questions and leaves historical variants in their Phase 0-3 logbooks.

| Family | Existing impl | Phase 6 role |
|---|---|---|
| **Connectome-constrained (302-neuron, Cook 2019)** | Not yet | **Focal architecture.** The wild-type topology with PPO-learned weights. The headline rank. |
| **MLP-PPO** | `MLPPPOBrain` | Strongest classical baseline (Phase 2 SOTA on foraging). |
| **LSTM / GRU-PPO** | `LSTMPPOBrain` | Strongest temporal baseline (Phase 3 reached 94% L500). |
| **Spiking (PPO-trained)** | `SpikingReinforceBrain` | Bridge to L4. STDP is the spiking-network plasticity rule; without spiking in the comparison, L4 has no L2 precursor on the same substrate. |
| **Reservoir** | `QRHBrain`, `CRHBrain` | Preserves Phase 2's pursuit-advantage finding (QRH +9.4pp). One row each; low marginal cost. |
| **Quantum** | `QVarCircuitBrain`, `QEF`, others | Phase 2's 300-session campaign is the baseline reference; Phase 6 confirms (or refines) its findings on continuous-physics tasks. |
| **Hybrid quantum-classical** | `HybridQuantum`, `HybridClassical` | Preserves Phase 2's SOTA finding (96.9% / 96.3%). One row to confirm at higher complexity. |
| **NEAT-evolved (topology + weights)** | Not yet (L3 deliverable) | The unconstrained-optimal baseline against which the connectome is ranked. |
| **Transformer / attention-based** | Not yet | ⭐ **MAY.** Optional addition if scope and engineering effort allow. Flagged so reviewers see it considered, not forgotten. |

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
| **2 (dynamic Fick's-law + adaptation) — Phase 6 target** | Source depletion when worms feed; source replenishment; decay terms for short-lived signals; signal-type-specific D values (food vs pheromone vs CO₂); paired with **log-concentration chemosensory adaptation kinetics** on AWC/AWA/ASE-style sensors | What the computational-chemotaxis field actually uses |
| 3 (multi-species + substrate) | Vector of chemical signals with cross-modal receptor overlap; substrate-varying diffusion; bacterial biofilm boundary layers | OpenWorm-level fidelity; specialist territory |

Rung 2 has two coupled components — environment dynamics AND chemosensory adaptation kinetics. They must be designed together; without log-concentration adaptation on the sensory side, the gradient realism is wasted (the brain sees a normalised scalar either way).

**Mechanosensation: corrected ASH/ADL contact-based nociception.** Phase 4's Logbook 011 surfaced that the current nociception model is biologically wrong — real *C. elegans* nociception is contact-based mechanosensation (ASH/ADL neurons), not chemosensory at distance. The corrected model is owed correctness work and lands in Phase 6's sensory-physics stack.

**Other sensors:** physical temperature fields with conduction; contact mechanics for mechanosensation with continuous bodies; realistic sensory ranges scaled to worm body length.

#### Architecture-comparison protocol

The architecture-comparison protocol is one experimental sweep across three dimensions: architecture family × behaviour × seed. The Phase 5 statistical bar carries forward — paired-seed Wilcoxon tests, bootstrap confidence intervals, n ≥ 4 seeds per condition with explicit power analysis when smaller. Phase 5's lag-matrix instrument and TEI fair-F0 paired comparison are the methodological templates.

The protocol explicitly answers the four primary Phase 6 research questions (connectome ranking; fitness landscape; architecture asymmetry under matched capacity; first closed-loop learning on the connectome) via the same sweep, not via separate experiments.

#### Built-in real-worm validation

Phase 6 validates *at least one* model output quantitatively against published real-worm data, as a Phase 6 exit criterion rather than a Phase 7 collaboration deliverable. Three data sources are concrete candidates:

- **Chemotaxis indices** (Bargmann lab + others, multiple published datasets).
- **Escape latencies** (mechanosensation literature). The corrected ASH/ADL nociception is the natural validation pair.
- **Whole-brain Ca²⁺ imaging correlation matrices** (Kavli / Janelia open data; BAAIWorm achieves 92.4% fidelity to this in the literature).

Internal validation against public data is required at Phase 6 close; external lab collaboration (Phase 7) is additive, not a precondition.

#### Phase 6 exit criteria

**Required (MUST):**

- ✅ L0 connectome substrate operational: ≥ 1 real connectome dataset (Cook 2019 or OpenWorm c302) imported, with documented topology and synaptic-weight provenance.
- ✅ L1 architecture-plugin interface accommodates the curated MUST set above. Adding a 9th architecture is ≤ 1 week of work (the plugin-parity test).
- ✅ L2 weight-search results across all MUST architectures on all three behaviours, at the Phase 5 statistical bar (paired-seed, bootstrap CIs, n ≥ 4 seeds per condition).
- ✅ L3 NEAT topology-search results comparing the wild-type connectome to NEAT-evolved topologies on at least one behaviour, with the lag-matrix or equivalent discriminative instrument.
- ✅ Rung 2 chemical gradients (dynamic Fick's-law + source dynamics + signal-type diffusion coefficients) operational, paired with log-concentration chemosensory adaptation kinetics.
- ✅ Corrected ASH/ADL contact-based nociception operational.
- ✅ At least one model output quantitatively validated against published real-worm data.

**Optional (MAY) — not phase exit criteria:**

- ⭐ Transformer / attention-based architecture added to the comparison sweep.
- ⭐ Connectome-learning platform paper drafted.
- ⭐ Connectome fitness-landscape science paper drafted.
- ⭐ Reproducibility artefacts (Docker, evaluation scripts) updated to current Phase 6 state.

Papers and external collaboration are explicitly optional — the project may pursue them when evidence and context justify, but pursuing them is not a precondition for closing Phase 6 or advancing to Phase 7.

#### Mid-phase decision gates

Phase 6 is long enough (~6-10 months) that mid-phase gates matter. Each gate produces a written go/no-go decision in the relevant OpenSpec change, not just an implicit continuation — the same discipline Phase 5 used.

- **Gate 1 (month ~2): L0 import working?** Connectome substrate loaded, validated, and basic-MLP-PPO baseline trainable on it. If not, trigger the L0 hand-curated-subset pivot (see Risk-mitigation below).
- **Gate 2 (month ~4-5): L1 plugin parity achieved?** Adding a new architecture demonstrably ≤ 1 week. If not, trigger the L1 refactor pivot.
- **Gate 3 (month ~7-8): L2 results across architectures?** Weight-search results across MUST architectures and all three behaviours in hand. If not, trigger the Phase 6a / Phase 6b sub-phase split.

The hard phase boundary between Phase 5 and Phase 6 protects the narrative arc (no Phase 6 work begins until Phase 5 is synthesised); the mid-phase gates protect the execution (fail-fast at the architecture and substrate level, not at the phase level). Both are intentional.

#### Risk-mitigation: failure modes and pivots

| Failure mode | Trigger | Pivot |
|---|---|---|
| **L0 c302 import takes > 2 months** | OpenWorm c302 / NeuroML integration proves harder than expected (format incompatibility, missing metadata, unclear synaptic-weight provenance) | Drop to a hand-curated subset of the Cook 2019 connectome — sensory-interneuron-motor subgraph for the three target behaviours, ~50-100 neurons. Document the subset choice; defer full 302-neuron import to Phase 7. The platform claim survives ("learning on a real *C. elegans* subgraph"); the comparative-completeness claim weakens. |
| **L1 architecture-plugin interface proves messy** | Multiple architecture families need bespoke plumbing; the interface accumulates per-architecture branches; "swap in another brain" stops being one-line | Pause architecture-sweep work; spend 2-4 weeks refactoring L1 toward genuine plugin parity. Better to delay L2/L3 results than to ship a "platform" that isn't one. |
| **L2 PPO-on-connectome fails to learn** | After reasonable hyperparameter search, no architecture family reaches the Phase 0-3 baselines on any of the three behaviours | Diagnostic sequence: (a) is the connectome topology dense enough to support gradient flow? (b) is the continuous action head the bottleneck? (c) is reward shaping the issue? If none resolve: the finding is "learning on the real connectome with PPO requires further substrate work" — itself a publishable negative result and a Phase 7 prerequisite. |
| **L3 NEAT produces no separation from connectome** | NEAT-evolved topologies and the wild-type connectome converge to indistinguishable performance | This is itself a finding: "the connectome is competitive with evolved topologies on these behaviours." The optimal-primary framing weakens; the connectome-primary framing strengthens. Acceptable outcome — pivot the headline framing if it lands. |
| **Phase 6 overshoots 10 months** | Scope creep, L1 refactor, multi-architecture debugging push timeline past 12 months | Trigger a sub-phase split: Phase 6a (L0+L1+L2) ships first as a standalone result; Phase 6b (L3 + NEAT topology search) becomes a follow-on. Pre-commit the split criterion: if L2 isn't producing publishable architecture comparisons by month 6, split the phase. |

The general principle: **fail-fast at the architecture and substrate level, not at the phase level.** A Phase 6 that fails a mid-phase gate produces a documented pivot, not a silent slide past the gate.

#### Compute / infrastructure planning

Phase 5 ran on CPU. Phase 6's L3 (NEAT topology search across many candidates × multiple architectures × multiple behaviours × multiple seeds) is substantially more compute-intensive. Three tiers should be considered explicitly, in order of preference:

1. **CPU + targeted parallelism (no new infra).** Sufficient for L0, L1, L2 on small-to-medium architecture sweeps. Probably not sufficient for full L3 NEAT search at population sizes the field uses (~1000+ genomes × generations).
2. **GPU (single or small cluster, including consumer-class cards).** The realistic baseline. Sufficient for L3 with TensorNEAT vectorisation (~500× speedup over neat-python via JAX/vmap on GPU is documented in the field); sufficient for full L2 sweeps across all architectures.
3. **HPC allocation (NERSC, JURECA, or similar).** Optional, not required for Phase 6. Pursue an allocation if a specific Phase 6 or Phase 7 need justifies it (e.g., Phase 7 pacificus comparative work, neuromorphic-hardware deployment). National HPC centres regularly fund computational-neuroscience time on this class of work — realistic, not aspirational, but not a precondition.

The roadmap encodes GPU as the realistic baseline. The L3 implementation choices (TensorNEAT, JAX vmap, batched fitness evaluation) flow from that.

#### Go / No-go decision

- **GO if**: L0+L1+L2 ship and the three mid-phase gates pass with results in hand. L3 either ships within Phase 6, or splits into Phase 6b under the sub-phase pivot.
- **PIVOT-narrative if**: L3 shows the connectome competitive-but-not-dominant with evolved alternatives. The headline framing shifts toward connectome-primary; the platform claim is unchanged.
- **PIVOT-scope if**: a mid-phase gate fails — execute the relevant Risk-mitigation row above and document the pivot in the OpenSpec change.
- **STOP if**: L0 connectome import is fundamentally infeasible after the hand-curated-subset pivot has also failed — at which point the diagnosis itself is the Phase 6 deliverable, and Phase 7 inherits a substrate-engineering question rather than a plasticity question.

______________________________________________________________________

### Phase 7: Deepen — Plasticity & Cross-Species Transfer

**Goal**: Add the L4 plasticity layer to the architecture-comparison platform (biologically-plausible learning rules on the connectome) and extend the comparison to a second nematode species (*P. pacificus*) using the Cook et al. 2025 connectome data. The headline framing is *deepen, don't broaden* — Phase 7 doesn't add new behaviours or new sensory modalities; it deepens the platform's biological-plausibility and species-coverage along the same three Phase 6 behaviours.

**Aspirational timeline**: ~8-12 months from Phase 6 close.

#### Required deliverables (MUST)

1. **L4 Plasticity Layer**

   Phase 6's L2 weight search uses PPO-family gradient learning. Phase 7's L4 adds biologically-plausible plasticity on the same connectome substrate, materialised in four sub-deliverables that must be designed together — STDP alone reproduces synaptic-level plasticity but misses the circuit-level behavioural plasticity the *C. elegans* literature is built around.

   - **Vanilla STDP / Hebbian rules** on the connectome topology (~2-4 months software-only).
   - **Diffusible-signal layer** modelling at least serotonin and dopamine concentrations as a function of internal state (food detection, sensory input, satiety where available).
   - **Receptor-class metadata** on the connectome neurons — which neurons express which receptor classes. CeNGEN gene-expression data is the reference source. *C. elegans* has documented behavioural roles for 5 serotonin, 4 dopamine, 4 tyramine, and 3 octopamine receptor classes per WormBook.
   - **Modulated STDP rules** — three-factor learning where the third factor is neuromodulator concentration. Receptor-class metadata determines which synapses see which modulators.

   Total scope: ~4-6 months software-only for vanilla STDP, ~6-8 months for full neuromodulator grounding. Loihi 2 / SpiNNaker 2 neuromorphic hardware deployment is a credible additional target if scope allows; the software-only path is fully sufficient for the headline claim.

2. ***P. pacificus* Connectome Transfer**

   Cook et al. 2025 (*Science*) published the full *P. pacificus* connectome with neuronal adjacency and connectivity data, accompanied by code (`stevenjcook/cook_et_al_2025_pristionchus`). Phase 7 uses this to ask: *do learned and evolved architectures from C. elegans transfer to P. pacificus's different connectome on shared tasks?*

   - Same three behaviours as Phase 6 (klinotaxis, thermotaxis, predator evasion). Species is the independent variable — clean comparison.
   - Import and validate *P. pacificus* connectome through the same L0 / L1 pipeline Phase 6 builds.
   - Evaluate architecture-comparison sweep on the new species; compare results to Phase 6's *C. elegans* baseline.
   - **MAY**: extend with one *pacificus*-distinctive behaviour (e.g., predatory-mouth-form switching, which *C. elegans* lacks) if connectome data and remaining scope support it.

   Transferring to *C. briggsae* is **not** a Phase 7 deliverable. As of project planning, *C. briggsae* lacks a high-quality published connectome (chromosome-level genomes only); briggsae transfer becomes scoped only when reference data appears.

#### Optional deliverables (MAY)

These are not Phase 7 exit criteria. The project may pursue any combination when evidence and context justify, but Phase 7 closes whether or not they ship.

- **Biological-validation collaboration.** Engage one *C. elegans* lab (Bargmann, Sengupta, Horvitz, Lockery, or a smaller group with relevant Ca²⁺ recording data) to test ≥ 1 model prediction against real worm data. At most one partnership if pursued — collaboration overhead grows non-linearly with multiple labs.
- **Paper drafts and submissions.** A platform paper, a connectome-learning paper, and a fitness-landscape paper are all plausibly publishable from Phase 6+7 results; the project may write them when evidence justifies. No specific venues are promised.
- **Neuromorphic deployment.** Loihi 2 / SpiNNaker 2 implementation of the L4 plasticity layer — credible exotic-hardware angle for STDP-on-connectome work.
- **Reproducibility artefacts updated** to the Phase 7 platform state.

#### Phase 7 exit criteria

**Required (MUST):**

- ✅ L4 plasticity layer operational: vanilla STDP/Hebbian rules + diffusible-signal layer + receptor-class metadata + modulated STDP, all on the connectome substrate, with results across the Phase 6 architecture-family set at the Phase 5/6 statistical bar.
- ✅ *P. pacificus* connectome imported through L0 / L1; architecture-comparison sweep on the same three behaviours; quantitative comparison to *C. elegans* Phase 6 baseline.

**Optional (MAY):**

- ⭐ Biological-validation collaboration completed; ≥ 1 model prediction tested against published or partner-lab worm data.
- ⭐ Paper drafts in flight.
- ⭐ Neuromorphic deployment demonstrated.
- ⭐ Reproducibility artefacts current.

#### Risk-mitigation: failure modes and pivots

| Failure mode | Trigger | Pivot |
|---|---|---|
| **L4 implementation overshoots ~6 months** | STDP + neuromodulator grounding more complex than estimated; receptor-class metadata harder to integrate; modulated rules harder to debug than vanilla | **Split Phase 7 into 7a / 7b.** Phase 7a ships L4 standalone as the headline deliverable; Phase 7b carries pacificus transfer + optional MAY items as the follow-on. The decision criterion: if L4-on-connectome is not learning demonstrably by month 6, split. Phase 7 closes whether it lands as one phase or two. |
| **Pacificus connectome data integration is problematic** | Cook et al. 2025 data is incomplete, format-incompatible, or has unclear synaptic-weight provenance once handled in detail | Sub-deliverable: produce the hand-curated *P. pacificus* subset that does work, document the gap, and ship transfer results on that subset. The cross-species transfer claim survives in restricted form. |
| **L4 plasticity fails to learn on the connectome** | Modulated STDP doesn't reach the L2 PPO baselines on the three behaviours after reasonable hyperparameter search | The finding itself is publishable — *"biologically-plausible plasticity on the C. elegans connectome requires further substrate work or different rule families."* Phase 7 closes with the negative result; future work picks up rule-family alternatives (e.g., reward-modulated Hebbian without spike timing). |

#### Go / No-go decision

- **GO if**: L4 lands on the connectome with results in hand, and pacificus transfer ships (or is well underway under the 7b pivot).
- **PIVOT-split if**: L4 overshoots month-6 milestone — execute Phase 7a / 7b split per the Risk-mitigation table above.
- **STOP if**: Both L4 implementation and pacificus transfer are infeasible at the substrate level — at which point the diagnosis itself is the Phase 7 deliverable, and follow-on work picks up alternative plasticity rule families or alternative connectome data sources.

______________________________________________________________________

## Architecture-Comparison Protocol

Phase 2's 300-session quantum architecture campaign — covering 15 quantum and hybrid variants against matched-capacity classical baselines — established that grid-world complexity is below the threshold for quantum advantage on every variant tested. Quantum is therefore *one architecture family among many* in the project's comparison sweep, not a separate goal or a separate phase.

The architecture-comparison protocol consolidates this into a single mechanism, applied at Phase 6 and again at Phase 7:

1. Run the full architecture-family sweep (MLP, recurrent, spiking, reservoir, quantum, hybrid, connectome-constrained, NEAT-evolved) on the three target behaviours under the current substrate (Rung 2 chemical gradients, continuous 2D, corrected nociception at Phase 6; same plus L4 plasticity at Phase 7).
2. Report results paired-seed at the Phase 5 statistical bar (Wilcoxon, bootstrap CIs, n ≥ 4 per condition).
3. The 300-session campaign's results carry forward as **baseline reference data** — quantum families are not re-evaluated from scratch unless the substrate change (continuous physics, connectome topology, neuromodulator-modulated STDP) plausibly changes the comparison.
4. If a future phase introduces a complexity dimension that *did* clear a quantum-advantage threshold in the Phase 2 campaign (e.g., long non-Markovian dependencies for QRH), revisit that family's evaluation at that phase. This is opportunistic, not scheduled.

There is no separate "quantum checkpoint" gate, and no "if classical drops below 70% then launch a quantum campaign v2." The optionality of revisiting quantum at higher complexity is preserved through the architecture-family sweep itself.

______________________________________________________________________

## Complexity Dashboard

A snapshot of where the platform sits across five complexity dimensions, tracked across phases. Each dimension matters because the architecture comparison's interpretation depends on it (high-dimensional + non-Markovian observations test different architectural assumptions than low-dimensional Markovian ones), not because any one dimension is a quantum-advantage gate.

| Dimension | Phase 0-2 | Phase 3 | Phase 4 | Phase 5 | Phase 6 target | Phase 7 target |
|---|---|---|---|---|---|---|
| **Input dimensionality** | 2-9D | ~15-20D | similar | similar | > 50D (continuous + sensory-physics) | similar; cross-species sensors |
| **Partial observability** | Viewport only | STAM temporal memory | Multi-agent fog-of-war | Generational uncertainty | Realistic sensing range + connectome | Adds modulator-state observability |
| **Multi-agent** | 1 | 1 | 5-10 | Single-agent populations | 1 (multi-agent deferred) | 1 (cross-species, not multi-agent) |
| **Temporal horizon** | Memoryless | STAM (~minutes) | STAM + social memory | Cross-generational | Full non-Markovian + plasticity-shaped | STDP-modulated long-horizon |
| **Classical ceiling on hardest task** | 94-98% (PPO foraging) | 94% (Mode A L500) | partially measured | n/a | TBD on continuous + connectome | TBD with plasticity |

Update protocol: after each phase's results are in, this dashboard records *measured* values for that phase's columns and pencils-in targets for the next. The dashboard documents the substrate's complexity profile, not a quantum-advantage threshold; quantum architectures appear in the architecture-family sweep regardless of where the substrate sits on any one row.

______________________________________________________________________

## Biological Fidelity

### Current snapshot

Where the platform sits across five fidelity dimensions, by phase. This view answers "what does the substrate actually look like today, and what is each forward phase deepening?" — the question the optimal-primary framing makes load-bearing.

| Dimension | Phase 0-4 | Phase 5 | Phase 6 target | Phase 7 target | Future |
|---|---|---|---|---|---|
| **Connectome topology** | None (MLP/LSTM/etc.) | None | 302-neuron Cook 2019 / OpenWorm c302 | + *P. pacificus* (Cook 2025) | + briggsae (gated on data) |
| **Sensory transduction** | Spatial gradient lookups | + klinotaxis head-sweep | Rung 2 dynamic Fick's-law + log-concentration adaptation kinetics | unchanged | + multi-species receptors |
| **Plasticity rules** | PPO / DQN / Reinforce | + Lamarckian inheritance, hyperparameter evolution | L2 PPO + L3 NEAT topology search | + L4 STDP + neuromodulator-modulated STDP | + reward-modulated Hebbian; alt rule families |
| **Body mechanics** | Discrete 4-action grid | Discrete | Continuous 2D + spatial scales; OpenWorm Sibernetic interop if needed | unchanged | Native undulation / omega turns / pirouettes; 3D |
| **Environment** | Grid; static gradients | + multi-agent, pheromones | Rung 2 dynamic gradients; corrected ASH/ADL contact nociception | unchanged | Bacterial lawns; energy/metabolic state; population dynamics |

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
| **6** | Future | 3D substrate (soil mechanics, fluid dynamics), native body mechanics, energy/metabolic model, population dynamics, life-cycle simulation, briggsae and other species |

Level 5 + Level 5+ together represent the project's target state. Level 6 is aspirational — see Future Directions for the technology selection and gated dependencies.

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
- **L4 plasticity overshoots Phase 7 month-6 milestone** → Phase 7a / 7b split. 7a ships L4 standalone; 7b carries pacificus transfer + optional MAY items. See Phase 7 Risk-mitigation.
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
| **Cook et al. 2025 pacificus data quality** (Phase 7) | Data incomplete or format-incompatible on detailed handling | Hand-curated *P. pacificus* subset; cross-species transfer claim survives on the subset. |
| **GPU / HPC access for L3 NEAT** (Phase 6) | TensorNEAT-scale population search needs GPU; HPC allocation overhead | GPU is realistic baseline (consumer-class cards sufficient with TensorNEAT vectorisation). HPC is optional, pursue only when a specific Phase 7 stretch need justifies. |
| **Neuromorphic deployment** (Phase 7 MAY) | Loihi 2 / SpiNNaker 2 access non-trivial | Software-only L4 path is fully sufficient for the headline claim. Neuromorphic is a stretch / publication enhancement, not a requirement. |

### Adaptive execution

- **Per-milestone logbooks** with audit findings, statistical evidence, and explicit GO/PIVOT/STOP verdicts. Phase 5's logbooks 012-021 are the template.
- **OpenSpec change per non-trivial milestone**, archived on close (proposal → design → tasks → implementation → verification → archive).
- **Mid-phase decision gates** for long phases (Phase 6's three gates; Phase 7's L4-month-6 split criterion).
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
- **Phase 7**: deepen biological validation with the L4 plasticity layer (does modulated STDP reproduce documented learning dynamics?) and with *P. pacificus* cross-species transfer. External lab partnership is optional (MAY); internal validation against published data is sufficient for the phase to close.

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

### 2. Architecture Comparison

**Definition**: how rigorously the platform ranks the connectome against unconstrained, evolved, quantum, and hybrid architectures.

**Metrics**:

- Architecture-family coverage in the Phase 6+ sweep (MUST set per Phase 6's architecture-families table).
- Statistical rigor: paired-seed Wilcoxon, bootstrap CIs, n ≥ 4 per condition.
- Plugin-parity test: adding a new architecture to L1 demonstrably ≤ 1 week.
- Lag-matrix or equivalent discriminative instrument for matched-capacity comparisons.

**Targets**:

- **Phase 6**: L2 weight-search results across the MUST architecture-family set on all three behaviours, at the Phase 5 statistical bar. L3 NEAT topology-search results comparing wild-type connectome to NEAT-evolved on ≥ 1 behaviour.
- **Phase 7**: same sweep with L4 plasticity added; cross-species transfer measured on the same architecture-family set.

### 3. Substrate Coverage

**Definition**: which behaviours, sensors, and substrate dimensions the platform supports at phase close.

**Metrics**:

- Behaviours operational on the connectome substrate (Phase 6 commits to three; Phase 7 retains the same three on a second species).
- Sensory-physics fidelity rung achieved (Phase 6 commits to Rung 2: dynamic Fick's-law + log-concentration chemosensory adaptation kinetics).
- Connectome species supported (Phase 6: *C. elegans*; Phase 7: + *P. pacificus*).

**Targets**:

- **Phase 6**: three behaviours × MUST architectures × continuous 2D + Rung 2 + corrected ASH/ADL nociception.
- **Phase 7**: same three behaviours × MUST architectures + L4 × *C. elegans* and *P. pacificus*.

### 4. Sample efficiency and convergence

**Definition**: how efficiently the architectures in the comparison reach the Phase 6+ behavioural targets.

**Metrics**:

- Episodes to convergence per architecture family.
- Generational convergence speed (evolutionary regimes).
- Phase 5's Lamarckian-inheritance speed gate (+5.25 generations) carries forward as the methodological template for evolutionary efficiency comparisons.

**Targets**:

- **Phase 6**: report convergence statistics per architecture family on all three behaviours; identify whether the connectome family shows characteristic sample-efficiency differences from unconstrained alternatives.
- **Phase 7**: compare L4 plasticity sample efficiency against L2 PPO on the same connectome substrate.

### 5. Robustness

**Definition**: performance under noise, missing sensors, or circuit ablation.

**Metrics**:

- Sensor dropout robustness (10%, 20%, 50% sensors disabled).
- Gaussian observation noise robustness.
- Circuit-ablation graceful degradation (Phase 6+ connectome architectures — matches biological lesion data).

**Targets**:

- **Phase 6**: connectome architecture demonstrates graceful degradation under circuit ablation comparable to documented biological lesion studies (qualitative match acceptable; quantitative match is a stretch).
- **Phase 7**: with L4 plasticity, test whether ablation-then-relearning approximates documented *C. elegans* recovery dynamics.

______________________________________________________________________

## Success Levels

Three levels of success, each representing a coherent and publishable scientific contribution. Higher levels do not invalidate lower ones.

### Minimum viable success

The platform exists and produces a defensible architecture-comparison result.

- **L0 connectome substrate operational** (at least the hand-curated subset under the L0 fallback pivot).
- **L1 architecture-plugin interface** at plugin-parity (adding a new architecture ≤ 1 week).
- **L2 weight-search results** on ≥ 1 behaviour across ≥ 4 architectures of the MUST set.
- **≥ 1 model output validated against published real-worm data.**
- **Phase 5's STOP findings preserved** as documented substrate-grounded diagnoses with reusable methodology.

### Target success

Phase 6-balanced ships cleanly; the first headline-positive result lands.

- All Phase 6 MUST exit criteria met: L0+L1+L2+L3 operational across the full MUST architecture-family set, three behaviours, Rung 2 gradients + adaptation kinetics, corrected ASH/ADL nociception, real-worm validation.
- A defensible answer to "is the wild-type connectome a local optimum?" lands, either as "yes, dominant" (connectome-primary headline) or "competitive but not dominant" (optimal-primary headline). Either framing is a contribution; the data picks the framing.
- Architecture-asymmetry question carried from Phase 5 M5 is tested under matched capacity (NEAT-vs-NEAT, connectome-vs-NEAT).
- Phase 7 has begun and L4 is on a credible trajectory.

### Stretch success

Phase 7 closes cleanly and external visibility follows.

- L4 plasticity (STDP + neuromodulator-modulated three-factor STDP) operational on the connectome with results across the architecture-family set.
- *P. pacificus* cross-species transfer ships; behavioural transfer is measurable across species; the connectome's role in transfer is characterised.
- At least one of the optional MAY items lands: biological-validation collaboration with a *C. elegans* lab; ≥ 1 paper drafted (platform paper, connectome-learning paper, or fitness-landscape paper); neuromorphic deployment demonstrated.
- Reproducibility artefacts updated to current platform state.

Publication, external collaboration, and community-launch metrics are not codified as success-level requirements — the project pursues them when evidence and context justify, not on a phase-locked schedule.

______________________________________________________________________

## Relationship to External Projects

### OpenWorm

**Focus**: cellular biophysics, muscle dynamics, 3D body simulation, *C. elegans* digital twin. Includes c302 (NeuroML-format connectome model) and Sibernetic (SPH-based body physics).

**Relationship**: **substrate dependency** at Phase 6 L0 + **interop boundary** for body-mechanics fidelity if needed.

OpenWorm has the connectome data and the body physics this project does not rebuild. The platform imports c302 as the canonical *C. elegans* topology source at Phase 6 L0 — OpenWorm is upstream infrastructure, not just a complementary project. If behavioural-fidelity claims later require native body mechanics, the platform interoperates with Sibernetic at the c302 boundary rather than re-implementing undulatory locomotion.

The Leeds physics group (Boyle, Bryden, Cohen) is complementary in the same way — best-in-class undulatory locomotion modelling, no learning or evolution. The platform interoperates with this body of work; it does not compete with it.

### Izquierdo & Beer (klinotaxis arc, Indiana)

**Focus**: evolved minimal circuits for klinotaxis; ensemble-of-models integrating connectome data; information-flow analysis through evolved circuits.

**Relationship**: **closest conceptual neighbour, deliberate methodological divergence**.

Their evolved minimal circuits are not the real connectome — they are abstract neural networks that behave like worms. The platform's L3 NEAT topology search produces directly comparable "what would evolution find?" results; placing the wild-type connectome (Phase 6 L0) and NEAT-evolved topologies (L3) in the same comparison sweep is the methodological extension. Their information-flow analysis tooling is a natural future-direction interop target.

### Cook et al. — *P. pacificus* connectome (Science, 2025)

**Focus**: full *P. pacificus* connectome from two adult hermaphrodite heads; neuronal adjacency + connectivity data; published code (`stevenjcook/cook_et_al_2025_pristionchus`).

**Relationship**: **substrate dependency** at Phase 7 cross-species transfer.

The 2025 data makes *C. elegans* → *P. pacificus* transfer feasible today with published reference data. *C. briggsae*, by contrast, lacks a high-quality published connectome as of project planning, so the briggsae direction stays in Future Directions until reference data appears.

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

### NematodeBench

The benchmark infrastructure (Docker images, evaluation scripts, leaderboard scaffolding, the 300-session campaign as reference data) persists as internal tooling — useful for reproducibility and for the architecture-comparison protocol itself. A public-facing community launch (external submissions, external leaderboard) is moved to Future Directions; benchmarks crystallise mature communities rather than bootstrap them.

______________________________________________________________________

## Future Directions

Beyond the current roadmap phases, potential research directions include:

### 1. Organism Transfer

- **Drosophila (fruit fly)**: ~100K neurons, similar sensory tasks, well-studied connectome (partial)
- **Zebrafish larvae**: ~100K neurons, visual predator avoidance, schooling behavior
- **Approach**: Transfer learned principles and architectural insights from C. elegans deep dive. Proof-of-concept, not full simulation.

### 2. Three-Dimensional Physics

- Full 3D substrate simulation (soil mechanics, agar surface, burrowing)
- Fluid dynamics for movement in aqueous media
- OpenWorm integration for body physics
- Significantly harder computationally but most biologically realistic

### 3. Clinical Applications

- Neurological disease models (C. elegans Alzheimer's, Parkinson's analogs)
- Drug discovery: Compound screening via behavioral phenotyping
- Aging studies: Age-dependent behavioral changes (C. elegans lifespan ~2-3 weeks)

### 4. Hybrid Behavioral-Cellular Models

- Combine behavioral abstraction with selective cellular models
- Example: RL-trained behavioral foraging + detailed AFD neuron biophysics for thermotaxis
- Cross-validation with OpenWorm's cellular-level predictions

### 5. Quantum Advantage for General AI

- Extract insights: When does quantum help for RL tasks?
- Apply findings to broader quantum ML research
- Contribute to theoretical understanding of quantum computational advantage

### 6. Ecological Co-Evolution

- Add state to predators (HP, satiety, death-by-starvation, kill-replenishes-energy) so the Phase 5 frozen-weight Red Queen substrate gains coupled population dynamics
- Lotka-Volterra-style oscillations: study whether predator-prey populations stabilise, oscillate, or collapse under learned policies
- Phase 5's `PredatorBrain` Protocol (M1) and `MLPPPOPredatorBrain` (M5) already supply the policy substrate; the new work is env-side state machinery + reward shaping
- Selection pressure shifts from "maximise kill-rate" to "maintain a viable population against prey escape velocity" — natural follow-up if Phase 5's co-evolution verdict motivates richer eco-dynamics

______________________________________________________________________

## Technical Debt & Maintenance

### Resolved (Phases 0-2)

- ~~QQLearningBrain completion~~ — Evaluated, not competitive; deprioritised
- ~~MLPReinforceBrain loss bug~~ — Investigated and documented
- ~~Grid size hardcoding~~ — Fixed
- ~~Statistical analysis framework~~ — Implemented (confidence intervals, significance tests)

### Active (Address by Phase 3)

1. **Sensory input refactoring** — Current spatial gradient inputs need parallel temporal derivative infrastructure
2. **Memory buffer architecture** — Design efficient STAM buffers compatible with all brain architectures
3. **Visualization improvements** — Gradient flow viz, spike raster plots, attention maps
4. **Documentation** — API documentation, tutorials, architecture guides (prerequisite for Phase 7)

### Lower Priority (Address as needed)

5. **Code quality** — Address remaining Ruff/Pyright warnings, increase test coverage
6. **Configuration system** — Hyperparameter search, experiment templates
7. **Performance profiling** — Optimization for multi-agent and continuous physics workloads

______________________________________________________________________

## Conclusion

This roadmap charts a milestone-based adaptive path from strong completed foundations (Phases 0-2) through progressively increasing biological fidelity toward the most complete C. elegans behavioral simulation available, while systematically testing quantum advantage hypotheses at each complexity milestone.

The dual-goal approach ensures scientific value regardless of quantum outcomes:

1. **If quantum shows advantage at higher complexity**: Groundbreaking result demonstrating that biological fidelity creates the conditions for quantum computational advantage — a new research direction linking quantum computing to neuroscience.

2. **If quantum shows no advantage**: The simulation itself is a major contribution to computational neuroscience, and the comprehensive characterisation of complexity thresholds (from 300+ sessions at grid-world complexity through enriched environments) provides definitive guidance for the quantum ML field.

The project's unique position — the only effort combining real connectome topology + reinforcement learning + quantum architecture comparison, grounded in the only organism with a fully mapped nervous system — provides unparalleled opportunities for breakthrough discoveries.

Key principles:

1. **Biological fidelity drives complexity**: Each phase makes the simulation more realistic AND creates harder computational problems
2. **Quantum checkpoints, not quantum phases**: Re-evaluation is triggered by measured complexity milestones
3. **C. elegans deep dive**: Stay with the only fully-mapped organism and go deeper than anyone else
4. **Community and validation**: External collaboration and open-source adoption amplify impact
5. **Adaptive execution**: Every phase has go/no-go gates; negative results are valuable when rigorous

By the completion of Phase 8, this project will have either demonstrated quantum advantage on biologically-grounded tasks or provided the most comprehensive negative result in quantum RL — along with the deepest C. elegans behavioral simulation, connectome-constrained architectures, and universal computational principles extracted from a uniquely tractable biological system.
