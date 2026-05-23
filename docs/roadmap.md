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
    - [Phase 2: Architecture Analysis & Standardization](#phase-2-architecture-analysis--standardization-substantially-complete) (SUBSTANTIALLY COMPLETE)
    - [Phase 3: Temporal Sensing & Memory](#phase-3-temporal-sensing--memory)
    - [Phase 4: Multi-Agent Complexity](#phase-4-multi-agent-complexity)
    - [Phase 5: Evolution & Adaptation](#phase-5-evolution--adaptation)
    - [Phase 6: Continuous Physics & Connectome](#phase-6-continuous-physics--connectome)
    - [Phase 7: Community, Validation & Publication](#phase-7-community-validation--publication)
    - [Phase 8: Integration & Comprehensive Evaluation](#phase-8-integration--comprehensive-evaluation)
05. [Quantum Re-evaluation Checkpoints](#quantum-re-evaluation-checkpoints)
06. [Complexity Dashboard](#complexity-dashboard)
07. [Biological Fidelity Ladder](#biological-fidelity-ladder)
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

### Phase 2: Architecture Analysis & Standardization (SUBSTANTIALLY COMPLETE)

**Status**: ✅ 300-session quantum architecture campaign complete. Carries forward as baseline reference in Phase 6's architecture-comparison protocol.

See [Current State — Phase 2](#phase-2--architecture-analysis) for the campaign summary; full results in [Logbook 008](experiments/logbooks/008-quantum-brain-evaluation.md). Key outcomes:

- 15 architecture variants systematically evaluated
- Strategic conclusion: environment complexity below quantum advantage thresholds
- Brain naming migration complete
- Statistical framework operational
- Clear complexity thresholds identified for quantum re-evaluation

**Remaining items** (moved to later phases):

- Interpretability framework → Phase 7 (publication readiness)
- Mechanism discovery protocol → Phase 7
- First biological prediction tested → Phase 7 (external collaboration)

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

### Phase 6: Continuous Physics & Connectome

**Goal**: Transition from grid-world to continuous 2D physics with realistic C. elegans locomotion, and introduce connectome-constrained architectures using the real 302-neuron wiring diagram. This is the largest single fidelity jump in the roadmap and creates the conditions for a major quantum re-evaluation.

**Aspirational timeline**: Q1-Q3 2027

#### Background

**Continuous physics**: C. elegans moves via sinusoidal body undulations, with a rich locomotion repertoire: forward crawling, reversals, omega turns (deep ventral bends reorienting 180°), pirouettes, and speed modulation. Current discrete 4-direction grid movement captures none of this. Continuous 2D dramatically increases both action space (speed + turning angle) and state space (continuous coordinates + heading + velocity).

**Connectome**: C. elegans has the only fully mapped connectome of any organism — 302 neurons connected by ~7,000 chemical synapses and ~900 gap junctions (Cook et al. 2019). This is the uniquely tractable advantage of choosing C. elegans. Using the real wiring diagram to constrain network architecture lets us ask: "Does biology's wiring learn better than arbitrary architectures?" and "Do quantum circuits on the real topology outperform classical ones?"

#### Deliverables

1. **Continuous 2D Environment** [CRITICAL]

   - Replace discrete grid with continuous 2D coordinates
   - Realistic C. elegans locomotion: sinusoidal crawling, reversals, omega turns, pirouettes
   - Continuous action space: speed (0 to max) + turning angle (-π to π)
   - Realistic spatial scales: ~1mm worm body on cm-scale plates
   - Physics: basic 2D kinematics, optional viscous medium effects

2. **Realistic Sensory Physics**

   - Diffusion-based chemical gradients (Fick's law, not superposition approximation)
   - Physical temperature fields with conduction
   - Contact mechanics for mechanosensation (collision detection with continuous bodies)
   - Realistic sensory ranges scaled to worm body length

3. **Full 302-Neuron Connectome-Constrained Architecture** [CORE DELIVERABLE]

   - Import C. elegans wiring diagram (Cook et al. 2019 / WormAtlas)
   - Build network where connections exist only where real synapses exist
   - Weights are learned via RL; topology is biologically fixed
   - Functional circuit modules: chemotaxis (AWC→AIY→RIB→motor), thermotaxis (AFD→AIY→AIZ→RIA), escape (ASH→AVA→motor)
   - Compare connectome-constrained vs. unconstrained architectures on identical tasks
   - Ablation: remove specific circuits and measure behavioral impact (matches biological lesion studies)

4. **Architecture Adaptation for Continuous Control**

   - Existing brain architectures (MLPPPOBrain, QVarCircuitBrain, HybridQuantum, QRH, etc.) were designed for discrete 4-action grid-worlds
   - Continuous action space (speed + turning angle) requires actor-critic variants with continuous action heads (e.g., Gaussian policy for PPO, continuous-output quantum circuits)
   - Adaptation strategy: extend PPO with continuous action head first (well-understood), then adapt quantum architectures
   - Benchmark discrete-trained vs. continuous-native architectures to quantify the impact of action space expansion

5. **Connectome + Quantum** [RESEARCH]

   - Build quantum circuit architectures (QSNN, variational) whose topology mirrors the real connectome
   - Test whether biologically-constrained quantum circuits outperform:
     - Unconstrained quantum circuits (arbitrary topology)
     - Biologically-constrained classical circuits (same topology, classical dynamics)
     - Unconstrained classical circuits (arbitrary MLP)

#### Metrics Focus

- **Locomotion fidelity**: Match real C. elegans movement statistics (speed distribution, turn angle distribution, reversal frequency)
- **Connectome advantage**: Performance gap between constrained and unconstrained architectures
- **Continuous complexity**: Quantify state/action space expansion vs. grid-world
- **Classical ceiling**: Does continuous action space + connectome create problems where classical approaches genuinely struggle?

#### Phase 6 Exit Criteria

- ✅ Continuous 2D environment operational with realistic locomotion (crawling, reversals, omega turns)
- ✅ Full 302-neuron connectome architecture benchmarked on at least 3 tasks
- ✅ Connectome-constrained vs. unconstrained comparison completed with statistical analysis
- ✅ Action space is continuous (speed + turning angle, not discrete directions)
- ✅ Locomotion statistics quantitatively compared to real C. elegans data

#### Quantum Checkpoint (Phase 6) — MAJOR

**Trigger**: Continuous environment + connectome operational, classical baselines established.

This is the primary quantum re-evaluation point:

- Continuous action/state space addresses the "action space too small" and "state space polynomial" limitations
- Connectome complexity adds structural constraints that may favour quantum representations
- High-dimensional continuous observations (>50D with temporal + multi-agent + continuous) address the "observation space too small" limitation

**Action**: If any classical architecture drops below 70% on hard continuous tasks, launch **full quantum campaign v2** — systematic re-evaluation of QRH, QEF, HybridQuantum, and potentially new architectures designed for continuous domains.

#### Go/No-Go Decision

**GO if**: Continuous physics creates meaningfully harder problems AND connectome architectures show interesting properties.
**PIVOT if**: Continuous physics doesn't increase difficulty → Classical approaches trivially handle continuous control. Focus on multi-agent + evolution as the primary complexity source.
**STOP if**: Continuous physics too expensive computationally → Optimize or simplify (reduce physics fidelity while keeping continuous action space).

______________________________________________________________________

### Phase 7: Community, Validation & Publication

**Goal**: Open-source launch, external validation, and publication campaign. Share the simulation, data, and findings with the research community.

**Aspirational timeline**: Q2-Q4 2027

**Note**: This phase runs partly in parallel with Phases 6-8. Community infrastructure (NematodeBench, docs, Docker) can begin before Phase 6 completes. Paper 1 (quantum evaluation) and Paper 2 (benchmark) can be drafted early. Paper 3 (connectome) depends on Phase 6 results. External collaboration outreach can begin at any time.

#### Deliverables

1. **NematodeBench Public Launch** [CORE]

   - Public benchmark suite with tasks spanning all complexity levels:
     - Basic: chemotaxis, foraging (grid-world)
     - Intermediate: thermotaxis, predator evasion, temporal sensing
     - Advanced: multi-agent coordination, continuous locomotion
   - Public leaderboard with standardized evaluation protocol
   - Docker-based reproducibility (containerized environments, seed management)
   - Submission guidelines and evaluation scripts
   - Include the 300+ session quantum evaluation dataset as a baseline reference
   - Target: Differentiated from standard RL benchmarks by biological grounding and validation targets

2. **Publication Campaign**

   - **Paper 1**: "Systematic Evaluation of Quantum Architectures for Biological Navigation: From Grid-World Parity to Enriched Environments" — The before/after contrast showing the 300+ session quantum campaign results at grid-world complexity, then what happens when environment complexity crosses thresholds. Valuable regardless of whether quantum shows advantage at higher complexity.
   - **Paper 2**: "NematodeBench: A Biologically-Grounded Benchmark for Comparative Computational Neuroscience" — Benchmark paper introducing the task suite, biological validation methodology, and baseline results across architectures.
   - **Paper 3**: "Connectome-Constrained Learning in C. elegans: Does Real Wiring Beat Arbitrary Architecture?" — Results from Phase 6 connectome experiments.
   - Target venues: Nature Methods, NeurIPS, ICML, eLife, PNAS

3. **External Collaboration**

   - **C. elegans labs**: Biological prediction validation
     - Target: Bargmann (Rockefeller), Sengupta (Brandeis), Horvitz (MIT), Lockery (Oregon)
     - Deliverable: ≥1 model prediction tested with real C. elegans (escape latencies, thermotaxis precision, foraging efficiency)
     - Collaboration model: We generate predictions → lab designs experiments → co-authored publication
   - **Quantum hardware providers**: QPU benchmarks on enriched tasks
     - IBM Quantum, Q-CTRL Fire Opal for error suppression
     - Deploy best quantum architectures on real hardware with enriched tasks
   - **OpenWorm**: Explore policy export → muscle control integration
     - Test whether RL-trained policies can control OpenWorm's simulated body

4. **Community Building**

   - Documentation: API docs, tutorials, architecture guides
   - Contribution guidelines: How to add new brain architectures, environments, sensory modules
   - Example notebooks: Reproduce key results, extend benchmarks
   - Target: ≥3 external research groups engaging with NematodeBench

#### Phase 7 Exit Criteria

- ✅ NematodeBench launched with ≥1 external submission
- ✅ ≥1 paper submitted to peer-reviewed venue
- ✅ ≥1 external collaboration established (lab partnership or quantum hardware access)
- ✅ 300+ session quantum evaluation dataset publicly available
- ✅ Documentation sufficient for external researchers to run experiments independently

#### Go/No-Go Decision

**GO if**: External interest demonstrated (submissions, citations, collaborations).
**PIVOT if**: No external adoption → Focus on internal research value. Re-evaluate benchmark design and accessibility.

______________________________________________________________________

### Phase 8: Integration & Comprehensive Evaluation

**Goal**: Full system integration and definitive quantum vs. classical comparison at maximum simulation complexity. Extract universal principles and scope future directions.

**Aspirational timeline**: Late 2027+

#### Deliverables

1. **Full Integration**

   - All systems running together: temporal sensing + multi-agent + continuous physics + memory + evolution
   - Unified configuration system for combined scenarios
   - Performance at scale: 5+ agents in continuous 2D with temporal sensing, memory, and predators

2. **Definitive Quantum vs. Classical Comparison**

   - Comprehensive evaluation at maximum complexity across all architecture families
   - **If classical ceiling \<70%**: Full quantum architecture campaign v2 — systematic evaluation of QRH, QEF, HybridQuantum, connectome-quantum, and new architectures on enriched tasks
   - **If classical still dominant**: Publish definitive characterisation: "Quantum advantage in biological RL requires complexity threshold X" — precise measurement of what's needed
   - Either outcome is a valuable scientific contribution

3. **Universal Principles Extraction**

   - What computational principles emerge from the C. elegans deep dive?
   - Identify domain-invariant insights: approach-avoidance conflicts, exploration-exploitation trade-offs, temporal credit assignment
   - Mathematical formalisation of principles applicable beyond C. elegans
   - Target: ≥3 universal principles documented and validated

4. **Applied Directions** [Exploratory]

   - Drug screening assays: Use simulation for compound screening (behavioral phenotyping)
   - Neurodegeneration models: C. elegans models of Alzheimer's, Parkinson's (age-dependent behavioral changes)
   - Brain-computer interfaces: Quantum-inspired neural decoding insights

5. **Future Scoping**

   - Organism transfer proof-of-concept: Simplified Drosophila (~100K neurons) to test principle generality
   - 3D physics aspiration: Define requirements for full 3D substrate simulation
   - Clinical applications: Drug discovery, behavioral assay automation
   - Clear roadmap for post-project research directions

#### Phase 8 Exit Criteria

- ✅ Fully integrated simulation operational (all systems combined)
- ✅ Definitive quantum vs. classical comparison at enriched complexity published
- ✅ ≥3 universal principles documented with supporting evidence
- ✅ Clear roadmap for post-project directions (organism transfer, 3D physics, clinical applications)

______________________________________________________________________

## Quantum Re-evaluation Checkpoints

Instead of concentrating quantum work in a single phase, quantum re-evaluation is distributed across the roadmap as complexity milestones are reached:

| Phase | Complexity Milestone | Quantum Re-evaluation | Rationale |
|-------|---------------------|----------------------|-----------|
| **3** | Non-Markovian temporal dependencies | Re-test QRH (temporal advantage), QEF on harder tasks | QRH showed genuine advantage on temporal pursuit; richer temporal structure may amplify |
| **4** | Exponential multi-agent state space | Evaluate quantum game theory, entangled strategies | Strongest theoretical case for quantum advantage (quantum game theory) |
| **6** | Continuous action/state + connectome | **MAJOR**: Full quantum campaign v2 if classical \<70% | All identified complexity thresholds potentially crossed |
| **8** | Full integration | Definitive comparison at maximum complexity | Final answer on quantum advantage at this simulation scale |

Each checkpoint follows the same protocol:

1. Establish classical baselines on enriched tasks
2. Measure classical ceiling (post-convergence success rate)
3. If ceiling \<threshold: launch targeted quantum evaluation
4. If ceiling remains high: document and move to next checkpoint

______________________________________________________________________

## Complexity Dashboard

A living metric tracking the five quantum advantage thresholds identified by the [strategic assessment](research/quantum-architectures.md#strategic-assessment-environment-complexity--quantum-advantage):

| Dimension | Phase 2 (current) | Phase 3 target | Phase 4 target | Phase 6 target | Quantum threshold |
|-----------|-------------------|----------------|----------------|----------------|-------------------|
| **Input dimensionality** | 2-9D | ~15-20D | >30D | >50D (continuous) | >30D with cross-modal correlations |
| **Partial observability** | Viewport only | Temporal memory limited | Multi-agent fog-of-war | Realistic sensing range | Information-theoretic limits on classical |
| **Multi-agent** | 1 | 1 | 5-10 | 5-10 | 5+ interacting agents |
| **Temporal horizon** | Memoryless | STAM (~minutes) | STAM + social memory | Full non-Markovian | Very long non-Markovian dependencies |
| **Classical ceiling** | 94-98% | Target \<85% | Target \<75% | Target \<70% | \<70% on challenging tasks |

**Update protocol**: After each phase's classical baselines are established, update this dashboard with **measured** values (replacing the targets). Quantum checkpoints activate when thresholds are crossed. Classical ceiling targets are aspirational — the key criterion is measurable difficulty increase, not hitting a specific number.

______________________________________________________________________

## Biological Fidelity Ladder

Progressive realism across the roadmap phases:

| Level | Phase | Fidelity Description |
|-------|-------|---------------------|
| **1** | 0-2 ✅ | Grid-world, spatial gradient sensing, stateless reflexes, single agent, discrete actions |
| **2** | 3 | Temporal sensing (dT/dt, dC/dt, dO2/dt), short-term memory, non-Markovian decisions |
| **3** | 4 | Multi-agent, pheromone communication, social dynamics, competitive/cooperative behaviors |
| **4** | 5 | Evolved behaviors, transgenerational epigenetic memory, co-evolutionary arms races |
| **5** | 6 | Continuous 2D physics, realistic locomotion, connectome-constrained 302-neuron architecture |
| **6** | Future | 3D substrate (soil mechanics), full sensory suite, population dynamics, life cycle simulation |

Each level builds on the previous, with the entire stack running simultaneously in later phases. Level 5 represents the target state for this roadmap; Level 6 is aspirational for future work.

______________________________________________________________________

## Adaptive Roadmap Philosophy

This roadmap is designed to be **adaptive, not linear**. Each phase includes explicit go/no-go decision gates that allow the project to pivot based on empirical findings.

### Decision Gate Principles

1. **Evidence-driven**: Decisions based on experimental results, not assumptions
2. **Fail fast**: If a key assumption fails, pivot immediately rather than continuing unproductively
3. **Multiple paths to impact**: Alternative success modes if primary hypotheses don't hold
4. **Scientific rigor**: Better to publish "quantum didn't work but here's why" than to force false claims
5. **Quantum checkpoints, not quantum phases**: Quantum re-evaluation is distributed across the roadmap, triggered by complexity milestones rather than calendar dates

### Potential Pivot Scenarios

- **Scenario 1: Quantum shows no advantage even at high complexity** → The C. elegans simulation is still the most complete ever built. Publish comprehensive characterisation of complexity thresholds. Focus on biological insights and computational neuroscience impact.
- **Scenario 2: Multi-agent complexity too high** → Deepen single-agent biological fidelity (continuous physics, connectome, richer sensing) instead.
- **Scenario 3: Temporal sensing doesn't increase difficulty** → Classical RNNs handle temporal derivatives trivially. Skip to multi-agent as the primary complexity driver.
- **Scenario 4: Connectome doesn't improve performance** → Valuable negative result: "Evolution's wiring is not optimal for RL." Publish and continue with unconstrained architectures.
- **Scenario 5: Continuous physics too expensive** → Keep continuous action space but simplify physics. The action space expansion alone may create sufficient complexity.
- **Scenario 6: External collaboration fails** → Focus on simulation-only insights. Use published C. elegans behavioral datasets for validation instead of lab partnerships.

Each pivot maintains scientific value and publishable outcomes.

### External Dependency Risk Mitigation

| Dependency | Risk | Mitigation |
|------------|------|------------|
| **Neuroscience lab collaboration** (Phase 7) | Labs decline or slow response | Use published behavioral datasets; partner with smaller labs or citizen science projects |
| **IBM Quantum access** (quantum checkpoints) | Queue times, access limits | Maintain simulator-first development; explore IonQ/Rigetti alternatives |
| **OpenWorm integration** (Phase 7) | Project inactive or incompatible | Develop minimal integration in-house; focus on connectome data (publicly available) |
| **External NematodeBench adoption** (Phase 7) | No uptake | Focus on internal research value; improve accessibility and documentation |

### Adaptive Execution

- **Phase reviews**: Assess progress against exit criteria at each phase boundary
- **Complexity dashboard updates**: Measure and record quantum advantage thresholds after each phase
- **Quantum checkpoints**: Triggered by measured complexity milestones, not calendar dates
- **Open science**: Public benchmarks and preprints enable community feedback and course correction

______________________________________________________________________

## Ongoing Validation Milestones

Throughout all phases, the following validation activities occur continuously:

### Biological Validation (Every Phase)

**Objective**: Ensure models align with real C. elegans biology and generate testable predictions.

- **Phases 0-2** ✅: Validated chemotaxis, thermotaxis, and predator evasion against published behavioral data
- **Phase 3**: Validate temporal sensing against published dT/dt, dC/dt behavioral data
- **Phase 4**: Validate social behaviors against aggregation and pheromone literature
- **Phase 5**: Validate evolved behaviors against natural C. elegans adaptations
- **Phase 6**: Validate locomotion statistics against real worm movement data
- **Phase 7**: First experimental collaboration — model prediction tested with real C. elegans
- **Phase 8**: Comprehensive biological validation across all behaviors

### Quantum Hardware Validation (At Checkpoints)

**Objective**: When quantum checkpoints activate, benchmark on real quantum devices.

- **Phase 3 checkpoint**: If triggered, run QRH on IBM Quantum with temporal tasks
- **Phase 6 checkpoint**: Major QPU deployment — all viable quantum architectures on enriched tasks
- **Phase 8**: Production deployment pipeline for final comparison
- **Error mitigation**: Q-CTRL Fire Opal integration for noise suppression

______________________________________________________________________

## Success Metrics Framework

The project tracks success across 6 primary dimensions:

### 1. Biological Fidelity

**Definition**: How accurately the simulation captures real C. elegans behavior.

**Metrics**:

- Biological fidelity level achieved (see [Biological Fidelity Ladder](#biological-fidelity-ladder))
- Quantitative match to published C. elegans behavioral data (chemotaxis indices, escape latencies, aggregation patterns)
- Number of biological predictions generated and tested

**Targets**:

- **Phase 3**: Temporal sensing matches published dT/dt sensitivity data
- **Phase 6**: Locomotion statistics match real C. elegans (speed, turn angle, reversal frequency)
- **Phase 7**: ≥1 biological prediction tested experimentally

### 2. Complexity Achievement

**Definition**: Progress toward quantum advantage complexity thresholds.

**Metrics**:

- Complexity dashboard values (5 dimensions)
- Classical ceiling on hardest tasks
- State/action space dimensionality

**Targets**:

- **Phase 3**: Input dimensionality >15D, classical ceiling measurably lower than Phase 2 baseline (aspirational: \<85%)
- **Phase 4**: Multi-agent ≥5, classical ceiling measurably lower than Phase 3 (aspirational: \<75%)
- **Phase 6**: Continuous action space, classical ceiling measurably lower than Phase 4 (aspirational: \<70%)

### 3. Architecture Insight

**Definition**: Understanding of when and why architectures succeed or fail.

**Metrics**:

- Quantum advantage demonstrated (task count, effect size, significance)
- Complexity threshold characterisation (precise boundaries)
- Connectome advantage (constrained vs. unconstrained gap)

**Targets**:

- **Phase 6**: Connectome-constrained vs. unconstrained comparison with p < 0.05
- **Phase 8**: Definitive quantum vs. classical comparison at maximum complexity

### 4. Sample Efficiency

**Definition**: Episodes required to achieve target performance.

**Metrics**:

- Episodes to convergence
- Performance per 1000 timesteps
- Memory utilisation efficiency (STAM/ITAM/LTAM impact)

**Targets**:

- **Phase 3**: Memory systems reduce sample complexity by ≥20%
- **Phase 5**: Evolved architectures converge in 50% fewer episodes than hand-designed

### 5. Community Impact

**Definition**: External adoption and scientific influence.

**Metrics**:

- NematodeBench external submissions
- Citations to project publications
- External research groups engaging
- Open-source contributions (PRs from external researchers)

**Targets**:

- **Phase 7**: ≥3 external groups engaging, ≥1 paper submitted
- **Phase 8**: ≥10 external groups, ≥100 citations

### 6. Robustness

**Definition**: Performance under noise, missing sensors, or adversarial conditions.

**Metrics**:

- Sensor dropout robustness (performance with 10%, 20%, 50% sensors disabled)
- Noise robustness (Gaussian noise on observations)
- Hardware noise tolerance (QPU performance vs. simulator)

**Targets**:

- **Phase 3**: ≥80% performance with 20% sensor dropout
- **Phase 6**: Connectome architecture graceful degradation under circuit ablation (matches biological lesion data)

______________________________________________________________________

## Success Levels

Three levels of success, each representing valuable scientific contribution:

### Minimum Viable Success

*Primary metrics: Biological Fidelity, Architecture Insight*

- Highest-fidelity C. elegans behavioral simulation (temporal sensing, memory, multi-agent)
- Comprehensive architecture comparison published (quantum negative result at grid-world complexity + enriched re-evaluation results)
- NematodeBench operational with external adoption
- Clear documentation of complexity thresholds for quantum advantage

### Target Success

*Primary metrics: Complexity Achievement, Community Impact*

- Continuous physics + connectome-constrained architectures operational
- Classical complexity ceiling identified (where classical genuinely struggles)
- Quantum advantage demonstrated on enriched tasks OR definitive characterisation of required complexity
- ≥1 biological prediction validated experimentally with lab collaboration
- ≥3 external research groups building on NematodeBench

### Stretch Success

*Primary metrics: All dimensions at stretch targets*

- Full 302-neuron connectome running in continuous environment with multi-agent dynamics
- Universal computational principles extracted and validated
- Quantum computational neuroscience recognised as research direction (workshop, special issue, or funding)
- External research groups actively extending the project
- Biological discoveries published in peer-reviewed journals

______________________________________________________________________

## Relationship to External Projects

### OpenWorm

**Focus**: Cellular biophysics, muscle dynamics, 3D body simulation, full C. elegans digital twin

**Relationship**: **Complementary with integration aspirations**

**Differentiation**:

- Quantum Nematode: Neural computation paradigms, behavioral optimization, quantum ML, RL-trained networks
- OpenWorm: Cellular-level simulation, muscle physics, connectome-based biophysical modeling

**Integration Points**:

1. Phase 6: Use the same connectome data (wiring diagram) but different approaches — we train weights via RL on the real topology; they simulate biophysics
2. Phase 7: Explore exporting RL-trained policies to control OpenWorm's simulated muscles
3. Cross-validation: Do our behaviorally-optimized networks predict the same circuit importance as their biophysical model?

### WormBot

**Focus**: Hardware embodiment, real-world sensors, physical nematode-inspired robots

**Relationship**: **Potential future validation platform**

**Integration Points** (Phase 7+):

1. Export optimized policies for robotic deployment
2. Sim-to-real transfer testing
3. Real-world benchmarks with physical sensors

### Quantum Computing Ecosystem

**IBM Quantum**: Hardware provider, Qiskit framework, QPU access
**Q-CTRL**: Quantum error suppression (Fire Opal), circuit optimization

**Engagement**: Benchmarking at quantum checkpoints (Phases 3, 4, 6, 8), not continuous hardware testing. Simulator-first development with QPU validation at milestones.

### Neuroscience Community

**Target Labs**: Bargmann (Rockefeller), Sengupta (Brandeis), Horvitz (MIT), Lockery (Oregon)

**Collaboration Model** (Phase 7):

1. We generate biological predictions from model analysis
2. Lab designs and executes experiments
3. Co-authored publications validating (or refuting) predictions
4. Iterative: experimental results → model updates → new predictions

**Value Proposition**: Computational predictions guide experiments; access to novel analysis tools; high-impact co-authored publications.

### NematodeBench Community

**Goal**: Build an open-source community around biologically-grounded RL benchmarks.

**Strategy**:

- Launch when simulation is sufficiently differentiated from standard RL benchmarks (Phase 7)
- Provide Docker images, evaluation scripts, submission guidelines
- Include the 300+ session quantum evaluation dataset as baseline reference
- Tutorials for extending the benchmark with new architectures, environments, or biological models

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
