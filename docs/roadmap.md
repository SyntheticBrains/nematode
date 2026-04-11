# Quantum Nematode Project Roadmap

**Vision**: Build the highest-fidelity *Caenorhabditis elegans* behavioral simulation as a rigorous testbed for comparative architecture analysis, with two co-equal goals: (1) simulate the full behavioral repertoire of C. elegans with increasing biological fidelity, and (2) progressively cross complexity thresholds where quantum approaches may demonstrate genuine advantages over classical methods.

**Version**: 3.0

**Last Updated**: March 2026

**Horizon**: Milestone-based (aspirational timeline ~2025-2028+, phases advance when exit criteria are met)

______________________________________________________________________

> **Disclaimer**: This roadmap describes a research program with hypotheses to be tested, not established results. Many claims about quantum advantages, biological insights, and scaling properties are research questions requiring empirical validation. Outcomes may differ significantly from projections as evidence accumulates. The adaptive decision gates throughout this document reflect our commitment to evidence-driven pivots when hypotheses are not supported by experimental results.

______________________________________________________________________

**Note**: This roadmap aims for world-class scientific impact through rigorous methodology, external validation, and transformative discoveries at the intersection of quantum computing, neuroscience, and artificial intelligence.

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

> This roadmap is milestone-based: phases advance when exit criteria are met, not when calendar dates arrive. Aspirational timelines are provided for planning purposes but the science dictates the pace.

| Phase | Aspirational Timeline | Focus | Status | Key Deliverable |
|-------|----------------------|-------|--------|-----------------|
| **0** | Q4 2025 - Q1 2026 | Foundation & Baselines | ✅ COMPLETE | Validated optimization methods, SOTA baselines, first QPU run |
| **1** | Q1 - Q2 2026 | Sensory & Threat Complexity | ✅ COMPLETE | Thermotaxis, enhanced predators, mechanosensation, HP system |
| **2** | Q2 - Q3 2026 | Architecture Analysis | ✅ SUBSTANTIALLY COMPLETE | 300+ session quantum evaluation, brain renaming, statistical framework |
| **3** | Q2 - Q3 2026 | Temporal Sensing & Memory | ✅ SUBSTANTIALLY COMPLETE | Temporal/derivative sensing, STAM, LSTM/GRU PPO brain (19th architecture). Temporal Mode A achieves 94% L500 on hardest environment. Oxygen sensing (aerotaxis) implemented with 5-zone system and combined thermal+oxygen environments. |
| **4** | Q3 - Q4 2026 | Multi-Agent Complexity | 🔨 IN PROGRESS | Multi-agent infrastructure (Deliverable 1 complete). Pheromone communication (Deliverable 2): PheromoneField with point-source decay, food-marking + alarm pheromones, 4 sensing modules (oracle + temporal), STAM 6-channel extension, CSV export with agent_id. Social dynamics (Deliverable 3): social feeding via satiety decay reduction (npr-1), aggregation pheromones (continuous ascaroside emission), 4 collective behavior metrics (aggregation index, alarm evasion, food sharing, social feeding events), STAM 7-channel extension. Competitive foraging and game-theoretic analysis pending. |
| **5** | Q4 2026 - Q1 2027 | Evolution & Adaptation | 🔲 PLANNED | Baldwin Effect, co-evolution, transgenerational memory |
| **6** | Q1 - Q3 2027 | Continuous Physics & Connectome | 🔲 PLANNED | Continuous 2D, realistic locomotion, full 302-neuron connectome |
| **7** | Q2 - Q4 2027 | Community & Publication | 🔲 PLANNED | NematodeBench launch, publication campaign, external collaboration |
| **8** | Late 2027+ | Integration & Evaluation | 🔲 PLANNED | Full integration, definitive quantum vs. classical comparison |

______________________________________________________________________

## Executive Summary

The Quantum Nematode project pursues two co-equal goals at the intersection of quantum computing, neuroscience, and artificial intelligence:

**Goal 1 — Biological Simulation**: Build the deepest, most complete behavioral simulation of *C. elegans* ever created. Progressively implement all sensory systems, survival behaviors, learning and memory, social dynamics, and realistic physics — grounded in a fully mapped 302-neuron connectome.

**Goal 2 — Quantum Architecture Analysis**: Systematically compare quantum, classical, and biologically-realistic architectures as environment complexity increases, identifying the precise conditions under which quantum approaches provide genuine advantages — or definitively characterising why they don't.

These goals are mutually reinforcing: biological fidelity creates the computational complexity needed to test quantum advantage hypotheses, while architecture analysis reveals which computational principles best capture biological intelligence.

### Key Differentiators

- **vs. OpenWorm**: Complementary focus on neural computation paradigms and behavioral optimization vs. cellular biophysics. Integration opportunity: optimized policies controlling OpenWorm's simulated muscles. Our connectome-constrained architectures (Phase 6) use the wiring diagram for RL-trained networks, not biophysical simulation.
- **vs. Standard RL Benchmarks**: Biologically-grounded tasks with real organism validation; the only benchmark suite where tasks are derived from documented C. elegans behaviors with quantitative biological validation targets.
- **vs. Quantum ML Research**: Embodied, ecologically-valid tasks requiring multi-objective optimization, temporal integration, and social dynamics — not toy gridworlds or random circuits. Systematic 300+ session evaluation campaign provides the most comprehensive quantum architecture comparison in biological RL to date.
- **Unique position**: The only project combining real connectome topology + reinforcement learning + quantum architecture comparison. C. elegans is the only organism with a fully mapped connectome, making this uniquely tractable.

### North Star

Demonstrate that progressively increasing biological fidelity in C. elegans simulation creates computational challenges where (1) quantum approaches provide measurable advantages over classical methods, or (2) we can precisely characterise what complexity thresholds are required for quantum advantage — either outcome being a valuable scientific contribution. Simultaneously, build the most complete C. elegans behavioral simulation available, advancing computational neuroscience regardless of quantum outcomes.

______________________________________________________________________

## Current State

### Completed Phases Summary

#### Phase 0 (Foundation & Baselines) — COMPLETE

All required and stretch exit criteria met:

- **6 brain architectures** implemented and benchmarked: QVarCircuitBrain, QQLearningBrain, MLPReinforceBrain, MLPDQNBrain, MLPPPOBrain, SpikingReinforceBrain
- **PPO validated** as classical SOTA (94-98% across thermotaxis configurations)
- **CMA-ES validated** for quantum circuits (88% success, 4x better than gradient-based)
- **Spiking neural networks** rewritten with surrogate gradient descent (73.3% success)
- **IBM QPU deployment** complete (first quantum hardware run)
- **14 benchmark categories** with automated experiment tracking

#### Phase 1 (Sensory & Threat Complexity) — COMPLETE

All core exit criteria met:

- **Thermotaxis** implemented with 9 configurations (3 sizes × 3 task variants), all baselines established (see [Logbook 007](experiments/logbooks/007-ppo-thermotaxis-baselines.md))
- **Mechanosensation** implemented (boundary + predator contact detection)
- **Enhanced predators**: Stationary + pursuit types with configurable behaviors
- **Health/HP system** with damage, healing, and strategic trade-offs
- Oxygen sensing deferred to Phase 3 (pairs naturally with temporal sensing infrastructure)

#### Phase 2 (Architecture Analysis) — SUBSTANTIALLY COMPLETE

Core quantum evaluation complete, remaining items folded into later phases:

- **300+ session quantum architecture campaign** across 15 architecture variants (see [Logbook 008](experiments/logbooks/008-quantum-brain-evaluation.md))
- **Brain naming migration** to paradigm-prefix scheme complete
- **Statistical framework** operational
- **Novel architectures evaluated**: QRH, QEF, HybridQuantum, HybridClassical, QSNN, QRC, QSNN-PPO, QSNNReinforce A2C, HybridQuantumCortex, CRH, and variants
- Remaining (folded into Phase 7): interpretability framework, mechanism discovery, biological predictions

### Architecture Evaluation Results (Logbook 008)

The 300+ session campaign across 15 architecture variants produced the following landscape (snapshot as of March 2026 — see [Logbook 008](experiments/logbooks/008-quantum-brain-evaluation.md) for full details):

```text
GRADIENT-BASED ONLINE LEARNING EFFECTIVENESS (March 2026 snapshot)
═══════════════════════════════════════════════════════════════════════════

Architecture                      Foraging   Pursuit Pred   Viable?
──────────────────────────────────────────────────────────────────────────
QRC                               0%         0%             NO
QSNN (Hebbian)                    0%         N/A            NO
QSNN-PPO Hybrid                   N/A        0%             NO
QVarCircuit (parameter-shift)     ~40%       Not tested     MARGINAL
QSNN (Surrogate gradient)         73.9%      0%             PARTIAL
QSNNReinforce A2C                 N/A        0.5%           NO
QVarCircuit (CMA-ES)              99.8%      76.1%          NOT ONLINE
QRH (quantum reservoir)           86.8%      41.2%          PARTIAL
CRH (classical reservoir)         N/A        31.8%          PARTIAL (CTRL)
QEF (entangled features)          N/A        93.0%          COMPETITIVE
HybridQuantum                     91.0%      96.9%          YES (BEST)
HybridClassical (ablation)        97.0%      96.3%          YES (CONTROL)
SpikingReinforceBrain             73.3%      ~61%           UNRELIABLE
MLPReinforceBrain                 95.1%      73.4%          YES
MLPPPOBrain                       96.7%      94.5%          YES
──────────────────────────────────────────────────────────────────────────
```

### Key Findings from Quantum Campaign

1. **HybridQuantum achieved SOTA (96.9%)** but classical ablation matches (96.3%) — the three-stage curriculum and mode-gated fusion drive performance, not the quantum component
2. **QRH shows genuine quantum advantage** on pursuit predators (+9.4pp, Domingo-confirmed) but at low absolute performance (41.2%)
3. **QEF competitive but not advantageous** — 24-phase optimisation, no significant improvement (p>0.05 on all tasks)
4. **QA-7 (Quantum Plasticity)**: Classical baselines show zero backward forgetting (11/12 seeds BF=0.0) — environment too simple for quantum anti-forgetting hypothesis
5. **Every trainable quantum component matches classical**: QLIF gates = classical gates, QEF ≈ MLP PPO, HybridQuantum ≈ HybridClassical

### Strategic Conclusion

Environment complexity (2-9D observations, 4 discrete actions, ~10K effective states) is fundamentally below the threshold for quantum advantage. See [quantum-architectures.md Strategic Assessment](research/quantum-architectures.md#strategic-assessment-environment-complexity--quantum-advantage) for full analysis.

**Path forward**: Advance biological fidelity to cross complexity thresholds, then re-evaluate quantum architectures at each milestone.

### Known Gaps & Technical Debt

**Resolved from Phase 0:**

- ~~QQLearningBrain incomplete~~ — evaluated but not competitive; low priority
- ~~No SOTA RL baselines~~ — PPO implemented and validated
- ~~No real C. elegans behavioral datasets~~ — integrated for validation

**Remaining:**

- No energy/metabolic model (satiety is abstract, not ATP-based)
- No neuromodulation (dopamine/serotonin mentioned but not simulated)
- Limited environmental diversity (no temporal dynamics beyond thermotaxis)
- Sensory inputs are spatial gradient lookups, not temporal derivatives (biologically inaccurate)
- ~~Single-agent only~~ — Multi-agent infrastructure implemented (Phase 4 Deliverable 1)
- Discrete grid-world (not continuous physics)
- No connectome-constrained architectures

### Research Questions

1. **Complexity Thresholds**: At what environment complexity do quantum approaches first outperform classical? (>30D inputs? Multi-agent? Non-Markovian? Continuous state spaces?)
2. **Biological Fidelity**: Does increasing biological realism (temporal sensing, connectome topology, realistic locomotion) create computational problems that favour quantum approaches?
3. **Connectome Advantage**: Does the real C. elegans wiring diagram produce better learning agents than arbitrary architectures?
4. **Universal Principles**: What computational principles emerge from a deep C. elegans simulation that generalise to other domains?
5. **Quantum Reservoir Memory**: Can QRH's demonstrated temporal advantage (pursuit predators) scale with richer temporal dependencies?

______________________________________________________________________

## Phase Roadmap

### Phase 0: Foundation & Baselines (COMPLETE)

**Status**: ✅ All required and stretch exit criteria met.

See [Current State](#completed-phases-summary) for achievements. Key breakthroughs:

- Evolutionary optimization (CMA-ES) achieving 4x better performance than gradient-based on quantum circuits
- Spiking neural network rewrite to surrogate gradient descent enabling viable learning
- PPO established as classical SOTA across all thermotaxis configurations

______________________________________________________________________

### Phase 1: Sensory & Threat Complexity (COMPLETE)

**Status**: ✅ All core exit criteria met. Oxygen sensing deferred to Phase 3.

See [Current State](#completed-phases-summary) for achievements. Key deliverables:

- Thermotaxis system with 9 validated configurations (Logbook 007)
- Mechanosensation with boundary and predator contact detection
- Stationary and pursuit predator types
- HP-based health system with strategic damage/healing trade-offs

______________________________________________________________________

### Phase 2: Architecture Analysis & Standardization (SUBSTANTIALLY COMPLETE)

**Status**: ✅ Core quantum evaluation complete (300+ sessions). Remaining items folded into Phases 7-8.

See [Current State](#architecture-evaluation-results-logbook-008) for the full architecture landscape. Key outcomes:

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

6. **Food Spatial Persistence** [DEFERRED — prerequisite for food-marking pheromone evaluation]

   - Current food respawns uniformly random after consumption, making food-marking pheromone trails point to stale locations (misleading signal)
   - **Food patches/hotspots**: Define regions where food spawns preferentially, modeled on bacterial lawns. Pheromone trails then correctly signal "productive region"
   - **Satiety-dependent foraging**: Agents should reduce food-seeking when sated (biological: npr-1 dwelling behavior). Prevents local monopoly where one agent depletes a patch before pheromone-guided agents arrive
   - **Biased respawn**: Food respawns with spatial correlation to consumed position (configurable locality parameter)
   - Required for meaningful food-marking pheromone evaluation; alarm and aggregation pheromones have valid signals without this

#### Metrics Focus

- **Emergent phenomena**: Identify behaviors not explicitly programmed (spontaneous aggregation, division of labor, communication strategies)
- **Cooperation quantification**: Cooperation intensity, stability, efficiency gains over individual foraging
- **State space explosion**: Quantify effective state space growth with agent count
- **Classical ceiling**: Do classical approaches struggle with multi-agent coordination?

#### Phase 4 Exit Criteria

- ✅ ≥5 agents running stably with independent brains
- ✅ ≥1 emergent behavior documented (spontaneous aggregation, information sharing, etc.)
- ✅ Pheromone communication functional (at least alarm + food-marking)
- ✅ Classical approaches show measurable strain on coordination tasks (quantified difficulty increase vs. single-agent baseline)

#### Quantum Checkpoint (Phase 4)

**Trigger**: Multi-agent operational with ≥5 agents, coordination metrics established.

Evaluate:

- **Quantum entangled strategy spaces**: Can quantum architectures represent correlated multi-agent strategies more efficiently?
- **Quantum game theory approaches**: Do quantum-enhanced Nash equilibrium solvers outperform classical?
- If classical approaches show measurable difficulty on coordination tasks (aspirational: ceiling \<85%), launch targeted quantum evaluation campaign

#### Go/No-Go Decision

**GO if**: Multi-agent scenarios reveal interesting emergent phenomena OR create measurably harder coordination problems.
**PIVOT if**: Multi-agent complexity too high or unstable → Deepen single-agent complexity (richer sensing, longer horizons).
**STOP if**: Infrastructure can't handle ≥3 agents → Re-architect for scalability before proceeding.

______________________________________________________________________

### Phase 5: Evolution & Adaptation

**Goal**: Evolve optimal learning strategies and study how learning guides evolution, including biologically-documented transgenerational memory.

**Aspirational timeline**: Q4 2026 - Q1 2027

**Prerequisites**: Phase 3 (memory infrastructure for transgenerational memory). Phase 4 multi-agent infrastructure required only for co-evolution (deliverable 4) — other deliverables can begin in parallel with Phase 4.

**Pilot-Then-Focus Approach**: Start with lightweight pilots of 2-3 evolutionary approaches using small populations and few generations. Based on pilot results, select 1-2 approaches for deep investigation.

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
- ✅ Baldwin Effect or Lamarckian inheritance demonstrated (learned behaviors become innate)
- ✅ Co-evolution produces arms race dynamics with measurable escalation
- ✅ Transgenerational memory functional (if biologically justified by pilot results)
- ✅ Generational fitness tracking shows continuous improvement over ≥50 generations

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
