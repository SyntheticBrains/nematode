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
| **3** | Q2 - Q3 2026 | Temporal Sensing & Memory | 🔲 NEXT | Temporal gradients (dT/dt, dC/dt), STAM, oxygen sensing |
| **4** | Q3 - Q4 2026 | Multi-Agent Complexity | 🔲 PLANNED | Multi-agent infrastructure, pheromones, social/competitive dynamics |
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

The 300+ session campaign across 15 architecture variants produced the following landscape:

```text
GRADIENT-BASED ONLINE LEARNING EFFECTIVENESS (March 2026)
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
- Single-agent only
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

- 11+ quantum architectures systematically evaluated
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

**Non-gradient modalities** (mechanosensation, nociception) are already biologically honest — they provide binary contact signals that the agent actually experiences. These remain unchanged.

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

#### Phase 3 Exit Criteria

- ✅ Biologically honest sensing (Mode A or B) operational for thermotaxis and chemotaxis
- ✅ STAM implemented with biologically-calibrated exponential decay rates
- ✅ Oracle vs. honest performance gap quantified (expected: significant drop in success rate)
- ✅ Classical approaches show measurable difficulty increase vs. oracle baseline (quantified)
- ✅ ≥1 associative learning paradigm functional (classical conditioning or aversive learning)
- ✅ Oxygen sensing implemented with honest temporal sensing

#### Quantum Checkpoint (Phase 3)

**Trigger**: Temporal sensing operational, classical ceiling measured.

Re-evaluate:

- **QRH** (showed genuine advantage on temporal pursuit tasks — does richer temporal structure amplify this?)
- **QEF** on temporally-enriched tasks
- If classical ceiling drops below ~80% on any enriched task, resume targeted quantum architecture search

#### Go/No-Go Decision

**GO if**: Temporal sensing creates measurably harder problems (classical ceiling drops ≥10 percentage points) AND STAM improves performance ≥10%.
**PIVOT if**: Temporal sensing doesn't change difficulty → Classical approaches may trivially handle temporal derivatives with simple RNNs. Focus on multi-agent complexity (Phase 4) as the primary difficulty driver.
**STOP if**: STAM infrastructure too complex or unreliable → Simplify to fixed-length observation windows.

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

#### Metrics Focus

- **Emergent phenomena**: Identify behaviors not explicitly programmed (spontaneous aggregation, division of labor, communication strategies)
- **Cooperation quantification**: Cooperation intensity, stability, efficiency gains over individual foraging
- **State space explosion**: Quantify effective state space growth with agent count
- **Classical ceiling**: Do classical approaches struggle with multi-agent coordination?

#### Phase 4 Exit Criteria

- ✅ ≥5 agents running stably with independent brains
- ✅ ≥1 emergent behavior documented (spontaneous aggregation, information sharing, etc.)
- ✅ Pheromone communication functional (at least alarm + food-marking)
- ✅ Classical approaches show measurable strain on coordination tasks (ceiling \<85% on hard multi-agent scenarios)

#### Quantum Checkpoint (Phase 4)

**Trigger**: Multi-agent operational with ≥5 agents, coordination metrics established.

Evaluate:

- **Quantum entangled strategy spaces**: Can quantum architectures represent correlated multi-agent strategies more efficiently?
- **Quantum game theory approaches**: Do quantum-enhanced Nash equilibrium solvers outperform classical?
- If classical ceiling drops below 85% on coordination tasks, launch targeted quantum evaluation campaign

#### Go/No-Go Decision

**GO if**: Multi-agent scenarios reveal interesting emergent phenomena OR create genuinely hard coordination problems (classical \<85%).
**PIVOT if**: Multi-agent complexity too high or unstable → Deepen single-agent complexity (richer sensing, longer horizons).
**STOP if**: Infrastructure can't handle ≥3 agents → Re-architect for scalability before proceeding.

______________________________________________________________________

### Phase 5: Evolution & Adaptation

**Goal**: Evolve optimal learning strategies and study how learning guides evolution, including biologically-documented transgenerational memory.

**Aspirational timeline**: Q4 2026 - Q1 2027

**Prerequisites**: Phase 4 multi-agent infrastructure (co-evolution requires populations)

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

#### Go/No-Go Decision

**GO if**: Evolution produces novel, high-performing behaviors OR demonstrates Baldwin Effect.
**PIVOT if**: Evolution plateaus quickly → Focus on hand-designed architectures. Document evolutionary limitations.
**STOP if**: Evolutionary algorithms fail to converge → Revisit fitness functions or population parameters.

______________________________________________________________________

### Phase 5: Social Complexity (Q1-Q2 2027)

**Goal**: Implement multi-agent scenarios to study cooperation, competition, and emergent collective behaviors.

**Transition Buffer**: Phase 5 start is conditional on Phase 4 progress. If Phase 4 is on track by mid-Q4 2026, Phase 5 infrastructure work can begin in parallel. If Phase 4 requires extension, Phase 5 start shifts to Q2 2027.

**Dependency Note**: Conditional on Phase 2 showing sufficient single-agent competence, and Phase 3 (Learning & Memory) providing memory systems needed for social behaviors. C. elegans social behavior depends on learned associations; implementing memory first ensures social agents can remember past interactions.

#### Deliverables

1. **Multi-Agent Infrastructure**

   - Multiple agents in same environment (2-10 agents)
   - Independent brains (each agent has own architecture instance)
   - Agent-agent interactions tracked (proximity, collisions, food competition)

2. **Cooperative Foraging**

   - **Social facilitation**: Feeding rate increases when near other agents (biological observation)
   - **Pheromone communication**: Agents deposit chemical trails marking food locations
   - **Shared food discovery**: When one agent finds food, others can follow pheromone trail
   - **Collective exploration**: Distributed search patterns (each agent explores different region)
   - **Emergent cooperation metrics**: food collection rate per agent with cooperation vs. alone

3. **Competitive Foraging**

   - **Zero-sum resource competition**: Limited food, agents compete for access
   - **Territorial behavior**: Agents defend food-rich zones
   - **Dominance hierarchies**: Stronger agents (better learners) access food first
   - **Mate competition**: Simulated reproductive success as fitness metric
   - **Game-theoretic analysis**: Nash equilibria, evolutionary stable strategies

4. **Collective Behaviors**

   - **Aggregation patterns**: C. elegans naturally clusters; can agents learn aggregation?
   - **Collective predator response**: Multiple agents coordinate to evade or mob predators
   - **Information sharing**: Do agents develop communication strategies (pheromone trails, proximity signaling)?
   - **Swarm intelligence**: Emergent optimization (find food faster as group than individuals)

5. **Mechanism Discovery from Emergent Behaviors**

   - Identify unexpected collective phenomena
   - Compare to swarm robotics literature (particle swarm optimization, ant colony optimization)
   - Extract general principles: "Cooperation emerges when [X conditions]"

6. **Architecture Comparisons in Multi-Agent Settings**

   - Quantum vs. classical in cooperative settings: which learns cooperation faster?
   - Mixed populations: quantum + classical agents competing
   - Evolutionary dynamics: which architectures dominate over many generations?

#### Metrics Focus

- **Emergent phenomena**: Identify behaviors not explicitly programmed (e.g., spontaneous aggregation, division of labor)
- **Cooperation quantification**: Measure cooperation intensity, stability, efficiency gains
- **Biological insight**: Do simulated behaviors match C. elegans social behavior literature?

#### Phase 5 Exit Criteria

- ✅ Multi-agent infrastructure supports ≥10 simultaneous agents with stable performance
- ✅ Cooperative, competitive, and collective behaviors all implemented and benchmarked
- ✅ ≥1 emergent behavior discovered and documented (e.g., spontaneous aggregation, division of labor, information sharing)
- ✅ Quantum vs. classical comparison in multi-agent settings completed
- ✅ Mechanism discovery yields insights applicable to swarm robotics or social foraging theory

#### Go/No-Go Decision

**GO if**: Multi-agent scenarios reveal interesting emergent phenomena OR scale successfully to ≥5 agents.
**PIVOT if**: Multi-agent complexity too high or unstable → Continue with single-agent complexity (deeper sensory integration, longer-horizon planning).
**STOP if**: Infrastructure can't handle ≥3 agents → Re-architect for scalability before proceeding.

______________________________________________________________________

### Phase 6: Quantum Frontiers (Q2-Q3 2027)

> **Note**: This phase involves advanced quantum computing concepts (VQC, QAOA, error mitigation, hardware deployment). The technical claims and feasibility assessments in this section should be reviewed by domain experts in quantum computing and quantum machine learning before implementation.

**Goal**: Push boundaries of quantum algorithms, deploy on real hardware, and demonstrate quantum advantages on biologically-relevant tasks.

#### Deliverables

1. **Advanced Quantum Algorithms**

   - **Variational Quantum Circuits (VQC)**: Continue developing parameterized quantum circuits optimized via evolution (building on Phase 0 success)
   - **Quantum Approximate Optimization Algorithm (QAOA)**: Explore whether multi-objective foraging can be framed as combinatorial optimization (research-level question)
   - **Quantum Neural Networks (QNN)**: Data reuploading circuits, hybrid quantum-classical architectures
   - **Quantum Reinforcement Learning**: Quantum policy representations, exploration of quantum-enhanced value estimation
   - **Quantum Natural Gradient (QNG) variants**: Momentum-QNG, Conjugate QNG (CQNG) with dynamic hyperparameters, QNG with geodesic corrections for faster convergence
   - **Quantum kernel methods**: Quantum Policy Gradient in RKHS, sparse non-parametric policies with tunable expressiveness

2. **Quantum Error Mitigation**

   - **Q-CTRL Fire Opal Integration**: Error suppression for NISQ devices
   - **Zero-noise extrapolation**: Estimate noiseless results from noisy runs
   - **Probabilistic error cancellation**: Mitigate gate errors
   - **Noise-robust circuit design**: Shorter circuits, native gate sets, error-aware compilation

3. **Real Quantum Hardware Deployment**

   - **IBM Quantum**: Monthly benchmarks on ibm_sherbrooke, ibm_kyiv, or latest backends
   - **IonQ**: Trapped-ion quantum computers (if accessible)
   - **Hardware performance tracking**: Noise levels, gate fidelities, decoherence times
   - **Hardware-specific optimization**: Topology-aware circuit mapping, qubit selection

4. **Theoretical Quantum Advantage Analysis**

   - Mathematical analysis: Identify structural features of foraging tasks that may benefit from quantum computation
   - Explore potential advantages: Grover-like search in large state spaces, quantum parallelism for multi-objective optimization
   - Note: Formal complexity-theoretic proofs (BQP vs. P) are unlikely for RL tasks, but empirical advantages may be demonstrable

5. **Quantum-Classical Hybrid Ensembles**

   - Voting ensembles: Quantum + classical models vote on actions
   - Hierarchical decision-making: Quantum for strategy, classical for tactics (or vice versa)
   - Adaptive switching: Use quantum when uncertain, classical when confident
   - Meta-learning: Learn which model to trust in which context

6. **Quantum Interpretability**

   - Visualize quantum state evolution during decision-making
   - Track entanglement entropy: How much are qubits correlated?
   - Gate importance: Which quantum gates are critical for performance?
   - Hypothesis: "Entanglement enables [X] computational capability"

#### Metrics Focus

- **Quantum advantage**: Demonstrate quantum > classical on specific task with statistical significance
- **Hardware viability**: Real QPU performance vs. simulation (noise gap)
- **Error mitigation efficacy**: Performance improvement from error suppression

#### Phase 6 Exit Criteria

- ✅ VQC and QAOA (if applicable) implemented and benchmarked on foraging tasks
- ✅ Real quantum hardware deployment successful: Achieves **>50% of classical baseline performance** on at least 1 task
- ✅ Quantum error mitigation (Fire Opal or similar) improves real hardware performance by ≥20%
- ✅ Monthly IBM Quantum benchmarks operational (automated deployment and tracking pipeline)
- ✅ At least 1 quantum advantage claim validated on physical QPU (not just simulation)
- ✅ Quantum advantage demonstrated on ≥1 biologically-relevant task with p < 0.01 statistical significance OR compelling explanation of why quantum doesn't provide advantages
- ✅ Theoretical framework published linking quantum computational principles to intelligent behavior

#### Go/No-Go Decision

**GO if**: Quantum hardware achieves ≥50% classical performance OR reveals insights into quantum computation despite performance gaps.
**PIVOT if**: Hardware too noisy for useful computation → Focus on quantum-inspired classical algorithms (variational methods, evolution strategies applied to classical nets).
**STOP if**: No theoretical or empirical quantum advantage after extensive testing → Conclude quantum not suitable for this domain, publish comprehensive negative result.

______________________________________________________________________

### Phase 7: Scaling & Real-World (Q3-Q4 2027)

> **Note**: This phase involves neuromorphic hardware deployment (Intel Loihi) and embodied robotics (WormBot). Neuromorphic expertise and robotics integration experience recommended before implementation.

**Goal**: Scale to large environments, deploy on neuromorphic hardware, validate with embodied robots, and achieve biological discoveries.

#### Deliverables

1. **Large-Scale Environments**

   - 200×200+ grid support (vs. current 50×50 max)
   - 100+ simultaneous food sources (vs. current ~20)
   - 10+ concurrent predators (vs. current 1-3)
   - Memory-efficient rendering (viewport-based, sparse representations)
   - Distributed computation: Multi-GPU training, parallel environment execution

2. **Neuromorphic Hardware Deployment**

   - **Intel Loihi**: Deploy SpikingReinforceBrain (LIF neurons with surrogate-gradient-trained weights) on neuromorphic chip
   - Event-driven computation efficiency: Compare power consumption (Joules per decision)
   - Spike-timing precision: Does hardware spike timing affect inference quality?
   - Comparison: Neuromorphic vs. GPU vs. CPU vs. QPU (energy, latency, throughput)
   - Note: Training done on GPU with surrogate gradients; inference on neuromorphic hardware

3. **WormBot Embodied Deployment**

   - Export optimized policies → control WormBot hardware platform
   - Real-world sensors: Physical chemical sensors (if available), contact sensors, IMU
   - Sim-to-real transfer: Does simulation training transfer to physical robot?
   - Embodied validation: Does robot behavior match simulation predictions?

4. **Biological Discovery Validation**

   - Collaborate with C. elegans neuroscience labs
   - Test ≥1 major prediction from model analysis (from Phase 2 or 3)
   - Experimental design: Control vs. treatment, statistical power analysis
   - Expected outcome: Model prediction confirmed → biological discovery

5. **Scalability Analysis**

   - **Theoretical**: Complexity analysis (time/space vs. neuron count)
   - **Empirical**: Benchmark on increasingly complex tasks (10×10 to 200×200 grids)
   - **Architectural**: Demonstrate composability (modules, hierarchies)
   - Path to larger systems: Demonstrate scaling principles from C. elegans (302 neurons) to larger invertebrates (Drosophila: ~100K neurons, honeybee: ~1M neurons) as intermediate steps

6. **Distributed Training Infrastructure**

   - Ray-based parallel experiment running
   - Multi-GPU classical training (PyTorch DDP)
   - Distributed quantum circuit evaluation (parallel Qiskit jobs)
   - Cloud deployment: AWS, GCP, or Azure for large-scale benchmarks

#### Metrics Focus

- **Scalability**: Performance on 200×200 grids, 100+ foods, 10+ predators
- **Real-world performance**: Sim-to-real transfer success rate
- **Energy efficiency**: Joules per decision (neuromorphic vs. classical vs. quantum)

#### Phase 7 Exit Criteria

- ✅ 200×200 environments operational with 100+ foods and 10+ predators
- ✅ SpikingReinforceBrain deployed on Intel Loihi with energy efficiency analysis (Joules per decision vs. GPU/CPU)
- ✅ [If quantum path continues from Phase 6] QPU energy efficiency comparison included
- ✅ Embodied robot successfully demonstrates at least 1 C. elegans behavior (chemotaxis, predator evasion, or foraging)
- ✅ WormBot controlled by optimized policy with successful sim-to-real transfer (>50% sim performance)
- ✅ Biological experiment collaboration yields quantitative validation (model prediction confirmed in peer-reviewed publication)
- ✅ Scalability path to larger invertebrate models (~100K-1M neurons) documented with proof-of-concept demonstration

#### Go/No-Go Decision

**GO to Phase 8 if**: External validation successful (embodied robot works OR biological discovery confirmed).
**PIVOT if**: Sim-to-real transfer fails → Focus on simulation-only insights, theoretical contributions. Still valuable without embodiment.
**STOP if**: Scalability analysis shows fundamental barriers to larger systems → Document limits, focus on C. elegans-scale insights only.

______________________________________________________________________

### Phase 8: Universality & Impact (Q4 2027 - Q2 2028)

**Goal**: Demonstrate universal principles by transferring to new organisms and domains, establish quantum computational neuroscience as a recognized research direction, and achieve paradigm-shifting impact.

#### Deliverables

1. **Transfer to New Organisms**

   - **Drosophila (fruit fly)**: 100,000 neurons, similar sensory tasks (chemotaxis, foraging, escape)
   - **Zebrafish larvae**: 100,000 neurons, visual predator avoidance, schooling behavior
   - **Honeybee**: Foraging, waggle dance communication, collective decision-making
   - **Zero-shot transfer**: Do C. elegans-trained architectures transfer to new organisms?
   - **Fine-tuning**: How much retraining is needed for new organism?

2. **Domain Transfer**

   - **Swarm robotics**: Multi-robot foraging, cooperative exploration, task allocation
   - **Financial trading**: Multi-objective optimization (profit vs. risk), temporal dynamics
   - **Network routing**: Packet routing with congestion avoidance (food = destination, predators = congestion)
   - **Autonomous vehicles**: Navigation with multi-objective constraints (speed vs. safety)
   - **Measure**: Performance on new domain without C. elegans-specific tuning

3. **Universal Principles Extraction**

   - Identify domain-invariant computational principles
   - Example: "Approach-avoidance conflicts optimally resolved by [X quantum mechanism]"
   - Example: "Exploration-exploitation trade-off follows [Y scaling law] across all architectures"
   - Mathematical formalization: General theorems applicable beyond C. elegans

4. **Scaling Theory Development**

   - **Theoretical scaling bounds**: Time/space complexity as function of neuron count
   - **Hierarchical composition**: Combine modules for larger-scale systems
   - **Modular quantum circuits**: Can we compose 1000+ qubit circuits from 10-qubit modules?
   - **Scaling trajectory**: Principles for moving from C. elegans (302 neurons) through Drosophila (~100K) toward larger systems

5. **Unified Theory Publication**

   - Mathematical framework connecting: quantum-inspired algorithms ↔ neural computation models ↔ intelligent behavior
   - Testable predictions across multiple levels (algorithmic, neural network, behavioral)
   - Computational implications: When do quantum-inspired representations provide advantages for modeling biological intelligence?

6. **Field Establishment: Quantum Computational Neuroscience**

   - Workshop organization: Invite quantum physicists, neuroscientists, AI researchers
   - Special journal issue: Guest edit special issue on "Quantum Computing for Neuroscience Applications"
   - Review article: "Quantum Machine Learning for Computational Neuroscience"
   - Funding: Apply for major grants (NSF, NIH, ERC) to establish research program

7. **Open Science Impact**

   - ≥10 external research groups using NematodeBench
   - ≥100 citations to project publications
   - Open-source contributions: PRs from external researchers
   - Educational impact: Used in university courses on quantum ML or computational neuroscience

#### Metrics Focus

- **Universal principles**: ≥3 domain-invariant principles extracted and validated
- **Transfer success**: ≥50% performance on new organisms/domains without retraining
- **Field recognition**: Invited talks, special sessions, funding awards

#### Phase 8 Exit Criteria

- ✅ Transfer to ≥2 new organisms (Drosophila, zebrafish, or honeybee) demonstrated with ≥50% performance
- ✅ Domain transfer to ≥2 non-biological domains (swarm robotics, financial trading, network routing, etc.)
- ✅ ≥3 universal principles extracted, formalized mathematically, and validated across multiple domains
- ✅ Unified theory published in peer-reviewed venue
- ✅ Quantum computational neuroscience recognized as research direction (workshop organized, special issue published, OR funding program established)
- ✅ External adoption (tiered): Minimum ≥3 / Target ≥10 / Stretch ≥25 external research groups using project tools/benchmarks (verified through citations, GitHub forks, or contributions)

#### Go/No-Go Decision

**SUCCESS (Phase complete) if**: Universal principles extracted AND recognized through publication in top-tier venue.
**PARTIAL SUCCESS if**: Transfer works but principles are domain-specific → Still valuable contribution to computational neuroscience and quantum ML.
**REFRAME if**: Universality doesn't hold → Focus on C. elegans-specific insights as deep case study. Publish comprehensive analysis of "when quantum works and why."

______________________________________________________________________

## Adaptive Roadmap Philosophy

This roadmap is designed to be **adaptive, not linear**. Each phase includes explicit go/no-go decision gates that allow the project to pivot based on empirical findings.

### Decision Gate Principles

1. **Evidence-driven**: Decisions based on experimental results, not assumptions
2. **Fail fast**: If a key assumption fails (e.g., quantum advantage), pivot immediately rather than continuing down an unproductive path
3. **Multiple paths to impact**: Alternative success modes if primary hypotheses don't hold
4. **Scientific rigor**: Better to publish "quantum didn't work but here's why" than to force false claims

### Potential Pivot Scenarios

The roadmap includes multiple pivot points to maintain scientific value even if primary hypotheses fail:

- **Scenario 1: Quantum shows no advantage** → Focus on spiking vs. classical architecture comparison. Publish comprehensive negative result: "Why Quantum Doesn't Help Biological Navigation: Lessons for Quantum ML"
- **Scenario 2: Multi-agent complexity too high** → Deepen single-agent biological fidelity, longer-horizon planning, more complex sensory integration
- **Scenario 3: Learning doesn't improve performance** → Focus on innate behavior repertoire mapping. Document evolutionary optimization of fixed policies.
- **Scenario 4: Hardware too noisy** → Quantum-inspired classical algorithms (evolutionary optimization, variational methods applied to classical networks)
- **Scenario 5: Scaling fails** → Deep dive into C. elegans-specific insights as tractable case study. Extract principles applicable at 302-neuron scale.
- **Scenario 6: Sim-to-real transfer fails** → Focus on simulation-only theoretical insights, mathematical frameworks, computational principles
- **Scenario 7: Universality doesn't hold** → C. elegans as deep case study in computational neuroscience, quantum ML benchmarking

Each pivot maintains scientific value and publishable outcomes. The goal is impactful science, not forcing a predetermined narrative.

### External Dependency Risk Mitigation

Several critical deliverables depend on external partners. Fallback strategies:

| Dependency | Risk | Mitigation |
|------------|------|------------|
| **Neuroscience lab collaboration** (Phases 2, 7) | Labs decline or slow response | Use published C. elegans behavioral datasets; partner with smaller labs or citizen science projects |
| **IBM Quantum access** (Phase 0+) | Queue times, access limits, service changes | Maintain simulator-first development; explore IonQ/Rigetti as alternatives; budget for paid access |
| **WormBot integration** (Phases 5-7) | Project inactive or incompatible | Develop minimal embodied testbed in-house; partner with swarm robotics labs as alternative |
| **Intel Loihi access** (Phase 7) | Hardware access denied | Use SpiNNaker or software neuromorphic simulators; focus on algorithmic insights over hardware deployment |

### Adaptive Execution

- **Quarterly reviews**: Assess progress against exit criteria, adjust timelines and priorities
- **Annual replanning**: Major roadmap revisions based on cumulative findings
- **Continuous validation**: Biological experiments, hardware tests, external collaborations throughout all phases (not just at the end)
- **Open science**: Public benchmarks and preprints enable community feedback and course correction

This adaptive approach maximizes the probability of **publishable, impactful outcomes** regardless of whether quantum provides advantages, while maintaining the ambitious north star goal of mapping C. elegans behaviors and extracting universal principles of biological intelligence.

______________________________________________________________________

## Ongoing Validation Milestones

Throughout all phases, the following validation activities occur continuously:

### Biological Validation (Every Phase)

**Objective**: Ensure models align with real C. elegans biology and generate testable predictions.

- **Phase 0-1**: Validate chemotaxis and predator evasion against published behavioral data
- **Phase 2**: First biological prediction tested experimentally (collaboration with neuroscience lab)
- **Phase 3**: Test memory timescale predictions (STAM, ITAM, LTAM) with real worms
- **Phase 4**: Validate evolved behaviors match natural C. elegans adaptations (e.g., foraging efficiency, predator avoidance latency, exploration patterns)
- **Phase 5**: Validate multi-agent behaviors against C. elegans social behavior literature
- **Phase 6**: Quantum hardware validation with biological task benchmarks
- **Phase 7**: Major biological discovery published (model → experiment → insight)
- **Phase 8**: Predictions generalize to other organisms (Drosophila, zebrafish)

**Partnerships**: Establish MOUs with 2-3 C. elegans labs (e.g., Bargmann Lab at Rockefeller, Sengupta Lab at Brandeis, Horvitz Lab at MIT)

### Quantum Hardware Validation (Every Phase)

**Objective**: Regularly benchmark on real quantum devices to track hardware progress and algorithm robustness.

- **Phase 0**: First successful QPU run (QVarCircuitBrain on IBM Quantum)
- **Phase 1-4**: Monthly benchmarks on available backends (track noise, fidelity, queue times)
- **Phase 5-6**: Weekly benchmarks as algorithm complexity increases
- **Phase 6**: Daily benchmarks during intensive quantum algorithm development
- **Phase 7-8**: Production deployment pipeline (automated QPU testing in CI/CD)

**Hardware Partners**: IBM Quantum, Q-CTRL (Fire Opal), IonQ (if accessible), Rigetti (if accessible)

### Embodied Testing (Phases 2+)

**Objective**: Validate that simulation-trained policies transfer to physical robots.

- **Phase 2**: Initial WormBot contact (explore collaboration, assess platform compatibility)
- **Phase 3**: Define sim-to-real transfer requirements, begin policy export prototyping
- **Phase 5**: Export first policy for WormBot testing (simple chemotaxis)
- **Phase 7**: Full WormBot deployment with multiple behaviors (foraging, evasion, multi-agent)
- **Phase 8**: Transfer to other robotic platforms (swarm robotics testbeds)

**Partnerships**: WormBot project, swarm robotics labs, soft robotics groups

______________________________________________________________________

## Success Metrics Framework

The project tracks success across 6 primary dimensions:

### 1. Generalization

**Definition**: Performance on unseen environments, tasks, or organisms without retraining.

**Metrics**:

- Zero-shot transfer accuracy (% of original performance)
- Fine-tuning efficiency (episodes needed to match original performance)
- Cross-organism transfer (C. elegans → Drosophila → zebrafish)
- Cross-domain transfer (foraging → robotics → finance)

**Targets** (Minimum / Target / Stretch):

- **Phase 1**: ≥50% / ≥70% / ≥85% performance on unseen predator types
- **Phase 3**: ≥40% / ≥60% / ≥75% performance on novel associative learning tasks
- **Phase 8**: ≥30% / ≥50% / ≥70% performance on new organisms without retraining

### 2. Hardware Efficiency

**Definition**: Computational resources required for training and inference.

**Metrics**:

- **Quantum**: Circuit depth, gate count, qubit count, shots per decision
- **Classical**: Parameter count, FLOPs, memory (MB), training time (hours)
- **Spiking**: Neuron count, synapse count, spike count, energy (Joules per decision)
- **All**: Wall-clock time to convergence, GPU-hours, QPU-hours, total cost ($)

**Targets**:

- **Phase 2**: Identify minimal sufficient architectures (fewest parameters for target performance)
- **Phase 6**: Quantum error mitigation reduces QPU resource usage by ≥20%
- **Phase 7**: Neuromorphic deployment achieves ≥10× energy efficiency vs. GPU

### 3. Biological Insight

**Definition**: Novel discoveries about C. elegans or general principles of biological intelligence.

**Metrics**:

- Biological predictions generated (count)
- Predictions tested experimentally (count)
- Predictions confirmed (%, significance level)
- Mechanistic insights (does model reveal "how" behavior emerges?)
- Publications in neuroscience journals (count)

**Targets**:

- **Phase 2**: ≥1 prediction tested and confirmed (p < 0.05)
- **Phase 7**: ≥1 major biological discovery published in peer-reviewed journal (e.g., Nature, Science, eLife, PNAS, Current Biology)
- **Phase 8**: ≥3 universal principles applicable to other organisms

### 4. Sample Efficiency

**Definition**: Episodes or timesteps required to achieve target performance.

**Metrics**:

- Episodes to convergence (when variance < 5% for 10 consecutive runs)
- Timesteps to first success (e.g., first food collected, first predator evaded)
- Data efficiency (performance per 1000 timesteps)

**Targets** (Minimum / Target / Stretch):

- **Phase 1**: Reduce convergence episodes by 15% / 30% / 50% through better algorithms
- **Phase 3**: Memory systems reduce sample complexity by 20% / 50% / 70% (leverage past experience)
- **Phase 4**: Evolved architectures converge in 50% fewer episodes than hand-designed

### 5. Robustness

**Definition**: Performance under noise, missing sensors, or adversarial conditions.

**Metrics**:

- Sensor dropout robustness (performance with 10%, 20%, 50% sensors disabled)
- Noise robustness (performance with Gaussian noise on observations)
- Adversarial robustness (performance under worst-case perturbations)
- Hardware noise tolerance (QPU performance degradation vs. simulator)

**Targets**:

- **Phase 2**: ≥80% performance with 20% sensor dropout
- **Phase 6**: Quantum error mitigation maintains ≥70% of simulator performance on real QPU
- **Phase 7**: Sim-to-real transfer maintains ≥60% of simulation performance

### 6. Interpretability

**Definition**: Ability to explain and understand model decisions.

**Metrics**:

- Human-understandable explanations (qualitative assessment)
- Feature attribution accuracy (measured by feature ablation)
- Hypothesis generation (count of testable hypotheses from model analysis)
- Mechanistic alignment (does explanation match known biology?)

**Targets**:

- **Phase 2**: All target architectures have operational interpretability tools
- **Phase 3**: Memory mechanisms interpretable and matched to biological substrates
- **Phase 8**: Universal principles extractable from interpretability analysis

______________________________________________________________________

## Success Levels

The project defines three levels of success, each representing valuable scientific contribution. These map to the [Success Metrics Framework](#success-metrics-framework) dimensions.

### Minimum Viable Success

*Primary metrics: Interpretability, Hardware Efficiency*

- Comprehensive architecture comparison (quantum vs. classical vs. spiking) with rigorous statistical analysis
- Public benchmark suite (NematodeBench) with external adoption
- Clear documentation of which optimization methods work for which architectures
- At least one peer-reviewed publication on architecture comparison methodology

### Target Success

*Primary metrics: Biological Insight, Generalization, Sample Efficiency*

- Demonstrated quantum advantage on at least one biologically-relevant task (or compelling explanation of why not)
- Biological predictions validated experimentally with C. elegans lab collaboration
- Memory and learning systems that improve agent performance over static policies
- Transfer to at least one new domain (embodied robot or other organism)

### Stretch Success

*Primary metrics: Generalization (cross-organism), Robustness, all dimensions at stretch targets*

- Universal computational principles extracted and validated across multiple domains
- Quantum computational neuroscience recognized as a research direction (workshop, special issue, or funding program)
- Sim-to-real transfer successful on embodied platform
- External research groups actively building on project tools and benchmarks

______________________________________________________________________

## Relationship to External Projects

### OpenWorm

**Focus**: Cellular biophysics, muscle dynamics, 3D body simulation, full C. elegans digital twin

**Relationship**: **Complementary with integration aspirations**

**Differentiation**:

- Quantum Nematode: Neural computation paradigms, behavioral optimization, quantum ML
- OpenWorm: Cellular-level simulation, muscle physics, connectome-based modeling

**Integration Points**:

1. Export optimized policies from Quantum Nematode → control OpenWorm's simulated muscles
2. Import OpenWorm's connectome data → seed quantum circuit topology
3. Validate: Do optimized behaviors match OpenWorm's biophysical predictions?

**Collaboration Opportunities**:

- Share behavioral datasets
- Cross-validate predictions (algorithm-level vs. cellular-level)
- Co-organize workshops on multi-scale modeling

### WormBot

**Focus**: Hardware embodiment, real-world sensors, physical nematode-inspired robots

**Relationship**: **Validation platform**

**Integration Points**:

1. Deploy Quantum Nematode-optimized policies on WormBot hardware
2. Test sim-to-real transfer (does simulation learning work on physical robot?)
3. Real-world benchmarks: Chemical sensing (if available), obstacle navigation, multi-robot coordination

**Collaboration Opportunities**:

- WormBot provides embodied validation testbed
- Quantum Nematode provides optimized control policies
- Joint experiments on real-world foraging tasks

### Quantum Computing Ecosystem

**IBM Quantum**: Hardware provider, Qiskit framework, access to real QPUs

**Q-CTRL**: Quantum error suppression (Fire Opal), circuit optimization

**Relationship**: **Technology partners**

**Engagement**:

- Regular benchmarking on IBM Quantum hardware
- Quantum algorithm development using Qiskit ecosystem
- Potential joint publications on quantum advantage in RL

### Neuroscience Community

**Target Labs**: Bargmann (Rockefeller), Sengupta (Brandeis), Horvitz (MIT), Lockery (Oregon)

**Relationship**: **Experimental validation partners**

**Collaboration Model**:

1. Quantum Nematode generates biological predictions from model analysis
2. Neuroscience lab designs and executes experiments
3. Co-authored publications validating (or refuting) predictions
4. Iterative refinement: Experimental results → model updates → new predictions

**Value Proposition for Labs**:

- Computational predictions guide experiments (hypothesis generation)
- Access to quantum ML expertise
- Novel analysis tools (interpretability, mechanism discovery)
- High-impact co-authored publications

______________________________________________________________________

## Future Directions

Beyond Phase 8 (2027+), potential research directions include:

### 1. Hybrid Behavioral-Cellular Models

- Combine behavioral abstraction (current approach) with selective cellular models
- Example: Behavioral foraging + detailed AFD neuron biophysics for thermotaxis
- Validation: Does behavioral optimization match cellular-level predictions?

### 2. Quantum-Classical Computational Comparisons

- Systematic benchmarking of quantum-inspired vs. purely classical algorithms
- Identify task features where quantum representations provide measurable advantages
- Develop theoretical frameworks explaining when and why quantum approaches help (or don't)

### 3. Clinical Applications

- Neurological disease models: C. elegans models of Alzheimer's, Parkinson's
- Drug discovery: Screen compounds using optimized behavioral assays
- Brain-computer interfaces: Quantum-inspired neural decoding

### 4. Larger-Scale Neural Systems

- Scaling toward larger invertebrate nervous systems (Drosophila ~100K neurons, honeybee ~1M neurons)
- Compositional reasoning: Combine learned modules for novel tasks
- Meta-learning: Learning to learn across domains

### 5. Quantum Advantage for AI

- Beyond C. elegans: Quantum advantages in general RL, NLP, computer vision?
- Theoretical foundations: When does quantum help? (BQP vs. P for AI tasks)
- Hardware roadmap: What quantum devices enable practical AI applications?

______________________________________________________________________

## Technical Debt & Maintenance

The following items are tracked as ongoing maintenance, not blocking new phases:

### High Priority (Fix in Phase 0)

1. **QQLearningBrain Completion**

   - Implement tracking metrics (episode data collection)
   - Add Qiskit runtime integration (for real hardware)
   - Fix parameter initialization (currently hardcoded)

2. **MLPReinforceBrain Loss Bug**

   - Investigate loss calculation (flagged in codebase)
   - Fix incorrect loss computation in later sessions
   - Add unit tests for loss calculation

3. **Grid Size Hardcoding**

   - QQLearningBrain: Grid size hardcoded to 10 instead of derived from environment
   - Fix: Read grid size from environment config

### Medium Priority (Address by Phase 3)

4. **Statistical Analysis Framework**

   - Add confidence intervals to all benchmarks
   - Implement significance testing (t-test, ANOVA, Bonferroni correction)
   - Effect size calculations (Cohen's d)

5. **Visualization Improvements**

   - Gradient flow visualization (show how gradients propagate through surrogate gradient layers)
   - Spike raster plots for spiking networks (membrane potential + spike times)
   - Attention maps for classical networks (if using attention)

6. **Documentation**

   - API documentation
   - Tutorials for new users
   - Video demos

### Lower Priority (Nice-to-Have)

7. **Code Quality**

   - Address remaining Ruff/Pyright warnings
   - Increase test coverage to ≥90%
   - Performance profiling and optimization

8. **Configuration System**

   - Hyperparameter search (grid search, Bayesian optimization)
   - Experiment templates for common tasks
   - Configuration validation improvements

______________________________________________________________________

## Conclusion

This roadmap charts a 3.5-year adaptive path from current strong foundations to potential paradigm-shifting discoveries in quantum computational neuroscience and biological intelligence. The incremental, evidence-driven, phase-based approach ensures:

1. **Scientific Rigor**: Each phase builds on validated results from previous phases
2. **External Validation**: Continuous testing with real C. elegans, quantum hardware, and embodied robots
3. **Theoretical Depth**: Not just empirical comparisons, but mathematical frameworks and universal principles
4. **Practical Impact**: Open-source benchmarks, reproducible science, and technological applications
5. **World-Class Ambition**: Explicitly targeting field-defining contributions and paradigm-shifting discoveries

The project's unique position at the intersection of quantum computing, neuroscience, and AI—grounded in a tractable biological organism (C. elegans) with a fully mapped connectome—provides unparalleled opportunities for breakthrough discoveries.

By systematically mapping C. elegans behaviors across multiple architectures, validating predictions with real biology, and extracting universal principles, this project aims to answer fundamental questions:

- **When do quantum-inspired algorithms outperform classical approaches for modeling biological intelligence?** (Empirical benchmarks)
- **What computational principles enable intelligent behavior?** (Theoretical frameworks)
- **How do we build efficient systems that capture biological decision-making?** (Engineering insights)

The roadmap is ambitious but achievable with disciplined execution, strategic partnerships, and commitment to open science. The adaptive decision gates ensure scientific value even if primary hypotheses fail: negative results are valuable when rigorously documented. Success at the Minimum Viable level would represent solid scientific contribution; Target Success would establish new research directions; Stretch Success would fundamentally advance our understanding of biological intelligence.
