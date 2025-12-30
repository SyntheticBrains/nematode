# Quantum Nematode Project Roadmap

**Vision**: Map all behaviors of *Caenorhabditis elegans* through comparative architecture analysis, with focus on quantum computational approaches, to discover fundamental principles of biological intelligence and demonstrate quantum advantages in cognitively-relevant tasks.

**Version**: 1.0

**Last Updated**: December 2025

**Horizon**: 2025-2028 (8 phases + future directions, ~3.5 years)

______________________________________________________________________

> **Disclaimer**: This roadmap describes a research program with hypotheses to be tested, not established results. Many claims about quantum advantages, biological insights, and scaling properties are research questions requiring empirical validation. Outcomes may differ significantly from projections as evidence accumulates. The adaptive decision gates throughout this document reflect our commitment to evidence-driven pivots when hypotheses are not supported by experimental results.

______________________________________________________________________

**Note**: This roadmap aims for world-class scientific impact through rigorous methodology, external validation, and transformative discoveries at the intersection of quantum computing, neuroscience, and artificial intelligence.

______________________________________________________________________

## Table of Contents

01. [Timeline Overview](#timeline-overview)
02. [Executive Summary](#executive-summary)
03. [Current State (Baseline)](#current-state-baseline)
04. [Phase Roadmap](#phase-roadmap)
    - [Phase 0: Foundation & Baselines](#phase-0-foundation--baselines-q4-2025---q1-2026)
    - [Phase 1: Sensory & Threat Complexity](#phase-1-sensory--threat-complexity-q1-q2-2026)
    - [Phase 2: Architecture Analysis](#phase-2-architecture-analysis-q2-q3-2026)
    - [Phase 3: Learning & Memory](#phase-3-learning--memory-q3-q4-2026)
    - [Phase 4: Evolution & Adaptation](#phase-4-evolution--adaptation-q4-2026---q1-2027)
    - [Phase 5: Social Complexity](#phase-5-social-complexity-q1-q2-2027)
    - [Phase 6: Quantum Frontiers](#phase-6-quantum-frontiers-q2-q3-2027)
    - [Phase 7: Scaling & Real-World](#phase-7-scaling--real-world-q3-q4-2027)
    - [Phase 8: Universality & Impact](#phase-8-universality--impact-q4-2027---q2-2028)
05. [Adaptive Roadmap Philosophy](#adaptive-roadmap-philosophy)
06. [Ongoing Validation Milestones](#ongoing-validation-milestones)
07. [Success Metrics Framework](#success-metrics-framework)
08. [Success Levels](#success-levels)
09. [Relationship to External Projects](#relationship-to-external-projects)
10. [Future Directions](#future-directions)
11. [Technical Debt & Maintenance](#technical-debt--maintenance)
12. [Conclusion](#conclusion)

______________________________________________________________________

## Timeline Overview

| Phase | Timeline | Focus | Key Deliverable |
|-------|----------|-------|-----------------|
| **0** | Q4 2025 - Q1 2026 | Foundation & Baselines | Validated optimization methods, SOTA baselines, first QPU run |
| **1** | Q1 - Q2 2026 | Sensory & Threat Complexity | Thermotaxis, enhanced predators, ablation toolkit |
| **2** | Q2 - Q3 2026 | Architecture Analysis | Interpretability framework, first biological prediction tested |
| **3** | Q3 - Q4 2026 | Learning & Memory | STAM/ITAM/LTAM, associative learning paradigms |
| **4** | Q4 2026 - Q1 2027 | Evolution & Adaptation | Baldwin Effect, evolutionary dynamics comparison |
| **5** | Q1 - Q2 2027 | Social Complexity | Multi-agent infrastructure, emergent behaviors |
| **6** | Q2 - Q3 2027 | Quantum Frontiers | Advanced quantum algorithms, hardware deployment |
| **7** | Q3 - Q4 2027 | Scaling & Real-World | Large environments, neuromorphic, WormBot |
| **8** | Q4 2027 - Q2 2028 | Universality & Impact | Cross-organism transfer, unified theory |

______________________________________________________________________

## Executive Summary

The Quantum Nematode project aims to achieve breakthrough discoveries at the intersection of quantum computing, neuroscience, and artificial intelligence by:

1. **Comparative Architecture Analysis**: Systematically comparing quantum (ModularBrain, QModularBrain), classical (MLPBrain, QMLPBrain), and biologically-realistic (SpikingBrain) architectures across increasingly complex C. elegans behaviors

2. **Incremental Behavioral Coverage**: Progressively implementing sensory systems (chemotaxis ✓, thermotaxis, oxygen sensing, mechanosensation), survival behaviors (foraging ✓, predator evasion ✓, learning/memory), and social dynamics (cooperation, competition, communication)

3. **Theoretical Foundations**: Developing mathematical frameworks linking quantum computational principles to biological intelligence, generating testable hypotheses for experimental validation

4. **External Validation**: Partnering with C. elegans neuroscience labs, deploying on real quantum hardware (IBM Quantum), and testing on embodied platforms (WormBot integration)

5. **Universal Principles**: Extracting domain-invariant computational principles through transfer learning to other organisms (Drosophila, zebrafish) and domains (swarm robotics, multi-objective optimization)

**Key Differentiators** from existing work:

- **vs. OpenWorm**: Complementary focus on neural computation paradigms vs. cellular biophysics; potential integration where optimized policies control OpenWorm's simulated muscles
- **vs. Standard RL Benchmarks**: Biologically-grounded tasks with real organism validation; quantum-classical comparative analysis
- **vs. Quantum ML Research**: Embodied, ecologically-valid tasks requiring multi-objective optimization, long-horizon planning, and sensory integration

**North Star Objective**: Demonstrate that quantum computational approaches can (1) match or exceed classical methods on biologically-relevant decision tasks, (2) reveal new insights into biological neural computation, and (3) establish theoretical foundations for quantum computational neuroscience as a new interdisciplinary field, achieving world-class scientific impact.

______________________________________________________________________

## Current State (Baseline)

### Implemented Capabilities

#### Brain Architectures (5 total)

**Quantum Approaches:**

1. **ModularBrain**: Quantum variational circuits with modular sensory processing, multi-layer entanglement, evolutionary optimization (CMA-ES validated), with parameter-shift rule gradients available for comparison

2. **QModularBrain**: Hybrid quantum-classical Q-learning (early stage, incomplete TODOs)

**Classical Approaches:**

3. **MLPBrain**: REINFORCE policy gradients with baseline subtraction, entropy regularization, learning rate scheduling

4. **QMLPBrain**: Deep Q-Network (DQN) with experience replay

**Biologically-Realistic:**

5. **SpikingBrain**: Leaky Integrate-and-Fire (LIF) neurons with **surrogate gradient descent**, REINFORCE policy gradients, backpropagation through time (BPTT), PyTorch autograd integration. Previous STDP implementation replaced due to fundamental learning failures.

#### Environments

1. **StaticEnvironment**: Single-goal maze navigation (legacy benchmark)

2. **DynamicForagingEnvironment**: Multi-food foraging with:

   - Satiety-based homeostasis (hunger decays, food replenishes)
   - Gradient-based chemotaxis (exponential decay superposition)
   - Viewport perception (no visual input, gradient-only)
   - Distance efficiency tracking
   - Exploration rewards for novel cell visits

3. **Predator Evasion** (added Nov 2025):

   - Multi-predator support with random movement
   - Detection radius and kill radius
   - Unified gradient field: food attraction + predator repulsion
   - Multi-objective learning: food collection vs. survival
   - Proximity penalties and death penalties
   - Predator metrics: encounters, evasions, survival rate

#### Benchmarking & Tracking

- 14 benchmark categories across static/dynamic, quantum/classical, with/without predators
- Automated experiment tracking with session IDs
- Per-run and session-level metrics (success rate, steps, convergence run, stability)
- CSV exports and matplotlib visualizations
- Leaderboard system with contributor attribution

#### Current Performance Snapshot (December 2025)

**Static Maze:**

- **Quantum (ModularBrain)**: 0.980 score (100% success, 34 steps, converge@20)
- **Classical (MLPBrain)**: 0.960 score (100% success, 24 steps, converge@20)
- **Spiking (SpikingBrain)**: 0.932 score (100% success, 67 steps, converge@34) - *surrogate gradients*

**Dynamic Small (≤20x20):**

- **Quantum (ModularBrain)**: 0.762 score (100% success, 207 steps, converge@20) - *CMA-ES evolution*
- **Classical (MLPBrain)**: 0.822 score (100% success, 181 steps, converge@20)
- **Spiking (SpikingBrain)**: 0.733 score (100% success, 267 steps, converge@22) - *surrogate gradients*

**Dynamic Predator Small (≤20x20):**

- **Quantum (ModularBrain)**: 0.675 score (95% success, 224 steps, converge@29) - *CMA-ES evolution*
- **Classical (MLPBrain)**: 0.740 score (92% success, 199 steps, converge@30)
- **Spiking (SpikingBrain)**: 0.556 score (63% success, 247 steps, converge@20) - *surrogate gradients*

**Key Findings**:

1. **Quantum-classical gap nearly closed**: With evolutionary optimization (CMA-ES), quantum achieves 0.762 vs classical 0.822 on dynamic foraging
2. **Spiking as a viable model**: Surrogate gradient approach achieving 0.733 on foraging tasks
3. **Predator tasks remain challenging**: All architectures show lower scores due to multi-objective complexity

### Known Gaps & Technical Debt

**Architecture Gaps:**

- QModularBrain incomplete (missing: tracking metrics, Qiskit runtime integration, parameter initialization)
- MLPBrain loss calculation bug (flagged in codebase)
- No multi-brain ensemble or hierarchical decision systems
- No transfer learning framework

**Simulation Gaps:**

- No energy/metabolic model (satiety is abstract, not ATP-based)
- No neuromodulation (dopamine/serotonin mentioned in docs but not simulated)
- Limited environmental diversity (no temporal dynamics, food quality variations)
- No thermotaxis, oxygen sensing, or mechanosensation implementations yet

**Experimental Gaps:**

- No SOTA RL baselines for comparison (PPO, SAC, TD3, DreamerV3)
- No real C. elegans behavioral datasets for validation
- Limited interpretability tools (no attention maps, gradient visualization, spike rasters)
- No statistical analysis framework (confidence intervals, effect sizes, significance testing)

### Research Questions

1. **Quantum Advantage**: Under what task conditions do quantum approaches outperform classical? (Multi-objective optimization? Uncertainty? Exploration?)
2. **Biological Plausibility**: Do optimized quantum circuits reveal insights into real C. elegans neural computation?
3. **Scalability**: Can these approaches scale beyond 302 neurons to larger invertebrate nervous systems (Drosophila ~100K, honeybee ~1M neurons)?
4. **Universality**: Do learned principles transfer to other organisms, domains, or tasks?
5. **Hardware Viability**: How do real quantum devices (IBM QPU) perform vs. simulation on these tasks?

### Recent Breakthrough: Evolutionary Optimization (December 2025)

**Key Finding**: Evolutionary optimization (CMA-ES) achieves **4x better performance** than gradient-based learning on quantum circuits for complex predator evasion tasks.

**Performance Data:**

- **Evolutionary optimization (CMA-ES)**: **88% success rate** on dynamic predator evasion (small grid)
- **Gradient-based learning**: Only **22.5% success rate** on the same task
- **Quantum-classical gap**: Reduced from **70%** (22.5% quantum vs 92% classical) to just **4%** (88% vs 92%)

**Critical Insight**: Gradient-based learning with parameter-shift rule shows high variance on quantum circuits due to statistical noise from finite shot counts in gradient estimation. This noise causes quantum circuits to converge to suboptimal local optima compared to gradient-free evolutionary methods.

**Implication for Roadmap**: Evolution-based optimization (CMA-ES, Genetic Algorithms) is now the **validated baseline** for training quantum variational circuits. Gradient methods remain effective for classical networks (MLPBrain, QMLPBrain) but should not be assumed to work for quantum approaches without extensive validation.

This breakthrough validates the adaptive, evidence-driven approach throughout this roadmap: methods are prioritized based on empirical results, not assumptions.

### Recent Breakthrough: Spiking Neural Network Rewrite (December 2025)

**Key Finding**: Complete architectural rewrite of SpikingBrain from STDP to **surrogate gradient descent** enables successful learning for the first time.

**Previous Implementation Problems:**

- **STDP (Spike-Timing Dependent Plasticity)** had critical bugs: wrong input preprocessing, broken credit assignment, missing weight updates
- **Limitations for this implementation**: Standard STDP without reward modulation cannot implement effective credit assignment for sparse-reward reinforcement learning tasks
- **Result**: 0% success rate over 400 episodes of testing

**New Implementation (Surrogate Gradient Descent):**

- **Differentiable spike function**: Smooth approximation enables backpropagation through spiking neurons
- **REINFORCE policy gradients**: Same proven algorithm as MLPBrain with discounted returns and baseline subtraction
- **LIF neurons preserved**: Maintains biological plausibility with Leaky Integrate-and-Fire dynamics
- **PyTorch autograd integration**: Standard gradient-based optimization with gradient clipping

**State-of-the-Art Alignment**: This approach matches modern neuromorphic research (SpikingJelly, snnTorch, Norse) which all use surrogate gradients for training spiking neural networks on complex tasks.

**Implication for Roadmap**: SpikingBrain is now a viable architecture for comparative analysis. Gradient-based methods (with surrogate gradients) work for spiking networks, complementing the evolution-first approach for quantum architectures.

______________________________________________________________________

## Phase Roadmap

### Phase 0: Foundation & Baselines (Q4 2025 - Q1 2026)

**Goal**: Establish rigorous baselines using validated optimization methods, fix critical technical debt, and create public benchmark infrastructure for reproducible science.

#### Deliverables

1. **Validated Optimization Baselines** (CRITICAL PRIORITY)

   - **PRIMARY: Evolutionary optimization** for at least one quantum architecture (ModularBrain, QModularBrain)
     - CMA-ES (validated: 88% success on predator tasks)
     - Genetic Algorithms (population-based search)
     - Compare variants: (μ, λ)-ES, Natural Evolution Strategies
   - **SECONDARY: Gradient methods** for at least one of each classical and spiking architectures (MLPBrain, QMLPBrain, SpikingBrain)
     - REINFORCE, DQN (already implemented for classical)
     - SpikingBrain: Surrogate gradient descent with REINFORCE (December 2025 rewrite)
     - Validate that gradients remain effective for non-quantum architectures
   - Systematic comparison across all benchmarked architectures with consistent evaluation protocol
   - Document which optimization methods work for which architecture types

2. **SOTA RL Baselines**

   - Implement modern RL algorithms: Proximal Policy Optimization (PPO), Soft Actor-Critic (SAC), Twin Delayed DDPG (TD3)
   - Run on all current benchmark categories for credible classical comparison
   - Establish performance ceiling: "What's the best classical performance we can achieve?"
   - Identify where quantum/classical/spiking excel relative to SOTA

3. **Real C. elegans Behavioral Datasets**

   - Curate datasets from literature: chemotaxis indices, foraging paths, predator escape trajectories
   - Define standard metrics matching biological experiments (e.g., chemotaxis index CI = (N_attractant - N_control) / N_total)
   - Create validation benchmarks: "Does our agent match real worm behavior?"

4. **NematodeBench: Public Benchmark Suite**

   - Standardized task suite with reproducible configurations
   - Public leaderboard (GitHub Pages or dedicated site)
   - Submission guidelines, evaluation scripts, and reproducibility requirements
   - Target: First external research group submission

5. **Technical Debt Resolution**

   - Fix QModularBrain TODOs (tracking, Qiskit runtime, initialization)
   - Fix MLPBrain loss calculation bug
   - Code quality improvements (address remaining Ruff/Pyright warnings)

6. **First Quantum Hardware Run**

   - Deploy ModularBrain on IBM Quantum backend (simulator → real QPU)
   - Measure performance degradation from noise
   - Establish hardware deployment pipeline for ongoing testing

#### Metrics Focus

- **Optimization validation**: Confirm evolutionary methods work for quantum, gradients for classical
- **SOTA baselines**: Establish performance ceiling with modern RL
- **Biological fidelity**: Correlation with real C. elegans data
- **Reproducibility**: External researchers can replicate results

#### Phase 0 Exit Criteria

**Required (must complete before Phase 1):**

- ✅ At least 1 quantum architecture (ModularBrain) achieves **>70% success** on at least 2 benchmark tasks using evolutionary optimization
- ✅ At least 1 classical architecture (MLPBrain or QMLPBrain) benchmarked with gradient-based methods
- ✅ Classical SOTA baseline (PPO or SAC) achieves **>85% success** on foraging tasks
- ✅ Clear documentation of which optimization methods work for which architectures
- ✅ At least 1 real C. elegans behavioral dataset integrated for validation

**Stretch (complete if time permits, can continue into Phase 1):**

- ✅ ModularBrain successfully runs on IBM QPU hardware (simulator → real QPU)
- ✅ SpikingBrain benchmarked alongside quantum and classical
- ✅ All 5 brain architectures benchmarked on consistent evaluation protocol
- ✅ At least 3 real C. elegans behavioral datasets integrated
- ✅ Public benchmark site launched with ≥1 external submission
- ✅ All critical technical debt resolved

#### Go/No-Go Decision

**GO if**: Quantum architectures show promise (>70% success) OR reveal interesting failure modes worth investigating.
**PIVOT if**: Quantum shows no promise → Focus Phase 1+ on classical vs. spiking comparison only, document "why quantum didn't work" for publication.
**STOP if**: No architecture generalizes to basic tasks → Re-evaluate fundamental approach.

______________________________________________________________________

### Phase 1: Sensory & Threat Complexity (Q1-Q2 2026)

**Goal**: Enrich sensory input and predator behaviors to match C. elegans multi-modal perception and ecological complexity.

**Pilot-Then-Focus Approach**: Start with chemotaxis (already implemented) + thermotaxis as the priority sensory pair. If architectures handle dual-modality well, add oxygen sensing and mechanosensation. For predators, start with stationary + pursuit types; add patrol and group hunting only if simpler predators work well. This avoids building complex sensory systems that architectures can't yet handle.

#### Deliverables

1. **Enhanced Predator Behaviors**

   - **Stationary Predators** \[Priority: Implement First\]: Nematode-trapping fungi (sticky patches at fixed locations), toxic bacteria zones (invisible danger fields)
   - **Active Pursuit** \[Priority: Implement Second\]: Predatory nematode tracking behavior (move toward agent within detection radius)
   - **Patrol Patterns** \[Conditional\]: Fixed routes, circular paths, territorial zones
   - **Group Hunting** \[Optional: if simpler predators work\]: Coordinated multi-predator attacks (2+ predators converge on target)
   - **Dynamic Threat Levels**: Predator danger scales with proximity (graduated penalties vs. binary)

2. **Health System Alternative**

   - Replace instant death with HP-based damage model
   - Predator encounters deal damage (e.g., -10 HP per collision)
   - Food provides healing (+5 HP per food item)
   - Strategic trade-offs: risky paths with more food vs. safe paths with less food
   - Configurable: toggle between instant-death and HP modes

3. **Thermotaxis System** [Priority: First new sensory modality]

   - Spatial temperature gradient fields (2D grid with temperature values)
   - AFD neuron simulation: temperature memory storage (cultivation temperature Tc)
   - Isothermal tracking behavior (move along preferred temperature contours)
   - Association learning: temperature ↔ food availability (high food at 20°C → prefer 20°C)
   - Integration with foraging: agents perceive food gradients + temperature gradients
   - **Validation gate**: If thermotaxis works (>60% success), proceed to oxygen sensing

4. **Oxygen Sensing System** [Conditional on thermotaxis success]

   - Oxygen concentration gradient fields (5-12% optimal range)
   - Hypoxia avoidance (\<5% O2) and hyperoxia avoidance (>12% O2)
   - URX/AQR/PQR neuron simulation (detect high O2)
   - BAG neuron simulation (detect low O2)
   - Multi-objective: balance food quality vs. oxygen comfort

5. **Mechanosensation (Touch Response)** [Parallel track: mechanistically simpler than gradient sensing]

   - Obstacle collision detection (walls, barriers)
   - Gentle touch: triggers local exploration (increased turning)
   - Harsh touch: triggers escape response (rapid reversal, omega turn)
   - Wall-following behavior option
   - Integration with predator evasion: physical contact with predator = harsh touch → immediate escape
   - **Note**: Can be developed in parallel with thermotaxis as a fallback if gradient-based sensing proves difficult

6. **Architecture Ablation Studies**

   - Systematically remove components from each architecture (e.g., remove entanglement from quantum circuits, remove hidden layers from MLP)
   - Measure performance degradation: which features are critical?
   - Cross-architecture feature importance analysis
   - Identify minimal sufficient architectures for each task

#### Metrics Focus

- **Generalization**: Transfer to unseen predator types, temperature ranges, oxygen levels
- **Biological fidelity**: Match C. elegans chemotaxis indices, thermotaxis precision, escape latencies
- **Architecture efficiency**: Which features are necessary vs. redundant?

#### Phase 1 Exit Criteria

- ✅ Thermotaxis implemented and integrated with chemotaxis (dual-modality tasks)
- ✅ At least 1 additional sensory modality (oxygen or mechanosensation) if thermotaxis succeeds
- ✅ ≥2 predator behavior types (stationary + pursuit) implemented and tested
- ✅ Health system option available and benchmarked
- ✅ Ablation toolkit operational with automated feature importance ranking
- ✅ Performance profiles documented: where each architecture excels (e.g., "quantum best at multi-objective, classical best at single-objective")
- ✅ Biological validation: At least 1 behavior quantitatively matches C. elegans data (e.g., thermotaxis precision, escape latency)

#### Go/No-Go Decision

**GO if**: At least one architecture generalizes well across multiple sensory modalities (>60% success on ≥3 tasks).
**PIVOT if**: No architecture generalizes → Focus on hybrid approaches combining architecture strengths, or specialize architectures for specific tasks.
**STOP if**: Sensory complexity breaks all architectures → Simplify task suite or re-examine architecture fundamentals.

______________________________________________________________________

### Phase 2: Architecture Analysis (Q2-Q3 2026)

**Goal**: Move beyond "which architecture wins?" to "why do architectures work?" through systematic analysis, interpretability, and mechanism discovery.

#### Deliverables

1. **Interpretability Framework**

   - **Quantum**: Circuit visualization, gate importance analysis (parameter sensitivity), superposition state tracking
   - **Classical**: Attention maps (if using attention mechanisms), activation analysis, saliency maps (which inputs drive decisions?)
   - **Spiking**: Spike raster plots, connectivity analysis, neuron firing patterns, membrane potential dynamics, surrogate gradient flow visualization
   - Unified API: `architecture.interpret(state, action)` returns explanation

2. **Feature Importance Across Architectures**

   - Which sensory inputs are most critical? (Chemotaxis gradient magnitude? Direction? Satiety level?)
   - How do architectures differ in feature reliance? (Does quantum use different cues than classical?)
   - Integrated Gradients, SHAP values, or similar for attribution
   - Cross-architecture comparison: "Quantum prioritizes gradient direction 60% vs. classical 40%"

3. **Mechanism Discovery Protocol**

   - Automated hypothesis generation from model analysis
   - Example: "Quantum model uses superposition to simultaneously explore approach/avoid strategies"
   - Translation to biological hypotheses: "Does C. elegans use [X] mechanism? Test with [Y] experiment"
   - Partnership with C. elegans labs to design validation experiments

4. **First Biological Prediction Tested**

   - Identify a novel prediction from model behavior (e.g., "optimal escape angle from predators is 135° based on quantum model")
   - Collaborate with neuroscience lab to test with real C. elegans
   - Publication: Model prediction → experimental validation loop

5. **Comparative Analysis Methodology**

   - Statistical testing framework: confidence intervals, effect sizes (Cohen's d), significance tests (t-test, ANOVA, Bonferroni correction)
   - Benchmark visualizations: heatmaps, performance profiles, scaling curves
   - Reproducibility toolkit: containerized environments, seed management, deterministic benchmarks

6. **Architecture Comparison Whitepaper**

   - Comprehensive analysis: When does each architecture excel? Why?
   - Computational cost comparison: parameters, FLOPs, wall-clock time, energy consumption
   - Scalability analysis: how performance changes with environment complexity (grid size, food count, predator count)

#### Metrics Focus

- **Interpretability**: Can we explain why an action was chosen?
- **Mechanism discovery**: Generate ≥1 testable biological hypothesis per architecture
- **Statistical rigor**: All comparisons have confidence intervals and significance tests

#### Phase 2 Exit Criteria

- ✅ Interpretability toolkit operational for all architectures
- ✅ Feature importance analysis reveals architecture-specific strategies (e.g., "quantum prioritizes gradient direction 60% vs. classical 40%")
- ✅ Comparative framework published: preprint or conference paper
- ✅ Quantum advantage clearly demonstrated on **at least 2 tasks** OR compelling negative result documented ("why quantum didn't provide advantages")
- ✅ Theory connecting architecture properties to task performance (e.g., "entanglement correlates with multi-objective optimization performance")
- ✅ ≥1 biological prediction tested and published in peer-reviewed journal
- ✅ Statistical analysis framework integrated into all benchmarks (confidence intervals, significance tests)
- ✅ Mechanism discovery protocol yields ≥3 testable biological hypotheses

#### Go/No-Go Decision

**GO if**: Quantum shows advantage on ≥2 tasks OR reveals interesting computational principles worth investigating further.
**PIVOT if**: No quantum advantage found → Reframe as "architecture comparison study" focusing on classical vs. spiking. Publish "Why Quantum Didn't Work: Lessons from Biological Navigation" as valuable negative result.
**STOP if**: Unable to explain why any architecture works → Need better interpretability tools or simpler tasks.

______________________________________________________________________

### Phase 3: Learning & Memory (Q3-Q4 2026)

**Goal**: Implement associative learning and memory systems matching C. elegans biological timescales and mechanisms.

**Pilot-Then-Focus Approach**: Start with Short-Term Associative Memory (STAM) as the foundation. If STAM demonstrates value (improves performance over static policies), extend to ITAM and LTAM. This avoids over-engineering memory systems before validating the core concept.

#### Deliverables

1. **Short-Term Associative Memory (STAM)** [Priority: Implement First]

   - Duration: Minutes to 30 minutes (timescale matches biology)
   - Timescale modeling: Exponential decay parameters calibrated to match biological cAMP-mediated memory (~minutes)
   - No protein synthesis required (immediate formation)
   - Use cases: Remember recent food locations, recent predator encounters
   - **Validation gate**: If STAM improves foraging efficiency by ≥10%, proceed to ITAM

2. **Intermediate-Term Associative Memory (ITAM)** [Conditional on STAM success]

   - Duration: 30 minutes to several hours
   - Timescale modeling: Two-pathway decay model inspired by cAMP + CaMKII signaling dynamics
   - Memory consolidation gate: Simulates protein synthesis requirement to extend beyond 30 min
   - Use cases: Learn temperature-food associations over multiple foraging bouts

3. **Long-Term Associative Memory (LTAM)** [Conditional on ITAM value]

   - Duration: Hours to days (persist across multiple simulation sessions)
   - Training paradigm: Spaced training (multiple sessions with intervals) vs. massed training (single long session)
   - Biological inspiration: Timescales matched to protein synthesis + CREB-dependent consolidation in real C. elegans
   - Memory traces stored to disk, reloaded in subsequent sessions
   - Use cases: Persistent pathogen avoidance, learned temperature preferences

4. **Associative Learning Paradigms**

   - **Classical conditioning**: Odor (CS) + food (US) → approach odor
   - **Operant conditioning**: Action (e.g., turn left) → reward (food) → repeat action
   - **Aversive learning**: Pathogen exposure → avoid pathogen
   - **Context conditioning**: Temperature + food → prefer that temperature
   - **Extinction learning**: Reward stops → unlearn association (forgetting)

5. **Memory Decay & Forgetting**

   - Protein synthesis for proper memory decay (biology: forgetting is active process)
   - Configurable decay rates: STAM decays in minutes, LTAM decays in days
   - Interference: new learning can overwrite old memories

6. **Theoretical Framework: Quantum-Inspired Memory Models**

   - Mathematical models exploring whether quantum-inspired representations offer advantages for memory encoding
   - Research question: Do quantum circuit representations capture memory uncertainty better than classical networks?
   - Compare quantum vs. classical memory architectures on associative learning benchmarks
   - Note: This explores computational advantages, not claims about quantum effects in biological neurons

7. **Neural Circuits for Learning**

   - RIM interneurons: Integrate chemosensory + mechanosensory for associative learning
   - NMDA receptors (NMR-1): Required for context conditioning
   - Dopaminergic neurons: Reward signaling, motivation
   - Serotonergic neurons: Modulate learning intensity

8. **Biological Accuracy Revisit: Temporal Sensory Systems**

   > **Note**: Once memory systems are operational, revisit the biological accuracy of sensory modalities implemented in Phase 1 and earlier. Current implementations use spatial gradient sensing (computationally equivalent for stateless brains), but real C. elegans uses temporal sensing for several modalities:
   >
   > - **Thermotaxis**: AFD neurons use temporal derivative (dT/dt) to detect temperature changes over time, not instantaneous spatial gradients
   > - **Chemotaxis**: ASE neurons perform temporal concentration comparisons during head sweeps
   > - **Oxygen sensing**: URX/BAG neurons integrate oxygen changes over time
   >
   > With STAM/ITAM memory systems available, implement biologically-accurate temporal sensing that compares current vs. recent sensory values. This would require memory buffers and temporal derivative computation, making it a natural extension of Phase 3 memory infrastructure.
   >
   > **Deliverable**: Temporal sensing module for thermotaxis (and optionally chemotaxis, aerotaxis) that uses memory to compute dT/dt, dC/dt gradients instead of spatial approximations.

#### Metrics Focus

- **Memory timescales**: Match biological STAM (minutes), ITAM (hours), LTAM (days)
- **Biological insight**: Do models reveal memory mechanisms? (e.g., role of protein synthesis, spaced training)
- **Learning efficiency**: Sample complexity to learn associations (fewer trials = better)

#### Phase 3 Exit Criteria

- ✅ STAM implemented and validated (improves performance over static policies by ≥10%)
- ✅ At least 1 additional memory timescale (ITAM or LTAM) if STAM proves valuable
- ✅ ≥2 associative learning paradigms functional (classical conditioning + one other)
- ✅ Meta-learning demonstrates improvement over static policies (agents that learn perform better than fixed policies)
- ✅ Memory persistence across sessions demonstrated (save/load memory traces work correctly)
- ✅ Quantum vs. classical memory comparison reveals mechanistic differences

#### Go/No-Go Decision

**GO if**: Learning and memory systems improve performance over static policies by ≥10%.
**PIVOT if**: Learning doesn't improve performance → Focus on innate behavior repertoire mapping. Document "why learning didn't help" as interesting negative result.
**STOP if**: Memory systems are too complex to implement reliably → Simplify to single timescale or remove temporal dynamics.

______________________________________________________________________

### Phase 4: Evolution & Adaptation (Q4 2026 - Q1 2027)

**Goal**: Implement breeding and evolutionary algorithms to evolve optimal learning strategies, comparing evolutionary dynamics across architectures.

**Building on Phase 0**: This phase extends the evolutionary optimization foundations established in Phase 0 (CMA-ES for quantum training) to full genetic algorithm systems with crossover, mutation, and population dynamics.

**Pilot-Then-Focus Approach**: Start with lightweight pilots of 2-3 evolutionary approaches (hyperparameter evolution, Lamarckian, Baldwin Effect) using small populations and few generations. Based on pilot results, select 1-2 approaches for deep investigation. This prevents over-investment in approaches that don't work for our specific architectures.

#### Deliverables

1. **Hyperparameter Evolution (Simplest)** [Priority: Pilot First]

   - Genome = learning rates, layer sizes, circuit depths, reward weights
   - Crossover: Blend parent hyperparameters (weighted average, uniform crossover)
   - Mutation: Random perturbations (Gaussian noise, random resampling)
   - Fitness metric: Final performance after fixed number of training episodes
   - Selection: Tournament selection, rank-based, fitness-proportionate
   - Use case: Find optimal hyperparameter sets for each architecture

2. **Lamarckian Evolution (Fast Convergence)** [Pilot alongside Hyperparameter]

   - Each agent learns during lifetime (current RL approach)
   - Genome = initial weights/parameters
   - Fitness = final performance after learning
   - Offspring inherit learned weights (not biologically accurate but fast)
   - Use case: Rapidly evolve high-performing initial conditions

3. **Baldwin Effect (Biologically Realistic)** [Conditional: if Lamarckian shows promise]

   - Each agent learns during lifetime
   - Genome = learning capacity (learning rates, architectural features)
   - Fitness = final performance
   - Offspring inherit *ability to learn*, not learned weights
   - Over generations: Learned behaviors become innate (genetic assimilation)
   - Use case: Study how learning guides evolution

4. **Architecture Evolution (NEAT-style)** [Optional: based on pilot findings]

   - Genome = network topology + weights
   - Crossover: Combine network structures, preserve innovations
   - Mutation: Add/remove neurons, connections, quantum gates
   - Speciation: Protect novel architectures from competition during early stages
   - Use case: Discover novel hybrid quantum-classical architectures

5. **Co-Evolution (Predators + Prey)** [Optional: if basic evolution works well]

   - Predators evolve hunting strategies simultaneously with prey evolving evasion
   - Red Queen dynamics: Arms race between predator and prey
   - Fitness: Prey = survival rate, Predators = kill rate
   - Use case: Study evolutionary pressures shaping intelligence

6. **Evolutionary Algorithm Comparisons**

   - Benchmark against CMA-ES, genetic programming, evolution strategies
   - Which EA works best for each architecture?

7. **Generational Fitness Tracking**

   - Visualize fitness curves over generations
   - Track diversity (genotypic and phenotypic)
   - Detect convergence, stagnation, speciation events
   - Archive best genomes for reproduction

#### Metrics Focus

- **Evolution of learning strategies**: Do populations discover better learning algorithms?
- **Convergence speed**: Generations to reach optimal performance
- **Architectural innovation**: Do new architectures emerge from NEAT-style evolution?

#### Phase 4 Exit Criteria

- ✅ ≥2 evolution approaches piloted with documented results
- ✅ At least 1 approach selected for deep investigation based on pilot performance
- ✅ Selected approach(es) produce novel behaviors not present in training set
- ✅ Baldwin Effect or Lamarckian inheritance demonstrated (learned behaviors become innate over generations) OR documented why neither worked
- ✅ Evolutionary dynamics comparison (quantum vs. classical) completed showing convergence rates and diversity
- ✅ Generational fitness tracking shows continuous improvement over ≥50 generations

#### Go/No-Go Decision

**GO if**: Evolution produces novel, high-performing behaviors OR demonstrates Baldwin Effect.
**PIVOT if**: Evolution plateaus quickly → Focus on hand-designed architectures and learning algorithms. Document evolutionary limitations.
**STOP if**: Evolutionary algorithms fail to converge or produce unstable results → Revisit fitness functions or population parameters.

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

   - **Intel Loihi**: Deploy SpikingBrain (LIF neurons with surrogate-gradient-trained weights) on neuromorphic chip
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
- ✅ SpikingBrain deployed on Intel Loihi with energy efficiency analysis (Joules per decision vs. GPU/CPU)
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

- **Phase 0**: First successful QPU run (ModularBrain on IBM Quantum)
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

1. **QModularBrain Completion**

   - Implement tracking metrics (episode data collection)
   - Add Qiskit runtime integration (for real hardware)
   - Fix parameter initialization (currently hardcoded)

2. **MLPBrain Loss Bug**

   - Investigate loss calculation (flagged in codebase)
   - Fix incorrect loss computation in later sessions
   - Add unit tests for loss calculation

3. **Grid Size Hardcoding**

   - QModularBrain: Grid size hardcoded to 10 instead of derived from environment
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
