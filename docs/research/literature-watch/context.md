# Project brief for literature triage

Hand-maintained. This file is the entire definition of "relevant" for the weekly watch — the
scoring model sees this text and nothing else about the project. Keep it to roughly this length:
it is read once per batch, and detail that does not change a score is detail that costs money.

**Review it at each phase close** — principle 13 of the [phase protocol](../phase-protocol.md).
A brief describing a phase that closed six months ago scores against questions nobody is asking any
more, and the digest will look fine while doing it.

Last reviewed: 2026-10-14 — re-aimed at the Phase 8 close (Logbook 094), for the questions the
phase after 8 opens on. Previous reviews 2026-10-03 (8a close) and 2026-09-20 (Phase 8's start).

## What the project is

A closed-loop sensory-motor simulation platform. A simulated *C. elegans* forages, evades
predators, and navigates thermal and oxygen gradients, while a pluggable brain architecture drives
it — feedforward, recurrent, spiking, reservoir, quantum, and, as the focal comparison, a network
constrained to the real 302-neuron connectome. Architectures are trained by reinforcement learning
or evolved, ranked under a paired-seed statistical protocol, and validated against published
*C. elegans* behavioural data (klinokinesis and weathervane bias curves, thermotaxis).

The organising question is which architecture best learns nematode behaviours, and whether the
real wiring confers an advantage over matched nulls. Quantum circuits are one architecture family
in that comparison, not the point of the project.

## What is open right now

These are the live questions — what [Phase 8's close](../../experiments/logbooks/094-phase8-synthesis.md)
leaves open, ahead of the next phase's plan review. A paper bearing on one of them is a 3.

- **What a "wiring advantage" over a null is made of.** A connectome-constrained network here learns
  faster than degree-preserving rewired nulls, and the advantage turned out to be the null's rewired
  *gap junctions* and its rewired sensory-to-motor boundary; against a null that keeps both, no
  interior advantage is detectable. What holds is **gap-junction placement**: a null whose electrical
  synapses alone are rewired learns slower, not because of each junction's strength. Anything on
  null-model design for connectomes is directly on point — degree-, strength-, sign- or
  boundary-preserving nulls, nulls that hold each neuron's total coupling, what a null fails to
  preserve, shortcut artefacts — and any structure-function claim about electrical synapses or gap
  junction placement in any connectome, especially one tested against a matched null.
- **Learning through a body.** The anatomical motor-to-muscle map drives a 12-segment kinematic body,
  and the wiring's gap-junction advantage survives it. Connectome-driven or learned locomotion
  controllers, forward/backward switching, reduced-order body models, and connectome-through-a-body
  work with a null. Compute is the binding constraint, so fast models are as interesting as accurate
  ones.
- **Why learned policies steer but never modulate runs.** Every substrate and sense here learns the
  weathervane and never klinokinesis (turning more when heading the wrong way). Work on how
  klinokinesis and weathervaning arise — reward, action space, timescale, sensory adaptation — in
  real worms or in models, and on thermotaxis strategies (AFD–AIY–AIZ, isothermal tracking).
- **Training stability over long horizons.** Policy-gradient learners here learn and then collapse
  late in long runs. Diagnoses and remedies for late policy collapse in PPO-like learners, especially
  in small biological networks.
- **Behaviour precise enough to validate a body against.** Locomotion kinematics, forward and reverse
  bouts, roaming/dwelling and patch-leaving on structured lawns, chemotaxis or thermotaxis
  quantification with magnitudes (not only signs), and methods for separating taxis from wall effects.

## Also worth knowing about

Score 2, not 3: useful without settling anything above.

- **Gap junctions as dynamics** — electrical synapses as ohmic coupling in a leaky network,
  electrical-synapse plasticity, stiffness of graded networks with strong coupling. *Demoted from 3
  on 2026-10-14: the dynamical rung here learned far slower than the memoryless one and closed
  (Logbook 080); plastic gap strengths on the settling substrate did not move the wiring effect.*
- **Operating point** — whether a wiring effect survives the learner's settings, connectome reservoir
  computing, hyperparameter sensitivity surfaces. *Demoted from 3 on 2026-10-14: the depth effect
  here was explained by the sensory-to-motor boundary.*
- **Cross-connectome transfer** — *C. elegans* to *P. pacificus* head circuits, the dauer wiring
  state, homology between nematode nervous systems, and wiring changes across development or life
  stage. *Demoted from 3 on 2026-09-20: Phase 8 stays on one species deliberately. Still worth
  seeing, because the question returns in a later phase and the data landscape moves meanwhile.*
- **Measured synaptic weights and signs on connectome edges** — fitted weights, functional
  connectivity atlases, per-connection sign predictions. *Demoted from 3 on 2026-10-03: measured
  weights did not move the wiring effect under gradient learning here, and the result under the
  local learner was unresolved, so the rung is closed for this phase; new data or a per-connection
  sign resource is still worth seeing.*
- **Neuromodulators on connectome models** — serotonin, PDF and other extrasynaptic signalling
  placed on wired networks, in-silico mutants graded against published phenotypes, and models of
  roaming/dwelling persistence (NSM, mutual inhibition). *Added 2026-10-11 after B.3 (Logbook 090):
  the trained connectome used both modulators, but not in the mutants' way, and weakened PDF's
  inhibition of NSM; work that constrains or imposes these pathways is what a follow-up builds on.*
- **Placed plasticity** — a learning rule at an anatomically identified site against a matched
  random subset — and diagnosed failures of local rules.
- New connectome datasets or releases for any organism, and the tooling to use them.
- Whole-organism or whole-brain simulation efforts in any organism, including ones with neither
  learning nor controls — they bound what this project can claim as novel.
- Three-factor, neuromodulated and eligibility-trace rules in general; reservoir computing and
  recurrent or state-space architectures evaluated on *biological* tasks, particularly negative
  results.
- Methods for comparing learned solutions across architectures; representational or dynamical
  similarity measures; statistical protocol for paired-seed model comparison.
- *C. elegans* behavioural quantification beyond the validation targets named above.
- Neuromorphic hardware running connectome-scale networks.

## What is not relevant

Score 0. These dominate the raw sweep.

- *C. elegans* as a genetics, ageing, toxicology, drug-screening, or disease-model organism, with
  no circuit, behaviour, or learning content. Most papers mentioning the worm are this.
- Machine learning with no biological grounding: benchmark results, language models, vision, and
  general deep-learning methods, including ones that borrow neuroscience vocabulary.
- Clinical and cognitive neuroscience in humans; neuroimaging; psychiatry.
- Quantum computing that is not about learning or neural architectures.
- Molecular and cellular neuroscience below the circuit level — receptor pharmacology, channel
  biophysics, synapse molecular composition — unless it constrains a network-level model.
