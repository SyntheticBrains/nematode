# Project brief for literature triage

Hand-maintained. This file is the entire definition of "relevant" for the weekly watch — the
scoring model sees this text and nothing else about the project. Keep it to roughly this length:
it is read once per batch, and detail that does not change a score is detail that costs money.

**Review it at each phase close** — principle 13 of the [phase protocol](../phase-protocol.md).
A brief describing a phase that closed six months ago scores against questions nobody is asking any
more, and the digest will look fine while doing it.

Last reviewed: 2026-10-03 — re-aimed at the 8a close and the 8b re-plan (roadmap v4.4). Previous
review 2026-09-20, at Phase 8's start.

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

These are the live questions — the second half of Phase 8 of the [roadmap](../../roadmap.md),
*ground, then embody*. A paper bearing on one of them is a 3.

- **What a "wiring advantage" over a null is made of.** A connectome-constrained network here
  reaches competence sooner than degree-preserving rewired nulls, and the advantage survived a
  shared-initialisation control — but about half of it came from how the null rewired the *gap
  junctions*, and the rest depends on the null manufacturing short sensory-to-motor routes the real
  wiring lacks. Anything on null-model design for connectomes is directly on point: degree-,
  strength-, sign- or boundary-preserving nulls, what a null fails to preserve, shortcut or
  path-length artefacts, shared-initialisation controls, and structure-function claims in any
  connectome that do or do not survive a stronger null. So is anything predicting *which* graphs
  learn faster from structure alone.
- **Gap junctions as dynamics.** Electrical synapses here are a fixed symmetric matrix; the next
  rung gives neurons time constants, couples them ohmically, and makes the gap junctions plastic
  under gradient learning. Gap-junction function in network models, electrical-synapse plasticity,
  synchronisation, stiffness and integration of graded networks with strong coupling, and intrinsic
  or node-level adaptation.
- **Learning through a body.** The next shipment drives the anatomical motor-to-muscle map into a
  kinematic two-dimensional body whose rhythm comes from a body-level generator, and asks whether
  the wiring matters for segmental drive and steering. Connectome-driven or learned locomotion
  controllers, central pattern generators and proprioceptive rhythm generation, forward/backward
  switching, resistive-force and reduced-order body models, muscle models and neuromuscular signs,
  and any connectome-through-a-body work with or without learning or a null — especially with one.
  Compute is the binding constraint, so fast models are as interesting as accurate ones.
- **Operating point.** Whether a wiring effect survives the learner's settings. Here it is
  depth-critical — present at four or more settling hops, reversed at two. Connectome reservoir
  computing (positive or negative), hyperparameter sensitivity surfaces, and claims that a wiring
  effect holds across an operating region rather than at one pinned setting.
- **Behaviour precise enough to validate a body against.** Locomotion kinematics (undulation
  frequency and amplitude, wavelength, speed, eigenworm spectra, omega-turn geometry), forward and
  reverse bout statistics, roaming/dwelling and patch-leaving on structured lawns, and
  chemotaxis or thermotaxis quantification — including methods for separating taxis from edge or
  wall effects in tracking data.

## Also worth knowing about

Score 2, not 3: useful without settling anything above.

- **Cross-connectome transfer** — *C. elegans* to *P. pacificus* head circuits, the dauer wiring
  state, homology between nematode nervous systems, and wiring changes across development or life
  stage. *Demoted from 3 on 2026-09-20: Phase 8 stays on one species deliberately. Still worth
  seeing, because the question returns in a later phase and the data landscape moves meanwhile.*
- **Measured synaptic weights and signs on connectome edges** — fitted weights, functional
  connectivity atlases, per-connection sign predictions. *Demoted from 3 on 2026-10-03: measured
  weights did not move the wiring effect here, so the question is answered for now; new data or a
  per-connection sign resource is still worth seeing.*
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
