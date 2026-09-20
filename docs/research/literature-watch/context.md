# Project brief for literature triage

Hand-maintained. This file is the entire definition of "relevant" for the weekly watch — the
scoring model sees this text and nothing else about the project. Keep it to roughly this length:
it is read once per batch, and detail that does not change a score is detail that costs money.

**Review it at each phase close** — principle 13 of the [phase protocol](../phase-protocol.md).
A brief describing a phase that closed six months ago scores against questions nobody is asking any
more, and the digest will look fine while doing it.

Last reviewed: 2026-09-20 — re-aimed at Phase 8 (*ground, then embody*) as ratified in roadmap v4.3.

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

These are the live questions — Phase 8 of the [roadmap](../../roadmap.md), *ground, then embody*. A
paper bearing on one of them is a 3.

- **Wiring advantage vs. initialisation.** A connectome-constrained network reaches competence
  sooner than degree-preserving rewired nulls in the current results, but the rewiring covaries
  with initialisation, and a fly connectome result reports exactly that advantage dissolving under
  shared initialisation plus a degree-preserving null. Anything on connectome-vs-null comparisons,
  degree-preserving or spectral nulls, wiring-versus-initialisation confounds, or structure-function
  claims in any connectome is directly on point. So is anything predicting *which* graphs learn
  faster from structure alone — input routing, confinement of activity to a core, which cells drive
  the dominant modes — because no graph property measured here predicts learning time.
- **Operating point.** Whether a wiring's apparent advantage survives the learner's settings. One
  inherited learning rate reversed the sign of a registered result here, and a connectome-reservoir
  study reports the same sensitivity independently. Connectome reservoir computing, hyperparameter
  robustness or sensitivity surfaces, and any claim that a wiring effect holds — or does not —
  across an operating region rather than at one pinned setting.
- **Measured weights on real edges.** This substrate is anatomically constrained in topology and
  *randomly initialised in weight*; the next rung replaces the draw with a measurement. Fitted
  synaptic weights or signs on connectome edges, optogenetic and signal-propagation measurements of
  functional connectivity, neuromuscular junction data, sex-specific wiring, and whole-brain imaging
  that constrains a model's parameters rather than describing activity.
- **Embodiment.** There is no body here: motor output is a learned readout over motor-neuron
  activations, and the next shipment drives the anatomical motor-to-muscle map into a
  two-dimensional body. Neuromechanical models, rod-chain and viscoelastic body mechanics,
  resistive-force and drag models, muscle models, proprioceptive feedback, and connectome-driven
  locomotion. Compute is the binding constraint — a body must survive thousands of training
  episodes per seed — so reduced-order and fast models are as interesting as accurate ones.
- **Dynamics beyond a fixed weight matrix.** Gap junctions as a dynamical term rather than a
  symmetric constant; intrinsic neuronal dynamics; node-level adaptation, which is the biologically
  faithful form of much *C. elegans* learning and sits in the neuron rather than the synapse; and
  how far a rate model can be pushed before spiking is required.
- **Plasticity, narrowed.** The uniform-rule programme closed here with a diagnosed cause: no
  biologically plausible rule that *writes* the chemical weights learns this substrate to any
  benefit. What stays live is narrower — plasticity **placed** at an anatomically identified site
  against a degree-matched random subset of the same size; **plastic electrical synapses**; and
  *negative* results or diagnosed failure modes for local rules. A three-factor or neuromodulated
  rule paper with none of those properties is a 2.
- **Behaviour precise enough to validate a body against.** Locomotion kinematics (undulation
  frequency and amplitude, crawling and swimming speeds, posture or eigenworm spectra, omega-turn
  geometry), forward and reverse bout statistics, and roaming/dwelling or patch-leaving on
  structured bacterial lawns — what a simulated worm with a body and an internal state is checked
  against.

## Also worth knowing about

Score 2, not 3: useful without settling anything above.

- **Cross-connectome transfer** — *C. elegans* to *P. pacificus* head circuits, the dauer wiring
  state, homology between nematode nervous systems, and wiring changes across development or life
  stage. *Demoted from 3 on 2026-09-20: Phase 8 stays on one species deliberately. Still worth
  seeing, because the question returns in a later phase and the data landscape moves meanwhile.*
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
