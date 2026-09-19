# Project brief for literature triage

Hand-maintained. This file is the entire definition of "relevant" for the weekly watch — the
scoring model sees this text and nothing else about the project. Keep it to roughly this length:
it is read once per batch, and detail that does not change a score is detail that costs money.

**Review it at every phase-plan review.** A brief describing a phase that closed six months ago
scores against questions nobody is asking any more, and the digest will look fine while doing it.

Last reviewed: 2026-09-20 (Phase 8 opening).

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

These are the live questions. A paper bearing on one of them is a 3.

- **Wiring advantage vs. initialisation.** A connectome-constrained network beats degree-preserving
  rewired nulls in the current results, but the rewiring covaries with initialisation, and a fly
  connectome result reports exactly that advantage dissolving under shared initialisation plus a
  degree-preserving null. Anything on connectome-vs-null comparisons, degree-preserving or
  spectral nulls, wiring-versus-initialisation confounds, or structure-function claims in any
  connectome is directly on point.
- **Biologically plausible plasticity that actually learns.** Rate-based three-factor rules on the
  connectome have not reached competence in this project: every learner that gets there leaves the
  chemical weights effectively frozen, and the rule programme has no positive control. Work on
  three-factor or neuromodulated rules, eligibility traces, anti-Hebbian or multi-site plasticity,
  pathway-specific rather than globally-broadcast third factors, credit assignment solved by
  wiring, and especially *negative* results or diagnosed failure modes for such rules.
- **Placed plasticity.** Whether a rule at an anatomically identified site beats the same rule at a
  degree-matched random subset. Anything locating plasticity to identified synapses or cell types.
- **Dynamics beyond a fixed weight matrix.** Gap junctions as a dynamical term rather than a
  symmetric constant; intrinsic neuronal dynamics; how far a rate model can be pushed before
  spiking is required.
- **Cross-connectome transfer.** *C. elegans* to *P. pacificus* head circuits, and the dauer wiring
  state as a within-species comparison. Comparative connectomics, homology between nematode
  nervous systems, and wiring-state changes across development or life stage.
- **Grounding the model in measurement.** Neuromuscular junction data, sex-specific wiring, fitted
  synaptic weights or signal-propagation measurements, whole-brain imaging that constrains a
  simulation's parameters rather than merely describing activity.
- **Embodiment.** Neuromechanical models, body-environment coupling, OpenWorm and c302, and
  simulation platforms a connectome model could be dropped into.

## Also worth knowing about

Score 2, not 3: useful without settling anything above.

- New connectome datasets or releases for any organism, and the tooling to use them.
- Reservoir computing, recurrent and state-space architectures evaluated on *biological* tasks,
  particularly negative results.
- Methods for comparing learned solutions across architectures; representational or dynamical
  similarity measures; statistical protocol for paired-seed model comparison.
- *C. elegans* behavioural quantification precise enough to validate a simulation against.
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
