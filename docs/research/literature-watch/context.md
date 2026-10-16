# Project brief for literature triage

Hand-maintained. This file is the entire definition of "relevant" for the weekly watch — the
scoring model sees this text and nothing else about the project. Keep it to roughly this length:
it is read once per batch, and detail that does not change a score is detail that costs money.

**Review it at each phase close** — principle 13 of the [phase protocol](../phase-protocol.md).
A brief describing a phase that closed six months ago scores against questions nobody is asking any
more, and the digest will look fine while doing it.

Last reviewed: 2026-10-15 — re-aimed at Phase 9's plan (roadmap v4.5): the gap-junction effect
screened for a model artefact and then taken to mechanism, the learner's late collapse, nulls from
biological variation, and the ranking through the body behind a trigger. Previous reviews
2026-10-14 (Phase 8 close), 2026-10-03 (8a close) and 2026-09-20 (Phase 8's start).

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

These are the live questions — [Phase 9's plan](../../roadmap.md), which screens the one wiring
effect Phase 8 left standing for a model artefact and then takes it to mechanism. A paper bearing
on one of them is a 3.

- **What a gap-junction effect in a learned connectome model is made of.** The one wiring effect
  standing here is gap-junction *placement*: the wild type learns faster than a null whose
  electrical synapses alone are rewired. It is being screened for a model artefact first — gap
  junctions enter as additive drive scaled by raw synapse counts, a few hub pairs sit far above a
  tanh unit's saturation, and the coupling scale was never swept — then taken to mechanism with
  nulls the field lacks: strength- or totals-preserving placement nulls for electrical synapses,
  boundary- and class-level gap nulls, contact-constrained nulls, size-matched random scatters. A 3:
  how electrical synapses should enter a learned or dynamical connectome model; strength-preserving
  randomisation of sparse, heavy-tailed networks; gap-junction structure-function against a matched
  null in any connectome; a few count outliers driving a wiring result.
- **Late policy collapse in long PPO runs.** Policy-gradient learners here learn and then collapse
  after a few thousand episodes, in every arm, on small tanh policies — an optimiser property, not
  a wiring one. Wanted: diagnostics that *lead* a collapse (entropy, KL, feature rank, saturated
  units, gradient norms), remedies with low comparison risk between arms (KL early stopping, weight
  clipping, optimiser moments, advantage normalisation), and collapse incidence reported as a
  statistic beside a learning curve. Loss of plasticity in continual learning is the same question.
- **Why learned policies steer but never modulate runs.** Every substrate and sense here learns the
  weathervane and never klinokinesis (turning more when heading the wrong way), which theory
  predicts for a learner given bearing information and a continuous turn. Live: bearing information
  crossed with a discrete reorientation primitive (pirouette, reversal-plus-reheading) in a
  reward-driven learner; unit-free klinokinesis measures (reorientation rate across the worm's own
  dC/dt quantiles, not a ratio in arbitrary units); thermal klinokinesis above against below the
  cultivation temperature; strategy re-weighting and state-switching in real worms.
- **Wiring contrasts against biological rather than synthetic variation.** Every null here is a
  shuffle. Now: a second individual's nerve ring transplanted into the wild type, the male's
  gap-junction layer read against its own shuffles, and the referee's bar for any two-connectome
  contrast — a between-individual variability baseline first. Inter-individual gap-junction
  variability data (none exist for adults), new adult or male *C. elegans* reconstructions with
  electrical synapses, and two-connectome contrasts that carry or lack such a baseline are 3s.
- **A placed, state-gated set-point memory at an identified synapse.** The worm's mapped learning
  sites are gains at identified synapses — the salt set-point in ASER decoded at AIB, the thermal
  set-point in the AFD→AIY gate — not Hebbian weight changes. The question is whether a rule
  placed there on the full connectome beats the same rule at a degree-matched random site, and
  whether the difference vanishes on rewired wiring. Models of these set-point circuits, above all
  any that compare a placed mechanism with a random placement, and new mechanism at either synapse.
- **Architecture rankings through a body, and fast overdamped worm bodies.** The anatomical
  motor-to-muscle map drives a 12-segment kinematic body with an imposed rhythm, and six
  architecture families are to be ranked through it once. Wanted: resistive-force rod chains at
  millisecond time steps (overdamped, no inertia), load-sensitive rhythm generation that yields
  the swim–crawl gait transition, and — the scoop watch — any paper training a policy on a
  *C. elegans* connectome through a body with a wiring control.

## Also worth knowing about

Score 2, not 3: useful without settling anything above.

- **Null-model methodology in other connectomes** — fly visual, walking and nerve-cord wiring
  against degree-, cost- or type-preserving nulls; shortcut artefacts; what a null fails to
  preserve. *Added 2026-10-15: the field posts one of these about weekly, all on chemical graphs.
  It is a 3 only where it reaches electrical synapses, strength preservation or count outliers.*
- **Gap junctions as dynamics** — electrical-synapse plasticity, stiffness and synchrony of graded
  networks under strong coupling. *Demoted from 3 on 2026-10-14: the dynamical rung here learned far
  slower than the memoryless one and closed (Logbook 080). The ohmic-coupling form returns at 3
  only as the first question above — how electrical synapses should enter the model.*
- **Operating point and reservoir results** — whether a wiring effect survives the learner's
  settings, hyperparameter sensitivity surfaces, connectome reservoir computing against nulls.
  *Demoted from 3 on 2026-10-14 (Logbook 079); the gap model's own operating point is the first
  question above, and the general question stays here.*
- **Cross-connectome transfer** — *C. elegans* to *P. pacificus* head circuits, the dauer wiring
  state, homology between nematode nervous systems, and wiring changes across development or life
  stage. *Demoted from 3 on 2026-09-20 and kept there 2026-10-15: the P. pacificus data carry no
  gap junctions and the dauer's are unreleased, so neither can carry the one wiring effect that
  holds. A release of either gap layer is a 3 under the biological-variation question above.*
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
- Connectome-driven locomotion controllers and neuromechanical models without a wiring control.
  *Demoted from 3 on 2026-10-15: the body is frozen for the phase; the scoop watch above is the 3.*
- New connectome datasets or releases for any organism, and the tooling to use them.
- Whole-organism or whole-brain simulation efforts in any organism, including ones with neither
  learning nor controls — they bound what this project can claim as novel.
- Three-factor, neuromodulated and eligibility-trace rules in general, and diagnosed failures of
  local rules; recurrent or state-space architectures evaluated on *biological* tasks, particularly
  negative results.
- Methods for comparing learned solutions across architectures; representational or dynamical
  similarity measures; statistical protocol for paired-seed model comparison.
- *C. elegans* behavioural quantification beyond the targets above — locomotion kinematics, bouts,
  roaming/dwelling, patch-leaving, taxis separated from wall effects. *Demoted from 3 on
  2026-10-15: the body's instruments exist; the repaired klinokinesis reference is the live one.*
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
