# Project brief for literature triage

Hand-maintained. This file is the entire definition of "relevant" for the weekly watch — the
scoring model sees this text and nothing else about the project. Keep it to roughly this length:
it is read once per batch, and detail that does not change a score is detail that costs money.

**Review it at each phase close** — principle 13 of the [phase protocol](../phase-protocol.md).
A brief describing a phase that closed six months ago scores against questions nobody is asking any
more, and the digest will look fine while doing it.

Last reviewed: 2026-10-23 — re-aimed at Phase 10's plan (roadmap v5.0): standard on-lawn assays
graded against real-worm data, controllers fitted to match the animal, and the wiring question asked
on behavioural fidelity. Previous reviews 2026-10-15 (Phase 9's start), 2026-10-14 (Phase 8 close),
2026-10-03 (8a close) and 2026-09-20 (Phase 8's start).

## What the project is

A closed-loop sensory-motor simulation of *C. elegans*. A simulated worm, a point agent in a 2D
arena with patchy bacterial lawns, is driven by a controller: a mechanistic circuit model, a network
constrained to the real 302-neuron connectome with neuromodulatory state, or a reward-trained
network. Its behaviour is graded against real-worm data from standard plate assays.

The organising question is **how faithfully a simulated worm reproduces the worm's standard
behaviours, and what the real wiring, synaptic and neuromodulatory, contributes to that fidelity**,
tested against wiring nulls fitted with the same budget. Quantum circuits and architecture rankings
on learning speed are no longer the point of the project.

## What is open right now

These are the live questions in [Phase 10's plan](../../roadmap.md). A paper bearing on one of them
is a 3.

- **Quantitative references for standard on-lawn behaviour.** Roaming and dwelling (state fractions,
  bout distributions, food-density dependence), lawn leaving (rates, its coupling to roaming, its
  dependence on depletion and food quality), local search after removal from food (the decay of the
  reorientation rate), and the navigation components of chemotaxis (the pirouette rate against
  dC/dt, weathervane steering). A 3: a new dataset or deposit of tracks for any of these, especially
  with mutants or batch-matched controls; between-cohort or between-lab variability in them; a
  standard pipeline that scores them from tracks. *(Added 2026-10-24.)* Also a 3: **locomotor
  kinematics on food against off food**, especially reversal and omega durations, speeds and rates.
  The simulator's movement primitives use published off-food values, and real worms reverse more
  briefly and more slowly on a lawn. And **differences between wild-type strains**, N2 against the
  PD1074 reference strain, in any of these behaviours: one mutant's batch-matched control is N2.
- **Models of these behaviours graded against data.** Mechanistic, circuit or connectome models of
  roaming/dwelling, lawn leaving, local search or pirouettes, in closed loop with food, that are
  graded quantitatively against real worms; above all any that hold out an assay or a mutant, or that
  compare the real wiring against a null. The scoop watch: an embodied or closed-loop connectome
  model graded on any on-lawn statistic.
- **Neuromodulatory state from connectome data.** How serotonin, PDF, tyramine and neuropeptides set
  persistent behavioural states, and how their receptor placement (monoamine and neuropeptide
  connectomes) can enter a network model. A 3: receptor-expression or neuropeptide-connectome
  releases; models that place neuromodulators from data; in-silico mutants graded against published
  phenotypes (tph-1, pdfr-1, tax-4, npr-1).
- **Fitting behavioural models to summary statistics.** Simulation-based inference, ABC and
  evolutionary fitting of animal behaviour models, and their failure under misspecification;
  parameter-recovery checks; imitation learning from animal tracks; separating fixed measured
  kinematics from an optimised policy.
- **Whether reward-trained policies can reproduce innate foraging strategies.** Learned policies here
  steer but never modulate turning, and leave food without roaming first. Wanted: the conditions under
  which a reward-maximiser acquires run modulation or state switching, and reasons it should not
  (innate versus learned strategies, normative accounts of worm foraging).

## Also worth knowing about

Score 2, not 3: useful without settling anything above.

- **Gap junctions in connectome models, and late PPO collapse.** *Demoted from 3 on 2026-10-23: Phase 9
  closed both, with the gap-placement effect traced to the model's normaliser and the collapse
  statistic shipped. A paper showing a connectome result driven by how a connection type is
  normalised is still worth seeing.*
- **Null-model methodology in other connectomes**: fly visual, walking and nerve-cord wiring against
  degree-, cost- or type-preserving nulls; what a null fails to preserve. A 3 only where it grades
  behaviour against real-animal data.
- **Thermotaxis and isothermal tracking.** *The second tier, deferred, not dropped (2026-10-23): new
  circuit mechanism or quantitative references at AFD and AIY are 2s.*
- **Neuromechanical worm bodies and posture**: rod-chain bodies, eigenworm statistics, omega-turn
  kinematics, and connectome-driven locomotion controllers. *The body and posture open the next
  phase.*
- **O₂ sensing, aggregation and bordering (npr-1), pathogen avoidance learning, satiety
  quiescence**: the second tier of standard assays.
- **Cross-connectome transfer and biological variation**: *P. pacificus*, the dauer, the male, new
  adult reconstructions, and between-individual variability.
- **Measured synaptic weights and signs**, and whole-brain activity atlases (signal propagation,
  fitted weights).
- New connectome datasets or releases for any organism, and the tooling to use them.
- Whole-organism or whole-brain simulation efforts in any organism, including ones with neither
  learning nor controls. They bound what this project can claim as novel.
- Three-factor, neuromodulated and eligibility-trace rules; recurrent or state-space models evaluated
  on *biological* tasks, particularly negative results.
- Statistical protocol for comparing models against animal data (paired designs, held-out
  validation, posterior-predictive checks).

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
