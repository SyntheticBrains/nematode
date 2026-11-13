# Fit-first investigation: brief

This is an investigation, not a build. The question is whether a different approach would get Quantum Nematode to its long-run goal faster and more reliably than the current phase and roadmap, and if so, where that work should live. Work on a new branch. Do not change the main code paths, the current phase's work, or any other repository.

First commit on the branch: save this brief verbatim as `investigation/BRIEF.md`.

## 1. The goal

The long-run goal is a high-fidelity simulation of the C. elegans nervous system: the connectome, electrical (gap junction) and chemical signalling, neuromodulators, neuroplasticity, and learning the way the worm learns naturally, through local mechanisms rather than reinforcement learning. Progress is being made, but slowly, and other groups in the field also seem to move slowly. The real question is whether that pace is inherent or partly a result of the method.

## 2. The hypothesis to test

**Fit-first:** fit a connectome-constrained model directly to recordings of neural activity, especially perturbation data (stimulate one neuron, record the rest), using gradient-based fitting in a differentiable simulator. Add biophysical fidelity only where it improves prediction. Behaviour benchmarks, including the new feeding and lawn-navigation benchmark, then become held-out tests the model was never fitted to.

**The alternatives must get a fair hearing:**
- The current approach may already be close to this, or may be the better route.
- Fit-first may only move the bottleneck. Calcium is a slow, indirect read of voltage, many neurons go unrecorded, worms differ from one another, and neuromodulators and plasticity are barely measured.
- A hybrid may beat both.

"Stay the course" is an acceptable conclusion. So is "fit-first for some layers only".

## 3. Part A: evidence (no code)

Complete these in order, then stop for review.

1. **`investigation/ORIENTATION.md`: the current approach, described honestly.**
   - How neuron and synapse parameters are set today, and where each value comes from.
   - Every benchmark that exists, what it measures, and how the current model scores on it.
   - The roadmap: phases done, the current phase, and what remains.
   - Pacing, measured from git history and the roadmap: elapsed time and main output for each completed phase, the recurring bottlenecks, and an estimate of time remaining under the current approach, with assumptions stated.
   - Whether the simulator could be made differentiable, and at what cost (JAX or PyTorch backend, an adapter, or a rewrite).

2. **`investigation/LANDSCAPE.md`: an independent review.**
   - Who fits connectome-constrained models to worm neural data, what worked, what failed, and what remains open.
   - Analogues in other organisms where they inform method, such as the fly visual system.
   - Which parts of the goal fit-first does not address, such as plasticity and neuromodulator dynamics, and how others have handled them.
   - For each source, record whether you read the full text, the abstract, or a second-hand report.
   - Give no verdict yet.

3. **`investigation/DATA_INVENTORY.md`: every candidate dataset.** For each one record:
   - what was measured (number of neurons, identified or not, immobilised or freely moving, stimuli, perturbations, behaviour recorded alongside, number of worms);
   - access, licence, size and format;
   - how it maps to model variables (calcium indicator kinetics and the voltage-to-calcium link);
   - known quality issues.

   Include the behaviour datasets already used in the project.

4. **`investigation/IDENTIFIABILITY.md`: the deciding analysis.** For each class of parameter, list which datasets constrain it and whether that is enough. Mark each class "well constrained", "constrained with priors", or "not constrained". The classes:
   - synaptic weights and signs;
   - gap junction conductances;
   - time constants;
   - intrinsic currents;
   - neuromodulator release and effects;
   - plasticity rules.

   For classes the data does not reach, name the priors that could help, such as per-neuron gene expression for ion channels and receptors, or neuropeptide receptor maps, and how much they would narrow the range.

5. **`investigation/SPIKE_PLAN.md`: a timeboxed technical spike, proposed but not yet run.**
   - **Minimum scope:** reproduce one published connectome-constrained fit on one public dataset in a differentiable framework.
     - Score it on held-out perturbations or held-out neurons.
     - Compare it with a degree-preserving shuffled-connectome control.
   - **Bridging test:** run the fitted model, unchanged, in the existing closed-loop environment on one existing behaviour benchmark, and report the result whatever it is.
   - **Budget:** state the time and compute, preferring local runs on an M5 Max MacBook Pro with 128 GB. If cloud GPU is needed, estimate its cost.
   - **Stop rule:** if reproduction fails within the timebox, record why and stop. Do not keep tuning.

6. **Stop and summarise for my review.**

Then I will paste a short list of my own seed sources and claims. Reconcile them in `investigation/RECONCILE.md`: what I had that you missed, what you found that I missed, and where we disagree.

## 4. Part B: spike and decision (after my approval)

1. Run the spike as approved and write `investigation/SPIKE_REPORT.md`, negative results included.

2. **`investigation/COMPARISON.md`.** Compare the current roadmap, fit-first and a hybrid against shared milestones, so that pacing can be compared like for like:
   - **M-a:** predicts responses to held-out perturbations.
   - **M-b:** predicts freely moving whole-brain activity on held-out worms.
   - **M-c:** passes the existing behaviour benchmarks in closed loop with no fitting to them.
   - **M-d:** reproduces a neuromodulator-driven state switch (for example, roaming and dwelling).
   - **M-e:** shows an experience-dependent change in behaviour (this links to the separate Wormturn project).

   For each approach and milestone, give:
   - estimated time at part-time pace;
   - fidelity reached;
   - main risks;
   - compute needs;
   - what is reused;
   - scientific credibility.

   Base the estimates on the spike and the pacing history, not on optimism.

3. **`investigation/PLACEMENT.md`.** If fit-first or a hybrid wins, decide where it lives:
   - a new phase or module inside Quantum Nematode;
   - a new repository that depends on Quantum Nematode for data, connectome and benchmarks;
   - a standalone repository.

   Judge them on:
   - **architecture:** whether the current simulation core can take a differentiable backend without distorting it;
   - **identity:** whether fit-first suits the project's stated scope, purpose and name, and how it would be positioned publicly;
   - **dependencies:** the weight of new ones, such as JAX;
   - **licences:** of code and data;
   - **maintenance:** load at part-time pace;
   - **integration:** how the sibling projects Wormlight (browser worm) and Wormturn (plasticity research) would consume a fitted model. Propose a versioned export format for fitted models.

   Name which existing parts carry over and which do not.

4. **`investigation/RECOMMENDATION.md`.** Give one recommendation: switch, hybrid, or stay. Include:
   - the first three milestones under it;
   - kill criteria;
   - what happens to the current phase's work in progress.

5. **Independent check.** Give the recommendation, the comparison and the spike report to a fresh subagent that has not seen this brief or my seeds. Ask it for the strongest argument against the recommendation, and include its answer unedited.

6. Stop and summarise.

## 5. Working agreement

- Cite only sources you opened. Never invent a citation. Say when you are unsure.
- Be candid about the current approach. Its pacing and its weaknesses are the point of this investigation.
- A negative spike, or a recommendation to stay the course, are valid outcomes.
- Log every non-trivial decision in `investigation/DECISIONS.md`.
- Ask before any major deviation from this brief.
- I work on this part-time, so keep each step reviewable in one sitting.
- Write docs in plain prose with no em dashes.
