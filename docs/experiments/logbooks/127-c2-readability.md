# 127: Phase 10 — C.2 Fitted on Every Family Reaches Five of Six, but Its Perturbations Stay Unreadable: Every Fit Switches Inhibition Off

**Status**: completed, as a diagnosis. No fit is registered.

After Logbook 126 the maintainer re-assessed Phase 10's plan (`supporting/127-c2-readability/plan_review.md`). The search problem was solved; the risk had become identifiability. Fitted on the lawn families alone, nothing chooses which internal route carries the worm's state, and a mutant removes a route. So W, fitted that way, would likely read null on every level for a reason about the fit, not the wiring. The maintainer chose to re-plan around a perturbation test, with a readability probe first.

- **C.2 can be fitted on every family, as C.1 is.** From Logbook 126's answer, the guarded search reaches 5 of 6 families by generation 150 on a fresh population, local search with presence (D 1.77, D_fit 1.69). Chemotaxis fails on its flip fraction alone (z +4.1) and on presence.

- **Its perturbations stay unreadable.** tph-1 is inert and pdfr-1 moves the wrong way; mgl-1, López-Cruz's glutamate receptor on AIA and ADE, does nothing. tax-4 alone passes, in the right direction.

- **Why mgl-1 is inert:** every sep-CMA fit drives the encoder's one inhibitory wiring scale to 0.005–0.09 of its [0, 0.75] box, with excitation at the top. The MGL-1 edges are inhibitory, so they carry 0.3% of AIA's input.

- **Restoring inhibition on that answer collapses its wild type,** and mgl-1 never falls below 0.8 of its wild type at any level. One fitted answer that kept inhibition at 0.09, the imitation's refinement, carries mgl-1 at 0.61 of its wild type's early search (real worms 0.31).

- **Three short screens** (inhibition floored, or tph-1 fitted) were inconclusive. Each started too far from its answer (σ₀ 0.1) and stopped inside the search's transient.

- **The maintainer's decision (2026-11-07): constrain the encoder with data, then fit.** C.2's change builds Logbook 126's procedure into the encoder, with one chemical scale, so the connectome's counts and signs set the balance of excitation and inhibition, and modulator signs fixed where receptor biology resolves them.

**Date**: 2026-11-06 to 2026-11-07.

**Records**: [`supporting/127-c2-readability/`](supporting/127-c2-readability/):

- `plan_review.md`: the re-assessment, as written before the probe;
- `probes/`: every probe's record, each with its script as run, and a README on re-running them.

The bounds, guards, tying and the mgl-1 encoding were applied inside the probe scripts; no package code changed.

## Objective

Logbook 126 fitted C.2 to v4's four lawn families from an imitation start, and its held-out families and mutants failed. The re-assessment asked whether W, as planned, could read the wiring at all, and found three problems:

- **Identifiability.** The fitted answer uses its modulator layer, but serotonin's pool sits nearly constant (0.0134 ± 0.0036 over a 30-min episode, `serotonin_check.json`), so knocking out tph-1 removes a near-constant offset: reversals 17.0% → 15.7% of seconds, runs 82.6% → 84.3%. Nothing in the lawn families chooses the route a mutant removes.
- **The held-out families sat at the floor.** The wild type failed every one, so no null could do measurably worse there.
- **C.1 and C.2 were fitted on different families.** C.1's ranking score covers all six; C.2's D_fit leaves out local search and chemotaxis, so C.4's contrast would compare a model fitted on search with one that never saw it.

The re-plan: fit C.2 on every family, as C.1 is; hold out the perturbations, adding López-Cruz's mgl-1 to Scheer's three mutants; and first check, in one probe, (a) whether C.2 can fit all six families and (b) whether the perturbations then move.

## Method

One sep-CMA restart on discarded seeds (`readability.py`), on Logbook 126's model and rules:

- the model: per-ligand tying with four pool centres (136 parameters), the soft cap and the graded box;
- the guards: saturation (+500) and responsiveness (+300), the latter over all four assays' recorded outputs;
- the search: population 48, σ₀ 0.1, from Logbook 126's guarded answer, 150 generations on 16 workers;
- the objective: v4's ranking score against the band centres at n = 200 on all six families. Presence is left out, as in C.1's fits.

At generations 0, 50, 100 and 150 the search's mean was graded on one fresh population of 200 with presence, with the three Scheer mutants on the small lawn and mgl-1 on López-Cruz's food removal.

**mgl-1's encoding.** The López-Cruz sign overrides sign 16 glutamatergic edges inhibitory, as a fast stand-in for MGL-1 (Logbook 122's addendum): the tax-4 sensory neurons onto AIA, and AVM and FLP onto ADE. mgl-1 removes them. Its reading is each 2-min bin's reorientations after removal, the early (8–12 min) and late (34–42 min) means, and mgl-1's ratio to its own wild type on the same animals. López-Cruz's own ratio, mgl-1 against N2 recorded on the same dates (figures 3D and 3E), is 0.31–0.32 early and 0.21–0.52 late (`readability.json`, `lopezcruz_ratio`). mgl-1 is read, not graded: no band has been built for it.

## Fitting every family

| Generation | D | D_fit | Families passing | Local search presence | Off-food reversal share |
|---|---|---|---|---|---|
| 0 (Logbook 126's answer) | 3.63 | 1.33 | the 4 lawn families | fails | 21% |
| 50 | 3.67 | 3.83 | activity | passes | 31% |
| 100 | 2.32 | 2.30 | local search | passes | 29% |
| 150 | **1.77** | **1.69** | **5 of 6** (all but chemotaxis) | passes | 22% |

- **The search first traded the lawn for search, then recovered both.** At generation 150 local search passes (γ 0.058 against 0.114, z −1.2; the log early/late ratio z −1.6), and so do the four lawn families.
- **Chemotaxis fails on one statistic:** the flip fraction, 0.107 against the real worms' 0.023 (z +4.1), and its presence. Its weathervane (z −0.4) and onset rate (z +1.8) pass.
- **It was still improving at the cap:** the median score fell to 2.6–2.9 over the last 30 generations, with σ at 0.045. C.1's continued v4 probe sits at D 1.28–1.44 (Logbook 125).
- **Cost:** 184 s a generation, 7.8 h in all.

## The perturbations

| | Gen 0 | Gen 50 | Gen 100 | Gen 150 | Real worms |
|---|---|---|---|---|---|
| tph-1, roaming change | +0.001 | +0.027 | +0.004 | +0.021 (z −4.9, fails) | roams more |
| pdfr-1, roaming change | +0.041 | +0.176 | +0.119 | +0.100 (z +3.4, fails) | roams less (graded by magnitude) |
| tax-4, roaming change | +0.001 | −0.018 | −0.038 | −0.051 (passes) | roams less |
| mgl-1, early search vs its wild type | 1.00× | 1.00× | 1.00× | 1.01× | 0.31× |

- **Fitting more families switched the slow pathways on, but did not pin which way they act.** pdfr-1's knockout now has a large effect, of the opposite sign to the real one, and tph-1 stays near zero.
- **tax-4 is the only perturbation the fitted wild type gets right,** and so the only one W could read its arms on today.

## Why mgl-1 is inert

`mgl1_diagnosis.json`, one 90-min reference episode of López-Cruz's food removal on the generation-150 answer, and `wiring_scales.json`:

1. **Every fit switches signed inhibition almost off.** The encoder has one scale for every excitatory chemical edge and one for every inhibitory one:

   | Answer | Excitatory | Inhibitory | Gap |
   |---|---|---|---|
   | Logbook 126's guarded fit (lawn families) | 0.75 | 0.03 | 0.66 |
   | its ablation without the imitation | 0.72 | 0.07 | 0.51 |
   | the imitation's refinement | 0.57 | 0.09 | 0.75 |
   | the imitation answer (gradient, before refining) | 0.75 | 0.75 | 0.75 |
   | this probe, generation 150 | 0.74 | **0.005** | 0.68 |

   The box is [0, 0.75] for each. The 16 MGL-1 edges are inhibitory, so in the generation-150 answer they carry 0.0066 of AIAL's 2.00 of input weight (chemical and gap), and similar shares for AIAR, ADEL and ADER. Removing them moves AIA's activity by 0.002 and the motor pools by at most 0.0002.

2. **AIA barely drives the motor pools.** A nudge to AIA at the first settling pass moves the pools by 0.005 per unit, against 0.15 for AVA, 0.04 for ADE and 0.02 for AIB, though AIA is two synapses from the motor neurons.

3. **Two secondary limits.** AVM and FLP receive no sensory injection, so they start each second at zero. And the network keeps no neural state between seconds: each step settles four passes from its injected sensors, and only the modulator pools persist. So there is no tonic glutamate for the mutant to lose.

Every perturbation that acts through inhibitory synapses is inert under these fits, not only mgl-1, and every W arm fitted the same way would share it.

## Restoring inhibition

**On the generation-150 answer** (`inhibition_check.json`; no fitting, 200 paired animals), only the inhibitory scale changed:

| Inhibitory scale | mgl-1 early ratio | Wild type: roaming | Time on lawn |
|---|---|---|---|
| 0.005 (as fitted) | 1.00 | 0.215 | 0.994 |
| 0.1 | 0.88 (late 1.33) | 0.172 | 0.991 |
| 0.25 | 1.00 | 0.055 | 0.919 |
| 0.5 | 1.00 | 0.000 | 0.273 |
| 0.74 (equal to excitation) | 1.00 | 0.000 | 0.238 |

The branch fixed before the run was to screen if any level reached 0.8; none did. Above about 0.25 the wild type stops roaming and leaves the lawn, so this answer, tuned with inhibition off, cannot show what a network fitted with it on would do.

**On answers that kept inhibition on** (`inhibition_check2.json`):

| Answer | Inhibitory scale | Wild type early / late | mgl-1 early ratio |
|---|---|---|---|
| the imitation answer | 0.75 | 5.65 / 5.62, no decay | 1.00 |
| the same, inhibition set to 0.005 | 0.005 | 5.77 / 3.89 | 1.00 |
| **the imitation's refinement** | **0.09** | 2.67 / 1.50 | **0.61** (late 1.02) |

A fitted network with some inhibition can carry mgl-1's early deficit, in the right direction and at about half the real size. Inhibition is necessary but not sufficient: the same level on the generation-150 answer does not.

## Three screens

`screens.py`, `screens_S1.json` to `screens_S3.json`. Each is a sep-CMA refinement on all six families at 64 animals a candidate, on the stored 64-animal bands, at population 24 and σ₀ 0.1, on 5 workers, the three in parallel. Each had a stop rule written before the run and readouts every 10 generations on 200 shared animals:

| Screen | Start and change | Stopped | Readout D | Its target |
|---|---|---|---|---|
| S1 | the imitation's refinement, inhibition floored at 0.09 | generation 20: mgl-1 above 0.9 at two readouts | 5.04 → 5.73 → 5.08 | mgl-1 0.62 → 13.2 (its wild type had stopped reorienting after removal) → 0.96 |
| S2 | generation 150, inhibition floored at 0.1 | generation 20: D more than 1 above its start | 1.91 → 3.35 → 4.30 | mgl-1 0.89 → 1.02 → 0.95 |
| S3 | generation 150, tph-1's lawn differences added to the objective | generation 30: tph-1's roaming z not 1 closer to 0 | 1.79 → 3.69 → 3.06 → 3.42 | tph-1 +0.026 → +0.017 → +0.013 → −0.001 |

**They did not test their questions.**

- Each began at σ₀ 0.1 from a fitted answer. At that step most candidates leave a statistic undefined: the screens' median scores stayed near 1,000, and near 2,000 in S3's first generation.
- The readability probe had shown the same transient. Its median sat near 1,000 for 20 generations and settled only after about 50 (`readability.json`).
- Each screen stopped inside that transient. A refinement screen needs a σ₀ near 0.02, or stop rules timed after the transient.

## What this establishes, and what it does not

- **C.2 can be fitted on all six families, as C.1 is:** 5 of 6 at generation 150, still improving. The start was warm: Logbook 126's answer, itself descended from the imitation of C.1. A fit from a clean start is untested.
- **Fitting more wild-type data does not identify the mechanisms the perturbations test.** Every fit drives the inhibitory scale near zero, and the fitted modulator signs are free. So pdfr-1 comes out reversed, tph-1 inert, and every inhibitory perturbation inert.
- **mgl-1 can be represented:** the imitation's refinement carries it at 0.61×. Whether it survives a fit of the six families is untested.
- **tax-4 is the one perturbation the fitted wild type gets right.**
- **Every reading is description:** one restart per design, on discarded seeds, under bounds applied in scripts.

## Cost

- **The readability probe:** 7.8 h on 16 workers, at 184 s a generation for population 48 on all four assays.
- **The checks:** the diagnosis, the inhibition checks and the serotonin check took seconds to minutes each.
- **The screens:** 0.7–1.0 h each, in parallel, at about 80 s a generation.

## Implications

- **The maintainer's decision (2026-11-07): constrain the encoder, then fit.** C.2's change builds Logbook 126's procedure into the encoder (the soft cap, the pool centres, per-ligand tying, the graded box, both guards and the wider population), each off by default and byte-identical. It adds two data constraints:
  - **one chemical scale for both signs,** so the connectome's own counts and signs set the balance of excitation and inhibition, and the fit cannot switch inhibition off;
  - **modulator signs fixed where receptor biology resolves them,** so a knockout's direction is not left to the fit.
- **Then a check, then one fit.** A minutes-long representability check on the constrained model comes first. Then one full fit on all six families, with readouts every 10 generations and stop rules timed after the search's transient.
- **The fit split** becomes all six families, as C.1's: a dated D37 amendment in that change. The perturbations are held out: Scheer's three mutants, and mgl-1 once its band is built.
- **W** waits for that fit, as the re-assessment planned: four arms, sized and its compute decided from the fit's cost.

## How the probes are run from here

The probe above took 7.8 h. Its mgl-1 half was answerable at generation 0 and by a 20-minute structural check: the inhibitory scale was already near zero in every committed answer. The maintainer asked for exploration to be staged more cheaply, and the remaining probes here follow that pattern:

1. **Representability first (minutes):** whether the model can produce the target at all, from existing answers, by inspection, sweeps or hand-set changes, with no fitting.
2. **Screens (about an hour, several in parallel):** 64 animals a candidate on the stored 64-animal bands, the assays the question needs, 30–40 generations from an existing answer, a target read every 10 generations, and a stop rule written before the run.
3. **Confirmation at full scale** only for an idea that passed a screen.
4. **Registration** as the phase protocol has it.

Each decision comes with its branches approved beforehand, so the machine does not wait on a reply. The screens here show the pattern's own trap: a screen that refines an answer must start near it.

## Seeds

All discarded:

- **The readability probe:** fits 726,000,000 + 200·k + i with k = 48(g − 1) + m; checkpoints 727,000,000 + 200·c + i; CMA 9479; the smoke 99,780,000.
- **The checks:** the mgl-1 diagnosis's episode 99,790,000; the inhibition checks 99,800,000 + i and 99,810,000 + i; the serotonin check on the profile seed 99,670,000.
- **The screens:** fits 730, 731 and 732 M (64·k + i), CMA 9480–9482; readouts 733,000,000 + i (shared); the smokes at k = 99,999 of 730 M and 732 M.
