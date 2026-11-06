# 126: Phase 10 — C.2's Fit: Seven Designs Stall at Constant Outputs; a Responsiveness Guard and a Wider Search From an Imitation Start Fit the Lawn, and the Held-Out Families and Mutants Fail

**Status**: completed, as a diagnosis. No fit is registered. **C.2's encoded connectome now fits benchmark v4's fitted families.** The search used a responsiveness guard and a population of 48, started from a network trained to imitate C.1. On 5 fresh populations it passes every fitted family (small lawn, uniform well, kinematics, activity) in all 5, at D_fit 1.29–1.40.

It passes the wild type in **0 of 5**. Every held-out family fails:

- **Local search:** no post-removal decay, and its presence test fails.
- **Chemotaxis:** no drift up the gradient, and it keeps reversing off food at its lawn rate.
- **The mutants:** tph-1 and tax-4 fail.

Seven designs before it ended in one basin: outputs held constant by the readout's biases, scoring about 9.3–10. **The imitation speeds the fit but is not what escapes the basin.** From stage 1's answer, with no teacher, the same guard and population reach 2 of the 4 fitted families by generation 250 (D_fit 2.15), still improving.

> | Design (one restart; v4's D_fit) | Start | Generations | Score at the end | What it found |
> |---|---|---|---|---|
> | stage 2, unbounded | stage 1's answer | 155 (stopped) | 8.8 | curvature pinned, reversals suppressed |
> | soft cap (±2) | stage 1's answer | 81 (stopped) | 9.4 | outputs parked at the cap; motor layer 97% saturated |
> | soft cap + pool centres | stage 1's answer | 80 (stopped) | 9.6 | the same |
> | graded box + saturation guard | stage 1's answer | 200 | 8.6 | responsive, unsaturated; the worm reverses half the time; D_fit 4.41 |
> | responsive stage 1, per-ligand tying | box centre | 47 + 200 | 9.3 | constant outputs from stage 1 on; leaves the lawn |
> | imitation of C.1, then sep-CMA | imitation answer | 200 | about 10 | D_fit 4.86 at the end |
> | **responsiveness guard, population 48** | **the imitation fit's answer** | **250** | **1.8** | **every fitted family passes; D_fit 1.33** |
> | the same, no imitation | stage 1's answer | 250 | 2.7, still falling | small lawn and uniform well pass; D_fit 2.15 |

**Date**: 2026-11-05 to 2026-11-06.

**Records**: [`supporting/126-c2-run-mode/`](supporting/126-c2-run-mode/):

- `probes/`: every probe's record, each with its script as run;
- `plan_review.md`: the plan review written mid-way, with its options and estimates.

The bounds, guards and tying were applied inside the probe scripts; no package code changed. `scripts/analysis/connectome_identity_check.py` now also digests the encoded brain's forward path, at two recorded vectors.

## Objective

C.2's staged fit on v3 held the lawn without running ([Logbook 123](123-c2-speed-noise.md)). Its speed and curvature outputs were pinned off the squash. Benchmark v4 then added an on-lawn activity family that catches that worm ([Logbook 124](124-benchmark-v4.md)). The question here was whether C.2 can be fitted to v4, and if not, why.

## Method

Every probe is one sep-CMA restart against v4's definedness-first ranking score on D_fit, on the corrected wiring, on discarded seeds. Every 50 generations its answer was profiled over a reference episode on Scheer's lawn. The profile records:

- each output's pre-squash 5th, 50th and 95th percentiles;
- the primitive shares and the time on the lawn;
- the motor neurons' saturation;
- each food feature's gain onto the four motor pools, by central differences.

The answer was also graded on D_fit's assays on one population.

## The readout pins the outputs

- **The box centre** (`profile_points.json`): every readout weight and bias is 0. The outputs are constant, mid-range and blind to the network.
- **The stage-1 answer** still runs, its speed output inside the squash's responsive range.
- **Stage 2 does the pinning** (`readout_anatomy.json`). The pooled motor rates sit off-centre and vary little, so the fit drives the readout weights to the ±3 bounds. On offset pools, those large weights act as a second bias, which pushes the outputs off the squash.

**Re-run on v4** (`stage2_v4_stopped.json`), stage 2 kept speed responsive, but pinned the curvature at its other extreme and the reversal hazard low. The worm held the lawn by running slowly in circles. The run was stopped at generation 155.

## Bounding the readout moves the saturation inward

**The soft cap and the pool centres** (`bounded_*_stopped.json`). A soft cap passes each output through 2·tanh(x/2), and fitted pool centres make the readout b + Σ w_j (pool_j − c_j). With either, the outputs settled at the cap within 30 generations, and the scores stalled at 9.4–9.6.

**The network's layers** (`network_gain.json`) show why:

| Answer | Sensory layer | Motor layer saturated | Sense → pool gain |
|---|---|---|---|
| stage 1 | 63% of neurons silent | 0% | ≈ 0, food contact 0.02 |
| v3 stage 2 | active | 5% | 0.15–2.3 |
| v4, unbounded | active | 8% | 0.04–0.32 |
| v4, soft cap | active | **97%** | ≈ 0.001 |
| v4, cap + centres | active | **97%** | ≈ 0.001–0.006 |

- At the box centre every sensory gain is 0, its box being symmetric, so the network gets no input.
- Stage 1 never fitted the 33 amphid gains.
- Once the readout was capped, the fit saturated the motor layer instead.
- The fits that switched the network on set wiring scales of 1.26–1.34 and sensory gains at their ±5 bound. The encoder's own graded range is 0.5–0.75.

## Responsive is not enough

- **The graded regime** (`graded.json`):

  - the wiring scales were held to [0, 0.75] and the sensory gains to ±2;
  - the cap and centres were kept;
  - a candidate with over half its motor neurons saturated scored 500 more.

  The network stayed responsive and unsaturated, with sense-to-pool gains up to 0.58. But the score fell only to 8.6 by generation 200, and the worm reversed up to 65% of seconds.

- **A stage 1 that fits every sense, and per-ligand tying** (`staged_bc.json`, 136 parameters). Stage 1 fitted the 77 parameters that carry the senses to the outputs, from the box centre. It still parked its outputs at the cap within 47 generations, and stage 2 stayed in that basin at 9.3.

Across these designs the basin is the same: constant outputs, set by the readout's biases, scoring about 9.3–10. A worm that varies its outputs scores worse there, so the search does not leave.

## An imitation start, then a guard against constant outputs

**The imitation** (`imitation_d.json`). The teacher was C.1's warm probe answer on v4 (Logbook 125), which drove the rollouts on D_fit's two assays. On the same senses, a shadow C.2 advanced its own modulators. C.2 was then trained by gradient descent to reproduce the teacher's actions, in three rounds of 20 epochs (about 3.3 h).

- **The fit to the teacher:** R² about 0.3 for speed and the reversal hazard; curvature's noise draw cannot be learned. C.2 learns the teacher's average action given the senses, not its roaming or dwelling state.
- **Closed-loop**, the imitation answer is responsive but leaves the lawn (D_fit 8.0).
- **sep-CMA refined it** to D_fit 4.86 at 200 generations. Its median score stayed near 10.

**The guard and a wider search** (`guarded_wide.json`), from the refined answer:

- a candidate whose speed or reversal output spans less than 0.5 between its 5th and 95th percentiles scores 300 more, alongside the saturation guard;
- the population is 48, against the earlier 16.

| Generation | Median score | Best | Responsive share | D_fit (one population) | Fitted families passing |
|---|---|---|---|---|---|
| 1 | 19.16 | 5.00 | 73% | — | — |
| 50 | 5.69 | 2.59 | 92% | 2.84 | 0 |
| 100 | 2.96 | 2.02 | 100% | 2.02 | 1 |
| 150 | 2.26 | 1.40 | 100% | 1.62 | 3 |
| 200 | 2.14 | 1.41 | 100% | 1.44 | **4** |
| 250 | 1.79 | 1.37 | 100% | **1.33** | **4** |

Its answer runs 83% of seconds and reverses 17%, with real worms at 14% on the lawn. Its motor neurons are unsaturated, and its sense-to-pool gains are 0.05–0.56. Its curvature output sits near the top of its range throughout.

## The full grade

`full_grade.json`: the guarded answer on 5 populations of 200, every family graded with presence, and the three mutants. tph-1 and pdfr-1 are encoded by their knockout genes; tax-4 by zeroing the seven tax-4 classes' gains.

- **Every fitted family passes in all 5 populations** (D_fit 1.29–1.40).

- **Local search fails in all 5.** Its log early/late ratio is about 0.00, against 1.39 ± 0.38 (z −3.7), and its presence test fails. Its reorientation rate does not decay after removal.

- **Chemotaxis fails in all 5:**

  - drift 0.004 mm/s against 0.026;
  - weathervane +4.1 to +4.3°/s against 0.36 (z +4.3 to +4.5);
  - onset rate 4.8 per minute against 1.1 (z +3.9);
  - flip fraction 0.144 against 0.023 (z +5.9);
  - presence fails.

  Off food it keeps its lawn reversal rate and does not climb the gradient.

- **The mutants:**

  - tph-1 fails in 5 of 5: its roaming changes by +0.003, where real tph-1 roams more (z −5.2);
  - tax-4 fails in 5 of 5, its roaming changing in the wrong direction;
  - pdfr-1 passes in 4 of 5.

- **D, all families, is 3.5–3.7**, and the wild type passes in 0 of 5. Its reversal share over the off-food windows is about 21%, against Roberts' 17.3%.

## The ablation: without the imitation

`guarded_stage1.json`: the guarded search with population 48, started from the staged pilot's stage-1 answer, with no teacher.

- **The start** is unresponsive: its speed and reversal outputs span 0.05 and 0.06. Penalised by the guard, it scores 309.9.
- **The search finds responsive candidates by itself:** 33% of the first generation's, and 100% by generation 50.

| Generation | Median score | Best | D_fit (one population) | Fitted families passing |
|---|---|---|---|---|
| 50 | 9.31 | 8.41 | 4.83 | 0 |
| 100 | 7.26 | 4.88 | 5.17 | 1 |
| 150 | 5.52 | 3.64 | 4.09 | 1 |
| 200 | 3.83 | 2.39 | 2.48 | 2 |
| 250 | 2.71 | 2.01 | 2.15 | 2 |

At generation 250:

- the small lawn and the uniform well pass;
- kinematics miss on the dwelling speed's 90th percentile (z +3.12) and on roaming survival (+3.55);
- activity misses on the onset rate (−4.3);
- the worm reverses 9% of seconds and spends 54% in delta turns.

Its median fell from 3.83 to 2.71 over the last 50 generations. The imitation start reached all four families by generation 200; this run reached two by 250, and how many more generations it needs is not measured.

## What this establishes, and what it does not

- **The run mode is solved on the fitted families,** but only with four changes acting together: the bounded readout, the graded box, the two guards and the wider search. Every partial combination stalled at constant outputs.

- **The imitation start speeds the fit; the guard and the wider search are what escape the basin.** Without the start, the fit is slower and had not reached all four families when stopped. So the fit that passes depends on C.1 only through its start, and a longer run without it may remove even that.

- **D_fit carries no signal for the held-out families,** and the fitted C.2 does not reproduce them:

  - it has no post-removal search, though a hand-set vector carries a 9-minute decay (Logbook 122), so the model can make one;
  - it has no chemotaxis;
  - its serotonin and tax-4 mutants show no effect, or the wrong one.

  Its fitted modulator layer is not doing what the wild type's does. That is a held-out result for this fit, not a property of C.2's model, since only one answer was graded.

- **Every reading is description:** one restart per design, on discarded seeds, under bounds applied in scripts.

## Cost

On 16 workers:

- about 25 s per generation at population 16 on D_fit's assays, and about 80 s at 48;
- the imitation's training took about 3.3 h, its refinement 1.4 h, and the guarded search 5.55 h. That is about 10 h for one fit as run, against Logbook 122's 4.8 h staged fit, which W's 232–815 h projection assumed.

## Implications

- **C.2's next change** implements the working procedure in the encoder and the fit, each piece off by default and byte-identical: the soft cap and pool centres, per-ligand tying, the graded box, the two guards, the wider population and its start. It then registers the readiness restarts, with their restart-to-restart spread read against W's minimum.
- **Its open questions:**
  - whether the imitation start is kept. Without it the fit escapes the basin, but reaches 2 of 4 families at 250 generations against 4 of 4 at 200 with it. A longer run without it decides whether C.2 can be independent of C.1;
  - whether C.2 is fitted on every family, as C.1 is, which would give up local search and chemotaxis as held-out tests, or keeps them held out and reports the failure;
  - the cost per fit, which decides what W can afford.
- **The slow-noise control** (Logbook 123) now has a fit whose speed responds, and is re-run on it in that change.

## Seeds

All discarded:

- **Fits:** 700, 702, 704, 706, 708, 710, 714, 716 and 722 M, with their checkpoints one block above (701, 703, 705, 707, 709, 711, 715, 719 and 725 M).
- **Imitation:** the rollouts 712 M and 712.1 M; the imitation answer's grade 715.9 M.
- **The full grade:** 720 M.
- **Profiles:** 99.66, 99.67, 99.69 and 99.71 M.
- **Smoke checks:** 99.68 M and 99.70–99.76 M.
- **CMA:** 9449 and 9471–9478.
