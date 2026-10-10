## Overview

D.1 builds what roaming and dwelling need on the point worm, then runs a positive control: can a
learner that reads its own satiety take worm-like roaming and dwelling states on patchy lawns? B.3
follows in its own change, gated on this control. The choices below were made with the user on
2026-10-10, and revised after the spec review and a first read of the reference data.

## Decisions

### Decision A: The point worm, set up for both states

Roaming and dwelling differ in speed and reorientation (Ben Arous et al. 2009; Flavell et al. 2013).
C.3 (Logbook 086) found two things about the kinematic body: its speed sat between 0.10 and 0.12 body
lengths/s in all 416 runs, trained or untrained, and its sharp turns were steering pivots without an
omega posture. The point worm sets its speed each step and turns freely. Its full speed is 0.2 mm/s,
one body length per 5 s step.

- **Speed.** Real roaming and dwelling differ several-fold. On Ji et al. 2021's deposited track,
  read at the point worm's 5 s step, the medians are 0.123 and 0.022 mm/s. The lawn cell sets
  `allow_reversal` and `signed_speed`, so zero speed is the centre of the action range rather than a
  limit approached as `tanh` saturates.

- **Reorientation.** On the same track, angular speed over a 5 s step has these medians and 90th
  percentiles:

  | state | median | 90th percentile |
  |---|---|---|
  | roaming | 2.1°/s | 6.6°/s |
  | dwelling | 4.5°/s | 35.4°/s |

  At its default `max_turn_rad` of 0.5 the point worm turns at most 5.7°/s. That covers roaming but not
  dwelling's tail, so the lawn cell sets `max_turn_rad` to π, up to 36°/s. A worm can reverse its heading
  within one step, as a dwelling worm does.

### Decision B: Lawns

`foraging.food_model` takes `points` (the default, unchanged) or `lawns`. Under `lawns`:

| key | meaning |
|---|---|
| `lawns.count` | the number of disc lawns |
| `lawns.radius_mm` | each lawn's radius |
| `lawns.min_separation_mm` | the least edge-to-edge gap, so lawns are patches, not one field |
| `lawns.cell_mm` | the density grid's cell size, about a body length |
| `lawns.quality` | each lawn's nutritional value per unit eaten, a list or a range drawn per lawn |
| `lawns.intake_fraction` | the share of a cell's density eaten per step on it |
| `lawns.reward_per_intake` | reward per unit eaten, times quality |
| `lawns.satiety_per_intake` | satiety gained per unit eaten, times quality, as a fraction of the maximum |
| `lawns.regrowth_per_step` | density regrown per step, 0 by default |

- **Placement.** The existing Poisson-disk sampler places the lawns away from the walls and the worm's
  start. Each starts at density 1 in every cell inside its disc.
- **The edge** is where the density grid ends. Beyond it there is odour but no food. Real lawns are
  thicker and lower in oxygen at the edge; that is not modelled.
- **The odour field** at a point sums, over every lawn cell, its remaining density times its share of
  its lawn's area, times the existing per-source kernel (`fick` or `exponential`) at the cell's distance.
  It is normalised as `get_food_concentration` already does. Weighting by area share makes a full lawn
  smell like one point source of strength 1, so the `tanh` normalisation does not saturate around a
  lawn. A grazed region smells weaker, so the gradient inside a lawn points toward what is left.
- **Every food sensor reads it.** `_compute_food_gradient_vector`, which the oracle modules use, sums
  over cells with the same weights. The klinotaxis and temporal modules sample concentration and need
  no change.
- **Quality does not change the odour.** A worm learns a lawn's quality by eating, as real worms do on
  foods of different quality (Shtonda & Avery 2006, to be verified in task 1).
- **Scope.**
  - Lawns run single-agent only; multi-agent configs with lawns are refused.
  - Lawns refuse the pixel renderers, which would draw no lawns, until the renderer draws them.
    Text themes are left alone, as they draw no food geometry anyway.
  - Predators and thermal fields stay off in the lawn configs.
  - `foods_on_grid`, `target_foods_to_collect` and their validators are ignored under lawns.

### Decision C: Intake, reward, satiety and outcomes

Each step, if the worm's position is inside a lawn, the cell it is on loses `intake_fraction` of its
density. The worm eats that amount, times the lawn's quality.

- **Reward and satiety.** Reward is `reward_per_intake` times what it ate. Satiety rises by
  `satiety_per_intake` times the same, clamped at the maximum. It decays as now, and starvation ends the
  episode as now.

- **Intake does not depend on speed.** Dwelling has to emerge from patchiness, depletion and satiety.

- **No shaping that favours either state.** Three reward terms would:

  - `penalty_stuck_position` punishes dwelling directly;
  - `reward_exploration` pays for visiting new cells, which rewards roaming;
  - `reward_distance_scale` rewards approach for its own sake.

  A config validator refuses `lawns` with any of them non-zero. Under lawns there is no capture event,
  so `reached_goal` is false, the goal bonus never applies, and intake is the only food reward.

- **Outcomes.** A lawn episode ends survived (at `max_steps`) or starved. Each episode's intake and
  outcome are written to the run's summary CSV and its log line. The analysis and the gate preflight
  read them through an intake reader.

### Decision D: The internal-state module

`internal_state` is a sensory module of width 1. It carries satiety as a fraction of its maximum. In
D.1 only MLP-PPO reads it. How internal state reaches the connectome is B.3's question: through the
modulator field's release neurons, not a direct sensory input.

### Decision E: The roaming/dwelling instrument and its reference data

**What the deposit holds** (Ji et al. 2021, Dryad 10.5061/dryad.3bk3j9kh3, CC0, read 2026-10-10):

- **One raw wild-type track** (`Data_F1.A`, about 68 minutes at 3 frames/s): position, speed, and the
  authors' roaming/dwelling label per frame, with 14 state changes.
- **Derived readings from the multi-worm patch foraging assay** (the paper's Fig. 7, the deposit's
  `Data_F6`), each a fraction of animals roaming:
  - wild type in the assay;
  - wild type on uniform sparse food;
  - *tph-1* and *pdfr-1* mutants;
  - wild type at two uniform food densities, with state durations.

**What it does not hold** is a population of raw tracks. The multi-worm data, segmented by Flavell et
al. 2013's speed and angular-speed method, are deposited only as fractions.

**The classifier** is Ji et al. 2021's own method for their tracked animals:

- each time point gets the median and variance of speed over a sliding 20-second window (four 5 s
  steps);
- a two-state hidden Markov model with Gaussian emissions is fitted on the deposited wild-type track,
  resampled to 5 s;
- its agreement with the authors' own labels is reported;
- it is applied unchanged to simulated worms.

The states are defined by speed, which the point worm controls fully. Angular speed per state is
reported beside, against the real track's.

**The reference readings are directions, graded on sign** as Logbook 035 graded its curves. The
deposited fractions come from a different classifier and assay geometry, so their absolute values are
described, not matched. Each direction's condition is pinned from the paper's legends in task 1:

| direction | deposited values |
|---|---|
| more roaming on sparse food when a dense patch is near than without one | wild type in the assay 0.37 against 0.18 |
| less roaming and longer dwelling at higher food density | 0.36 to 0.20 |
| *tph-1* roams more | 0.73 against 0.37; for B.3 |
| *pdfr-1* roams less | 0.22 against 0.37; for B.3 |

The duration arrays' unit is unstated in the legends; task 1 pins it or reports durations only as
ratios.

**One animal is thin for a fit.** The fitted model's fraction on the deposited track is checked against
the authors' labels, and against the deposited wild-type fractions as description. If the fit does not
reproduce the authors' labels on that track, the instrument is not used, and the change stops for a
decision.

**States are read on lawns only.** Both states are on-food states. Off a lawn, worms search and
disperse (Ji et al. 2021 and Flavell et al. 2013 assay on food), so off-lawn windows are reported as
description, never classified into a verdict.

**The model is written here.** A two-state Gaussian HMM fitted by expectation-maximisation is about 80
lines and testable on generated sequences. No dependency is added.

### Decision F: The positive control

| arm | role |
|---|---|
| MLP-PPO with `internal_state` | the learner |
| the same, untrained policy | its floor, paired by seed |
| MLP-PPO without `internal_state` | does reading satiety change the states? |

The registration is written after a pilot on seeds disjoint from the band. It fixes:

- **The learning gate**: intake per episode above the floor, paired by seed.

- **The positive control's question, on lawns.** While on lawns:

  - does the learner take both states, as the real-calibrated model classifies them, with each state's
    mean duration above its floor's;
  - and does its roaming fraction rise as the cells around it deplete?

  The minimum, sizing and verdict map come from the pilot.

- **Readings beside**, graded on sign against Decision E's directions where the cell can express them:

  - a richer lawn holds the worm in dwelling longer (density);
  - a lower-quality lawn is left sooner, once verified in task 1;
  - the with/without-`internal_state` contrast in time on lawns as satiety changes.

**If the control fails**, B.3 does not start on this readout. The logbook records why, and the readout
is revised or closed with a reason.

### Decision G: Feasibility

- **States last minutes.** Read in task 1 from the deposited durations, in whatever unit task 1 pins.
  An episode of 720 steps is an hour of worm time; the real track holds 14 state changes in 68 minutes.
- **Satiety must matter within an episode.** Decay and intake are set so an idle worm starves in about
  half an episode, and a lawn can sustain a worm for a few minutes before its nearby cells are grazed
  out. The pilot checks both.
- **Geometry.** A 20 mm arena holds three to five lawns of 2–3 mm radius at 1 mm cells, about 30 cells
  per lawn. The odour field costs about 150 kernel evaluations per sensing query, negligible beside a
  step.

## Risks

- **Dwelling may not emerge.** The positive control exists to say so, and the result is informative
  either way.
- **One real track calibrates the classifier.** The check against the authors' labels guards it, and
  the reference directions do not depend on matching absolute fractions.
- **The literature's directions must be checked**, not recalled: M.7 found five of five chemotaxis
  reference values unsourced. Task 1 verifies every reference reading against its source before the
  registration cites it.
