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

- **Moving costs energy**, added after the first pilot (see Decision G). It is counted in intake value:
  each millimetre moved costs `lawns.movement_cost_per_mm`, taken from reward and satiety alike, so
  the learner sees what the worm pays. Standing still costs only the basal decay every worm pays.

- **No shaping that favours either state.** Four reward terms would:

  - `penalty_stuck_position` punishes dwelling directly;
  - `penalty_anti_dithering` fires whenever the worm is where it was two steps before, which a
    worm staying put always is, so it punishes dwelling too;
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

**Two open deposits**, both CC0, chosen with the user on 2026-10-10:

- **The classifier's source: Scheer & Bargmann 2023** (eLife 88657; Dryad 10.5061/dryad.47d7wm3jf,
  mirrored on Zenodo 8310289).
  - **Animals:** 1,586 wild-type animals, 40 minutes each at 3 frames/s, one per well on a small
    bacterial lawn.
  - **Per animal:** midbody speed, angular speed, and whether it is in the lawn, plus the authors'
    roaming/dwelling label per 10-second bin.
  - **The model:** the authors' two-state roaming/dwelling HMM.
- **The directions' source: Ji et al. 2021** (Dryad 10.5061/dryad.3bk3j9kh3). Its patch foraging
  assay, food-density and *tph-1* / *pdfr-1* fractions of animals roaming.

**The authors' method**, read from their code (MIT):

1. Each 10-second bin's midbody speed and angular speed (degrees between consecutive frames' motion
   vectors, averaged) is a roaming observation when `speed × 450 > angular speed`.
2. A two-state categorical HMM over those binary observations, decoded within each run of in-lawn
   bins, gives the states.

The HMM's parameters are vendored from the deposited model, read without executing it:

| | dwelling | roaming |
|---|---|---|
| stay per 10 s | 0.979, mean bout about 8 min | 0.898, mean bout about 1.6 min |
| emits a roaming observation | 1% | 53% |

**Transferred to the point worm.**

- **The measures.** The point worm moves in 5 s steps. Its angular speed is the angle between
  consecutive steps' displacements, which is coarser than the authors' frame-to-frame measure, so
  their slope of 450 does not transfer.
- **Calibrating the slope.** The real tracks are resampled to 5 s and measured exactly as simulated
  tracks are (two-step windows, the three-point turn). The slope is then calibrated on real worms:
  the value whose decoded states best agree with the authors' labels (Cohen's κ). It is fitted on a
  calibration half of the animals, split by animal with a fixed seed.
- **The gate, fixed before calibration.** On the held-out half, agreement must reach **κ ≥ 0.6**,
  "substantial" (Landis & Koch 1977). Otherwise the instrument is not used and the change stops for a
  decision.
- **What carries over unchanged.** The authors' HMM smooths the binary observations, decoded within
  each run of on-lawn windows. Simulated worms are then read with the same slope and model, never
  refitted.

**The first calibration failed its gate, and one retry is registered.** Calibrated on 793 animals, the
line reached held-out κ = 0.49 (accuracy 90.1%, roaming 11.3% against the authors' 10.6%). At a 5 s
step, a dwelling worm's turn is close to random, so the line separates the states mostly on speed.
With the user, on 2026-10-10, one retry was registered. It was written down before any of it was
computed:

- **Same data:** the same 10-second windows, split (seed 2026) and labels.

- **Features:** each window's log(speed + 0.001 mm/s) and angular speed. A window with no measurable
  turn gets angular speed 0, as the line treated it.

- **Model:** a two-state HMM with Gaussian emissions, fitted with the labels on the calibration
  half:

  - each state's mean and full covariance come from its labelled on-lawn windows;
  - the transition probabilities are counted from consecutive labelled windows within on-lawn runs;
  - the initial distribution is the labels' state shares.

  This is deterministic; no iterative fitting.

- **Decoding:** Viterbi, within each on-lawn run.

- **Gate:** held-out κ ≥ 0.6, one attempt. **If it fails, D.1's roaming/dwelling readout closes**,
  with its reason and a destination, and B.3's start is decided separately.

**The retry passed.** Its held-out κ is 0.632, against the gate of 0.6, with accuracy 91.1%; that is
close to the bar, and is said wherever the instrument is cited. It calls more roaming than the
authors: 17.4% of on-lawn windows, against their 10.6%. So the cell's readings stay on directions and
within-cell contrasts, and absolute fractions are only described. This model is the instrument. The
line is kept beside it, unused for verdicts.

**States are read on lawns only.** Both states are on-food states. Off a lawn, worms search and
disperse, so off-lawn windows are reported as description, never classified into a verdict.

**Vendored.** The HMM parameters (`data/roaming_dwelling/reference_hmm.json`), and a derived file of
the wild-type animals' windows: each window's speed and angular speed in the point worm's measure,
its in-lawn flag and the authors' label. Both come with provenance. The 4.9 GB source pickle is not
vendored. It is read once, after its opcodes are checked to reference only numpy and pandas types.

**The reference readings are directions, graded on sign**, as Logbook 035 graded its curves. The
classifier's own fractions on real worms are reported beside. Absolute fractions from other assays
are described, never matched. Each direction's condition is pinned from the papers' legends in task
1:

| direction | deposited values |
|---|---|
| less roaming and longer dwelling at higher food density | Ji 2021: 0.36 to 0.20 |
| more roaming on sparse food when a dense patch is near than without one | Ji 2021: 0.37 against 0.18 |
| *tph-1* roams more | Ji 2021: 0.73 against 0.37; for B.3 |
| *pdfr-1* roams less | Ji 2021: 0.22 against 0.37; for B.3 |

**Reorientation, checked on the population.** Decision A's `max_turn_rad` of π was set from Ji et al.'s
one track. Task 1 checks it against the wild-type population's 5 s turning distribution in each
state.

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

### Decision G: Feasibility, and what the first pilot changed

**The first pilot, 2026-10-10** (seeds 9101–9104, 3,000 episodes, the original cell: intake 10% of a
cell per step, movement free):

| arm | intake per episode, final quarter |
|---|---|
| with `internal_state` | 13.4 |
| without it | 13.4 |
| untrained floor | 4.4 |

- **Both learners passed the learning gate.**
- **Neither dwelt.** Each was classified as roaming in 98–99% of on-lawn windows, with no complete
  bout of either state.
- **The cause is the cell's economics.** With movement free and a cell losing a tenth of its density
  per step, a worm moving at full speed reaches a fresh cell every step, so continuous roaming is
  optimal and dwelling never pays.

With the user, the cell now charges for movement and depletes slowly, as real lawns do over minutes.
Its numbers, per step, in fractions of maximum satiety:

| | per step |
|---|---|
| eating on a fresh cell (intake 2% of the cell, `satiety_per_intake` 0.3) | +0.006 |
| moving 1 mm (`movement_cost_per_mm` 0.013) | −0.0039 |
| basal decay | 0.0027 |

- **A spot is worth staying on** until its density falls to about 0.35: some 52 steps, 4.3 minutes,
  the order of a real dwelling bout.
- **Roaming through a lawn without stopping slowly starves the worm.**
- **Intake still does not depend on speed.** The cost is locomotion's, not a reward for slowness, and
  off a lawn it makes searching costly too.

**The second pilot, 2026-10-10** (seeds 9105–9108, with the movement cost):

| arm | intake per episode, final quarter | result |
|---|---|---|
| with `internal_state` | 0.15 | stood still from the start and starved at about 366 steps on basal decay alone |
| without it | 0.87 | stayed at its floor's level (floor 0.88) |

Neither passed the learning gate. The worm started at least 2 mm from any lawn. A random policy paid
for every move at once and found food only after several millimetres, so PPO settled on not moving.
The movement cost's economics are right on food, but they turned finding food into an exploration
trap.

With the user, **the worm now starts on a lawn** (`lawns.start_on_lawn`), as assays place worms on
food. The first lawn is centred on its start, and the others are placed clear of it. Dwelling or
leaving pays from the first step, and leaving for another lawn still crosses costly gaps. A third
pilot, on fresh seeds, checks it.

- **States last minutes.** In the reference model a dwelling bout averages about 8 minutes and a
  roaming bout about 1.6. An episode of 720 steps is an hour of worm time, room for several of each.
- **Satiety must matter within an episode.** Decay and intake are set so an idle worm starves in about
  half an episode, and a lawn can sustain a worm for a few minutes before its nearby cells are grazed
  out. The pilot checks both.
- **Geometry.** A 20 mm arena holds three to five lawns of 2–3 mm radius at 1 mm cells, about 30 cells
  per lawn. The odour field costs about 150 kernel evaluations per sensing query, negligible beside a
  step.

### Decision H: Patch-leaving is deferred to D.1b, with a destination

Decided with the user on 2026-10-11.

**Why D.1's cell cannot test leaving.** The worm starts on a lawn of about 21 cells, and that lawn
holds about three times the food it needs to survive an episode. It cannot be grazed out within 720
steps, so a run succeeds without ever leaving, and leaving never pays. This is right for D.1's
question, because roaming and dwelling are within-lawn states: Ji et al.'s and Scheer & Bargmann's
reference worms forage on a single lawn.

**D.1b's design.**

- **Its own cell:** smaller lawns, about 1.5 mm in radius, so that leaving is needed.
- **Its readings:** the leaving rate, and whether leaving comes out of roaming rather than
  dwelling. That coupling is Scheer & Bargmann 2023's central result.
- **Its reference:** the deposit already downloaded holds lawn exits and entries for all 1,586
  wild-type animals.
- **What it answers:** Al-Asmar & Pérez-Escudero's patch-leaving framing, named in the roadmap's D.1
  validation.

**Its order.** D.1b runs after D.1's within-lawn positive control passes, and before B.3, without
blocking it. B.3's references are within-lawn roaming fractions, so B.3 does not need leaving, but if
the mutants shift roaming it is worth knowing whether leaving shifts with it. If D.1's control fails,
D.1b has no states to couple leaving to.

## Risks

- **Dwelling may not emerge.** The positive control exists to say so, and the result is informative
  either way.
- **The calibrated line may not reproduce the authors' labels** at the point worm's coarser step. The
  κ ≥ 0.6 gate on held-out animals catches that before anything is built on it.
- **The literature's directions must be checked**, not recalled: M.7 found five of five chemotaxis
  reference values unsourced. Task 1 verifies every reference reading against its source before the
  registration cites it.
