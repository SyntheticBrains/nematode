# benchmark-grading Specification

## Purpose

Grade simulated worms against real worms on standard on-lawn and search assays. Real and simulated
tracks are read through one window rule and one classifier. Each statistic has a band built from
real worms' between-date spread, on band cohorts only. There is a per-statistic pass at a
family-wise bar, mutant differences are graded against the worm's own wild type, and a z distance
summarises the fit.

## Requirements

### Requirement: A classifier from the band cohorts

The benchmark's roaming/dwelling classifier SHALL be the two-state Gaussian HMM of
`validation/roaming_dwelling.py`. It SHALL be fitted against the authors' labels on two thirds of the
OD 2 wild type's band dates, drawn by date with a fixed seed. It SHALL be gated at Cohen's κ ≥ 0.6 on
the remaining third.

Only windows on the lawn by the window rule and labelled roaming or dwelling SHALL enter the fit, as
runs of consecutive such windows. The gate SHALL decode the same runs. The fitted model SHALL be
vendored with its split, its gate's reading and its provenance. No held-out date SHALL be read. D.1's
model and its records SHALL be unchanged.

#### Scenario: The gate is read on dates the fit never saw

- **WHEN** the calibration script runs
- **THEN** its fit and gate dates SHALL be disjoint band dates, and no held-out date SHALL be loaded

#### Scenario: A failed gate stops the calibration

- **WHEN** κ on the gate dates is below 0.6
- **THEN** the script SHALL exit nonzero without writing the model, and name the registered fallback

### Requirement: One window rule for real and simulated tracks

Real and simulated tracks SHALL be read through the same functions:

- positions sampled each second, read at 5 s;
- 10-s windows;
- a window on the lawn when every 1-s sample in its span is on the lawn;
- a window with a missing sample unclassified.

On the small lawns, a real worm's sample SHALL be on the lawn when its midbody lies inside Scheer's
boundary for its background slot. On the uniform well, whose bacteria cover it, every real and
simulated sample SHALL be on the lawn. A simulated worm's SHALL be on the lawn when the capture marks
it so: inside the lawn's footprint, in a phase with lawns.

#### Scenario: A worm on the boundary

- **WHEN** a window's samples lie inside the lawn except one just outside it
- **THEN** the window SHALL be neither wholly on nor wholly off the lawn

#### Scenario: The uniform well has no boundary

- **WHEN** a uniform-lawn track, whose boundary radii are missing, is read
- **THEN** every window without a missing sample SHALL be on the lawn and classified

#### Scenario: Agreement with D.1's runs is reported

- **WHEN** the bands are built
- **THEN** the record SHALL report the share of band windows on which the geometric rule and the
  authors' in-lawn runs agree

### Requirement: Phase-aware readers

A simulated run SHALL be read only from its assay phase's `analysis_from_s` to the phase's end. A run
without an assay, or whose analysis window is shorter than its reference's, SHALL be refused, naming
both lengths. A window longer than the reference's SHALL be read for the reference's length only.

#### Scenario: The small lawn reads its last 40 minutes

- **WHEN** a `scheer_lawn` capture is graded
- **THEN** exactly its minutes 20–60 SHALL be read: 2,400 samples, giving the same 239 complete windows
  a real 40-min track gives

### Requirement: Censoring-aware bout means

A bout SHALL be a maximal run of one state inside one on-lawn run. It SHALL be completed when it ends
in the other state, and censored when the run ends. A state's mean bout duration SHALL be its total
time over its completed bouts. A state with no completed bout SHALL have no mean.

#### Scenario: A bout cut by leaving is censored

- **WHEN** a dwelling bout of 300 s ends because the worm leaves the lawn
- **THEN** it SHALL add 300 s to dwelling's time and nothing to its count of completed bouts

### Requirement: The graded statistics

The wild type SHALL be graded on 11 statistics:

- Scheer's small lawn: roaming fraction, roaming bout mean, dwelling bout mean, sustained exits per
  on-lawn hour, pre-exit enrichment (30–180 s before each exit), and time on the lawn;
- the uniform lawn: roaming fraction, roaming bout mean and dwelling bout mean;
- López-Cruz's local search: γ over 8–44 min after removal with t₀ = 9 min, and the early/late ratio
  (8–14 min over 34–44 min). γ SHALL be undefined when the fit finds no decay: the fitted γ within 1%
  of the search grid's bound, or a fitted α no larger than 1e-9 of the curve's level.

Each mutant SHALL be graded on differences from its same-date wild type: tph-1 (against N2), pdfr-1
and tax-4 (against PD1074), on Δ roaming fraction and Δ exits per on-lawn hour. pdfr-1's López-Cruz
Δγ and Δ early/late SHALL be reported, not graded, since no held-out cohort can positive-control
them.

A statistic that cannot be computed SHALL fail its band. On real tracks every value SHALL come from
this pipeline, never from the authors' labels or published values.

#### Scenario: A worm that never leaves

- **WHEN** a simulated worm makes no exit
- **THEN** its exits per on-lawn hour SHALL be 0 and graded
- **AND** its pre-exit enrichment SHALL fail, entering the distance at the cap

#### Scenario: A flat search curve has no decay rate

- **WHEN** a population's onsets per bin do not decay
- **THEN** its γ SHALL be undefined and fail at the cap, and its early/late ratio SHALL still be graded

### Requirement: Bands from the band dates

For a statistic and a graded population of n animals, the band SHALL be read from a cluster bootstrap
over band dates. Each resample draws dates with replacement, taking each drawn date's animals until n
is reached, subsampling the last. There SHALL be 2,000 resamples with a fixed seed.

- The band's centre SHALL be the statistic pooled over every band animal, and its scale the bootstrap's
  SD.
- A mutant's resample SHALL draw paired dates and take Δ over them.
- An arm with fewer than five band dates SHALL take the larger of its own scale and the OD 2 wild
  type's at the same n, and √2 times that for a mutant's Δ.
- López-Cruz's band SHALL be the mean and SD of per-batch fits over its band batches: distinct date
  strings, a third held out by a fixed seed. pdfr-1's paired batches SHALL be kept as band data and
  left out of the draw.

The registered retry of 2026-10-24 SHALL also apply:

- every Scheer scale SHALL be κ_s · √(σ_boot(n)² + σ_centre²), with σ_centre the pooled statistic's
  bootstrap SD over the band dates at their full size (for a mutant, the paired dates at the mutant's
  full size);
- κ_s SHALL be the SD of that statistic's z over 200 half-splits of the OD 2 band dates by a fixed
  seed, under the centre-corrected scale, floored at 1;
- local search's scale SHALL be the band batches' SD × √(1 + 1/k) × t\_{k−1}(1 − 0.05/22) / z\*(11),
  for k band batches.

No held-out date or batch SHALL be read.

#### Scenario: A band grows as the population shrinks

- **WHEN** the roaming fraction's band is built at n = 64 and at n = 200
- **THEN** its scale at 64 SHALL be larger

#### Scenario: A thin arm is inflated

- **WHEN** the uniform lawn's band is built
- **THEN** each scale SHALL be at least the OD 2 wild type's for the same statistic and n

### Requirement: The pass rule

Each statistic's z SHALL be `(s − μ)/σ`, with |z| capped at 10.

- **The wild type** SHALL pass when every one of its 11 statistics has |z| ≤ 2.84, Bonferroni for a
  95% family-wise rate.
- **Each mutant** SHALL be its own family of two statistics. It passes when each has |z| ≤ 2.24.
- **Direction:** a reference Δ's direction SHALL be resolved when a two-sided paired t-test of its
  per-date differences, over the paired band dates, rejects zero at 0.05 with dates − 1 degrees of
  freedom. Where it is resolved, the graded Δ SHALL also have its sign. A single paired date SHALL
  resolve no direction.

#### Scenario: Normal leaving is an equivalence band

- **WHEN** a mutant's reference Δ exits is not resolved by the paired t-test
- **THEN** that statistic SHALL be graded by magnitude alone

#### Scenario: A wrong direction fails inside the band

- **WHEN** a reference Δ roaming is resolved below zero, and a graded Δ is positive with |z| ≤ 2.24
- **THEN** the mutant SHALL fail

#### Scenario: The direction reads the dates, not the band's scale

- **WHEN** a reference Δ's paired t-test rejects zero but its centre lies within 1.96 of the band's
  scale of zero
- **THEN** its direction SHALL be resolved

### Requirement: The z distance

The distance D SHALL be the weighted RMS of the wild-type z's. Each assay SHALL weigh 1/3, split equally
among its statistics. D_fit SHALL drop local search and renormalise the rest.

The record SHALL register the rule for W's minimum, to be read in B.4:

- split the held-out wild-type dates into halves by date, 1,000 times;
- for each split, take z = (s_A − s_B)/σ, σ the band's scale at a half's size, and form D;
- W's minimum is that distribution's 95th percentile.

#### Scenario: A worm at every centre

- **WHEN** every graded statistic equals its band's centre
- **THEN** D and D_fit SHALL be 0

### Requirement: The rule's own check

Before any simulated worm is graded, two checks SHALL be read on the band data alone, at the wild-type
z\* and over 200 half-splits by a fixed seed:

- **the small lawn:** bands built on half the OD 2 band dates at n = 200 SHALL grade a 200-animal
  population drawn from the other half by the same date resampling, with a joint pass rate of at least
  0.90 over the six statistics;
- **López-Cruz:** a band built on half the band batches SHALL grade each batch of the other half alone,
  with a per-batch joint pass rate of at least 0.90.

The uniform lawn SHALL be left out, since two dates cannot be halved, and the record SHALL say so.

The retry's check SHALL be read on half-splits drawn by a seed other than κ's, with κ fixed.

#### Scenario: A failing check stops the bands

- **WHEN** either joint pass rate is below 0.90
- **THEN** the bands script SHALL exit nonzero without writing the bands

### Requirement: A preflight before the controls register

ConstantHazards (run speed 0.15 mm/s; hazards 0.02, 0.01 and 0.01 per second) and PirouetteHazards
(the same, with β = 3) SHALL each be graded at n = 64 on pilot seeds 9101–9164, on the small lawn, the
uniform well and López-Cruz's assay, under the amended primitives. The result SHALL be
recorded as description. The bands SHALL not be changed by it.

#### Scenario: A Poisson worm passes the preflight

- **WHEN** ConstantHazards passes the wild type
- **THEN** the record SHALL say so, and B.4 SHALL not register until the maintainer decides

### Requirement: Reported beside the grading

The bands record SHALL report, without grading:

- the roaming fraction at OD 1, OD 2 and OD 4;
- the authors' labels' readings beside ours;
- Flavell 2013's published bout timescales, without a simulated reading;
- B.2's pooled-curve γ (0.114) beside the band's centre;
- the quintile ratio's reference.

The mutant rows are reported under "Reported beside the mutant grading".

#### Scenario: The density arms are not graded

- **WHEN** a worm is graded
- **THEN** no OD 1 or OD 4 statistic SHALL enter its pass or its distance

### Requirement: Grading at each family's own size

A population SHALL be graded at its own size, family by family. Where `bands.json` holds bands at a
family's n, they SHALL be used. Otherwise the family's bands SHALL be built at n by the same procedure,
resamples, with κ from the bands' record. Each (family, n) and (mutant, n) SHALL draw from its own
generator seeded by the family and n, so that a band does not depend on what was built before it.
Building the uniform well or a mutant at n SHALL build the small lawn at n first. Families in one
grading MAY differ in n. López-Cruz's band SHALL not depend on n.

#### Scenario: Bands built on demand match the vendored ones

- **WHEN** the small lawn's bands are built on demand at n = 200, after other bands were built
- **THEN** they SHALL equal `bands.json`'s at n = 200 exactly

#### Scenario: A real control at its own size

- **WHEN** a population of 31 animals is graded on the uniform well and one of 573 on the small lawn
- **THEN** each family SHALL be graded on bands at its own n

### Requirement: Reported beside the mutant grading

The bands record SHALL report, without grading:

- pdfr-1's López-Cruz Δγ and Δ early/late against its paired control;
- Ji 2021's patch-assay roaming for pdfr-1 and tax-4 against its wild type's, as directions only;
- each mutant statistic's paired t, its dates, and the direction it resolves;
- the roadmap's expected mutant directions, beside the paired t's resolution.

#### Scenario: pdfr-1's search is not graded

- **WHEN** a simulated pdfr-1 is graded
- **THEN** no López-Cruz statistic SHALL enter its pass

### Requirement: Local search must be present

A population's local search SHALL be present when its early/late ratio exceeds 1 at its own one-sided
97.5% lower bound. The bound comes from a bootstrap over animals for a simulated population, and from
normal draws at each bin's mean and SEM for a real batch: 2,000 resamples with a fixed seed. The
local-search family SHALL pass only when every one of its statistics passes and search is present.
Presence SHALL not enter the distance.

#### Scenario: A flat population fails local search

- **WHEN** a population's onsets per bin do not decline from early to late
- **THEN** its local search SHALL be absent, and the local-search family SHALL fail, whatever its z's

#### Scenario: The band batches are present

- **WHEN** the bands are built
- **THEN** the record SHALL report each band batch's presence bound, and every one SHALL exceed 1

### Requirement: Simulated tracks are measured as the real tracker measures

Before windowing, every simulated track graded against a measured rig SHALL receive that rig's position noise (on Scheer's rig, resampled from the real tracker's readings of a still worm), and SHALL lose seconds at the tracker's measured rates, conditioned on whether the simulated primitive that second is an omega. Both SHALL be measured from band cohorts and recorded with the bands. Real and imported tracks SHALL NOT be altered. Checks SHALL show the model reproduces the tracker without a smoothed truth: real tracks with their gaps filled, passed through the model's losses, SHALL give every kinematics statistic within |z| ≤ 1 of the raw tracks' value; the model's jitter on a still worm SHALL reproduce the real tracker's readings on the authors' paused stretches to within one graded-size band scale of the statistics they bias; and Chen's smoothed fragments with the model's jitter SHALL give every chemotaxis statistic within |z| ≤ 1. Local search, graded against counts from a rig no track measures, SHALL stay exact.

#### Scenario: An omega second can go missing

- **WHEN** a simulated animal is in an omega bout for a second
- **THEN** that second SHALL be missing with the measured omega-second probability, from the animal's own stream

#### Scenario: Real tracks are untouched

- **WHEN** imported or vendored real tracks are graded
- **THEN** no noise SHALL be added and no sample SHALL be dropped

### Requirement: The kinematics family

B v2 SHALL grade a kinematics family on the small lawn: the median and 90th-percentile 10-s window speed in dwelling and in roaming windows, and Kaplan–Meier roaming-bout survival at 5 min. Detected reversal onsets per on-lawn minute SHALL be computed and reported beside the grade, not graded, since no open-loop point worm reaches it through the tracker. It SHALL be a family of equal weight in D, and part of D_fit for every controller.

#### Scenario: A worm that roams too fast

- **WHEN** a controller's roaming windows are much faster than the band cohorts'
- **THEN** its roaming speed statistics SHALL fail, whatever its roaming fraction

### Requirement: Versioned bands and bar

B v1's bands and rule SHALL remain available for every past verdict. B v2's bands SHALL be built from band cohorts only, with the new statistics' generators after v1's, and its bar SHALL be Bonferroni over its 19 wild-type statistics.

#### Scenario: A past verdict is reproduced

- **WHEN** grading is asked for v1
- **THEN** it SHALL reproduce v1's values exactly

### Requirement: The chemotaxis family

B v2 SHALL grade a chemotaxis family on a linear-gradient assay matched to Chen et al. 2025's naive N2 butanone plates, from positions and the known gradient alone, with bands from its six standard-protocol recordings validated leave-one-recording-out, registered before any statistic is computed. It SHALL be held out of D_fit (a prediction for C.2 and C.3) and fitted by C.1, and simulated tracks on that assay SHALL receive that tracker's measured jitter and fragmentation. The simulated arena SHALL be the full plate, with samples outside the imaged field missing, so no statistic reads a wall. Like local search, chemotaxis SHALL pass only where it is present: up-gradient drift and the log quintile ratio each above zero at the 2.5% quantile of 2,000 resamples of the population's units (animals for a simulated population, fragments for a real recording).

#### Scenario: A worm that ignores the gradient

- **WHEN** a controller's up-gradient drift and turning bias match a random walk's
- **THEN** its chemotaxis statistics SHALL fail
- **AND** this SHALL hold even where its values lie inside the bands, through the presence test

### Requirement: The lawn rig's tracker noise is resampled real noise

B v2's lawn measurement model SHALL draw each simulated track's position noise from a library of the real tracker's readings of a still worm: the offsets from their own mean of PD1074's paused stretches on half of its band dates. The draw SHALL splice stretches at the nearest stored offset (the best of 50 drawn stretches), from the animal's own stream, with no other transformation. The model SHALL pass the jitter check on the other half of the band dates, which the library never saw, before any further B v2 reading. The check's gate SHALL be calibrated on the tracker's own date-to-date variation.

#### Scenario: A still simulated worm reads like a still real worm

- **WHEN** a still point is measured through the lawn model and read on the check half's paused-stretch statistics
- **THEN** each of its onsets per paused minute and its 10-s window speed's median and 90th percentile SHALL differ from the real values by no more than the central 95% of the differences between random halves of the real band dates

#### Scenario: The library never sees the check

- **WHEN** the library is built
- **THEN** it SHALL hold stretches from the library half's dates only, and the jitter check SHALL read the check half's only

#### Scenario: An animal is measured the same way alone or in a population

- **WHEN** an animal with seed s is measured alone and within a population
- **THEN** its noise track SHALL be identical, drawn from `default_rng([s, 2033])`

### Requirement: The reachability oracle's worm can reach its bar

The reachability oracle SHALL read an open-loop two-state worm whose per-state speeds, speed spreads (per 10-s window), curvature and hazards are fitted through the measurement model on discarded seeds. Before registration, the fitted worm SHALL reach the five gated statistics at a median RMS z of at most 1 on fresh discarded populations. If it does not, B v2 SHALL stop before registration, with the statistics it misses as the diagnosis.

#### Scenario: The matched worm misses with no noise

- **WHEN** B.4's matched worm reads outside the bands with no measurement noise
- **THEN** it SHALL NOT be the reachability oracle's worm, and the fitted worm SHALL take its place

#### Scenario: No open-loop worm reaches a statistic

- **WHEN** the fitted worm's median RMS z on fresh populations exceeds 1
- **THEN** the change SHALL stop before registration, and the statistic it misses SHALL be recorded as unreachable by an open-loop point worm

### Requirement: The on-lawn reversal rate is reported above the tracker's noise, not graded

The kinematics family's on-lawn reversal rate SHALL be read with the off-food detector at a step floor of 0.03 mm: the smallest 0.01-mm floor at which the real tracker's readings of a still worm, on the library half, detect reversals at under 1% of the real on-lawn rate. Because no open-loop point worm reaches it through the tracker, it SHALL be reported beside the grade and not graded, and B v2's bar SHALL be Bonferroni over 19 statistics. Off-food uses of the detector SHALL be unchanged.

#### Scenario: Tracker noise does not read as reversals

- **WHEN** a still worm's real readings are passed to the on-lawn reversal detector
- **THEN** it SHALL detect reversals at under 1% of the real on-lawn rate

### Requirement: Only Scheer's rig is measured by the lawn model

The lawn measurement model SHALL apply to the families graded against Scheer's rig (the small lawn, the uniform well and kinematics). Simulated local-search tracks SHALL stay exact, since López-Cruz's reorientation counts come from a rig whose tracker no track measures.

#### Scenario: Local search stays exact

- **WHEN** a simulated local-search population is graded on v2
- **THEN** its tracks SHALL NOT pass through the lawn model

### Requirement: A fit's objective may grade without presence

Grading on B v2 MAY skip the chemotaxis and local-search presence tests when only the statistics and the distance are read, as a fit's objective does. The statistics, D and D_fit SHALL then be identical to a grading with presence. Any grading read against a registered gate SHALL include presence.

#### Scenario: Presence off changes no number

- **WHEN** one simulation is graded with presence and without
- **THEN** every statistic, D and D_fit SHALL be equal, and only the presence records and family passes SHALL differ

### Requirement: The measurement model is faster with identical output

`measure` SHALL return, for every step, the same field values as before. Positions SHALL gain the same noise, and lost seconds SHALL hold NaN positions. Each noise splice SHALL make the same random draws and choose the same continuation, the first minimum in row-major order over its (candidate, offset) grid.

#### Scenario: A measured track is the same

- **WHEN** a track long enough to splice, with lost seconds, is measured by the new and the reference code from the same seed
- **THEN** every field of every step SHALL be equal

#### Scenario: Stored simulations re-simulate exactly

- **WHEN** stored or baseline simulations are re-run after the change: v2 with and without presence, and v1
- **THEN** every graded statistic, D and presence reading SHALL equal the stored record exactly
