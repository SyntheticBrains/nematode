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
