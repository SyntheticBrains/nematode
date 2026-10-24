## ADDED Requirements

### Requirement: A classifier from the band cohorts

The benchmark's roaming/dwelling classifier SHALL be the two-state Gaussian HMM of
`validation/roaming_dwelling.py`. It SHALL be fitted against the authors' labels on two thirds of the
OD 2 wild type's band dates, drawn by date with a fixed seed. It SHALL be gated at Cohen's κ ≥ 0.6 on
the remaining third.

The fitted model SHALL be vendored with its split, its gate's reading and its provenance. No held-out
date SHALL be read. D.1's model and its records SHALL be unchanged.

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
- a window on the lawn when every sample in it is on the lawn;
- a window with a missing sample unclassified.

A real worm's sample SHALL be on the lawn when its midbody lies inside Scheer's boundary for its
background slot. A simulated worm's SHALL be on the lawn when the capture marks it so: inside the lawn's
footprint, in a phase with lawns.

#### Scenario: A worm on the boundary

- **WHEN** a window's samples lie inside the lawn except one just outside it
- **THEN** the window SHALL be neither wholly on nor wholly off the lawn

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
- **THEN** exactly its minutes 20–60 SHALL be read: 240 windows

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
  (8–14 min over 34–44 min).

Each mutant SHALL be graded on differences from its same-date wild type:

- tph-1 (against N2), pdfr-1 and tax-4 (against PD1074): Δ roaming fraction and Δ exits per on-lawn
  hour;
- pdfr-1 also on López-Cruz: Δγ and Δ early/late against its paired control.

A statistic that cannot be computed SHALL fail its band. On real tracks every value SHALL come from
this pipeline, never from the authors' labels or published values.

#### Scenario: A worm that never leaves

- **WHEN** a simulated worm makes no exit
- **THEN** its exits per on-lawn hour SHALL be 0 and graded
- **AND** its pre-exit enrichment SHALL fail, entering the distance at the cap

### Requirement: Bands from the band dates

For a statistic and a graded population of n animals, the band SHALL be read from a cluster bootstrap
over band dates. Each resample draws dates with replacement, taking each drawn date's animals until n
is reached, subsampling the last. There SHALL be 2,000 resamples with a fixed seed.

- The band's centre SHALL be the statistic pooled over every band animal, and its scale the bootstrap's
  SD.
- A mutant's resample SHALL draw paired dates and take Δ over them.
- An arm with fewer than five band dates SHALL take the larger of its own scale and the OD 2 wild
  type's at the same n, and √2 times that for a mutant's Δ.
- López-Cruz's band SHALL be the mean and SD over its band batches: distinct date strings, a third held
  out by a fixed seed.

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
- **Each mutant** SHALL be its own family. It passes when each of its statistics has |z| ≤ z\*, with
  z\* = 2.24 for two statistics and 2.50 for four.
- **Direction:** where the reference's 95% bootstrap interval for a Δ excludes zero, the graded Δ SHALL
  also have its sign.

#### Scenario: Normal leaving is an equivalence band

- **WHEN** a mutant's reference Δ exits interval includes zero
- **THEN** that statistic SHALL be graded by magnitude alone

#### Scenario: A wrong direction fails inside the band

- **WHEN** a reference Δ roaming interval lies wholly below zero, and a graded Δ is positive with
  |z| ≤ z\*
- **THEN** the mutant SHALL fail

### Requirement: The z distance

The distance D SHALL be the weighted RMS of the wild-type z's. Each assay SHALL weigh 1/3, split equally
among its statistics. D_fit SHALL drop local search and renormalise the rest.

The record SHALL register the rule for W's minimum, to be read in B.4:

- split the held-out wild-type dates into halves by date, 1,000 times;
- for each split, take z = (s_A − s_B)/σ at each half's size and form D;
- W's minimum is that distribution's 95th percentile.

#### Scenario: A worm at every centre

- **WHEN** every graded statistic equals its band's centre
- **THEN** D and D_fit SHALL be 0

### Requirement: The rule's own check

Before any simulated worm is graded, the OD 2 wild type's band dates SHALL be split in halves 200 times
by a fixed seed, and López-Cruz's band batches likewise. Bands built on one half, at the other half's
size, SHALL grade the other half at the wild-type z\*. The joint pass rate SHALL be at least 0.90. The
uniform lawn SHALL be left out, since two dates cannot be halved, and the record SHALL say so.

#### Scenario: A failing check stops the bands

- **WHEN** the joint pass rate is below 0.90
- **THEN** the bands script SHALL exit nonzero without writing the bands

### Requirement: A preflight before the controls register

ConstantHazards and PirouetteHazards SHALL each be graded at n = 64 on pilot seeds 9101–9164, on the
small lawn, the uniform lawn and López-Cruz's assay, under the amended primitives. The result SHALL be
recorded as description. The bands SHALL not be changed by it.

#### Scenario: A Poisson worm passes the preflight

- **WHEN** ConstantHazards passes the wild type
- **THEN** the record SHALL say so, and B.4 SHALL not register until the maintainer decides

### Requirement: Reported beside the grading

The bands record SHALL report, without grading:

- the roaming fraction at OD 1, OD 2 and OD 4;
- the authors' labels' readings beside ours;
- Flavell 2013's bout timescales;
- the quintile ratio's reference;
- the roadmap's expected mutant directions, beside the references' intervals.

#### Scenario: The density arms are not graded

- **WHEN** a worm is graded
- **THEN** no OD 1 or OD 4 statistic SHALL enter its pass or its distance
