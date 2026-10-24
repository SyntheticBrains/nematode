## MODIFIED Requirements

### Requirement: The graded statistics

The wild type SHALL be graded on 11 statistics:

- Scheer's small lawn: roaming fraction, roaming bout mean, dwelling bout mean, sustained exits per
  on-lawn hour, pre-exit enrichment (30–180 s before each exit), and time on the lawn;
- the uniform lawn: roaming fraction, roaming bout mean and dwelling bout mean;
- López-Cruz's local search: γ over 8–44 min after removal with t₀ = 9 min, and the early/late ratio
  (8–14 min over 34–44 min). γ SHALL be undefined when the fit finds no decay: the fitted γ within 1%
  of the search grid's bound, or a fitted α ≤ 0.

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

#### Scenario: A flat search curve has no γ

- **WHEN** a population's onsets per bin do not decay
- **THEN** its γ SHALL be undefined and fail at the cap, and its early/late ratio SHALL still be graded

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

## ADDED Requirements

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
