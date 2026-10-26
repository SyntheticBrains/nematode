# controller-fit Specification

## Purpose

Fitting a controller to the real wild type by the validated method, under a registered verdict, and
the mutant encoding table that every controller's predictions are graded under.

## Requirements

### Requirement: The fit and the mutant encoding table are registered before any fit

A launch record SHALL fix C.1's objective, sep-CMA settings, restarts, plateau criterion, grading population, verdict and verdict map before any fit simulation runs. An encoding table, committed with it, SHALL fix every controller's encoding of tph-1, pdfr-1 and tax-4, what each mutant is graded on, and the `unidentified` rule. An encoding SHALL be a structural knockout with no free magnitude. A mutant with no counterpart in a controller SHALL be recorded as not encodable.

#### Scenario: The fit is guarded

- **WHEN** `run` starts while the launch record is uncommitted or changed, or the encoding table differs from its version in the launch record's commit
- **THEN** it SHALL refuse to run

#### Scenario: A mutant with no counterpart

- **WHEN** a controller has no mechanism a mutant's gene acts on
- **THEN** its row SHALL read not encodable, and that controller SHALL make no prediction for that mutant

### Requirement: C.1 is fitted to the wild type by sep-CMA

C.1 SHALL be fitted by sep-CMA with 8 restarts from the box centre (σ₀ 0.2, population 16, bounds [0, 1]). The objective SHALL be the benchmark's D on the 11 wild-type statistics against the band cohorts at n = 200, for 200 fresh animals per evaluation. Each restart SHALL stop at the plateau rule or at 210 generations. A restart stopped by the cap SHALL NOT be chosen. The chosen restart SHALL be the plateaued restart with the lowest mean median objective over its last 20 generations.

#### Scenario: No restart plateaus

- **WHEN** every restart stops at the cap
- **THEN** no verdict SHALL be read, and the run SHALL return to the maintainer

### Requirement: The wild-type verdict is the benchmark's rule

Every restart's answer SHALL be graded on five registered populations of 200 animals. A population SHALL pass when every family passes: each statistic at |z| ≤ 2.84, and local search only where its presence test holds. The chosen restart SHALL pass the wild type when at least 3 of its 5 populations pass. Every restart's verdict and D, the restart spread and the parameters SHALL be reported beside the verdict.

#### Scenario: C.1 fails the wild type

- **WHEN** fewer than 3 of the chosen restart's 5 populations pass
- **THEN** the failing statistics SHALL be recorded as the diagnosis, and the roadmap's pivot SHALL go to the maintainer before C.2

#### Scenario: One unlucky population does not fail a faithful controller

- **WHEN** one of five populations fails a family and the other four pass
- **THEN** the chosen restart SHALL pass the wild type

### Requirement: Mutant predictions that depend on the restart are unidentified

Among a controller's restarts whose answers pass the wild type (3 of 5 populations), a mutant prediction whose verdict differs between restarts SHALL be `unidentified`.

#### Scenario: Two passing restarts disagree

- **WHEN** one passing restart's tph-1 prediction passes and another's fails
- **THEN** the tph-1 prediction SHALL be recorded as `unidentified`
