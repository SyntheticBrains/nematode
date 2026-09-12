## MODIFIED Requirements

### Requirement: Curriculum-Then-Integrated Cell Structure

The cross-architecture comparison SHALL evaluate each architecture across a three-cell curriculum: a foraging-only smoke (C1, n=1 seed, short budget), a foraging+predator smoke (C2, n=1 seed), and an integrated foraging+predator+thermotaxis primary cell (C3, n ≥ 4 seeds per planning decision T4.0b). Only the C3 cells carry the ranked comparison; C1 and C2 SHALL be treated as de-risking smokes whose failure SHALL block launching the corresponding C3 for that architecture until the failure is diagnosed. The smoke-only status of C1 and C2 applies to the **architecture ranking**. A **wiring contrast** — the same substrate under two wirings — run on a single-behaviour cell at full seed count is governed by the wiring-contrast requirements below and is not a ranking result.

#### Scenario: C1 smoke runs before C2 smoke before C3 cell for each architecture

- **GIVEN** an architecture queued for the comparison sweep
- **WHEN** the architecture's C3 (primary) cell is queued to launch
- **THEN** the architecture's C1 smoke SHALL have completed without error
- **AND** the architecture's C2 smoke SHALL have completed without error
- **AND** C3 SHALL NOT launch until both smokes are green

#### Scenario: C3 is the primary cell with n ≥ 4 seeds; C1 and C2 are throwaway

- **WHEN** the comparison sweep records per-architecture results
- **THEN** the architecture ranking + paired-seed statistics SHALL be computed from C3 results only
- **AND** C1 and C2 results SHALL be retained as smoke verification only (no statistical aggregation, no ranking impact)

#### Scenario: A wiring contrast on a single-behaviour cell is not a ranking result

- **GIVEN** a wiring contrast run on a single-behaviour cell at full seed count
- **WHEN** its statistics are computed
- **THEN** it SHALL be reported under the wiring-contrast requirements, and SHALL NOT enter or alter
  the architecture ranking

## ADDED Requirements

### Requirement: A wiring contrast is run on a cell matched to the behaviour under claim

Where a comparison asks whether a biological wiring is load-bearing, the cell it is measured on
SHALL be matched to the behaviour the claim is about, and the record SHALL state which behaviours
the cell demands. A contrast measured only on a multi-objective cell SHALL NOT be generalised beyond
that cell's demand, and where the substrate's measured deficit is concentrated in one component of
such a cell, that component SHALL be separated before a null on the cell is generalised to the
wiring. A verdict already committed on such a cell stands as committed, scoped to its cell.

#### Scenario: A multi-objective result is scoped to its cell

- **GIVEN** a wiring contrast measured on a cell demanding several behaviours
- **WHEN** the result is recorded
- **THEN** it SHALL be scoped to that cell's demand, and SHALL NOT be reported as a general
  statement about the wiring

#### Scenario: The component carrying the deficit is separated

- **GIVEN** a substrate whose measured deficit on a multi-objective cell is concentrated in one
  component
- **WHEN** a wiring contrast on that cell returns a null
- **THEN** the null SHALL NOT be generalised beyond that cell until the contrast has been run on a
  cell without that component, and the committed verdict SHALL stand scoped to its cell

### Requirement: A wiring contrast is gated on the cell showing learning

Each arm of a wiring contrast SHALL be paired with its own floor on the same seeds — the same
substrate with the optimiser disabled — and the contrast SHALL NOT be read unless the arm carrying
the claim beats that floor by the registered test. Where it does not, the result SHALL be recorded
as a finding about the platform and SHALL license no conclusion about the wiring.

A ceiling threshold SHALL be registered before any data exists, and where both arms of a contrast
reach it the contrast SHALL be recorded as unresolvable on that cell, with the remedy named in
advance rather than chosen after the outcome is known.

#### Scenario: A contrast whose arm did not learn is not read

- **GIVEN** a wiring contrast whose claim-carrying arm does not beat its own frozen floor
- **WHEN** the family is scored
- **THEN** the cell's verdict SHALL be that no learning occurred, and the contrast SHALL NOT be
  assigned a wiring verdict

#### Scenario: A saturated cell is named, not re-tuned

- **GIVEN** both arms of a contrast at or above the registered ceiling threshold
- **WHEN** the family is scored
- **THEN** the cell SHALL be recorded as unresolvable and the registered remedy applied, and the
  recipe SHALL NOT be adjusted until the arms separate

#### Scenario: A minimum effect is registered beside significance

- **GIVEN** a paired contrast at a sample size where a rank test fires on the consistency of the
  sign rather than the size of the shift
- **WHEN** the primary is registered
- **THEN** a minimum effect SHALL be registered with it, and a significant result below that minimum
  SHALL be recorded as such and SHALL license nothing on its own
