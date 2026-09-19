## ADDED Requirements

### Requirement: A phase close assigns every exit criterion a status from a fixed vocabulary

Where a phase closes, its synthesis SHALL assign exactly one status from a fixed vocabulary to every
criterion the phase lists, and SHALL NOT leave one unmarked. Required and recommended criteria are
**gates** on the close; optional criteria are not gates and SHALL be marked without being treated as
blocking. The vocabulary SHALL distinguish a criterion that is still a question from one that is not.

#### Scenario: Deferred and superseded are not interchangeable

- **GIVEN** an exit criterion that was not attempted
- **WHEN** the close assigns its status
- **THEN** it SHALL be recorded as **deferred** only where the question remains live and a destination
  is named, and as **superseded** only where a committed result removed the question, with that result
  named and the reason its gate no longer exists stated
- **AND** where nothing in the phase could have made the criterion attemptable, it SHALL be recorded
  as **unreachable** with what was missing named, rather than as deferred

#### Scenario: A criterion's status is argued, not asserted

- **GIVEN** a status of superseded or unreachable
- **WHEN** the close records it
- **THEN** the record SHALL cite the committed result or the missing precondition that licenses the
  status
- **AND** a status that rests on a judgement SHALL say so where it is made

#### Scenario: The roadmap and the tracker agree

- **GIVEN** a criterion whose status differs between the phase tracker and the roadmap
- **WHEN** the close is written
- **THEN** the disagreement SHALL be resolved in both places rather than in one
- **AND** the close SHALL NOT leave a criterion showing unresolved in one artefact and settled in
  another

### Requirement: A shipped result with an uncontrolled confound carries it as a standing condition

Where a result ships with a confound its own record registered but did not control, the confound SHALL
be carried as a **standing condition** stated in the same sentence as the claim at every citation
site, and the control SHALL be named in the successor phase's opening scope rather than left as an
open caveat.

#### Scenario: The condition travels with the claim

- **GIVEN** a shipped result whose record names an uncontrolled confound
- **WHEN** the result is cited in a synthesis, a tracker, a roadmap or a later record
- **THEN** the condition SHALL appear in the same sentence as the claim, not in a separate caveat
  section
- **AND** the citation SHALL NOT state the effect size without it

#### Scenario: An external precedent for the confound raises its standing

- **GIVEN** published work that applies the missing control to the same kind of claim
- **WHEN** the close assesses the confound
- **THEN** that work SHALL be named, and the confound SHALL be treated as load-bearing rather than
  residual
- **AND** the control SHALL be scheduled ahead of further results that would inherit the confound

#### Scenario: A control needing a design decision opens a phase rather than closing one

- **GIVEN** a missing control whose specification is itself ambiguous
- **WHEN** the close schedules it
- **THEN** the ambiguity SHALL be stated as the reason it is the successor phase's work
- **AND** the close SHALL NOT report the control as a small remaining task
