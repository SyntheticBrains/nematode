## ADDED Requirements

### Requirement: An annealed perturbation clears the control before the assay

A variant that schedules its perturbation scale SHALL clear the rule's positive control under that
schedule before it is run through the clone assay, and the order SHALL be that one. The control run
SHALL use the same three validity arms, the same seeds and the same pass rule as the control the
constant-scale variant cleared, with the schedule compressed to the control's episode budget, and
SHALL report the update's alignment to the analytic policy gradient separately over the decay
and over the floor rather than as one mean. The bounds and length SHALL be the registered ones:
initial 0.2, final 0.02, decay over the first half of the budget.

A failure at the control SHALL stop the sequence, and SHALL be reported as a property of the
registered schedule rather than resolved by re-tuning its bounds or its length.

#### Scenario: The control gates the assay

- **GIVEN** a variant scheduling its perturbation scale
- **WHEN** it has not cleared the positive control under that schedule
- **THEN** its clone-assay result SHALL NOT be reported as evidence about retention

#### Scenario: The alignment is reported across the schedule

- **WHEN** an annealed arm's control run is recorded
- **THEN** the record SHALL carry the gradient alignment over the decay and over the floor
  separately, and the score over the floor, so that a schedule which anneals away its own signal —
  a decay-phase alignment that does not rise and a floor-phase score below the bar — is
  distinguishable from one whose floor-phase alignment is low only because the estimator is nearly
  silent there by construction

### Requirement: A scheduled arm's frozen control runs the same schedule

Where the clone assay screens an arm whose perturbation scale follows a schedule, the frozen control
required of a perturbing variant SHALL run that identical schedule with updates frozen. Its score
SHALL be read as a trajectory over the schedule rather than as a single endpoint, since a frozen
arm under a decaying scale recovers as the scale falls. Both arms' curves SHALL be binned in eight
equal parts of the budget with the scheduled scale stated per bin, and the learning arm SHALL be
read against the frozen arm bin by bin.

#### Scenario: The control anneals too

- **GIVEN** an annealed screening arm
- **WHEN** its frozen control is configured
- **THEN** the control SHALL carry the same initial scale, final scale and anneal length, and SHALL
  differ from the screening arm only in that updates are frozen

#### Scenario: The comparison is against the trajectory

- **WHEN** an annealed arm's assay result is reported
- **THEN** it SHALL be reported beside the frozen control's binned trajectory over the same
  schedule, and a claim that the rule damaged the policy SHALL require the learning arm to fall
  below the frozen arm in the floor-phase bins rather than below the committed comparator alone
