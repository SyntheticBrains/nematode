## ADDED Requirements

### Requirement: An annealed perturbation clears the control before the assay

A variant that schedules its perturbation scale SHALL clear the rule's positive control under that
schedule before it is run through the clone assay, and the order SHALL be that one. The control run
SHALL use the same three validity arms, the same seeds and the same pass rule as the control the
constant-scale variant cleared, with the schedule compressed to the control's episode budget, and
SHALL report the update's alignment to the analytic policy gradient at both ends of the schedule
rather than only at its end.

A failure at the control SHALL stop the sequence, and SHALL be reported as a property of the
registered schedule rather than resolved by re-tuning its bounds or its length.

#### Scenario: The control gates the assay

- **GIVEN** a variant scheduling its perturbation scale
- **WHEN** it has not cleared the positive control under that schedule
- **THEN** its clone-assay result SHALL NOT be reported as evidence about retention

#### Scenario: The alignment is reported across the schedule

- **WHEN** an annealed arm's control run is recorded
- **THEN** the record SHALL carry the gradient alignment at the schedule's initial and final scales,
  so that a schedule which anneals away its own signal is distinguishable from one which does not

### Requirement: A scheduled arm's frozen control runs the same schedule

Where the clone assay screens an arm whose perturbation scale follows a schedule, the frozen control
required of a perturbing variant SHALL run that identical schedule with updates frozen. Its score
SHALL be read as a trajectory over the schedule rather than as a single endpoint, since a frozen arm
under a decaying scale recovers as the scale falls, and the learning arm SHALL be read against that
trajectory.

#### Scenario: The control anneals too

- **GIVEN** an annealed screening arm
- **WHEN** its frozen control is configured
- **THEN** the control SHALL carry the same initial scale, final scale and anneal length, and SHALL
  differ from the screening arm only in that updates are frozen

#### Scenario: The comparison is against the trajectory

- **WHEN** an annealed arm's assay result is reported
- **THEN** it SHALL be reported beside the frozen control's trajectory over the same schedule, and a
  claim that the rule damaged the policy SHALL require the learning arm to fall below that
  trajectory rather than below the committed comparator alone
