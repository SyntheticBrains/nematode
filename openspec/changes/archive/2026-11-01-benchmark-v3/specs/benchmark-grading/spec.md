## ADDED Requirements

### Requirement: Benchmark v3 grades reorientation from positions

Benchmark v3 SHALL grade 21 statistics at z\* = 3.0381:

- v2's small lawn, uniform well and kinematics, unchanged;
- local search's γ and its **log** early/late ratio;
- chemotaxis's drift, log quintile ratio, weathervane slope, **onset rate per minute** and **direction-flip fraction**.

The direction-flip fraction SHALL be the share of consecutive 1-s displacement pairs, both at least 0.03 mm, whose directions differ by more than 135°. Every new statistic SHALL be read the same way on real tracks and on simulated tracks after the same measurement model. v1 and v2 SHALL be unchanged.

#### Scenario: A flat search curve fails

- **WHEN** a population's early/late ratio is near 1 while the real worms' log ratio is about 1.39
- **THEN** its log early/late z SHALL exceed the bar

#### Scenario: Frequent direction flips fail

- **WHEN** a population's 1-s steps reverse direction several times as often as Chen's real recordings
- **THEN** its direction-flip z SHALL exceed the bar

#### Scenario: v2 is unchanged

- **WHEN** any population is graded on v2 after v3 is added
- **THEN** every v2 statistic, D and presence reading SHALL be byte-identical to before

### Requirement: The share of time reversing is reported, never graded

Where a population's steps carry primitive labels, the grade SHALL report the share of seconds in a reversal bout over the local-search and chemotaxis windows, beside Roberts' off-food tracker reference. It SHALL take no part in any pass, z or distance.

#### Scenario: An imported track reports no share

- **WHEN** imported positions are graded on v3
- **THEN** no reversal share SHALL be reported, and the grade SHALL be otherwise complete
