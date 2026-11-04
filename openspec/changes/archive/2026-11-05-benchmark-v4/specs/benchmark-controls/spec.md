## ADDED Requirements

### Requirement: B v4 is valid only if real worms pass, the run-less worm fails, and activity is reachable

Benchmark v4 SHALL be VALID only if every registered control passes:

- the activity family's rule check;
- the track-import round trip;
- the reachability oracle on the pause share and the flip fraction;
- the fresh-N2 control;
- the adversary;
- the negatives.

Otherwise it SHALL be VOID, and v3 SHALL stay the default. C.1's v3 answer, PD1074's held-out cohorts,
W's minimum and the 3-of-5 probability SHALL be reported beside the verdict.

#### Scenario: The run-less worm is caught

- **WHEN** C.2's stage-2 answer is graded on 5 fresh populations
- **THEN** it SHALL fail at least one graded activity statistic in at least 3 of them, or v4 SHALL be
  VOID

#### Scenario: The onset rate is unreachable

- **WHEN** the oracle's fitted worm fails the onset rate's mean-z or 90% rule on its 40 populations,
  while passing the pause share and the flip fraction
- **THEN** the onset rate SHALL be reported, not graded, and v4 SHALL grade 23 statistics

#### Scenario: Activity is unreachable

- **WHEN** the oracle's fitted worm fails the pause share's or the flip fraction's rule
- **THEN** the read SHALL stop before any other control is read, and return to the maintainer

#### Scenario: v4 becomes the default

- **WHEN** `controls_v4.json` reads VALID with `bands_v4.json`'s SHA-256
- **THEN** `benchmark_grade.py` SHALL grade on v4 by default, and `--bands v3` SHALL reproduce every v3
  reading
