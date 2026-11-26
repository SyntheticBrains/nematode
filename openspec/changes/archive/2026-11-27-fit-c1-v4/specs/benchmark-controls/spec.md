## MODIFIED Requirements

### Requirement: The new statistics are validated on unread data

The new statistics SHALL be validated on N2's held-out dates that no reading has touched, against N2 bands from N2's band dates, read once under a guard. PD1074's held-out cohorts SHALL NOT be read again by the benchmark's validation. The one exception SHALL be a registered controller fit's held-out reading under D37 as amended (2026-11-10), read as description in that fit's `read` stage under its committed launch record. A smoke or test of such a reading SHALL use band-cohort stand-ins.

#### Scenario: A held-out PD1074 cohort is requested

- **WHEN** any v2 stage would read a PD1074 held-out date
- **THEN** it SHALL refuse

#### Scenario: A registered fit reads the held-out dates once

- **WHEN** C.1's v4 refit reads its held-out reading in `read`, under launch record 136
- **THEN** it SHALL read N2's and PD1074's held-out dates as description, and no other stage of that fit SHALL load a held-out row
