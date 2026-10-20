## ADDED Requirements

### Requirement: A panel longer than the collapse hazard may read its verdict before the hazard

A registered panel whose runs extend past the episode at which late collapse begins MAY fix its verdict
on the primary metric computed over a window that ends before that episode. The window SHALL be fixed
in the registration, before any run, and SHALL NOT be moved after any data are seen. The gates SHALL be
evaluated on the same window as the verdict.

The remainder of the horizon SHALL be reported beside the verdict, never as a verdict. That includes the
primary metric over the whole horizon and over the remainder, and the collapse statistic for every arm.

#### Scenario: The verdict window is moved after the data

- **WHEN** a reading computes its verdict on a window other than the registered one
- **THEN** the reading SHALL be refused as a re-read

#### Scenario: The late window disagrees with the verdict

- **WHEN** the primary metric over the remainder of the horizon points elsewhere than the registered
  verdict
- **THEN** the verdict SHALL stand, and the late window SHALL be reported beside it with the collapse
  statistic for every arm

#### Scenario: The gates are read on another window

- **WHEN** a panel's gates are evaluated on a window other than the verdict's
- **THEN** the gate evidence SHALL be refused
