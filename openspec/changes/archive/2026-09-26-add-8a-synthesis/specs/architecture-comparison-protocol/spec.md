## ADDED Requirements

### Requirement: A panel sized from a proxy reports its achieved sensitivity and never re-reads a verdict against it

Where a campaign's sensitivity is registered from a proxy — another campaign's per-seed spread, or a
pilot — the launch record SHALL name the proxy and the direction it is expected to err, if any, and
the logbook SHALL report the sensitivity the campaign itself achieved beside the registered figure. A
verdict SHALL be read against the registered minimum only; the achieved sensitivity SHALL NOT move a
reading from one state to another. A panel that turned out sharper or blunter than planned is
reported as such, and its verdict stands as registered.

#### Scenario: The proxy and its expected error are named in advance

- **GIVEN** a campaign whose minimum detectable effect is computed from committed data other than its
  own
- **WHEN** it is registered
- **THEN** the launch record SHALL name the source of the spread and state whether it is expected to
  overstate, understate or neither, with the reason

#### Scenario: The achieved sensitivity is reported

- **GIVEN** a campaign that has read out
- **WHEN** its logbook is written
- **THEN** the sensitivity computed from the campaign's own per-seed spread SHALL be reported beside
  the registered one

#### Scenario: A verdict is not re-read against the achieved sensitivity

- **GIVEN** a reading classified against the registered minimum
- **WHEN** the achieved sensitivity differs from the registered one
- **THEN** the reading SHALL stand as classified
- **AND** the difference SHALL be reported as a fact about the panel, not as a reason to change the
  verdict
