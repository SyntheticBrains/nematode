## ADDED Requirements

### Requirement: A substrate result is read against the rule's positive control

A registered result about whether a substrate's wiring is legible to a local plasticity rule SHALL
be read against that rule's positive control. Where the control has not been run, or has not
passed, the record SHALL state that a null is consistent both with the wiring carrying no signal
and with the rule not learning, and SHALL NOT attribute it to the wiring alone.

The control's pass rule SHALL be fixed before it runs: a stated margin over the computed
cue-blind floor on a stated number of seeds, with the reference and floor arms deciding validity.

#### Scenario: A null is not attributed to the wiring while the instrument is unverified

- **GIVEN** a registered substrate result that does not confirm its primary test
- **WHEN** it is written up and the rule's positive control has not passed
- **THEN** the record SHALL state that the null is consistent with the rule not learning

#### Scenario: The pass rule precedes the run

- **WHEN** the control is launched
- **THEN** its margin, seed count and budget SHALL already be recorded
