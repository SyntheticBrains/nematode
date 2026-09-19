## ADDED Requirements

### Requirement: A positive result carrying an inherited learner setting is re-read at a calibrated operating point before a synthesis cites it

Where a positive structure result was obtained under a learner setting inherited from a different
capacity or substrate rather than calibrated for the arms it ran on, and a later campaign shows the
setting moves the outcome, the record SHALL re-read the result's **registered primary** at the
calibrated setting before the result enters a phase synthesis, and SHALL carry the outcome as a
condition beside the committed verdict rather than as a rewrite of it.

#### Scenario: The re-read is the registered primary at one setting, not a new question

- **GIVEN** a committed positive whose primary was a crossed interaction
- **WHEN** it is re-read at the calibrated setting
- **THEN** the primary SHALL be the same interaction at that setting, with the cells already measured
  there reused only under the committed baseline-reuse requirement — one seed per reused arm re-run on
  the current path and compared on **every field the analysis parses** — and only the missing cells run
- **AND** the record SHALL state what that check establishes and what it does not: it establishes that
  the current path reproduces the committed run on every quantity any analysis in the programme reads,
  at the seed checked; it is **parsed-field identity, not byte equality** of logs, exports, weights or
  configuration, and it does not extend to the seeds left unchecked
- **AND** where a parsed field cannot be compared because the committed side no longer holds the
  artefact it derives from, that field SHALL be named as uncompared rather than counted as matching
- **AND** a minimum effect SHALL be registered as a fraction of the committed effect, with the
  reading that a significant result below it receives named before the runs, **and registered for
  both directions** where the reading is two-sided

#### Scenario: A headline form that is already known not to hold is stated before the runs

- **GIVEN** a committed positive with a registered primary and a more striking form it happened to
  take (a sign flip, a lead at one level)
- **WHEN** committed data already shows that form absent at the calibrated setting
- **THEN** the design SHALL say so before the runs, and SHALL read the registered primary
- **AND** a positive SHALL be reported as the weaker claim it is, never as the headline form
  reproduced

#### Scenario: The verdict is conditioned, not rewritten

- **GIVEN** any reading of the re-read
- **WHEN** the record is written
- **THEN** the committed verdict SHALL stand as read at its own setting
- **AND** the tracker, the original logbook and the roadmap SHALL each carry the condition as a dated
  note in the same place the verdict is cited
- **AND** the synthesis SHALL state the condition in the same sentence as the claim

#### Scenario: The re-read does not decide the setting and does not ablate against it

- **GIVEN** a re-read showing both arms learn better at the calibrated setting
- **WHEN** the record draws consequences
- **THEN** it SHALL NOT declare either setting correct, and SHALL leave the choice to the calibration
  of the rung that next runs there
- **AND** no ablation SHALL be read against the calibrated baseline inside the same campaign that
  establishes it
