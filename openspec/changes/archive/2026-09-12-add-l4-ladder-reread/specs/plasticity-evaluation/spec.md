## ADDED Requirements

### Requirement: A re-read establishes what a body of results is evidence about

Where a body of registered results is re-read after the instrument that produced them has been
tested, the re-read SHALL classify each result by what would have had to be true for its null to be
informative, and SHALL NOT assume that "about the question" and "about the instrument" exhaust the
possibilities. Where a result's premise was never established — where the effect it sought has not
been demonstrated by any method, including one known to work — it SHALL be classified as
uninformative about both rather than attributed to the instrument.

The re-read SHALL first classify each result by the kind of question it asked, since the premise
of a wiring contrast (that learning finds an advantage) and the premise of a retention assay (that a
policy can be held) are different claims established by different evidence, and a result that
involved no learning at all has neither. A result with no learning SHALL survive as a finding about
the substrate. Within a kind, the order SHALL place a failed premise before a failed instrument,
since repairing an instrument would not have changed a null whose premise never held.

#### Scenario: A result whose premise no method supports is not blamed on the instrument

- **GIVEN** a contrast on which a method known to solve the task shows no effect in the sought
  direction
- **WHEN** a null on that contrast under a different method is re-read
- **THEN** it SHALL be classified as uninformative about both the question and the instrument, and
  the record SHALL state that a working instrument would not have changed it

#### Scenario: A no-learning result survives regardless of the instrument

- **GIVEN** a registered result that compared frozen substrates with no rule running
- **WHEN** it is re-read after the rule is found not to learn
- **THEN** it SHALL be classified as a finding about the substrate, unchanged

#### Scenario: A result measured under a different rule is not classified by the tested rule's premise

- **GIVEN** a registered contrast measured under a rule other than the one the positive control
  tested, and evidence that this other rule reaches competent behaviour on the task
- **WHEN** it is re-read
- **THEN** it SHALL be classified on its own rule's premise, which is met, and SHALL NOT be
  evaluated under the tested rule's premise or attributed to the tested rule's failure

#### Scenario: A retention assay whose premise was met is not filed as a premise failure

- **GIVEN** a retention assay on a substrate shown to hold a competent policy under frozen weights
- **WHEN** it is re-read
- **THEN** its premise SHALL be recorded as met, and its null SHALL be attributed to the rule

#### Scenario: The classification order is stated and applied

- **WHEN** a result satisfies both the failed-premise and failed-instrument conditions
- **THEN** it SHALL be classified by the premise, and the record SHALL state the order used

### Requirement: A re-read carries its committed verdicts and its own corrections

Each committed verdict SHALL be carried unchanged beside its re-read, in the units and under the
rule it was registered with. A re-read SHALL NOT convert a negative result into a positive one, and
SHALL NOT license work that the results themselves do not license.

Where the re-read corrects an earlier reading made during the programme, the correction SHALL be
recorded with what it changes, rather than the earlier reading being silently dropped.

#### Scenario: A committed verdict survives its re-read

- **GIVEN** a committed verdict whose re-read classifies it differently
- **WHEN** the record is written
- **THEN** both SHALL appear, the committed verdict SHALL be unchanged, and the record SHALL state
  which is the verdict

#### Scenario: A superseded reading is corrected rather than dropped

- **GIVEN** a reading made earlier in the programme that the re-read finds unsupported
- **WHEN** the record is written
- **THEN** it SHALL state the earlier reading, what the evidence actually shows, and what followed
  from the earlier reading that no longer holds
