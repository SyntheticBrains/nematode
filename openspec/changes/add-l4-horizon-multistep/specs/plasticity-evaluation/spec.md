## ADDED Requirements

### Requirement: A setting found limiting on a control is tested where it was found to matter

Where a control establishes that a pinned setting limits the rule, that finding SHALL be tested on
a task of the kind whose failure motivated it before it is carried into a synthesis as an
explanation. The test SHALL run on the cheapest platform that poses the question, and a platform
whose runs are orders more costly SHALL NOT be used until the cheaper one has shown the setting
moves anything.

Where every existing result ran at the setting's default, the default SHALL be one of the arms, so
that the comparison includes the condition the record was made under.

#### Scenario: The cheap platform goes first

- **GIVEN** two platforms that pose the question, differing by orders of magnitude in cost per run
- **WHEN** the setting is tested
- **THEN** the cheaper platform SHALL be run first, and the costlier one SHALL be registered only if
  the cheaper one shows the setting moves the outcome

#### Scenario: The pinned value is an arm

- **WHEN** a setting whose default every committed result used is examined
- **THEN** that default SHALL be one of the arms

### Requirement: An arm at the metric's floor is scored on the graded reading

Where the primary metric is at its floor for both arms of a comparison, the comparison SHALL be
made on the graded reading rather than reported as no difference, since a metric that is zero for
both cannot separate them. The competence-dependent contrasts SHALL be reported as undefined where
no seed reaches competence, and SHALL NOT contribute a result.

#### Scenario: A floored primary metric does not decide the comparison

- **GIVEN** two arms whose full-clear rates are both at the floor, with no seed competent
- **WHEN** the comparison is made
- **THEN** it SHALL be made on the graded reading, and the competence-dependent contrasts SHALL be
  reported as undefined

### Requirement: A perturbing arm is compared with a control at its own setting

Where a rule's setting is varied and the rule perturbs the policy it runs, each setting SHALL have
its own frozen control at that same setting, and the learning arm SHALL be compared with it. A
single control at one setting SHALL NOT serve every arm, since what the perturbation costs a policy
need not be constant across the setting being varied.

#### Scenario: Each setting carries its own control

- **GIVEN** a grid over a rule setting, with a perturbing learning arm
- **WHEN** the arms are registered
- **THEN** each cell SHALL have a frozen control at that cell's setting, and the learning arm SHALL
  be scored against it rather than against a control from another cell

#### Scenario: A committed table from another rule is not the comparator

- **GIVEN** committed values for the same platform recorded under a different learning rule
- **WHEN** the comparison is made
- **THEN** those values MAY be reported as a descriptive reference and SHALL NOT be the comparator
