## ADDED Requirements

### Requirement: A perturbing rule's endpoint is evaluated with the perturbation off

Where a plastic rule perturbs its units during training, its endpoint weights SHALL be evaluated
with updates frozen and the perturbation removed, under the clone assay's comparator protocol —
the comparator's own configuration with only the weights changed — so that what the rule learned is
measured apart from the exploration noise it learned under. The evaluation SHALL use the assay's
registered pass rule and comparator unchanged, and SHALL be reported beside the under-perturbation
score from the assay as a descriptive annotation.

The outcomes and what each licenses SHALL be recorded before the evaluation runs, and the
registration that follows SHALL be the one the outcome selects.

#### Scenario: The endpoint is the registered arm's

- **GIVEN** a perturbing arm the clone assay has scored
- **WHEN** its endpoint is staged for evaluation
- **THEN** the staged weights SHALL be that arm's auto-saved final weights, per seed, with the
  source of each recorded, and SHALL NOT be re-trained or selected

#### Scenario: The evaluation runs the comparator's condition

- **WHEN** the endpoint is evaluated
- **THEN** updates SHALL be frozen, the perturbation scale SHALL be zero, and every other key SHALL
  equal the comparator's, so that the configuration differs from the comparator's in the weights
  alone

#### Scenario: The rule is the assay's

- **WHEN** the endpoint's scores are assessed
- **THEN** the pass rule, comparator values, budget and metric SHALL be those registered for the
  clone assay, and the verdict SHALL be one of the assay's own

#### Scenario: The outcome selects the next registration

- **GIVEN** the outcomes and their consequences recorded before the run
- **WHEN** the verdict is known
- **THEN** the change authored next SHALL be the one that verdict licenses, and the other SHALL NOT
  be authored on the same evidence
