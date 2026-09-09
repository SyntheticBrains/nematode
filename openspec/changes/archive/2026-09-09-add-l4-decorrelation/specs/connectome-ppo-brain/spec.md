## ADDED Requirements

### Requirement: Grounded signs are available to the rule without being enforced

When the connectome brain grounds its synapse signs, it SHALL supply the sign vector to the
plasticity rule whether or not Dale's law is enforced. Enforcement constrains where a weight may
go; a decorrelating variant reads the same identities to decide which way an update points, and
the two SHALL remain independently selectable and composable.

#### Scenario: A grounded brain hands its signs to the rule

- **GIVEN** a brain configured with atlas-grounded synapse signs and enforcement off
- **WHEN** the plastic rule is constructed
- **THEN** the rule SHALL hold the brain's sign vector

#### Scenario: An ungrounded brain hands the rule no signs

- **GIVEN** a brain whose synapse signs are drawn rather than grounded
- **WHEN** the plastic rule is constructed
- **THEN** the rule SHALL hold no sign vector
