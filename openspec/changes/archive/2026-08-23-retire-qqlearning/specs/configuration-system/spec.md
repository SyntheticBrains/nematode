## RENAMED Requirements

- FROM: `### Requirement: Brain Type Enumeration Extension`
- TO: `### Requirement: Brain Type Validation Against the Registry`

## MODIFIED Requirements

### Requirement: Brain Type Validation Against the Registry

The brain type validation SHALL accept exactly the brain types registered in the
brain plugin registry, and SHALL reject any other value with an error rather than
falling back to a default architecture.

The set of valid types is defined by the registry itself (`BrainType` /
`_REGISTRY`, kept in agreement by `assert_registry_matches_enum`), not by a list
maintained in this specification. Two previous enumerations here drifted from the
code — `qqlearning` outlived its retirement, and the legacy aliases "modular",
"qmodular", "mlp", "qmlp" and "ppo" were named here while existing in no
`BrainType` value and no alias mapping — so the specification now states the
contract and defers the membership to the single source of truth.

`qqlearning` is **not** a valid brain type. `QQLearningBrain` was retired after the
roadmap recorded it as "evaluated, not competitive; deprioritised"; a configuration
naming it fails at load.

#### Scenario: Brain Type Validation

- **GIVEN** a configuration specifying a brain type that the registry contains
- **WHEN** validation occurs
- **THEN** the configuration SHALL load and resolve to that architecture
- **AND** "spikingreinforce", "qvarcircuit", "mlpreinforce", "mlpppo" and "mlpdqn"
  remain valid, as registered members — stated here as examples, not as the
  authoritative list

#### Scenario: An unregistered brain type is rejected

- **GIVEN** a configuration specifying a brain type absent from the registry —
  whether a name that never existed, a retired architecture, or a legacy alias
- **WHEN** validation occurs
- **THEN** loading SHALL fail with an error naming the unknown type
- **AND** it SHALL NOT silently substitute a default or nearest-match architecture

#### Scenario: Registry membership changes without a configuration-schema change

- **GIVEN** a brain architecture is added to or removed from the registry
- **WHEN** a configuration naming a still-registered type is loaded
- **THEN** it SHALL continue to load unchanged
- **AND** a configuration naming the removed type SHALL be rejected per the
  scenario above, with no code change required in the configuration layer beyond
  the registration itself
