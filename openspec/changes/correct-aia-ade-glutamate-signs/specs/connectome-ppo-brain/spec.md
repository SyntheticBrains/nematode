## ADDED Requirements

### Requirement: Sign overrides excluded by citation

`ConnectomePPOBrainConfig` SHALL accept `sign_override_exclude`, a list of citation keys, empty by
default. Under `synapse_signs: per_connection` the brain SHALL build its signs from the table with those
citations' overrides ignored. A key the table does not cite SHALL be refused when the config is
validated, and a non-empty list SHALL be refused unless `synapse_signs` is `per_connection`.

#### Scenario: Excluding the López-Cruz overrides reproduces the earlier brain

- **WHEN** a per-connection brain is built with `sign_override_exclude: [lopezcruz2019]`
- **THEN** its chemical weights SHALL be byte-identical to a brain built from the same config and seed
  before the López-Cruz overrides were added

#### Scenario: The encoded brain changes only on the corrected edges

- **WHEN** an encoded brain is built from the same config and parameters with and without the
  exclusion
- **THEN** its fixed normalised counts SHALL differ in exactly the 16 López-Cruz entries, each moving
  from the excitatory matrix to the inhibitory one at the same magnitude, so that it takes the
  inhibitory scale

#### Scenario: An unknown or misplaced key is refused

- **WHEN** `sign_override_exclude` names a key the table does not cite, or is set with `synapse_signs`
  other than `per_connection`
- **THEN** the config SHALL be refused, naming the key or the setting
