## ADDED Requirements

### Requirement: Connectome PPO Weight Persistence

`ConnectomePPOBrain` SHALL implement the `WeightPersistence` protocol. `get_weight_components`
SHALL return a `topology` component holding the topology module's complete state, including the
chemical mask and gap-junction buffers, a `training_state` component recording the episode
count, `continuous_std_mode`, `learning_rule`, `wiring`, `weight_init` and `connectome_source`,
and, only while the PPO rule is live, `value` and `optimizer` components. `load_weight_components`
SHALL validate the std mode and the wiring (the saved chemical mask and gap-junction buffers
equal the receiving brain's) before mutating any state and SHALL raise on a mismatch; it SHALL
then load `topology`, load `value` and `optimizer` only when present and the PPO rule is live,
reset the rollout buffer, and under the plastic rule reset the rule's running state. A save
followed by a load into a brain built from the same configuration and seed SHALL reproduce every
parameter bit for bit. A brain that never saves or loads SHALL be bit-identical to the brain
without this requirement.

#### Scenario: Round trip is bit-identical

- **GIVEN** a connectome brain under either rule after some episodes
- **WHEN** its weights are saved and loaded into a fresh brain from the same configuration and seed
- **THEN** every parameter and wiring buffer SHALL be bit-identical between the two brains

#### Scenario: A wiring mismatch is refused before mutation

- **GIVEN** weights saved from a wild-type brain
- **WHEN** they are loaded into a rewired-null brain, or the reverse
- **THEN** loading SHALL raise and the receiving brain's parameters SHALL be unchanged

#### Scenario: PPO components follow the rule

- **WHEN** components are requested from a brain under the plastic rule
- **THEN** `value` and `optimizer` SHALL be absent
- **AND** a plastic-rule brain given a file that carries them SHALL ignore them
- **AND** a PPO brain given a file without them SHALL keep its fresh critic and optimiser

#### Scenario: Loading resets transient state

- **WHEN** weights are loaded
- **THEN** the rollout buffer SHALL be empty and, under the plastic rule, the rule's baseline,
  scales and traces SHALL be at their initial values
