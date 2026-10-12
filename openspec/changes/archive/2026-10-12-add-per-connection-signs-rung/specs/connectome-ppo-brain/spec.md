## ADDED Requirements

### Requirement: Per-connection synapse signs

The connectome PPO brain SHALL offer `synapse_signs: per_connection`, signing each chemical synapse
from the vendored per-connection sign table while keeping its drawn magnitude, and leaving a synapse
the table gives no sign with its drawn value. On a rewired null, each edge SHALL take the sign of the
wild-type edge whose place it took in the rewiring. Every check and path that accepts the atlas sign
model SHALL accept the per-connection model, including the sign hold under PPO. With
`synapse_signs` at `random` or `atlas`, the brain SHALL be byte-identical to before.

#### Scenario: A measured inhibitory connection is inhibitory

- **WHEN** the wild-type connectome is built with per-connection signs
- **THEN** AWC → AIY SHALL carry a negative sign, where the atlas model gives it a positive one

#### Scenario: Magnitudes match the random-sign build

- **WHEN** a brain is built with per-connection signs and another with random signs at the same seed
- **THEN** every chemical weight SHALL have the same magnitude in both

#### Scenario: The models differ only where the table corrects the rule

- **WHEN** the wild type is built with per-connection signs and with atlas signs at one seed
- **THEN** the signs SHALL differ on exactly the connections whose per-connection sign is opposite
  to the per-neuron rule's

#### Scenario: A null's signs travel with its edges

- **WHEN** a rewired null is built with per-connection signs
- **THEN** each neuron's multiset of outgoing signs SHALL equal the wild type's

#### Scenario: The atlas model is unchanged on a null

- **WHEN** a rewired null is built with atlas signs
- **THEN** its sign buffer SHALL be identical to the one built before this change

#### Scenario: PPO holds per-connection signs

- **WHEN** a per-connection-signed connectome with enforcement trains under PPO
- **THEN** no grounded synapse's weight SHALL cross zero

### Requirement: Chemical weight scale

The connectome PPO brain SHALL accept `chemical_weight_scale`, a positive multiplier applied to every
chemical weight once drawn, signed and placed, so a grounded sign model can be run at the same
operating point as random signs. At its default of 1.0 the brain SHALL be byte-identical to before.

#### Scenario: Unit scale is the unscaled build

- **WHEN** a brain is built with `chemical_weight_scale` 1.0
- **THEN** its chemical weights SHALL equal those of a brain built without the setting

#### Scenario: A scale multiplies every weight and keeps the signs

- **WHEN** a brain is built with `chemical_weight_scale` 0.42
- **THEN** every chemical weight SHALL be 0.42 times the unscaled build's, and every synapse SHALL
  keep its sign
