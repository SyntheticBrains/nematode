## ADDED Requirements

### Requirement: Configurable initial action noise on the plastic brains

Every brain that offers the plasticity rules SHALL accept `initial_log_std` from the shared
plasticity configuration mixin, default `0.0`, and SHALL initialise its state-independent
continuous `log_std` parameter to that value at construction. Under the gradient rule the
parameter SHALL still train from that value; under a plastic rule it SHALL stay at that value, as
every non-plastic parameter does. With the default, construction SHALL be bit-identical to the
brain without this requirement.

#### Scenario: The configured value is the initial parameter on both brains

- **WHEN** a connectome brain and an MLP brain are built with `initial_log_std` set to a value
- **THEN** each brain's `log_std` SHALL equal that value at construction
- **AND** with the default the parameter SHALL be zero and the build bit-identical to today's

#### Scenario: Paired arms share the noise

- **GIVEN** the plastic wild-type and plastic rewired-null arms built at one seed with the same
  `initial_log_std`
- **WHEN** their action-noise parameters are compared
- **THEN** they SHALL be bit-identical

### Requirement: Configurable activation on the MLP brain

The MLP brain configuration SHALL accept `activation`, `relu` (default) or `tanh`, and SHALL
build its actor and critic hidden layers with the matching module and initialise their weights
with the matching orthogonal gain (`√2` for `relu`, `5/3` for `tanh`). With the default the
build SHALL be bit-identical to the brain without this requirement.

#### Scenario: Default is bit-identical

- **WHEN** an MLP brain is built with the default activation
- **THEN** its modules and initial weights SHALL be bit-identical to today's

#### Scenario: Tanh builds bounded units

- **WHEN** an MLP brain is built with `activation: tanh`
- **THEN** every hidden non-linearity SHALL be a Tanh module
- **AND** the hidden layers' weights SHALL be orthogonal with gain `5/3`
