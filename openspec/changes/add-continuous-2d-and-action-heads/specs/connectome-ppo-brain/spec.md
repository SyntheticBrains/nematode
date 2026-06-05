## ADDED Requirements

### Requirement: Continuous-output adapter on the motor readout

The connectome-PPO brain SHALL provide a continuous-output adapter that maps the pooled motor-neuron activations to a 2-dimensional Gaussian head (`mean`, `log_std`) via the shared action-policy module when running on the continuous-2D substrate, while retaining the discrete categorical readout on the grid substrate.

#### Scenario: Continuous head from motor pool

- **WHEN** the connectome brain runs in the continuous-2D environment
- **THEN** the motor-neuron pool is projected to a 2-D Gaussian `(mean, log_std)` and a continuous action is sampled via the shared policy module

#### Scenario: Discrete readout retained on grid

- **WHEN** the connectome brain runs in the grid environment
- **THEN** the existing 4-way categorical motor readout is used unchanged

### Requirement: Strict-mask and gap junctions preserved under continuous output

The continuous-output adapter SHALL NOT alter the chemical-synapse strict-mask or the fixed gap-junction couplings. The strict-mask SHALL continue to pin non-existent chemical synapses to zero in the forward pass and after every optimiser step, independent of the output mode.

#### Scenario: Strict-mask invariant across output modes

- **WHEN** the connectome brain trains with the continuous-output adapter under strict-mask mode
- **THEN** non-existent chemical synapses remain zero after each optimiser step, identically to the discrete-output case

#### Scenario: Gap junctions remain fixed

- **WHEN** the connectome brain trains with the continuous-output adapter
- **THEN** the gap-junction couplings remain non-learnable and unchanged from their physiologically-informed initialisation
