## ADDED Requirements

### Requirement: Atlas-grounded chemical synapse signs

The connectome brain configuration SHALL accept `synapse_signs`, `random` (default) or `atlas`.
Under `random` the chemical weights SHALL be initialised exactly as before. Under `atlas` the same
draws SHALL be taken in the same order from the same generator and each weight SHALL become the
magnitude of its draw carrying the sign the substrate's transmitter table gives its pre-synaptic
neuron; a weight whose pre-synaptic neuron has no derived sign SHALL keep the sign it drew. Weight
magnitudes, the per-neuron initialisation scale, the RNG stream and every other parameter SHALL be
unchanged by the setting, so two brains differing only in it differ only in sign structure. With
the default, construction SHALL be bit-identical to the brain without this requirement.

#### Scenario: Grounding changes signs and nothing else

- **GIVEN** two brains from the same configuration and seed, one `random` and one `atlas`
- **WHEN** their chemical weights are compared
- **THEN** the absolute values SHALL be equal entry for entry
- **AND** every weight whose pre-synaptic neuron has a derived sign SHALL carry that sign
- **AND** every weight whose pre-synaptic neuron has none SHALL be identical between the two

#### Scenario: The default is bit-identical

- **WHEN** a connectome brain is built with the default `synapse_signs`
- **THEN** its chemical weights SHALL be bit-identical to today's at the same seed

### Requirement: Dale's law enforcement during plasticity

The plasticity configuration SHALL accept `enforce_synapse_signs`, default false. When true, each
plastic update SHALL be projected back onto its synapse's derived sign — a positive synapse
floored at zero, a negative synapse ceilinged at zero, a synapse without a derived sign
unconstrained — after the decay term and before the magnitude clamp, so that the homeostatic
rescale acts on projected weights. Enabling it without atlas-grounded signs SHALL be refused at
construction. With it false, updates SHALL be bit-identical to the rule without this requirement.

#### Scenario: No grounded sign is ever violated

- **GIVEN** a plastic brain with grounded signs and enforcement on
- **WHEN** it runs many updates
- **THEN** every synapse with a derived sign SHALL still carry that sign, after homeostasis
- **AND** synapses without a derived sign MAY have changed sign

#### Scenario: Enforcement requires grounding

- **WHEN** a brain is configured with enforcement on and `synapse_signs: random`
- **THEN** construction SHALL raise

#### Scenario: Off is unchanged

- **WHEN** enforcement is off
- **THEN** the rule's updates SHALL be bit-identical to the rule without this requirement
