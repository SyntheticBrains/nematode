## ADDED Requirements

### Requirement: The third factor may be routed through the instructive pathway

The three-factor rule SHALL offer a **third-factor routing mode** selected by a configuration value
shared through the plasticity configuration mixin, one of `global` or `pathway`, defaulting to
`global`. With `global` selected the rule SHALL be bit-identical to the rule without this
requirement.

With `pathway` selected, the rule SHALL apply the neuromodulatory factor **only at synapses the
substrate's instructive pathway marks**, and SHALL apply `1.0` in its place elsewhere — the
unmodulated Hebbian term the panel already registers as a floor. No other term SHALL change: the
weight decay, any decorrelating term, any consolidation term, the mask, Dale's-law projection, the
homeostatic rescale and the magnitude clamp SHALL keep their order and meaning, with the clamp
last.

Selecting `pathway` SHALL be rejected where the substrate exposes no instructive pathway, and
SHALL be rejected with the unmodulated rule, where the modulator is already `1.0` everywhere and
routing would be a no-op presented as a distinct arm. The rule SHALL receive the pathway as an
aligned per-tensor list of boolean masks at construction, as it receives synapse signs, and
SHALL NOT read a substrate's attributes to obtain it.

#### Scenario: The default path is unchanged

- **WHEN** the routing mode is `global`
- **THEN** the weight trajectory SHALL be bit-identical to the rule without routing

#### Scenario: Credit reaches only the synapses the wiring instructs

- **GIVEN** `pathway` selected and a step whose modulator differs from `1.0`
- **WHEN** the update is applied
- **THEN** the update at each instructed synapse SHALL equal the update the global rule would have
  applied there
- **AND** the update at each uninstructed synapse SHALL equal the update the **unmodulated** rule
  would have applied there

#### Scenario: Routing the unmodulated rule is refused

- **GIVEN** a configuration selecting `pathway` with the unmodulated learning rule
- **WHEN** the configuration is loaded
- **THEN** loading SHALL fail with a message saying the combination is the plain Hebbian floor

#### Scenario: Routing without a pathway is refused

- **GIVEN** a configuration selecting `pathway` on a substrate that exposes no instructive pathway
- **WHEN** the configuration is loaded
- **THEN** loading SHALL fail with a message saying the mode needs a derived pathway

### Requirement: Routing telemetry

The rule SHALL report the **instructed fraction** of the substrate and the share of the update's
absolute magnitude carried by instructed synapses, measured on the effective update, beside its
existing plasticity telemetry and recorded by the shared plasticity report. Under `global` the
share SHALL be reported as the whole. The share reports how the step's magnitude split between
the two sets; confinement of the modulated part to the instructed set is asserted by test, not by
this quantity. The fraction makes a pathway that turned out to cover nearly all or nearly none of
the substrate visible in the result.

#### Scenario: The instructed split is reported per step

- **WHEN** a run under `pathway` is read
- **THEN** the instructed fraction and the instructed share of the update SHALL be available per
  step
