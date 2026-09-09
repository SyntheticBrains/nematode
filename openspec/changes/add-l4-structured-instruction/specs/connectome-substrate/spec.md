## ADDED Requirements

### Requirement: Co-transmitter identities are read from every identity column of the atlas

The atlas reader SHALL read every neurotransmitter identity column the sheet carries, not the
first alone, and the classification table SHALL record for each neuron every **release** identity
the atlas assigns it, so that a neuron carrying more than one — ADF and HSN releasing serotonin
beside acetylcholine, RIM releasing tyramine beside glutamate — carries both. The primary identity
used for sign grounding SHALL be unchanged. Uptake-only annotations SHALL NOT yield a release
identity, and entries the atlas itself qualifies as an alternative synthesis/uptake mechanism, a
precursor, or a sex-specific note SHALL be excluded from release identities by a rule stated where
the reader is defined, with the excluded names listed.

#### Scenario: A co-transmitting neuron carries both identities

- **WHEN** the classification table is read for `ADFL`, `HSNL` or `RIML`
- **THEN** it SHALL carry the first-column identity the sign-grounding work used
- **AND** it SHALL also carry the serotonergic or tyraminergic release identity the atlas assigns

#### Scenario: Uptake and qualified entries yield no release identity

- **WHEN** the table is read for a neuron whose only aminergic annotation is uptake, an alternative
  mechanism, a precursor or a sex-specific note
- **THEN** it SHALL carry no aminergic release identity

#### Scenario: Sign grounding is unchanged

- **WHEN** synapse signs are derived from the extended table
- **THEN** every grounded sign SHALL equal the sign the single-column table produced

### Requirement: The instructive aminergic pathway is derived from the atlas and the wiring

The substrate SHALL derive, from the vendored neurotransmitter atlas and the connectome's chemical
edges, the set of neurons that receive chemical input from an **aminergic** neuron — one carrying a
dopamine, serotonin, octopamine or tyramine release identity in any of its identities — and
SHALL expose it as a per-synapse
boolean mask over the chemical weight matrix, true where the synapse's **post-synaptic** neuron is
in that set. The derivation SHALL be a pure function of the classification table and the loaded
connectome, so a rewired substrate derives its own pathway rather than inheriting the wild type's.

The substrate SHALL report the **instructed fraction** — the share of chemical synapses the mask
selects — so that a build routing everything or nothing is visible without inspecting the mask.

The derivation is a model of aminergic reach by **synaptic connectivity**. Aminergic transmission
in this animal is substantially extrasynaptic, so the wired reach is a lower bound and a modelling
choice; the implementation SHALL say so where the derivation is defined, and SHALL NOT describe
the mask as the set of neurons the amines act on.

#### Scenario: The pathway is derived from both sources

- **WHEN** the instructive pathway is derived
- **THEN** a neuron SHALL be in the instructed set exactly when the connectome carries a chemical
  edge to it from a neuron the classification table marks as releasing dopamine, serotonin,
  octopamine or tyramine, in any of that neuron's identities

#### Scenario: The mask keys on the post-synaptic neuron

- **WHEN** the per-synapse mask is built
- **THEN** an entry SHALL be true exactly when its post-synaptic neuron is in the instructed set,
  regardless of what its pre-synaptic neuron releases

#### Scenario: A rewired substrate derives its own pathway

- **GIVEN** a degree-preserving rewired connectome
- **WHEN** the pathway is derived
- **THEN** it SHALL be computed from that connectome's own edges

#### Scenario: The instructed fraction is reported

- **WHEN** a substrate derives the pathway
- **THEN** the share of chemical synapses selected SHALL be available without inspecting the mask
