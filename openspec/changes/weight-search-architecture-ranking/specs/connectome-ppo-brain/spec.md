## ADDED Requirements

### Requirement: Connectome PPO Predator Sensor Projection

The `ConnectomePPOBrain` SHALL provide a learnable predator-sensor projection that routes the corrected two-channel predator-sensing biology (per the `predator-sensing-biology` capability) onto the canonical *C. elegans* mechanosensory + nociceptor neurons. This projection complements the food-chemotaxis projection shipped at T2 and lights up the connectome architecture for predator-evasion behaviours.

The projection SHALL map:

- **Distal-chemosensation** (`predator_distal_concentration` + `predator_distal_dconcentration_dt`) onto **ASH + ASI** sensory neurons. ASH is the canonical polymodal nociceptor (Hilliard et al. 2005); ASI carries the Liu et al. 2018 sulfolipid distal-chemosensory signal.
- **Anterior-zone contact-mechanosensation** (`predator_contact_intensity` when `predator_contact_zone == ANTERIOR`, plus `predator_mechano_dintensity_dt`) onto **ALM + AVM** sensory neurons. ALM and AVM are the canonical anterior touch receptors (Pirri & Alkema 2012).
- **Posterior-zone contact-mechanosensation** (same fields when `predator_contact_zone == POSTERIOR`) onto **PLM** sensory neurons. PLM is the canonical posterior touch receptor.
- **Lateral-zone contact-mechanosensation** (same fields when `predator_contact_zone == LATERAL`) degenerately onto **ALM + PLM at half-weight** (no canonical lateral-only mechanosensor; ALM+PLM together carry the lateral signal).

The projection SHALL be PPO-learnable separately from the food projection and separately from the chemical-synapse weight matrix.

#### Scenario: Distal-chemosensation routes to ASH + ASI

- **WHEN** `run_brain()` receives a `BrainParams` with `predator_distal_concentration` and `predator_distal_dconcentration_dt` populated by an active predator-distal sensor module
- **THEN** these features SHALL be additively injected onto the ASH and ASI sensory neurons' input vector via a learnable gain matrix of shape `(2, 2)` (2 distal features × 2 sensory neurons)
- **AND** the gain matrix SHALL be PPO-learnable independently of the food-chemotaxis gain matrix and independently of the chemical-synapse weight matrix

#### Scenario: Anterior-zone contact routes to ALM + AVM

- **WHEN** `run_brain()` receives a `BrainParams` with `predator_contact_intensity > 0` and `predator_contact_zone == ContactZone.ANTERIOR` (plus the derivative `predator_mechano_dintensity_dt`)
- **THEN** these features (intensity + derivative) SHALL be additively injected onto the ALM and AVM sensory neurons' input vector via a learnable gain matrix of shape `(2, 2)`
- **AND** the gain matrix SHALL be PPO-learnable independently of the other projections

#### Scenario: Posterior-zone contact routes to PLM

- **WHEN** `run_brain()` receives a `BrainParams` with `predator_contact_intensity > 0` and `predator_contact_zone == ContactZone.POSTERIOR`
- **THEN** the intensity + derivative features SHALL be additively injected onto the PLM sensory neuron's input scalar via a learnable gain matrix of shape `(2, 1)`
- **AND** the gain matrix SHALL be PPO-learnable independently of the other projections

#### Scenario: Lateral-zone contact routes degenerately to ALM + PLM at half-weight

- **WHEN** `run_brain()` receives a `BrainParams` with `predator_contact_intensity > 0` and `predator_contact_zone == ContactZone.LATERAL`
- **THEN** the intensity + derivative features SHALL be additively injected onto BOTH the ALM and PLM sensory neurons' input vectors, each at half the learnable gain that the anterior/posterior projections use
- **AND** the half-weight factor SHALL be a fixed constant (not learnable) so the projection's degenerate routing remains explicit

#### Scenario: No predator inputs leaves the projection inactive

- **WHEN** `run_brain()` receives a `BrainParams` with `predator_contact_intensity == 0.0` AND `predator_distal_concentration == 0.0` (no predator-sensor modules active in the config)
- **THEN** the predator projection SHALL contribute zero additive input to ASH, ASI, ALM, AVM, PLM
- **AND** the brain's behaviour SHALL be byte-identical to a `ConnectomePPOBrain` instance whose config omits predator-sensor modules (the projection is inert when no predator signal exists)

#### Scenario: Predator projection does not affect food-projection or chemical-synapse weights

- **GIVEN** a `ConnectomePPOBrain` config with both food-chemotaxis and predator sensor modules active
- **WHEN** PPO updates run for any number of steps
- **THEN** the food-projection gain matrix's training trajectory SHALL be independent of the predator-projection gain matrices' values (no shared parameter tensor; gradients flow through disjoint parameter sets)
- **AND** the chemical-synapse weight matrix's strict-mask invariant SHALL continue to hold (the predator projection adds input to sensory neurons but does not introduce new chemical-synapse edges)
