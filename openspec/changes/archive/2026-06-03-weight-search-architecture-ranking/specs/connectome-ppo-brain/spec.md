## ADDED Requirements

### Requirement: Connectome PPO Predator Sensor Projection

The `ConnectomePPOBrain` SHALL provide a learnable predator-sensor projection that routes the corrected two-channel predator-sensing biology (per the `predator-sensing-biology` capability) onto the canonical *C. elegans* mechanosensory + nociceptor neurons. This projection complements the food-chemotaxis projection shipped at T2 and lights up the connectome architecture for predator-evasion behaviours.

**Neuron-name convention.** The Cook 2019 hermaphrodite connectome (consumed at `packages/quantum-nematode/quantumnematode/connectome/neurons.py`) registers bilateral pairs as separate neurons with `L`/`R` suffixes. The canonical predator-circuit neurons are: `ASHL`, `ASHR` (polymodal nociceptors); `ASIL`, `ASIR` (distal sulfolipid sensors per Liu et al. 2018); `ALML`, `ALMR` (anterior touch receptors per Pirri & Alkema 2012); `AVM` (anterior touch receptor, **unilateral** — only one exists in the hermaphrodite); `PLML`, `PLMR` (posterior touch receptors).

The projection maps each feature additively onto BOTH members of a bilateral pair via the same learnable gain (the implementation MAY share a single column of the gain matrix across L/R targets, or use distinct columns initialised identically — see "Bilateral broadcast convention" scenario below). The gain matrix dimensions in the scenarios below count each L/R member as a separate target column.

The projection mapping:

- **Distal-chemosensation** (`predator_distal_concentration` + `predator_distal_dconcentration_dt`, 2 features) onto **ASHL, ASHR, ASIL, ASIR** (4 targets).
- **Anterior-zone contact-mechanosensation** (`predator_contact_intensity` when `predator_contact_zone == ContactZone.ANTERIOR`, plus `predator_mechano_dintensity_dt`, 2 features) onto **ALML, ALMR, AVM** (3 targets).
- **Posterior-zone contact-mechanosensation** (same fields when `predator_contact_zone == ContactZone.POSTERIOR`, 2 features) onto **PLML, PLMR** (2 targets).
- **Lateral-zone contact-mechanosensation** (same fields when `predator_contact_zone == ContactZone.LATERAL`, 2 features) degenerately onto **ALML, ALMR, PLML, PLMR at half-weight** (no canonical lateral-only mechanosensor; ALM + PLM together carry the lateral signal).

The projection SHALL be PPO-learnable separately from the food projection and separately from the chemical-synapse weight matrix.

#### Scenario: Bilateral broadcast convention

- **GIVEN** the projection's gain matrix for any of the four routing targets above
- **WHEN** the implementation constructs the gain matrix
- **THEN** the matrix SHALL inject the feature additively onto BOTH members of a bilateral pair (L + R) using the same learnable gain magnitude at initialisation
- **AND** the implementation MAY use either of two equivalent constructions: (a) one gain column per L/R member with identical initial values that diverge under PPO updates, OR (b) a single shared gain column broadcast across the L/R targets
- **AND** the chosen construction SHALL be documented inline in `ConnectomePPOBrain` so future readers know which to expect; both choices conform to this requirement

#### Scenario: Distal-chemosensation routes to ASHL + ASHR + ASIL + ASIR

- **WHEN** `run_brain()` receives a `BrainParams` with `predator_distal_concentration` and `predator_distal_dconcentration_dt` populated by an active predator-distal sensor module
- **THEN** these 2 features SHALL be additively injected onto the `ASHL`, `ASHR`, `ASIL`, `ASIR` sensory neurons' input vector via a learnable gain matrix of shape `(2, 4)` (2 distal features × 4 sensory neurons, bilateral pairs counted separately per the convention above)
- **AND** the gain matrix SHALL be PPO-learnable independently of the food-chemotaxis gain matrix and independently of the chemical-synapse weight matrix

#### Scenario: Anterior-zone contact routes to ALML + ALMR + AVM

- **WHEN** `run_brain()` receives a `BrainParams` with `predator_contact_intensity > 0` and `predator_contact_zone == ContactZone.ANTERIOR` (plus the derivative `predator_mechano_dintensity_dt`)
- **THEN** these 2 features (intensity + derivative) SHALL be additively injected onto the `ALML`, `ALMR`, `AVM` sensory neurons' input vector via a learnable gain matrix of shape `(2, 3)` (AVM is unilateral; ALM is bilateral)
- **AND** the gain matrix SHALL be PPO-learnable independently of the other projections

#### Scenario: Posterior-zone contact routes to PLML + PLMR

- **WHEN** `run_brain()` receives a `BrainParams` with `predator_contact_intensity > 0` and `predator_contact_zone == ContactZone.POSTERIOR`
- **THEN** the 2 features (intensity + derivative) SHALL be additively injected onto the `PLML`, `PLMR` sensory neurons' input vector via a learnable gain matrix of shape `(2, 2)`
- **AND** the gain matrix SHALL be PPO-learnable independently of the other projections

#### Scenario: Lateral-zone contact routes degenerately to ALM + PLM bilaterally at half-weight

- **WHEN** `run_brain()` receives a `BrainParams` with `predator_contact_intensity > 0` and `predator_contact_zone == ContactZone.LATERAL`
- **THEN** the 2 features (intensity + derivative) SHALL be additively injected onto `ALML`, `ALMR`, `PLML`, `PLMR` (4 targets), each at half the learnable gain that the anterior/posterior projections use
- **AND** the half-weight factor SHALL be a fixed constant (not learnable) so the projection's degenerate routing remains explicit
- **AND** the implementation SHALL reuse the existing anterior + posterior gain matrices scaled by 0.5 at injection time, rather than introducing a separate lateral-only gain matrix

#### Scenario: No predator inputs leaves the projection inactive

- **WHEN** `run_brain()` receives a `BrainParams` with `predator_contact_intensity == 0.0` AND `predator_distal_concentration == 0.0` (no predator-sensor modules active in the config)
- **THEN** the predator projection SHALL contribute zero additive input to `ASHL`, `ASHR`, `ASIL`, `ASIR`, `ALML`, `ALMR`, `AVM`, `PLML`, `PLMR`
- **AND** the brain's behaviour SHALL be functionally equivalent to a `ConnectomePPOBrain` instance whose config omits predator-sensor modules (any RNG-stream perturbation from `nn.Parameter` construction is allowed; the test bar SHALL be "last-25-mean klinotaxis success within ±3 percentage points of the R2b reference baseline at the equivalent seed," not byte-identical activations)

#### Scenario: ContactZone.NONE with active distal channel routes only distal, not mechano

- **WHEN** `run_brain()` receives a `BrainParams` with `predator_contact_intensity == 0.0` AND `predator_contact_zone == ContactZone.NONE` AND `predator_distal_concentration > 0.0` (the agent is in-config for both channels but currently outside the predator's damage radius)
- **THEN** the distal projection SHALL contribute non-zero additive input to `ASHL`, `ASHR`, `ASIL`, `ASIR` per the distal-chemosensation routing
- **AND** the anterior + posterior + lateral mechanosensation projections SHALL contribute zero additive input to `ALML`, `ALMR`, `AVM`, `PLML`, `PLMR` (the gain matrices receive zero `predator_contact_intensity` so the multiplicative product is zero)
- **AND** the brain SHALL NOT raise an exception when `predator_contact_zone == ContactZone.NONE` is the active zone (the routing logic SHALL treat `ContactZone.NONE` as "no mechano injection," consistent with the env-side semantics that NONE means "out of damage radius / no contact")

#### Scenario: Predator projection does not affect food-projection or chemical-synapse weights

- **GIVEN** a `ConnectomePPOBrain` config with both food-chemotaxis and predator sensor modules active
- **WHEN** PPO updates run for any number of steps
- **THEN** the food-projection gain matrix's training trajectory SHALL be independent of the predator-projection gain matrices' values (no shared parameter tensor; gradients flow through disjoint parameter sets)
- **AND** the chemical-synapse weight matrix's strict-mask invariant SHALL continue to hold (the predator projection adds input to sensory neurons but does not introduce new chemical-synapse edges)

### Requirement: Connectome PPO Thermotaxis Sensor Projection

The `ConnectomePPOBrain` SHALL provide a learnable thermotaxis-sensor projection that routes the klinotaxis temperature signal onto the canonical *C. elegans* thermosensory neurons. This complements the food-chemotaxis projection (shipped at T2) and the predator projection, and lights up the connectome architecture for thermotaxis behaviours — a prerequisite for the integrated food + predator + thermotaxis comparison cells.

**Neuron-name convention.** The canonical thermosensory neurons are `AFDL`, `AFDR` — the AFD bilateral pair, the dominant *C. elegans* thermosensor (Mori & Ohshima 1995; ~0.01°C sensitivity). Targeting AFD alone follows the same primary-role-only convention the food projection (ASE/AWC/AWA) and predator projection (ASH/ASI/ALM/PLM) use. Secondary thermosensory contributors (AWC per Kuhara et al. 2008; AWB; ASI) are deliberately NOT modelled — AWC's dual odor+temperature role is a polymodal-integration refinement deferred to later cellular-realism work.

The projection consumes the klinotaxis temperature feature triple `[temp_deviation, temperature_lateral_gradient, temperature_ddt]` (mirroring the env-side `thermotaxis_klinotaxis` sensory module), where `temp_deviation = clip((temperature − cultivation_temperature) / 15, −1, 1)`. The projection is opt-in via the `enable_thermotaxis_projection` config flag (default off so foraging-only configs construct byte-identical parameter sets to pre-projection builds), and is PPO-learnable separately from the food, predator, and chemical-synapse weight matrices.

#### Scenario: Thermotaxis features route to AFDL + AFDR

- **WHEN** `run_brain()` receives a `BrainParams` with `temperature` populated and `enable_thermotaxis_projection=True`
- **THEN** the 3 thermotaxis features SHALL be additively injected onto the `AFDL`, `AFDR` sensory neurons' input vector via a learnable gain matrix of shape `(3, 2)` (3 thermo features × 2 AFD targets, bilateral pair counted separately)
- **AND** the gain matrix SHALL be PPO-learnable independently of the food, predator, and chemical-synapse weight matrices

#### Scenario: Bilateral broadcast convention (thermotaxis)

- **GIVEN** the thermotaxis gain matrix
- **WHEN** the implementation constructs the matrix
- **THEN** it SHALL use one independent learnable column per L/R member (AFDL, AFDR) initialised to identical small-magnitude values that diverge under PPO updates — the same option-(a) construction the food and predator projections use

#### Scenario: No thermotaxis inputs leaves the projection inactive

- **WHEN** `run_brain()` receives a `BrainParams` with `temperature == None` (no thermal stimulus / isothermal-with-unset-temperature) under `enable_thermotaxis_projection=True`
- **THEN** the thermotaxis projection SHALL contribute zero additive input to `AFDL`, `AFDR` (the gain matrix receives a zero-filled feature vector so the product is zero)
- **AND** the brain SHALL NOT raise an exception

#### Scenario: Foraging-only configs preserve the pre-projection food path

- **WHEN** a `ConnectomePPOBrain` is constructed with `enable_thermotaxis_projection=False` (the default)
- **THEN** the topology SHALL allocate zero thermotaxis-related `nn.Parameter` objects
- **AND** the brain's behaviour SHALL be functionally equivalent to a build predating the thermotaxis projection (any RNG-stream perturbation is disallowed because no thermotaxis parameters are allocated; the foraging-only parameter set is byte-identical)

#### Scenario: Thermotaxis projection does not affect food/predator-projection or chemical-synapse weights

- **GIVEN** a `ConnectomePPOBrain` config with food, predator, and thermotaxis projections all active
- **WHEN** PPO updates run for any number of steps
- **THEN** the thermotaxis gain matrix's training trajectory SHALL be independent of the food + predator gain matrices' values (no shared parameter tensor; gradients flow through disjoint parameter sets)
- **AND** the chemical-synapse weight matrix's strict-mask invariant SHALL continue to hold (the thermotaxis projection adds input to AFD sensory neurons but does not introduce new chemical-synapse edges)
