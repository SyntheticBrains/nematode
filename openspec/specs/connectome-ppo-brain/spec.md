# connectome-ppo-brain Specification

## Purpose

The `connectome-ppo-brain` capability provides a PPO-trainable brain architecture whose topology is the wild-type *C. elegans* Cook 2019 hermaphrodite connectome consumed from the [connectome-substrate](../connectome-substrate/spec.md) capability. Chemical synapses are subject to a strict-mask: only edges present in the wild-type adjacency carry PPO-learnable weights, and non-existent edges remain pinned to zero across every optimiser step. Gap junctions are non-learnable, with weights fixed to Cook 2019 synapse counts and symmetric fan-in normalised at construction time.

The brain registers through the [brain-architecture](../brain-architecture/spec.md) plugin registry as `connectomeppo` (`BrainType.CONNECTOMEPPO`, family `classical`) and is instantiable through the same `instantiate_brain(...)` code path as every other brain. Two `sensing_mode` variants ship: `oracle` consumes a 2-feature `[strength, angle]` food-chemotaxis vector, `klinotaxis` consumes the env-side klinotaxis sensory-module 3-feature emission `[concentration, lateral_gradient, dC/dt]`. Motor readout pools VB / DB / VA / DA motor-class activations and projects them to the 4-action `DEFAULT_ACTIONS` set via a learnable 4×4 matrix. Proprioception / mechanosensation / nociception projections are out of scope for this capability and live in downstream work.

The capability is the first closed-loop learning result on the real *C. elegans* connectome in this codebase, and the first non-trivial consumer of the [brain-architecture](../brain-architecture/spec.md) `BrainTopology` Protocol.

## Requirements

### Requirement: Connectome PPO Brain Architecture

The system SHALL provide a PPO-trainable brain whose topology is the *C. elegans* Cook 2019 hermaphrodite connectome consumed from the `quantumnematode.connectome` data model. Chemical synapses SHALL be subject to a strict-mask: only edges present in the wild-type adjacency are PPO-learnable; non-existent edges are pinned to zero. Gap junctions SHALL be non-learnable, with weights fixed to Cook 2019 synapse counts and fan-in normalised at construction time.

#### Scenario: Brain construction loads the connectome

- **WHEN** a brain config specifies `name: connectomeppo` with `connectome_source: "cook_2019_hermaphrodite"`
- **THEN** the brain SHALL call `load_cook_2019_hermaphrodite()` to obtain a `Connectome` instance
- **AND** the brain SHALL build its chemical-synapse weight tensor and strict-mask from `connectome.chemical_synapses`
- **AND** the brain SHALL build its gap-junction weight tensor from `connectome.gap_junctions`, symmetric across the two endpoints

#### Scenario: Chemical-synapse strict-mask enforces wild-type adjacency

- **GIVEN** a `ConnectomePPOBrain` constructed with `chemical_mask_mode: "strict"`
- **WHEN** the topology forward pass evaluates the chemical drive term
- **THEN** the forward SHALL use `W_chem * M_chem` so that backpropagation's chain rule pins gradients on positions where `M_chem[i, j] = False` to exactly zero
- **WHEN** the PPO learning rule completes any gradient step on the chemical-synapse weight tensor
- **THEN** the brain SHALL additionally apply `topology.apply_weight_mask(...)` to project the updated weights onto the strict-mask (defence-in-depth; combined with the forward-pass masking this guarantees the strict-mask invariant holds across every training step)
- **AND** all weight values along non-existent chemical-synapse edges (where `M_chem[i, j] = False`) SHALL be exactly zero at every step
- **AND** weight values along existing edges SHALL be the unprojected PPO update value

#### Scenario: Gap-junction weights remain fixed across PPO updates

- **GIVEN** a `ConnectomePPOBrain` with `enable_gap_junctions: true`
- **WHEN** the PPO learning rule completes any number of gradient steps
- **THEN** the gap-junction weight tensor `G_gap` SHALL be byte-identical to its construction-time value
- **AND** the gap-junction tensor's `requires_grad` attribute SHALL be `False`
- **AND** the symmetry `G_gap[a, b] == G_gap[b, a]` SHALL hold for every pair (a, b)

#### Scenario: Gap-junction fan-in normalisation at construction

- **WHEN** the brain constructs the gap-junction weight tensor from `connectome.gap_junctions`
- **THEN** each entry `G_gap[i, j]` SHALL be divided by `sqrt(max(1, d_i) * max(1, d_j))` where `d_i` is the gap-junction degree (number of non-zero entries) of neuron `i`
- **AND** the symmetric scaling SHALL preserve the symmetry `G_gap[a, b] == G_gap[b, a]` of the underlying bidirectional physics (a pure row-or-column scaling would break the symmetry constraint asserted by the "Gap-junction weights remain fixed across PPO updates" scenario above)
- **AND** the scaling SHALL be applied exactly once at construction time

#### Scenario: Sensor projection routes env input to canonical sensory neurons

- **WHEN** `run_brain()` receives a `BrainParams` with food-chemotaxis input
- **THEN** the food-chemotaxis feature vector SHALL be additively injected onto the ASEL, ASER, AWCL, AWCR, AWAL, AWAR sensory neurons' input vector via a learnable gain matrix of shape `(n_food_features, 6)`
- **AND** the gain matrix SHALL be PPO-learnable (separate from the chemical-synapse weight matrix)
- **AND** the value of `n_food_features` SHALL be determined by the `sensing_mode` config field: `2` in `oracle` mode (features `[strength, angle]`), `3` in `klinotaxis` mode (features `[concentration, lateral_gradient, dC/dt]` matching the env-side klinotaxis sensory-module emission shape)
- **AND** proprioception / mechanosensation / nociception sensor projections SHALL NOT be implemented at this revision — they are out of scope for the T2 connectome-PPO brain and ship in T3 (corrected ASH/ADL nociception) and T4 (sensor-projection ablation)

#### Scenario: Motor readout aggregates motor-class activations

- **WHEN** the connectome forward pass produces a 302-dim activation vector
- **THEN** the motor readout SHALL pool activations by motor-neuron class (VB, DB, VA, DA) via mean pooling, producing a 4-vector of class activations
- **AND** the class activations SHALL be projected to the 4 `DEFAULT_ACTIONS` ([FORWARD, LEFT, RIGHT, STAY]) via a learnable 4×4 readout matrix
- **AND** the readout matrix SHALL be PPO-learnable (separate from the chemical-synapse weight matrix)

#### Scenario: Forward-pass depth K is configurable

- **WHEN** the brain config specifies `forward_pass_depth: K` for an integer K ≥ 1
- **THEN** the forward pass SHALL iterate the connectome update `h = activation((W_chem * M_chem)ᵀ @ h + G_gapᵀ @ h)` exactly K times before the motor readout
- **AND** the default value of K SHALL be 4 to match the canonical klinotaxis pathway depth (sensory → primary-interneuron → command-interneuron → motor; K=1 produces a degenerate output because the food signal cannot reach motor neurons in one chemical-synapse hop)

### Requirement: ConnectomePPOBrainConfig

The brain SHALL be configured via a Pydantic `ConnectomePPOBrainConfig` model that exposes the connectome source, the gap-junction enablement flag, the chemical-mask mode, the forward-pass depth, the frozen-updates flag, and PPO hyperparameters mirroring `MLPPPOBrainConfig`.

#### Scenario: Required fields

- **WHEN** `ConnectomePPOBrainConfig` is constructed
- **THEN** the config SHALL accept the following fields:
  - `connectome_source: Literal["cook_2019_hermaphrodite"]`
  - `enable_gap_junctions: bool` (default `True`)
  - `chemical_mask_mode: Literal["strict", "soft_prior"]` (default `"strict"`)
  - `forward_pass_depth: int` (default `4`, must be ≥ 1 — see the "Forward-pass depth K is configurable" scenario above for the rationale)
  - `freeze_updates: bool` (default `False`, drives the Gate 1 G1.c paired control)
  - `sensing_mode: Literal["oracle", "klinotaxis"]` (default `"oracle"`) — controls the env-side feature shape consumed by the sensor projection. In `oracle` mode the brain reads `[food_gradient_strength, food_gradient_direction]` (2 features); in `klinotaxis` mode the brain reads `[food_concentration, food_lateral_gradient, food_dconcentration_dt]` (3 features) matching the env-side klinotaxis sensory-module emission shape. The learnable food-gain matrix is sized to match (`2 × 6` or `3 × 6`).
- **AND** the config SHALL accept the PPO hyperparameters used by `MLPPPOBrainConfig` (learning rate, clip range, value-loss coefficient, entropy coefficient, gradient-clip norm, batch size, etc.)

#### Scenario: Soft-prior mode allows new chemical edges to grow

- **GIVEN** a `ConnectomePPOBrainConfig` with `chemical_mask_mode: "soft_prior"`
- **WHEN** the topology forward pass evaluates the chemical drive term
- **THEN** the forward SHALL use the raw `W_chem` tensor (not `W_chem * M_chem`), so backpropagation produces non-zero gradients on every entry of `W_chem` — including positions where `M_chem[i, j] = False`
- **AND** the brain's update loop SHALL skip the post-optimiser-step strict-mask projection (the projection only runs under `chemical_mask_mode: "strict"`)
- **AND** new non-zero weights MAY therefore appear along edges where `M_chem[i, j] = False` as PPO optimises
- **AND** the initial weight tensor SHALL still be initialised from the wild-type chemical-synapse adjacency (so the prior persists at the start of training)

#### Scenario: Frozen-updates flag drives the Gate 1 G1.c paired control

- **GIVEN** a `ConnectomePPOBrainConfig` with `freeze_updates: true`
- **WHEN** the PPO learning rule's `step()` is invoked at any point during training
- **THEN** the call SHALL be a no-op (gradient computation skipped, optimiser update skipped)
- **AND** all PPO-learnable parameters (chemical-synapse weights, sensor gains, motor readout) SHALL remain byte-identical to their construction-time random initialisation across all episodes
- **AND** the brain SHALL still produce action samples via the forward pass using the frozen-random weights

### Requirement: Connectome PPO Brain Registration

The `ConnectomePPOBrain` SHALL be registered through the brain plugin registry (see `brain-architecture` capability) and SHALL be instantiable through the same `instantiate_brain(...)` code path as every other brain.

#### Scenario: ConnectomePPOBrain self-registers at import

- **WHEN** `quantumnematode.brain.arch.connectome_ppo` is imported
- **THEN** the registry SHALL contain a registration tuple for `("connectomeppo", ConnectomePPOBrainConfig, ConnectomePPOBrain, BrainType.CONNECTOMEPPO, ("classical",))`
- **AND** `list_registered_brains()` SHALL include `"connectomeppo"`

#### Scenario: Connectome brain instantiates through the same code path as MLPPPO

- **WHEN** `instantiate_brain("mlpppo", mlpppo_cfg)` and `instantiate_brain("connectomeppo", connectome_cfg)` are called from the same caller
- **THEN** both calls SHALL return Brain Protocol-conforming instances
- **AND** neither call SHALL require a brain-specific branch in the caller's code (the dispatch happens inside `instantiate_brain` via the registry only)

### Requirement: Connectome Brain Forward-Pass Numerical Invariants

The `ConnectomePPOBrain` SHALL produce finite, non-degenerate outputs across training and SHALL satisfy structural invariants on its weight tensors.

#### Scenario: Forward-pass output is finite

- **WHEN** `run_brain()` is called with any well-formed `BrainParams`
- **THEN** the returned action logits SHALL be finite (no NaN, no Inf)
- **AND** the per-step 302-dim activation vector SHALL be finite

#### Scenario: Forward-pass output has non-degenerate variance

- **WHEN** the brain is run for at least one episode on the klinotaxis smoke config
- **THEN** the variance across the 4-action logits over a sample of ≥ 100 forward passes SHALL be strictly greater than zero
- **AND** the logits SHALL not collapse to a constant action across that sample

#### Scenario: Strict-mask invariant holds after every training step

- **WHEN** the brain is queried via a test helper after any number of PPO updates in `chemical_mask_mode: "strict"`
- **THEN** the chemical-synapse weight tensor SHALL satisfy `(W_chem * ~M_chem).abs().max() == 0.0`
- **AND** the inverse condition (existing-edge weights are unconstrained by the mask) SHALL hold (i.e. at least one weight along an existing edge changes from its initialisation)

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

### Requirement: Degree-preserving rewired-null wiring option

`ConnectomePPOBrainConfig` SHALL expose a `wiring` selector with values `wild_type` (default) and
`rewired_degree_preserving`, plus a `rewire_seed` (integer, or unset to derive from the run seed).
When `wiring` is `rewired_degree_preserving`, the loaded `Connectome` SHALL be transformed **before**
the topology is constructed: its chemical-synapse edge set SHALL be replaced by a **directed**
degree-preserving edge-swapped set (each neuron's out-degree and in-degree preserved exactly) and its
gap-junction edge set by an **undirected** degree-preserving edge-swapped set (each neuron's gap degree
preserved exactly). The neuron set and ordering SHALL be unchanged, so per-post fan-in — and hence the
`w_chem` initialisation scale and the `g_gap` fan-in normalisation — are preserved; only *which*
neurons are connected changes. The transform SHALL be deterministic given `rewire_seed`, SHALL reject
self-loops and duplicate edges, and SHALL NOT silently reseed on a pathological draw.

When `wiring` is `wild_type` the transform SHALL be a no-op, leaving the built strict-mask, weight
initialisation, and gap-junction buffer byte-identical to the pre-change connectome brain.

#### Scenario: Rewiring preserves each neuron's in/out degree

- **WHEN** a connectome is rewired with `wiring: rewired_degree_preserving`
- **THEN** every neuron's chemical out-degree and in-degree, and its gap-junction degree, SHALL equal its wild-type values, while the connected pairs differ

#### Scenario: Rewiring produces a simple graph

- **WHEN** a connectome is rewired
- **THEN** the rewired chemical and gap-junction edge sets SHALL contain no self-loops and no duplicate edges

#### Scenario: Node set and ordering are preserved

- **WHEN** a connectome is rewired
- **THEN** the neuron set and its sorted ordering SHALL be identical to wild-type, so the strict-mask, `w_chem` init scale, and `g_gap` normalisation derive from the same per-neuron fan-in

#### Scenario: Rewiring is deterministic under the seed

- **WHEN** two connectomes are rewired with the same `rewire_seed`
- **THEN** their rewired edge sets SHALL be identical; different seeds SHALL (with overwhelming probability) differ

#### Scenario: Wild-type wiring is byte-identical

- **WHEN** the brain is built with `wiring: wild_type`
- **THEN** the strict-mask `m_chem`, the `w_chem` initialisation, and the `g_gap` buffer SHALL be identical to the pre-change connectome brain (no behaviour change to the existing ranking cell)
