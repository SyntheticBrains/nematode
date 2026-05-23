# connectome-ppo-brain Specification Delta

## ADDED Requirements

### Requirement: Connectome PPO Brain Architecture

The system SHALL provide a PPO-trainable brain whose topology is the *C. elegans* Cook 2019 hermaphrodite connectome consumed from the `quantumnematode.connectome` data model. Chemical synapses SHALL be subject to a strict-mask: only edges present in the wild-type adjacency are PPO-learnable; non-existent edges are pinned to zero. Gap junctions SHALL be non-learnable, with weights fixed to Cook 2019 synapse counts and fan-in normalised at construction time.

#### Scenario: Brain construction loads the connectome

- **WHEN** a brain config specifies `name: connectomeppo` with `connectome_source: "cook_2019_hermaphrodite"`
- **THEN** the brain SHALL call `load_cook_2019_hermaphrodite()` to obtain a `Connectome` instance
- **AND** the brain SHALL build its chemical-synapse weight tensor and strict-mask from `connectome.chemical_synapses`
- **AND** the brain SHALL build its gap-junction weight tensor from `connectome.gap_junctions`, symmetric across the two endpoints

#### Scenario: Chemical-synapse strict-mask enforces wild-type adjacency

- **GIVEN** a `ConnectomePPOBrain` constructed with `chemical_mask_mode: "strict"`
- **WHEN** the PPO learning rule completes any gradient step on the chemical-synapse weight tensor
- **THEN** the brain SHALL apply `topology.apply_weight_mask(...)` to project the updated weights onto the strict-mask
- **AND** all weight values along non-existent chemical-synapse edges (where `M_chem[i, j] = False`) SHALL be exactly zero
- **AND** weight values along existing edges SHALL be the unprojected PPO update value

#### Scenario: Gap-junction weights remain fixed across PPO updates

- **GIVEN** a `ConnectomePPOBrain` with `enable_gap_junctions: true`
- **WHEN** the PPO learning rule completes any number of gradient steps
- **THEN** the gap-junction weight tensor `G_gap` SHALL be byte-identical to its construction-time value
- **AND** the gap-junction tensor's `requires_grad` attribute SHALL be `False`
- **AND** the symmetry `G_gap[a, b] == G_gap[b, a]` SHALL hold for every pair (a, b)

#### Scenario: Gap-junction fan-in normalisation at construction

- **WHEN** the brain constructs the gap-junction weight tensor from `connectome.gap_junctions`
- **THEN** each row `G_gap[i, :]` SHALL be scaled by `1 / max(1, sum(G_gap[i, :]))` so per-neuron total gap-junction input is bounded
- **AND** the scaling SHALL be applied exactly once at construction time

#### Scenario: Sensor projection routes env input to canonical sensory neurons

- **WHEN** `run_brain()` receives a `BrainParams` with food-chemotaxis input
- **THEN** the food-chemotaxis signal SHALL be additively injected onto the ASE-left, ASE-right, AWC-left, AWC-right, and AWA sensory neurons' input vector
- **AND** the injection SHALL be scaled by a per-input learnable gain (PPO-learnable scalar parameter, separate from the chemical-synapse weight matrix)

#### Scenario: Motor readout aggregates motor-class activations

- **WHEN** the connectome forward pass produces a 302-dim activation vector
- **THEN** the motor readout SHALL pool activations by motor-neuron class (VB, DB, VA, DA) via mean pooling, producing a 4-vector of class activations
- **AND** the class activations SHALL be projected to the 4 `DEFAULT_ACTIONS` ([FORWARD, LEFT, RIGHT, STAY]) via a learnable 4×4 readout matrix
- **AND** the readout matrix SHALL be PPO-learnable (separate from the chemical-synapse weight matrix)

#### Scenario: Forward-pass depth K is configurable

- **WHEN** the brain config specifies `forward_pass_depth: K` for an integer K ≥ 1
- **THEN** the forward pass SHALL iterate the connectome update `h = activation((W_chem * M_chem)ᵀ @ h + G_gapᵀ @ h)` exactly K times before the motor readout
- **AND** the default value of K SHALL be 1

### Requirement: ConnectomePPOBrainConfig

The brain SHALL be configured via a Pydantic `ConnectomePPOBrainConfig` model that exposes the connectome source, the gap-junction enablement flag, the chemical-mask mode, the forward-pass depth, the frozen-updates flag, and PPO hyperparameters mirroring `MLPPPOBrainConfig`.

#### Scenario: Required fields

- **WHEN** `ConnectomePPOBrainConfig` is constructed
- **THEN** the config SHALL accept the following fields:
  - `connectome_source: Literal["cook_2019_hermaphrodite"]`
  - `enable_gap_junctions: bool` (default `True`)
  - `chemical_mask_mode: Literal["strict", "soft_prior"]` (default `"strict"`)
  - `forward_pass_depth: int` (default `1`, must be ≥ 1)
  - `freeze_updates: bool` (default `False`, drives the Gate 1 G1.c paired control)
- **AND** the config SHALL accept the PPO hyperparameters used by `MLPPPOBrainConfig` (learning rate, clip range, value-loss coefficient, entropy coefficient, gradient-clip norm, batch size, etc.)

#### Scenario: Soft-prior mode allows new chemical edges to grow

- **GIVEN** a `ConnectomePPOBrainConfig` with `chemical_mask_mode: "soft_prior"`
- **WHEN** the PPO learning rule completes a gradient step
- **THEN** `topology.apply_weight_mask(...)` SHALL be a no-op (the candidate weight tensor is returned unchanged)
- **AND** new non-zero weights MAY appear along edges where `M_chem[i, j] = False`
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
