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
