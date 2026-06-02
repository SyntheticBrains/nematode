## ADDED Requirements

### Requirement: Equivariant Quantum Brain Architecture

The system SHALL provide a quantum brain `EquivariantQuantumPPOBrain` (config `name: equivariantquantum`,
`brain_type: EQUIVARIANT_QUANTUM_PPO`) whose actor is a **bilateral-symmetry (Z₂) equivariant
parameterised quantum circuit**: an equivariant classical pre-encoder (a parity-block-structured linear
map) SHALL compress the sensory observation to `num_qubits` Z₂-typed latent features (split into `k_even`
even and `k_odd` odd channels); a **Z₂-equivariant** circuit SHALL angle-encode those latents with data
re-uploading and apply a variational ansatz drawn from the `U_R`-invariant gate set (single-qubit
rotations plus entangling `IsingXX` / same-parity `IsingZZ` couplings), where `U_R = ⊗ X` over the
odd-parity qubits realises the reflection; and an **equivariant readout** SHALL produce action logits over
the 4-action `DEFAULT_ACTIONS` set such that the left–right mirror swaps the `LEFT`/`RIGHT` logits and
fixes `FORWARD`/`STAY`. A plain-ANN critic SHALL estimate state value from the detached latent. The brain
SHALL be trained by PPO. Input dimensionality SHALL be derived from the active sensory-module
configuration at construction time, matching the other classical PPO brains.

#### Scenario: Brain construction builds the equivariant pre-encoder, circuit, and readout

- **WHEN** a brain config specifies `name: equivariantquantum`
- **THEN** the brain SHALL construct a parity-block-structured equivariant linear pre-encoder to
  `num_qubits = k_even + k_odd` Z₂-typed latents, a Z₂-equivariant parameterised quantum circuit with
  `num_layers` data-re-uploading layers, an equivariant Pauli-observable readout to the 4 action logits,
  and a critic MLP over the detached latent
- **AND** the circuit SHALL prepare a `U_R`-invariant reference state before encoding (odd-parity qubits
  in `|+⟩`, even-parity qubits in `|0⟩`), without which the end-to-end mirror-equivariance property does
  not hold
- **AND** the input dimensionality SHALL be derived from the active sensory modules at construction time
- **AND** no Qiskit/PennyLane dependency SHALL be required (the statevector simulator is in-repo torch)

#### Scenario: Forward pass produces finite logits and a value over the action set

- **WHEN** `run_brain()` is called with any well-formed `BrainParams`
- **THEN** the returned action logits SHALL be a 4-vector matching the `DEFAULT_ACTIONS` order
  ([FORWARD, LEFT, RIGHT, STAY])
- **AND** the logits and the critic value SHALL be finite (no NaN, no Inf)
- **AND** the brain SHALL sample the action by applying softmax to the logits and drawing from the
  resulting categorical distribution

#### Scenario: Forward pass produces non-degenerate variance across an episode

- **WHEN** the brain is run for at least one episode on the smoke config with non-zero environmental
  gradients
- **THEN** the variance across the 4-action logits over a sample of ≥ 100 forward passes SHALL be
  strictly greater than zero
- **AND** the logits SHALL not collapse to a constant action across that sample

#### Scenario: Entanglement is load-bearing

- **GIVEN** a constructed `equivariantquantum` brain with its entangling couplings (`IsingXX` /
  same-parity `IsingZZ`) present
- **WHEN** the entangling couplings are removed (rotation-only / product-state circuit) with all else
  fixed
- **THEN** the circuit's output state SHALL be separable and the two configurations SHALL produce
  measurably different action-logit distributions on a fixed input batch (the entanglement changes the
  represented policy)

#### Scenario: PPO update runs and leaves parameters finite

- **GIVEN** a full rollout buffer of recorded steps
- **WHEN** the PPO update runs
- **THEN** it SHALL recompute action log-probabilities and values with gradient flowing through the
  statevector simulator (backprop-through-simulator), apply the clipped PPO objective plus an entropy
  bonus and a value loss with gradient-norm clipping at `max_grad_norm` over `num_epochs`
- **AND** complete without error and leave the network and quantum parameters finite

#### Scenario: Entropy-schedule fields are validated as a pair

- **WHEN** a brain config sets exactly one of `entropy_coef_end` / `entropy_decay_episodes`
- **THEN** construction SHALL raise a clear error stating the two fields must be set together
- **AND WHEN** neither is set
- **THEN** the entropy coefficient SHALL be flat at `entropy_coef` for the whole run

### Requirement: Bilateral Z₂ Policy Equivariance

The `EquivariantQuantumPPOBrain` (with `equivariant: true`) SHALL implement a policy that is exactly
equivariant under the left–right bilateral mirror Z₂: reflecting the observation by the verified parity
operator `R = diag(p)` SHALL swap the `LEFT`/`RIGHT` action probabilities and leave `FORWARD`/`STAY`
unchanged. The observation parity vector `p` SHALL be derived empirically from the live sensory code (not
hand-assigned), and any feature that does not transform as a clean `±1` SHALL be symmetrised or routed as
an even side-input with the deviation recorded.

#### Scenario: The observation parity vector is validated by a mirror-consistency check across all headings

- **GIVEN** the parity vector `p` assigned from the sensory-module layout (each lateral-gradient *angle*
  feature `−1`; strength / temporal-derivative / fore-aft-zone / proprioception / STAM features `+1`),
  sized so `len(p)` matches `get_classical_feature_dimension`
- **WHEN** the sensory inputs are reflected across the agent's forward (egocentric left–right) axis for
  each of the four agent headings (UP/DOWN/LEFT/RIGHT) and the observation is recomputed via the live
  feature-extraction pipeline
- **THEN** the recomputed observation SHALL equal `R · obs` for the single assigned `p` (within
  floating-point tolerance) for every heading — confirming `R = diag(p)` is heading-independent
- **AND** any feature that does not transform as its assigned `±1` SHALL fail the test (catching drift
  between the assigned parity and the live sensory code)
- **AND** a construction-time guard SHALL fail loudly if `len(p) != input_dim`

#### Scenario: The policy is exactly mirror-equivariant end-to-end

- **GIVEN** a constructed `equivariantquantum` brain with `equivariant: true`
- **WHEN** action probabilities are computed for a random observation `s` and for its mirror `R · s`
- **THEN** the probability of `LEFT` at `s` SHALL equal the probability of `RIGHT` at `R · s` (and vice
  versa), and the probabilities of `FORWARD` and `STAY` SHALL be unchanged, within floating-point
  tolerance
- **AND** this property SHALL hold across a sample of ≥ 100 random observations

### Requirement: Equivariant and Quantum Ablation Flags

The brain SHALL expose two boolean config flags — `equivariant` (default `true`) and `quantum` (default
`true`) — that select ablation siblings sharing the env interface, readout shape, and PPO loop, so the
comparison can attribute effects to the symmetry prior and to the quantum circuit separately.

#### Scenario: Non-equivariant quantum ablation drops the symmetry constraint

- **WHEN** a config sets `equivariant: false`
- **THEN** the actor SHALL be an unstructured parameterised quantum circuit (arbitrary single-qubit
  rotations + `CZ` entanglers, with a free measurement→logit linear head)
- **AND** the end-to-end mirror-equivariance property SHALL NOT be required to hold (the ablation is
  genuinely unstructured)

#### Scenario: Classical-equivariant ablation drops the quantum circuit

- **WHEN** a config sets `quantum: false`
- **THEN** the actor SHALL be an equivariant classical MLP (parity-block-structured weights) with the
  same parity-typed readout
- **AND** the end-to-end mirror-equivariance property SHALL hold (the classical equivariant control is
  still exactly equivariant)

### Requirement: In-Repo Statevector Differentiation

The quantum actor SHALL be simulated by an **in-repo torch statevector simulator** that supports
backprop-through-simulator gradients (no parameter-shift, no external quantum-simulation dependency). The
simulator SHALL be validated against a reference (Qiskit `Statevector`) on a set of fixed circuits.

#### Scenario: Simulator matches a reference on fixed circuits

- **GIVEN** a set of fixed parameterised circuits (encoding + ansatz) over `num_qubits` qubits
- **WHEN** the in-repo torch simulator and the reference statevector simulator evaluate the same circuit
  and observable expectations
- **THEN** the results SHALL agree within floating-point tolerance
- **AND** the torch simulator's outputs SHALL be differentiable with respect to the circuit parameters
  (a finite, non-NaN gradient is produced by autograd)

### Requirement: WeightPersistence Protocol Conformance

The `EquivariantQuantumPPOBrain` SHALL implement the `WeightPersistence` protocol so its weights —
pre-encoder, quantum circuit parameters, readout scales/biases, and critic — can be serialised and
restored (checkpoint round-trip).

#### Scenario: Brain exposes and restores weights as named WeightComponents

- **GIVEN** an instance of `EquivariantQuantumPPOBrain`
- **WHEN** `brain.get_weight_components()` is called
- **THEN** the returned dict SHALL contain an `actor` component (the pre-encoder + quantum circuit
  parameters + readout scales, as a single module state) and a `critic` component — plus `optimizer` and
  `training_state` — each a `WeightComponent` carrying a `state` dict of named `torch.Tensor` parameters
- **AND WHEN** a second brain with identical topology loads that output via `load_weight_components()`
- **THEN** the two brains SHALL produce identical action logits for the same input (within
  floating-point tolerance)

### Requirement: Plugin Registry Registration

The `EquivariantQuantumPPOBrain` SHALL register through the `@register_brain` plugin registry with a
unique `BrainType` (`EQUIVARIANT_QUANTUM_PPO`) and SHALL load through the standard simulation launcher
without any per-architecture conditional branch.

#### Scenario: Brain loads through the plugin registry

- **GIVEN** the `quantumnematode.brain.arch` package is imported
- **WHEN** a simulation config names brain `equivariantquantum`
- **THEN** the launcher SHALL instantiate `EquivariantQuantumPPOBrain` via the registry lookup
- **AND** no per-architecture conditional SHALL be required in the launcher or brain factory
