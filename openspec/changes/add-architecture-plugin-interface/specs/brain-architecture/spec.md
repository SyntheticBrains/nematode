# brain-architecture Specification Delta

## ADDED Requirements

### Requirement: Brain Plugin Registry

The system SHALL provide a decorator-registration brain plugin registry as the single source of truth for the mapping from a brain name to its config class, Brain class, and `BrainType` enum member. Adding a new architecture SHALL touch at most six files (the new brain module, its config, the `BrainType` enum addition, `__init__.py` re-exports, the test module, and an example YAML config) and SHALL introduce zero per-architecture branches in the simulation loop or training loop.

#### Scenario: Decorator registers a brain

- **WHEN** a brain module declares `@register_brain(name="<name>", config_cls=<Config>, brain_type=BrainType.<X>, families=(...))` on its Brain class at module import time
- **THEN** the registry SHALL record the `(name, config_cls, brain_cls, brain_type, families)` tuple
- **AND** the brain SHALL be retrievable via `get_registration(name)` returning the same tuple
- **AND** the brain name SHALL appear in `list_registered_brains()`

#### Scenario: Duplicate name detection

- **WHEN** a second `@register_brain(name="X", ...)` call uses a name already in the registry
- **THEN** the registry SHALL raise `ValueError` at import time with a message identifying the conflicting brain modules

#### Scenario: Unknown name lookup raises

- **WHEN** `instantiate_brain(name="not_a_brain", config=...)` is called with a name not in the registry
- **THEN** the call SHALL raise `ValueError` with a message listing the available registered names

#### Scenario: Registry instantiates any registered brain through one code path

- **WHEN** `instantiate_brain(name, config, **infra_kwargs)` is called with `name` in the registry
- **THEN** the function SHALL return a `Brain` Protocol-conforming instance of the registered Brain class
- **AND** the function SHALL apply the registered config-class type-check before instantiation
- **AND** no per-brain `if`/`elif` branches SHALL be required in the function body

#### Scenario: All architecture modules self-register at import time

- **WHEN** `quantumnematode.brain.arch` is imported
- **THEN** every architecture module under `brain/arch/` SHALL load and self-register
- **AND** `list_registered_brains()` SHALL return a set containing every member of the `BrainType` enum's string-value namespace

### Requirement: BrainType Enum and Registry Consistency

The `BrainType` enum SHALL remain the public typed dispatch key for callers outside the brain-architecture subsystem (evolution framework, predator-brain factory). Its membership SHALL be kept consistent with the registry via a startup-time consistency check.

#### Scenario: Enum and registry agree at import time

- **WHEN** `quantumnematode.brain.arch` finishes importing
- **THEN** the set of `BrainType` enum string values SHALL equal the set of registered brain names
- **AND** a mismatch SHALL raise a clear exception identifying which names are missing from which side

#### Scenario: Family type-aliases derived from registry metadata

- **WHEN** `BRAIN_TYPES` Literal, `QUANTUM_BRAIN_TYPES` set, `CLASSICAL_BRAIN_TYPES` set, and `SPIKING_BRAIN_TYPES` set are inspected
- **THEN** their contents SHALL be derived from the `families` metadata on each `@register_brain(...)` registration
- **AND** SHALL NOT require hand-maintenance separate from the per-brain decorator

### Requirement: Brain Topology Protocol

The system SHALL provide a `BrainTopology` Protocol that exposes the structural and forward-pass interface of a brain's network, factored out from learning-rule concerns (optimisers, replay buffers, value heads).

#### Scenario: Topology exposes shape attributes

- **WHEN** a `BrainTopology` implementation is inspected
- **THEN** the instance SHALL expose `n_inputs: int`, `n_outputs: int`, and `n_hidden: int` attributes

#### Scenario: Topology computes a forward pass

- **WHEN** `topology.forward(x)` is called with a Tensor of shape compatible with `n_inputs`
- **THEN** the call SHALL return a Tensor of shape compatible with `n_outputs`
- **AND** the call SHALL be free of optimiser, replay-buffer, or value-head state changes

#### Scenario: Topology applies weight mask

- **WHEN** `topology.apply_weight_mask(weights)` is called with a candidate weight tensor
- **THEN** the call SHALL return the weight tensor projected onto the topology's allowed connectivity manifold
- **AND** for dense topologies the default implementation SHALL be the identity function
- **AND** for sparse / strict-mask topologies (e.g. connectome-constrained) the call SHALL zero out weights along non-existent edges

### Requirement: Learning Rule Protocol

The system SHALL provide a `LearningRule` Protocol that encapsulates how a topology's weights are updated from experience, factored out from topology concerns.

#### Scenario: Rule owns the optimiser state

- **WHEN** a `LearningRule` implementation is constructed
- **THEN** the rule instance SHALL own its optimiser, value head (if any), replay buffer (if any), advantage estimator (if any), and gradient clipper (if any)
- **AND** these objects SHALL NOT be exposed on the `BrainTopology` interface

#### Scenario: Rule advances the topology weights

- **WHEN** `rule.step(topology, batch)` is called with an experience batch
- **THEN** the rule SHALL compute gradients with respect to `topology` parameters
- **AND** SHALL apply optimiser updates to those parameters
- **AND** SHALL apply `topology.apply_weight_mask(...)` to any updated weights that are subject to a topology mask
- **AND** SHALL return a `RuleStepReport` summarising the update (loss components, gradient norms)

#### Scenario: Rule resets per-episode state

- **WHEN** `rule.reset_episode()` is called at the start of a new episode
- **THEN** the rule SHALL clear any per-episode state (advantage estimator buffers, recurrent-state caches owned by the rule)
- **AND** SHALL NOT clear the optimiser state or the persistent replay buffer

### Requirement: External Brain Protocol Surface is Unchanged

The external `Brain` Protocol surface (`run_brain` / `update_memory` / `prepare_episode` / `post_process_episode` / `copy` plus `history_data` and `latest_data` attributes) SHALL remain unchanged. Every registered brain SHALL satisfy the existing `Brain` Protocol exactly as defined at [packages/quantum-nematode/quantumnematode/brain/arch/\_brain.py:346-368](../../../packages/quantum-nematode/quantumnematode/brain/arch/_brain.py).

#### Scenario: Existing Brain Protocol consumers continue to work

- **WHEN** existing simulation / training / evolution code that calls `brain.run_brain(...)` / `brain.update_memory(...)` / etc. is executed against a registry-instantiated brain
- **THEN** the brain SHALL respond to every Protocol method with identical signatures and identical observable behaviour to its pre-refactor implementation
- **AND** no Protocol method SHALL be added, removed, or renamed

### Requirement: Migration Regression Bar — MLPPPO and LSTMPPO Byte-Equivalence

The MLPPPO and LSTMPPO brains, as the Phase 6 Gate 1 G1.d MUST architectures, SHALL produce byte-identical training trajectories and parameter tensors pre-refactor and post-refactor on at least one smoke config each, with all RNG seeds pinned.

#### Scenario: MLPPPO byte-equivalence on klinotaxis smoke config

- **GIVEN** a captured fixture of MLPPPO training trajectory + final parameter tensor on a klinotaxis smoke config with seed pinned, recorded pre-refactor
- **WHEN** the same config is re-run post-refactor with the same seed
- **THEN** every per-step action probability tensor SHALL satisfy `torch.equal(post, pre)`
- **AND** every per-episode parameter tensor SHALL satisfy `torch.equal(post, pre)`

#### Scenario: LSTMPPO byte-equivalence on klinotaxis smoke config

- **GIVEN** a captured fixture of LSTMPPO training trajectory + final parameter tensor on a klinotaxis smoke config with seed pinned, recorded pre-refactor
- **WHEN** the same config is re-run post-refactor with the same seed
- **THEN** every per-step action probability tensor + LSTM hidden state SHALL satisfy `torch.equal(post, pre)`
- **AND** every per-episode parameter tensor SHALL satisfy `torch.equal(post, pre)`

### Requirement: Migration Regression Bar — Other 17 Architectures Numerical Equivalence

The remaining 17 brain architectures SHALL produce parameter tensors after a 5-step smoke training pre-refactor and post-refactor that satisfy `torch.allclose(rtol=0, atol=1e-7)`, with all RNG seeds pinned.

#### Scenario: Per-architecture numerical-equivalence smoke

- **GIVEN** a brain architecture in `{QVARCIRCUIT, QQLEARNING, MLP_REINFORCE, MLP_DQN, QRC, QRH, QEF, CRH, SPIKING_REINFORCE, QSNN_REINFORCE, QSNN_PPO, HYBRID_QUANTUM, HYBRID_CLASSICAL, HYBRID_QUANTUM_CORTEX, QLIF_LSTM, QRH_QLSTM, CRH_QLSTM}`
- **AND** a 5-step smoke training config with a pinned seed
- **WHEN** the brain is trained for 5 steps pre-refactor and post-refactor on that config
- **THEN** every Pydantic-exposed parameter tensor SHALL satisfy `torch.allclose(post, pre, rtol=0, atol=1e-7)`
- **AND** any architecture exceeding the tolerance SHALL be either fixed or have its tolerance widened with explicit written justification in the T2 logbook

#### Scenario: Quantum architectures use deterministic simulator

- **WHEN** a quantum-family architecture's regression-equivalence test is executed
- **THEN** the test SHALL use a deterministic statevector simulator (not the noisy AerSimulator)
- **AND** SHALL pin shot-RNG seeds so QPU-shot variance does not introduce drift

## MODIFIED Requirements

### Requirement: Brain Type Registry

The `BrainType` enum SHALL remain the canonical typed dispatch key and SHALL include every supported brain architecture as enum members. Its membership SHALL be kept consistent with the brain plugin registry via a startup-time consistency check.

#### Scenario: CRH Brain Type Registration

- **WHEN** the brain type system is initialized
- **THEN** `BrainType.CRH` SHALL exist with value `"crh"` in the BrainType enum
- **AND** CRH SHALL be included in `CLASSICAL_BRAIN_TYPES`
- **AND** `"crh"` SHALL be included in the `BRAIN_TYPES` Literal type alias

#### Scenario: CRH Brain Factory

- **WHEN** `instantiate_brain("crh", config)` is called
- **THEN** the registry SHALL return a `CRHBrain` instance
- **AND** SHALL accept `CRHBrainConfig` for configuration

#### Scenario: CRH Config Loading

- **WHEN** a YAML config specifies `brain.name: crh`
- **THEN** the config loader SHALL parse brain config using `CRHBrainConfig`
- **AND** SHALL support all CRH-specific fields plus inherited ReservoirHybridBase fields

#### Scenario: QLIF-LSTM Brain Type Registration

- **WHEN** the brain type system is initialized
- **THEN** `BrainType.QLIF_LSTM` SHALL exist with value `"qliflstm"` in the BrainType enum
- **AND** QLIF_LSTM SHALL be included in `QUANTUM_BRAIN_TYPES`
- **AND** `"qliflstm"` SHALL be included in the `BRAIN_TYPES` Literal type alias

#### Scenario: QLIF-LSTM Config Loading

- **WHEN** a YAML config specifies `brain.name: qliflstm`
- **THEN** the config loader SHALL parse brain config using `QLIFLSTMBrainConfig`
- **AND** SHALL support all QLIF-LSTM-specific fields

#### Scenario: QLIF-LSTM Brain Factory Instantiation

- **WHEN** the brain factory receives a `QLIFLSTMBrainConfig`
- **THEN** the factory SHALL instantiate a `QLIFLSTMBrain` with the provided configuration

#### Scenario: QEF Brain Type Registration

- **WHEN** the brain type system is initialized
- **THEN** `BrainType.QEF` SHALL exist with value `"qef"` in the BrainType enum
- **AND** QEF SHALL be included in `QUANTUM_BRAIN_TYPES`
- **AND** `"qef"` SHALL be included in the `BRAIN_TYPES` Literal type alias

#### Scenario: QEF Brain Factory

- **WHEN** `instantiate_brain("qef", config)` is called
- **THEN** the registry SHALL return a `QEFBrain` instance
- **AND** SHALL accept `QEFBrainConfig` for configuration
- **AND** SHALL raise ValueError if config is not QEFBrainConfig

#### Scenario: QEF Config Loading

- **WHEN** a YAML config specifies `brain.name: qef`
- **THEN** the config loader SHALL parse brain config using `QEFBrainConfig`
- **AND** SHALL support all QEF-specific fields plus inherited ReservoirHybridBaseConfig fields

#### Scenario: ConnectomePPO Brain Type Registration

- **WHEN** the brain type system is initialized
- **THEN** `BrainType.CONNECTOMEPPO` SHALL exist with value `"connectomeppo"` in the BrainType enum
- **AND** CONNECTOMEPPO SHALL be included in `CLASSICAL_BRAIN_TYPES`
- **AND** `"connectomeppo"` SHALL be included in the `BRAIN_TYPES` Literal type alias

#### Scenario: ConnectomePPO Brain Factory

- **WHEN** `instantiate_brain("connectomeppo", config)` is called
- **THEN** the registry SHALL return a `ConnectomePPOBrain` instance
- **AND** SHALL accept `ConnectomePPOBrainConfig` for configuration

#### Scenario: ConnectomePPO Config Loading

- **WHEN** a YAML config specifies `brain.name: connectomeppo`
- **THEN** the config loader SHALL parse brain config using `ConnectomePPOBrainConfig`
- **AND** SHALL support all ConnectomePPO-specific fields

#### Scenario: Registry-enum consistency check at import time

- **WHEN** `quantumnematode.brain.arch` finishes importing
- **THEN** the set of string-values across `BrainType` enum members SHALL equal the set of names registered via `@register_brain(...)`
- **AND** any mismatch SHALL raise an exception that identifies which names are present on each side
