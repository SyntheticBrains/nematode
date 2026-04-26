## ADDED Requirements

### Requirement: Genome Encoder Protocol

The system SHALL provide a `GenomeEncoder` protocol allowing any brain implementing `WeightPersistence` to be serialized into a flat parameter genome and reconstructed from one. Concrete encoders SHALL be registered in a central `ENCODER_REGISTRY` keyed by the brain's name.

#### Scenario: Encoder round-trip preserves brain behaviour

- **GIVEN** a `MLPPPOBrain` instance built from a known config
- **WHEN** the brain is encoded to a `Genome` via `MLPPPOEncoder.initial_genome()` and decoded back via `MLPPPOEncoder.decode()`
- **THEN** the decoded brain SHALL produce identical first-step actions as the original brain on identical seeded inputs

#### Scenario: Encoder excludes runtime training state from the genome

- **GIVEN** a brain with `training_state` containing a non-zero `_episode_count`
- **WHEN** the brain is encoded to a `Genome`
- **THEN** the genome `params` array SHALL NOT include `_episode_count` or any optimizer state
- **AND** when the genome is decoded, the resulting brain SHALL have `_episode_count` reset to `0`

#### Scenario: Genome dimension matches flattened weight component shape

- **GIVEN** a brain config and the encoder for its brain type
- **WHEN** `encoder.genome_dim(brain_config)` is called
- **THEN** the returned integer SHALL equal the total number of float parameters across all weight components selected by the encoder

#### Scenario: Encoders discover weight components dynamically via denylist

- **GIVEN** any brain implementing `WeightPersistence`
- **WHEN** the encoder serializes the brain
- **THEN** the encoder SHALL call `brain.get_weight_components()` to retrieve the full set of components
- **AND** SHALL include in the genome every component whose name is NOT in the denylist `{"optimizer", "actor_optimizer", "critic_optimizer", "training_state"}`
- **AND** SHALL NOT hardcode an allowlist of component names

#### Scenario: Conditional weight components are picked up automatically

- **GIVEN** a `MLPPPOBrain` configured with `_feature_gating: true` (which adds a `gate_weights` component)
- **WHEN** the encoder serializes the brain
- **THEN** the genome `params` array SHALL include the parameters from `gate_weights`
- **AND** decode SHALL restore them correctly so that the gated and ungated brain produce identical first-step actions

#### Scenario: Encoder registry membership is sufficient for dispatch

- **GIVEN** any brain name X registered in `ENCODER_REGISTRY`
- **WHEN** `ENCODER_REGISTRY[X]()` is invoked
- **THEN** the call SHALL produce a working `GenomeEncoder` instance for that brain
- **AND** that encoder SHALL be the single dispatch point used by `EvolutionLoop`

#### Scenario: Unsupported brain name fails clearly

- **GIVEN** a brain name (e.g. `"qvarcircuit"`) NOT in `ENCODER_REGISTRY`
- **WHEN** the evolution loop attempts to construct an encoder
- **THEN** a `ValueError` SHALL be raised whose message lists the registered brain names
- **AND** the message SHALL note that quantum brain support is deferred to a future Phase 6 re-evaluation

### Requirement: Lineage Tracking

The system SHALL maintain a single CSV file per evolution run recording every fitness evaluation with parent→child genealogy, written in append mode so resume operations do not lose history.

#### Scenario: Lineage CSV records every fitness evaluation

- **GIVEN** an evolution run with population P and G generations
- **WHEN** the run completes
- **THEN** `evolution_results/<session_id>/lineage.csv` SHALL contain `P × G` rows (plus one header row)
- **AND** each row SHALL have columns `generation, child_id, parent_ids, fitness, brain_type`

#### Scenario: parent_ids is populated for non-zero generations

- **GIVEN** a lineage CSV row with `generation > 0`
- **WHEN** the row is read
- **THEN** the `parent_ids` field SHALL be a `;`-joined string of parent genome IDs from generation N-1
- **AND** for generation 0, `parent_ids` SHALL be empty

#### Scenario: Append mode preserves history across resume

- **GIVEN** an evolution run that wrote 5 generations to lineage.csv before crashing
- **WHEN** the run is resumed and completes the remaining 5 generations
- **THEN** the final CSV SHALL contain all 10 generations of rows in chronological order
- **AND** the header row SHALL appear exactly once

### Requirement: Evolution Loop Checkpoint and Resume

The `EvolutionLoop` SHALL pickle optimiser state at a configurable interval and SHALL support resuming from a checkpoint without altering the deterministic behaviour of the run.

#### Scenario: Checkpoint contains optimiser, generation, RNG state, and version

- **GIVEN** an evolution run with `checkpoint_every: 5` configured
- **WHEN** generation 5 completes
- **THEN** `output_dir/checkpoint.pkl` SHALL exist
- **AND** the pickled object SHALL include keys `optimizer`, `generation`, `rng_state`, `lineage_path`, `checkpoint_version`

#### Scenario: Resume continues from last checkpoint

- **GIVEN** a run was killed after writing a checkpoint at generation 5
- **WHEN** the run is invoked with `--resume <path>`
- **THEN** the loop SHALL resume from generation 6
- **AND** the optimizer's internal state (CMA-ES covariance matrix or GA population) SHALL be restored from the checkpoint

#### Scenario: Incompatible checkpoint version is rejected

- **GIVEN** a checkpoint pickle whose `checkpoint_version` does not match the current loop's expected version
- **WHEN** the loop attempts to resume
- **THEN** an error SHALL be raised with both the expected and found version numbers
- **AND** the loop SHALL NOT silently continue

### Requirement: Evolution Configuration Block

The `SimulationConfig` SHALL accept an optional `evolution` block; existing scenario configs without an `evolution` block SHALL load unchanged.

#### Scenario: Existing scenario config loads without evolution block

- **GIVEN** a scenario YAML file (e.g. `configs/scenarios/foraging/mlpppo_small_oracle.yml`) with no `evolution:` top-level key
- **WHEN** the config is loaded
- **THEN** `SimulationConfig.evolution` SHALL be `None`
- **AND** no error SHALL be raised

#### Scenario: Evolution config block parses into populated EvolutionConfig

- **GIVEN** a YAML file with an `evolution:` block specifying `algorithm`, `population_size`, `generations`, `episodes_per_eval`
- **WHEN** the config is loaded
- **THEN** `SimulationConfig.evolution` SHALL be an `EvolutionConfig` instance with the specified fields populated
- **AND** unspecified fields SHALL fall back to `EvolutionConfig` defaults

#### Scenario: CLI flags override YAML for the same field

- **GIVEN** a YAML file with `evolution.generations: 50`
- **WHEN** the user invokes `scripts/run_evolution.py --config <yaml> --generations 10`
- **THEN** the loop SHALL run for 10 generations
- **AND** the YAML value SHALL be ignored for that field only

### Requirement: First-Class Encoder Coverage for Phase 5 Brains

The encoder registry SHALL include `MLPPPOEncoder` and `LSTMPPOEncoder` at minimum, so that Phase 5 milestones M2 (hyperparameter pilot), M3 (Lamarckian pilot), M4 (Baldwin), M5 (co-evolution prey), and M6 (transgenerational memory) can target classical brains without further framework changes.

#### Scenario: MLPPPO encoder is registered

- **GIVEN** the `quantumnematode.evolution` module is imported
- **WHEN** `ENCODER_REGISTRY["mlpppo"]` is accessed
- **THEN** the value SHALL be the `MLPPPOEncoder` class

#### Scenario: LSTMPPO encoder is registered

- **GIVEN** the `quantumnematode.evolution` module is imported
- **WHEN** `ENCODER_REGISTRY["lstmppo"]` is accessed
- **THEN** the value SHALL be the `LSTMPPOEncoder` class

#### Scenario: LSTMPPO encoder includes all recurrent and feed-forward components

- **GIVEN** an `LSTMPPOBrain` instance (whose `get_weight_components()` returns `{"lstm", "layer_norm", "policy", "value", "actor_optimizer", "critic_optimizer", "training_state"}`)
- **WHEN** the encoder serializes the brain
- **THEN** the serialized weight components SHALL include `"lstm"`, `"layer_norm"`, `"policy"`, and `"value"` (all four learned-weight components)
- **AND** SHALL NOT include `"actor_optimizer"`, `"critic_optimizer"`, or `"training_state"` (denylist)
- **AND** the per-episode hidden state (`_pending_h_state`, `_pending_c_state`) SHALL NOT be part of the genome (it is reset at `prepare_episode()` per existing brain code)
