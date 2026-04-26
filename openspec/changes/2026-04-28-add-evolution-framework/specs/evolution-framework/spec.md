## ADDED Requirements

### Requirement: Genome Encoder Protocol

The system SHALL provide a `GenomeEncoder` protocol allowing any brain implementing `WeightPersistence` to be serialized into a flat parameter genome and reconstructed from one. Concrete encoders SHALL be registered in a central `ENCODER_REGISTRY` keyed by the brain's name. Encoder methods SHALL take the full `SimulationConfig` (not just `BrainConfig`), since fresh-brain construction requires fields scattered across multiple top-level config sections (`shots`, `qubits`, `device`, `learning_rate`, `gradient`, `parameter_initializer`).

#### Scenario: Encoder methods accept SimulationConfig

- **GIVEN** an encoder for a registered brain
- **WHEN** the encoder methods are called
- **THEN** their signatures SHALL be `initial_genome(sim_config: SimulationConfig, *, rng) -> Genome`, `decode(genome: Genome, sim_config: SimulationConfig) -> Brain`, and `genome_dim(sim_config: SimulationConfig) -> int`
- **AND** brain construction inside `decode` SHALL be delegated to a single helper `evolution.brain_factory.instantiate_brain_from_sim_config(sim_config)` so all encoders share one source of truth for fresh-brain construction

#### Scenario: Encoder round-trip preserves brain behaviour

- **GIVEN** a `MLPPPOBrain` instance built from a known `SimulationConfig`
- **WHEN** the brain is encoded to a `Genome` via `MLPPPOEncoder.initial_genome(sim_config)` and decoded back via `MLPPPOEncoder.decode(genome, sim_config)`
- **THEN** the decoded brain SHALL produce identical first-step actions as the original brain on identical seeded inputs

#### Scenario: Encoder excludes runtime training state from the genome and resets it on decode

- **GIVEN** a brain with `training_state` containing a non-zero `_episode_count`
- **WHEN** the brain is encoded to a `Genome`
- **THEN** the genome `params` array SHALL NOT include `_episode_count` or any optimizer state
- **AND** when the genome is decoded, the resulting brain SHALL have `_episode_count == 0`
- **AND** the resulting brain SHALL have its learning rate updated to match `_episode_count == 0` (i.e. the encoder calls `_update_learning_rate()` after the count reset, so a freshly-decoded brain is in the same state as a freshly-constructed one)

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

### Requirement: Frozen-Weight Fitness Evaluation

The `EpisodicSuccessRate` fitness function SHALL evaluate a genome by running its decoded brain through `episodes_per_eval` complete episodes WITHOUT calling `brain.learn()` or `brain.update_memory()` on any termination path. Calling `agent.run_episode()` directly is insufficient because [`StandardEpisodeRunner._terminate_episode`](packages/quantum-nematode/quantumnematode/agent/runners.py#L155) defaults `learn=True` on the success path. The framework SHALL provide a `FrozenEvalRunner` class that subclasses `StandardEpisodeRunner` and forces `learn=False, update_memory=False` on every termination.

#### Scenario: Fitness function never invokes brain.learn()

- **GIVEN** a `MLPPPOBrain` or `LSTMPPOBrain` (both `ClassicalBrain` subclasses)
- **WHEN** `EpisodicSuccessRate.evaluate()` runs `episodes_per_eval` episodes (including episodes that succeed)
- **THEN** `brain.learn()` SHALL NOT be called at any point during the evaluation
- **AND** `brain.update_memory()` SHALL NOT be called at any point during the evaluation

#### Scenario: Fitness function never invokes brain.learn() even on successful episodes

- **GIVEN** a genome whose decoded brain successfully completes the foraging task (terminates with `TerminationReason.COMPLETED_ALL_FOOD`)
- **WHEN** `EpisodicSuccessRate.evaluate()` processes that episode
- **THEN** `brain.learn()` SHALL NOT be called (even though the standard runner would default to `learn=True` on this path)

#### Scenario: Success detection uses TerminationReason

- **GIVEN** an `EpisodeResult` returned by `FrozenEvalRunner.run()`
- **WHEN** the fitness function determines whether the episode succeeded
- **THEN** success SHALL be defined as `result.termination_reason == TerminationReason.COMPLETED_ALL_FOOD`
- **AND** the fitness function SHALL NOT reference any `result.episode_success` attribute (which does not exist on `EpisodeResult`)

#### Scenario: Fitness returns ratio in unit interval

- **GIVEN** an evaluation of `episodes_per_eval = K` episodes where `S` succeed
- **WHEN** the fitness is returned
- **THEN** the value SHALL equal `S / K` exactly
- **AND** SHALL be a finite float in `[0.0, 1.0]`

### Requirement: Lineage Tracking

The system SHALL maintain a single CSV file per evolution run recording every fitness evaluation with parent→child genealogy, written in append mode so resume operations do not lose history. Generation indexing SHALL be 0-based: a run with `generations: G` populates rows for generations `0, 1, …, G-1`.

#### Scenario: Lineage CSV records every fitness evaluation

- **GIVEN** an evolution run with `population_size = P` and `generations = G`
- **WHEN** the run completes
- **THEN** `evolution_results/<session_id>/lineage.csv` SHALL contain exactly `P × G` data rows (plus one header row)
- **AND** the `generation` column SHALL take values in the inclusive range `[0, G-1]` with each value appearing exactly `P` times
- **AND** each row SHALL have columns `generation, child_id, parent_ids, fitness, brain_type`

#### Scenario: parent_ids is populated for non-zero generations

- **GIVEN** a lineage CSV row with `generation > 0`
- **WHEN** the row is read
- **THEN** the `parent_ids` field SHALL be a `;`-joined string of parent genome IDs from generation `N-1`
- **AND** for `generation == 0`, `parent_ids` SHALL be empty

#### Scenario: Append mode preserves history across resume

- **GIVEN** an evolution run with `generations: 10` that wrote rows for generations `0..4` to lineage.csv before crashing (5 generations × P rows = 5P rows + header)
- **WHEN** the run is resumed and completes the remaining generations `5..9`
- **THEN** the final CSV SHALL contain `10 × P` data rows in chronological generation order
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
