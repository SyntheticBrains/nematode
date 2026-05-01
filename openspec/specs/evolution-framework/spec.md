# evolution-framework Specification

## Purpose

Brain-agnostic evolutionary optimisation of classical brain initial-weight genomes. Provides the encoder protocol (any brain implementing `WeightPersistence` plugs in via a one-class registration), the frozen-weight fitness function (`EpisodicSuccessRate` — runs episodes via `FrozenEvalRunner` without ever calling `.learn()`), append-only lineage tracking, parallel fitness evaluation, and pickle checkpoint/resume. `MLPPPOEncoder` and `LSTMPPOEncoder` are registered out of the box; future inheritance strategies (Lamarckian, Baldwin), co-evolution loops, and additional encoders extend this framework behind the existing protocols.

## Requirements

### Requirement: Genome Encoder Protocol

The system SHALL provide a `GenomeEncoder` protocol allowing any brain implementing `WeightPersistence` to be serialized into a flat parameter genome and reconstructed from one. Concrete encoders SHALL be registered in a central `ENCODER_REGISTRY` keyed by the brain's name. Encoder methods SHALL take the full `SimulationConfig` (not just `BrainConfig`), since fresh-brain construction requires fields scattered across multiple top-level config sections (`shots`, `qubits`, `device`, `learning_rate`, `gradient`, `parameter_initializer`).

#### Scenario: Encoder methods accept SimulationConfig

- **GIVEN** an encoder for a registered brain
- **WHEN** the encoder methods are called
- **THEN** their signatures SHALL be `initial_genome(sim_config: SimulationConfig, *, rng) -> Genome`, `decode(genome: Genome, sim_config: SimulationConfig, *, seed: int | None = None) -> Brain`, and `genome_dim(sim_config: SimulationConfig) -> int`
- **AND** brain construction inside `decode` SHALL be delegated to a single helper `evolution.brain_factory.instantiate_brain_from_sim_config(sim_config, *, seed=seed)` so all encoders share one source of truth for fresh-brain construction
- **AND** the wrapper SHALL patch `BrainConfig.seed` (extracted from `sim_config.brain.config`) when `seed` is supplied, NOT `SimulationConfig.seed` — the brain reads its seed from `BrainConfig.seed`, matching the established pattern in `scripts/run_simulation.py`
- **AND** the wrapper SHALL force `BrainConfig.weights_path = None` so the genome (loaded via `load_weight_components` after construction) is the sole weight source

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
- **AND** the message SHALL state that quantum brains are not currently supported

### Requirement: Frozen-Weight Fitness Evaluation

The `EpisodicSuccessRate` fitness function SHALL evaluate a genome by running its decoded brain through `episodes_per_eval` complete episodes WITHOUT calling `brain.learn()` or `brain.update_memory()` at any point during the evaluation. The standard runner calls `learn` in TWO places — per-step (at [runners.py:747](packages/quantum-nematode/quantumnematode/agent/runners.py#L747), which fires every step regardless of any kwarg) AND per-termination (via [`_terminate_episode`](packages/quantum-nematode/quantumnematode/agent/runners.py#L155), which defaults `learn=True` on the success path) — so a kwarg-only override is insufficient. The framework SHALL provide a `FrozenEvalRunner` class that subclasses `StandardEpisodeRunner` and intercepts BOTH call sites: `run()` temporarily replaces `agent.brain.learn` and `agent.brain.update_memory` with no-ops for the duration of each episode (restored in a `finally` block) to neutralise the per-step call, AND `_terminate_episode` forces `learn=False, update_memory=False` as a belt-and-braces guard on the termination path.

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

#### Scenario: Fitness is deterministic given a fixed seed

- **GIVEN** the same `genome`, `sim_config`, `encoder`, `episodes`, and `seed`
- **WHEN** `evaluate()` is invoked twice
- **THEN** the two returned fitness values SHALL be byte-identical
- **AND** the fitness function SHALL apply `seed` by calling `encoder.decode(genome, sim_config, seed=seed)`, which forwards `seed` through `instantiate_brain_from_sim_config(sim_config, seed=seed)` to patch `BrainConfig.seed` before brain construction
- **AND** the fitness function SHALL pass the same `seed` to `create_env_from_config(env_config, seed=seed)` for environment RNG
- **AND** the fitness function SHALL NOT call `torch.manual_seed(seed)` or assign `brain.rng` directly, AND SHALL NOT call `sim_config.model_copy(update={"seed": seed})` — those would either be no-ops (the brain's `set_global_seed` later clobbers them, or `SimulationConfig.seed` is not the field the brain reads) or duplicates of work the wrapper does

#### Scenario: Fitness `seed` parameter overrides BrainConfig.seed from YAML

- **GIVEN** a `sim_config` whose nested `brain.config.seed` field is `0`
- **WHEN** `evaluate()` is invoked with `seed=1` and again with `seed=2`
- **THEN** the two fitness values MAY differ
- **AND** this proves the fitness function's `seed` parameter — not the YAML-configured `BrainConfig.seed` — controls the per-evaluation RNG state via the wrapper's seed-patching

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

#### Scenario: parent_ids convention for CMA-ES and GA

- **GIVEN** an evolution run using either CMA-ES or GA (neither optimiser exposes per-child parent provenance)
- **WHEN** lineage rows are written for generation N (where N > 0) with population size P
- **THEN** every gen-N row's `parent_ids` SHALL be the `;`-joined set of ALL P genome IDs from generation N-1 (not just a subset)
- **AND** this SHALL be true regardless of which optimiser is in use — the convention is uniform across CMA-ES and GA so downstream tooling does not need an algorithm-specific code path
- **AND** for CMA-ES this is semantically accurate (every gen-N-1 candidate contributes to the distribution that samples gen-N); for GA it is a slight over-approximation since true crossover parents are a 2-element subset, but the GA optimiser does not expose them

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

### Requirement: Optimiser Portfolio

The framework SHALL provide multiple optimisers behind a uniform `EvolutionaryOptimizer` ask/tell interface so the loop is optimiser-agnostic. At minimum: `CMAESOptimizer` (recommended for unbounded continuous search at moderate-to-large genome dim, including weight evolution at n>~100), `GeneticAlgorithmOptimizer` (simpler, more interpretable), and `OptunaTPEOptimizer` (Bayesian-style sampler for small-genome bounded hyperparameter search). The user-facing `evolution.algorithm` field SHALL accept `"cmaes"`, `"ga"`, or `"tpe"`. Each optimiser SHALL respect the framework's minimisation sign convention (lower fitness = better; the loop pre-negates success rates upstream).

#### Scenario: TPE optimiser is selectable via the algorithm field

- **GIVEN** an `EvolutionConfig` with `algorithm: "tpe"` and a `hyperparam_schema` block defining bounded parameters
- **WHEN** the loop constructs the optimiser
- **THEN** an `OptunaTPEOptimizer` SHALL be instantiated with per-parameter bounds derived from the encoder's `genome_bounds(sim_config)`
- **AND** the optimiser SHALL implement the same `ask`/`tell`/`stop`/`result` contract as `CMAESOptimizer` and `GeneticAlgorithmOptimizer`
- **AND** the loop's per-generation flow SHALL be identical regardless of which optimiser is in use

#### Scenario: Encoders expose per-parameter bounds via genome_bounds

- **GIVEN** the `GenomeEncoder` protocol
- **WHEN** an encoder is asked for `genome_bounds(sim_config)`
- **THEN** it SHALL return a `list[tuple[float, float]] | None` where each entry is the `(low, high)` range for the corresponding genome dimension, in the SAME float-space the genome itself lives in (e.g. log-space for `log_scale=True` schema entries; bin-index range for categoricals)
- **AND** weight encoders (whose genomes are unbounded network weights) SHALL return `None`
- **AND** the `HyperparameterEncoder` SHALL return a fully-populated list whose length matches `genome_dim(sim_config)`

#### Scenario: TPE rejects unbounded encoders at construction

- **GIVEN** a config selecting `algorithm: "tpe"` paired with an encoder whose `genome_bounds` returns `None` (e.g. a weight encoder)
- **WHEN** the loop attempts to construct the optimiser
- **THEN** construction SHALL fail with a clear error stating that TPE requires per-parameter bounds, and pointing the user to CMA-ES or GA for unbounded weight evolution
- **AND** the error SHALL fire BEFORE any fitness evaluation runs

### Requirement: First-Class Encoder Coverage for Classical Brains

The encoder registry SHALL include `MLPPPOEncoder` and `LSTMPPOEncoder` at minimum, so that downstream evolutionary work — hyperparameter sweeps, inheritance-strategy pilots, co-evolution, and transgenerational memory experiments — can target classical brains without further framework changes.

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

### Requirement: Hyperparameter Encoding

The system SHALL provide a `HyperparameterEncoder` that conforms to the existing `GenomeEncoder` protocol and encodes brain hyperparameters (rather than weights) as a flat float vector with a per-slot schema stored in `Genome.birth_metadata`. The encoder SHALL be brain-agnostic — it works for any brain via `sim_config.brain.config` patching — and therefore SHALL NOT be registered in `ENCODER_REGISTRY` (which is keyed by brain name). Encoder selection happens at the dispatch layer (`evolution.encoders.select_encoder`), not via registry lookup. Each genome SHALL produce a fresh brain from the genome's hyperparameter values via `model_copy(update={...})` on the brain config + `instantiate_brain_from_sim_config`; no weights from the genome are loaded.

#### Scenario: Hyperparameter encoder is NOT in the brain-keyed registry

- **GIVEN** the `quantumnematode.evolution` module is imported
- **WHEN** `ENCODER_REGISTRY` is inspected
- **THEN** `"hyperparam"` SHALL NOT be a key in `ENCODER_REGISTRY`
- **AND** `ENCODER_REGISTRY` SHALL contain only brain-name keys (`"mlpppo"`, `"lstmppo"`)
- **AND** `HyperparameterEncoder` SHALL be importable directly from `quantumnematode.evolution.encoders` for callers that construct it explicitly
- **AND** `HyperparameterEncoder.brain_name` SHALL equal the empty string `""` (to satisfy the `runtime_checkable` `GenomeEncoder` protocol's `brain_name: str` typing while signalling brain-agnosticism)

#### Scenario: Hyperparameter run records the actual brain name in lineage and best_params

- **GIVEN** a hyperparameter-evolution run where `HyperparameterEncoder` is selected (its `brain_name == ""`) and the YAML's `brain.name` is e.g. `"mlpppo"`
- **WHEN** the loop writes a `lineage.csv` row OR writes `best_params.json`
- **THEN** the `brain_type` field in both artefacts SHALL be the YAML's `sim_config.brain.name` (`"mlpppo"` in this example), NOT the encoder's empty `brain_name`
- **AND** the loop SHALL resolve the field via `self.encoder.brain_name or self.sim_config.brain.name` so that brain-keyed M0 encoders continue to write their own `brain_name` (M0 back-compat) while brain-agnostic M2 encoders fall through to the YAML

#### Scenario: Per-type decode transforms

- **GIVEN** a `Genome` with a `param_schema` containing one entry per supported type (`float`, `int`, `bool`, `categorical`)
- **WHEN** `HyperparameterEncoder.decode()` interprets each genome slot
- **THEN** float slots SHALL be clipped to `bounds` and (when `log_scale: true`) exponentiated from log-space
- **AND** int slots SHALL be clipped to `bounds` and rounded via `int(round(value))`
- **AND** bool slots SHALL decode to `value > 0.0`
- **AND** categorical slots SHALL decode to `values[int(round(value)) mod len(values)]`

#### Scenario: Decode round-trips through brain config

- **GIVEN** an `initial_genome(sim_config, rng)` that samples a value of `0.001` for a float slot named `learning_rate`
- **WHEN** the genome is decoded via `decode(genome, sim_config)`
- **THEN** the resulting brain's `BrainConfig.learning_rate` field SHALL equal `0.001` within float tolerance
- **AND** any other brain config field NOT named in the schema SHALL retain its YAML-configured value unchanged

#### Scenario: Hyperparameter genome does not load weights

- **GIVEN** any genome produced by `HyperparameterEncoder.initial_genome` or sampled by the optimiser
- **WHEN** the genome is decoded via `decode(genome, sim_config)`
- **THEN** the resulting brain's weights SHALL be the result of fresh construction via `instantiate_brain_from_sim_config`
- **AND** `WeightPersistence.load_weight_components` SHALL NOT be called by this encoder
- **AND** the genome `params` array length SHALL equal `len(param_schema)`, NOT the brain's weight-component parameter count

#### Scenario: Encoder dispatch when hyperparam_schema is set

- **GIVEN** a `SimulationConfig` whose top-level `hyperparam_schema` field is not `None`
- **WHEN** `evolution.encoders.select_encoder(sim_config)` is called (the public dispatch entry point used by `scripts/run_evolution.py` and any programmatic caller)
- **THEN** `HyperparameterEncoder` SHALL be returned regardless of `sim_config.brain.name`
- **AND** when `hyperparam_schema is None`, `select_encoder` SHALL fall back to `get_encoder(sim_config.brain.name)` — the existing M0 brain-name → encoder lookup
- **AND** the dispatch SHALL succeed even for brains that exist in `BRAIN_CONFIG_MAP` but NOT in `ENCODER_REGISTRY` (e.g., a future hyperparameter pilot against a brain like `qvarcircuit` that has no weight encoder), so brain-agnostic hyperparameter evolution works for any brain with a registered config
- **AND** the existing M0 gate at `scripts/run_evolution.py` (which rejects brain names not in `ENCODER_REGISTRY`) SHALL be replaced by a `select_encoder` call that subsumes both the dispatch and the error-surfacing — preserving the M0 user-facing error (registered-brains list) when `hyperparam_schema is None` and the brain has no weight encoder

### Requirement: Learned-Performance Fitness

The system SHALL provide a `LearnedPerformanceFitness` that conforms to the existing `FitnessFunction` protocol and computes a genome's fitness by running K training episodes (where `brain.learn()` IS called) followed by L frozen eval episodes (using the existing `FrozenEvalRunner`). K is read from `evolution.learn_episodes_per_eval` (no CLI override). L is `evolution.eval_episodes_per_eval` if set in YAML, else falls back to the protocol's `episodes` kwarg (which the loop wires from the resolved `evolution_config.episodes_per_eval`, including any `--episodes` CLI override). The score SHALL be the eval-phase success ratio `eval_successes / L`. `LearnedPerformanceFitness` SHALL be a peer of M0's `EpisodicSuccessRate` and the choice between them SHALL be controllable from the CLI. Optionally, `evolution.warm_start_path` MAY name a `.pt` checkpoint (produced by the existing `save_weights` helper); when set, each genome's brain SHALL be loaded with the checkpoint's weights AFTER `encoder.decode` and BEFORE the K train phase, so the K episodes fine-tune the checkpoint under the genome's evolved hyperparameters rather than training from scratch.

#### Scenario: Train phase mutates weights, eval phase does not

- **GIVEN** a fitness evaluation invoked with `learn_episodes_per_eval=K, eval_episodes_per_eval=L` (both > 0)
- **WHEN** the K-then-L flow runs to completion
- **THEN** during the K train episodes the brain's `learn()` method SHALL be called per-step (and the brain's weights SHALL be observed to change between the start of train episode 0 and the start of eval episode 0)
- **AND** during the L eval episodes the brain's `learn()` method SHALL NOT be called (verified via mock as in M0's `test_frozen_eval_runner_never_calls_learn`)

#### Scenario: Eval phase starts with a fresh environment

- **GIVEN** a fitness evaluation invoked with `learn_episodes_per_eval=K > 0`
- **WHEN** the train phase finishes and the eval phase begins
- **THEN** the eval phase SHALL build a fresh environment via `create_env_from_config(sim_config.environment, seed=seed, theme=Theme.HEADLESS)` rather than reusing the train phase's environment
- **AND** the train phase's residual env state (consumed food, agent position, HP, body length) SHALL NOT influence eval-phase episodes
- **AND** the brain — including its learned weights — SHALL carry over from train to eval unchanged

#### Scenario: K=0 raises with a clear error

- **GIVEN** an `EvolutionConfig` with `learn_episodes_per_eval=0`
- **WHEN** `LearnedPerformanceFitness.evaluate` is invoked
- **THEN** a `ValueError` SHALL be raised
- **AND** the error message SHALL state that `EpisodicSuccessRate` is the correct fitness for frozen-weight evaluation

#### Scenario: Missing evolution block raises with a clear error

- **GIVEN** a `SimulationConfig` whose `evolution` field is `None` (i.e. the YAML omitted the `evolution:` block)
- **WHEN** `LearnedPerformanceFitness.evaluate` is invoked
- **THEN** a `ValueError` SHALL be raised before any episode runs
- **AND** the error message SHALL name the missing `evolution:` YAML block as the cause and `learn_episodes_per_eval` as the field that needs setting

#### Scenario: Missing environment or reward block raises with a clear error

- **GIVEN** a `SimulationConfig` whose `environment` field is `None` OR whose `reward` field is `None` (with `evolution` populated so the first guard passes)
- **WHEN** `LearnedPerformanceFitness.evaluate` is invoked
- **THEN** a `ValueError` SHALL be raised before any episode runs, with the same load-time-clear-error contract as the missing `evolution:` block scenario
- **AND** the error message style SHALL mirror M0's `EpisodicSuccessRate.evaluate` (which guards the same two fields), so users see a consistent message across both fitness functions
- **AND** the function SHALL NOT crash with a raw `AttributeError` from `create_env_from_config(None, ...)` or `runner.run(agent, None, ...)`

#### Scenario: eval_episodes_per_eval=None falls back to the episodes kwarg

- **GIVEN** an `EvolutionConfig` with `learn_episodes_per_eval=10, eval_episodes_per_eval=None`
- **AND** the fitness function is invoked with `episodes=5` (the protocol kwarg, which the `EvolutionLoop` wires from its resolved `evolution_config.episodes_per_eval`, including any CLI overrides such as `--episodes`)
- **WHEN** `LearnedPerformanceFitness.evaluate` is invoked
- **THEN** the eval phase SHALL run exactly 5 episodes
- **AND** the returned fitness SHALL be `eval_successes / 5`
- **AND** when `evolution.eval_episodes_per_eval` IS set in YAML, that value SHALL win over the `episodes` kwarg (the YAML field is the explicit-override path for hyperparameter pilots that want different train and eval budgets)

#### Scenario: Score uses TerminationReason for success counting

- **GIVEN** the L eval-phase episodes
- **WHEN** the fitness counts successes
- **THEN** an episode SHALL count as a success if and only if `result.termination_reason == TerminationReason.COMPLETED_ALL_FOOD`
- **AND** the same convention used by M0's `EpisodicSuccessRate` SHALL apply (codebase-wide consistency)

#### Scenario: Fitness selectable via CLI

- **GIVEN** the `scripts/run_evolution.py` CLI
- **WHEN** the user invokes the script with `--fitness learned_performance`
- **THEN** the loop SHALL use `LearnedPerformanceFitness` as the fitness function
- **AND** when `--fitness success_rate` (or omitted), `EpisodicSuccessRate` SHALL be used (M0 default)
- **AND** invoking `--fitness learned_performance` against a config with `learn_episodes_per_eval=0` SHALL be rejected at startup with an error pointing the user to set the field

#### Scenario: Learned-performance fitness requires hyperparameter encoding

- **GIVEN** a `SimulationConfig` with `hyperparam_schema is None` (i.e. a weight-evolution config — the M0 pattern)
- **WHEN** the user invokes the CLI with `--fitness learned_performance`
- **THEN** startup SHALL fail with a clear error stating that learned-performance fitness is only valid for hyperparameter evolution, with the rationale that the weight-encoder + learned-performance combination would amount to Lamarckian inheritance (M3 scope, not M2)
- **AND** the error SHALL point the user to `EpisodicSuccessRate` (frozen-weight) for weight-evolution campaigns or to authoring a `hyperparam_schema:` block to switch to hyperparameter evolution
- **AND** the schema-presence check SHALL fire BEFORE the `learn_episodes_per_eval > 0` check, so a config with neither `hyperparam_schema` nor `learn_episodes_per_eval` set produces the schema-missing error (the more fundamental issue), not the field-missing error

#### Scenario: Warm-start path loads weights before train phase

- **GIVEN** an `EvolutionConfig` with `warm_start_path` set to a valid `.pt` checkpoint produced by `save_weights`
- **AND** a `hyperparam_schema` whose entries name only non-architecture brain-config fields (i.e. fields that do not change tensor shapes — `actor_lr`, `critic_lr`, `gamma`, `entropy_coef`, `num_epochs`, etc.; NOT `actor_hidden_dim`, `lstm_hidden_dim`, `rnn_type`, `num_hidden_layers`)
- **WHEN** `LearnedPerformanceFitness.evaluate` is invoked
- **THEN** after `encoder.decode(genome)` produces the fresh brain and BEFORE the first train episode, `load_weights(brain, warm_start_path)` SHALL be called
- **AND** the K train episodes SHALL run against the warm-started brain (so per-genome learning starts from the checkpointed weights, not random init)
- **AND** the genome's evolved hyperparameters (e.g. `actor_lr`) SHALL govern the K train episodes — optimiser state on the loaded checkpoint is reset by the existing `load_weights` semantics, so each genome fine-tunes from the checkpointed weights under its own hyperparameters
- **AND** when `warm_start_path is None` (the default), behaviour SHALL be identical to today: fresh random weights from `encoder.decode`, no load step

#### Scenario: Warm-start incompatible with architecture-changing schema entries

- **GIVEN** a `SimulationConfig` with `evolution.warm_start_path` set
- **AND** a `hyperparam_schema` containing at least one entry whose `name` references a brain-config field that changes tensor shapes (e.g. `actor_hidden_dim`, `critic_hidden_dim`, `num_hidden_layers`, `actor_num_layers`, `critic_num_layers`, `lstm_hidden_dim`, `rnn_type`)
- **WHEN** the `SimulationConfig` is validated at YAML load time
- **THEN** validation SHALL fail with a clear error naming the offending fields and explaining that warm-start cannot load a fixed-shape checkpoint into a brain whose architecture varies per-genome
- **AND** the error SHALL point the user to either (a) drop the architecture fields from the schema, or (b) drop `warm_start_path` (returning to fresh-init evaluation)

#### Scenario: Warm-start path missing or unreadable raises a clear error

- **GIVEN** an `EvolutionConfig` with `warm_start_path` set to a path that does not exist on disk
- **WHEN** the first `LearnedPerformanceFitness.evaluate` call begins
- **THEN** a `FileNotFoundError` SHALL be raised with a message naming `warm_start_path` and the resolved absolute path
- **AND** the error SHALL fire BEFORE any train or eval episode runs (so a 100-genome × 20-generation run does not waste hours discovering a missing file mid-campaign)

### Requirement: Hyperparameter Schema YAML

The `SimulationConfig` SHALL accept an optional top-level `hyperparam_schema:` list of `ParamSchemaEntry` items. Each entry SHALL declare a `name`, `type` (`Literal["float", "int", "bool", "categorical"]`), and the type-appropriate metadata (`bounds` for float/int, `values` for categorical, `log_scale` for float). The schema SHALL be validated at YAML load time: every `name` SHALL correspond to a real field on the resolved brain config Pydantic model, and every entry's metadata SHALL match its declared `type`. Invalid schemas SHALL be rejected with a clear error before any optimiser code runs.

#### Scenario: hyperparam_schema parses into typed entries

- **GIVEN** a YAML file with a top-level `hyperparam_schema` list containing entries of mixed types
- **WHEN** `load_simulation_config` is called
- **THEN** `SimulationConfig.hyperparam_schema` SHALL be a list of `ParamSchemaEntry` Pydantic instances with the declared fields populated

#### Scenario: Schema name validation rejects non-existent brain config fields

- **GIVEN** a `hyperparam_schema` entry with `name: "actor_hidden_dimm"` (typo) and `brain.name: mlpppo`
- **WHEN** the YAML is loaded
- **THEN** YAML loading SHALL raise a Pydantic `ValidationError`
- **AND** the error message SHALL name the offending field (`actor_hidden_dimm`) and SHALL list at least three valid alternatives from the resolved brain config's `model_fields`

#### Scenario: Schema requires a brain block

- **GIVEN** a YAML with `hyperparam_schema:` populated but no `brain:` block (i.e. `sim_config.brain is None`)
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a `ValidationError` whose message names `brain:` as the required missing block
- **AND** the validator SHALL NOT crash with a raw `AttributeError` on `None.name`

#### Scenario: Schema with unknown brain name fails clearly

- **GIVEN** a YAML with `hyperparam_schema:` populated and `brain.name: bogus_brain` (not in `BRAIN_CONFIG_MAP`)
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a `ValidationError` whose message names the unknown brain
- **AND** the message SHALL list the registered brain names from `BRAIN_CONFIG_MAP` so the user can correct the typo
- **AND** the validator SHALL NOT crash with a raw `KeyError` on the registry lookup

#### Scenario: Schema entry with type-mismatched metadata is rejected

- **GIVEN** a `hyperparam_schema` entry with `type: float` but missing `bounds`
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a `ValidationError`
- **AND** the error SHALL name the entry and the missing field

#### Scenario: Categorical schema requires at least 2 values

- **GIVEN** a `hyperparam_schema` entry with `type: categorical, values: [only_one]`
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a `ValidationError` indicating categorical entries require ≥2 distinct values

#### Scenario: Schema absence preserves M0 dispatch

- **GIVEN** any existing scenario or evolution YAML with no `hyperparam_schema:` key
- **WHEN** loaded and run via `scripts/run_evolution.py`
- **THEN** `SimulationConfig.hyperparam_schema` SHALL be `None`
- **AND** the loop SHALL select an encoder via the M0 `brain.name` → `ENCODER_REGISTRY` lookup, NOT `HyperparameterEncoder`

#### Scenario: Schema travels with the genome to workers

- **GIVEN** a parallel evolution run with `parallel_workers > 1` and a populated `hyperparam_schema`
- **WHEN** the `EvolutionLoop` constructs a `Genome` for the optimiser's sampled parameter vector
- **THEN** the loop SHALL populate `genome.birth_metadata["param_schema"]` with a list of plain dicts (NOT Pydantic model instances) — one per schema entry, produced via `entry.model_dump()` — so the genome pickles cheaply across worker processes without requiring a Pydantic dependency in the worker's decode path
- **AND** the worker's `HyperparameterEncoder.decode` SHALL read the schema from `genome.birth_metadata["param_schema"]` WITHOUT requiring a separate side-channel or re-loading the YAML
- **AND** when `hyperparam_schema is None` on `SimulationConfig`, the loop SHALL leave `genome.birth_metadata` empty (preserves M0 weight-evolution behaviour)
