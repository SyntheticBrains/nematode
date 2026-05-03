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

The system SHALL maintain an append-only `lineage.csv` per evolution run that records, for every evaluated genome, the generation index, genome ID, parent IDs, fitness score, brain type, and the parent genome ID this child inherited weights from (`inherited_from`). The `inherited_from` column SHALL be the empty string when the strategy is `NoInheritance` OR when the child is in generation 0 OR when the child fell back to from-scratch due to a missing parent file. Generation indexing SHALL be 0-based: a run with `generations: G` populates rows for generations `0, 1, …, G-1`. CSV writes SHALL survive process kill and resume — appending continues from the last-written row on reload.

#### Scenario: Lineage CSV records every fitness evaluation

- **GIVEN** an evolution run with `population_size = P` and `generations = G`
- **WHEN** the run completes
- **THEN** `evolution_results/<session_id>/lineage.csv` SHALL contain exactly `P × G` data rows (plus one header row)
- **AND** the `generation` column SHALL take values in the inclusive range `[0, G-1]` with each value appearing exactly `P` times
- **AND** each row SHALL have columns `generation, child_id, parent_ids, fitness, brain_type, inherited_from`

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

#### Scenario: inherited_from column is populated under Lamarckian inheritance

- **GIVEN** a Lamarckian run with `inheritance_elite_count: 1`, `population_size: 4`, `generations: 3`
- **WHEN** the run completes
- **THEN** `lineage.csv` SHALL include an `inherited_from` column in its header
- **AND** all 4 gen-0 rows SHALL have `inherited_from` empty
- **AND** all 4 gen-1 rows SHALL have `inherited_from` equal to the genome ID of gen 0's highest-fitness genome
- **AND** all 4 gen-2 rows SHALL have `inherited_from` equal to the genome ID of gen 1's highest-fitness genome

#### Scenario: inherited_from column is empty under NoInheritance

- **GIVEN** an `inheritance: none` run (or a run with no `inheritance` key — same default)
- **WHEN** the run completes
- **THEN** `lineage.csv` SHALL still include the `inherited_from` column in its header (single fixed schema across runs)
- **AND** every row's `inherited_from` field SHALL be the empty string

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

The `SimulationConfig` SHALL accept an optional `evolution` block; existing scenario configs without an `evolution` block SHALL load unchanged. The `evolution` block SHALL include `inheritance: Literal["none", "lamarckian"]` (default `"none"`) and `inheritance_elite_count: int >= 1` (default 1). Validators SHALL reject the following invalid combinations at YAML load time, before any optimiser code runs:

1. `inheritance != "none"` AND `learn_episodes_per_eval == 0`.
2. `inheritance != "none"` AND `warm_start_path is not None`.
3. `inheritance != "none"` AND `hyperparam_schema is None`.
4. `inheritance != "none"` AND `hyperparam_schema` contains any field in `_ARCHITECTURE_CHANGING_FIELDS` (the existing denylist that M2.10's static warm-start uses).
5. `inheritance_elite_count > population_size`.
6. `inheritance != "none"` AND `inheritance_elite_count != 1` (M3-only restriction; multi-elite parent selection is M4-or-later scope, see "Multi-elite inheritance is rejected in this milestone" scenario).

#### Scenario: Existing scenario config loads without evolution block

- **GIVEN** a scenario YAML file (e.g. `configs/scenarios/foraging/mlpppo_small_oracle.yml`) with no `evolution:` top-level key
- **WHEN** the config is loaded
- **THEN** `SimulationConfig.evolution` SHALL be `None`
- **AND** no error SHALL be raised

#### Scenario: Evolution config block parses into populated EvolutionConfig

- **GIVEN** a YAML file with an `evolution:` block specifying `algorithm`, `population_size`, `generations`, `episodes_per_eval`
- **WHEN** the config is loaded
- **THEN** `SimulationConfig.evolution` SHALL be an `EvolutionConfig` instance with the specified fields populated
- **AND** unspecified fields SHALL fall back to `EvolutionConfig` defaults (including `inheritance: "none"` and `inheritance_elite_count: 1`)

#### Scenario: CLI flags override YAML for the same field

- **GIVEN** a YAML file with `evolution.generations: 50`
- **WHEN** the user invokes `scripts/run_evolution.py --config <yaml> --generations 10`
- **THEN** the loop SHALL run for 10 generations
- **AND** the YAML value SHALL be ignored for that field only
- **AND** the same override pattern SHALL apply to `--inheritance {none,lamarckian}` against `evolution.inheritance`

#### Scenario: Inheritance fields default to no-op

- **GIVEN** a YAML evolution block that omits both `inheritance` and `inheritance_elite_count`
- **WHEN** the config is loaded
- **THEN** `EvolutionConfig.inheritance` SHALL equal `"none"`
- **AND** `EvolutionConfig.inheritance_elite_count` SHALL equal `1`
- **AND** the run SHALL be byte-equivalent to a pre-M3 run with the same other fields

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

The `evaluate` method SHALL additionally accept two optional kwargs that the `EvolutionLoop` MAY pass per-genome (defaulting to `None` when omitted, preserving the existing single-arg call shape):

- `warm_start_path_override: Path | None` — when set, the brain SHALL be loaded from this path INSTEAD of `evolution_config.warm_start_path`. The two SHALL be mutually exclusive at YAML load time (validator on `EvolutionConfig` rejects the combination), so in practice only one of the two paths is ever active. The override exists so the loop's inheritance step can supply per-genome parent checkpoints without mutating run-wide config.
- `weight_capture_path: Path | None` — when set, the post-K-train brain weights SHALL be written to this path via `save_weights` AFTER the K train loop completes and BEFORE the L eval phase begins. This captures the policy as-trained, not as-eval'd. The path's parent directory SHALL be created if missing. When `None` (default), no capture occurs (M2 behaviour).

`EpisodicSuccessRate.evaluate` does NOT accept these kwargs and the loop SHALL NOT pass them when `--fitness success_rate` is selected — the new kwargs are specific to the train+eval split that `LearnedPerformanceFitness` exposes.

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
- **THEN** startup SHALL fail with a clear error stating that learned-performance fitness is only valid for hyperparameter evolution
- **AND** the error SHALL point the user to `EpisodicSuccessRate` (frozen-weight) for weight-evolution campaigns or to authoring a `hyperparam_schema:` block to switch to hyperparameter evolution
- **AND** the schema-presence check SHALL fire BEFORE the `learn_episodes_per_eval > 0` check, so a config with neither `hyperparam_schema` nor `learn_episodes_per_eval` set produces the schema-missing error (the more fundamental issue), not the field-missing error

#### Scenario: Warm-start path loads weights before train phase

- **GIVEN** an `EvolutionConfig` with `warm_start_path` set to a valid `.pt` checkpoint produced by `save_weights`
- **AND** a `hyperparam_schema` whose entries name only non-architecture brain-config fields (i.e. fields that do not change tensor shapes — `actor_lr`, `critic_lr`, `gamma`, `entropy_coef`, `num_epochs`, etc.; NOT `actor_hidden_dim`, `lstm_hidden_dim`, `rnn_type`, `num_hidden_layers`)
- **WHEN** `LearnedPerformanceFitness.evaluate` is invoked
- **THEN** after `encoder.decode(genome)` produces the fresh brain and BEFORE the first train episode, `load_weights(brain, warm_start_path)` SHALL be called
- **AND** the K train episodes SHALL run against the warm-started brain (so per-genome learning starts from the checkpointed weights, not random init)
- **AND** the genome's evolved hyperparameters (e.g. `actor_lr`) SHALL govern the K train episodes — optimiser state on the loaded checkpoint is reset by the existing `load_weights` semantics, so each genome fine-tunes from the checkpointed weights under its own hyperparameters
- **AND** when `warm_start_path is None` AND no `warm_start_path_override` is passed (the default), behaviour SHALL be identical to M0: fresh random weights from `encoder.decode`, no load step

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

#### Scenario: warm_start_path_override takes precedence per-genome

- **GIVEN** an `EvolutionConfig` where `warm_start_path` is `None` and the loop is using a Lamarckian inheritance strategy
- **WHEN** the loop invokes `LearnedPerformanceFitness.evaluate(..., warm_start_path_override=Path("inheritance/gen-002/genome-abc.pt"))` for a particular child
- **THEN** that child's brain SHALL be loaded from the override path (NOT from `evolution_config.warm_start_path`, which is `None` anyway under the validator)
- **AND** OTHER children in the same generation invoked with `warm_start_path_override=Path("inheritance/gen-002/genome-xyz.pt")` SHALL be loaded from THEIR own override path (the override is per-call, not run-wide)
- **AND** when `warm_start_path_override=None` is passed (e.g. for gen-0 children), the brain SHALL be fresh-init from `encoder.decode` (no load step)

#### Scenario: weight_capture_path captures post-train weights before eval

- **GIVEN** a fitness evaluation invoked with `weight_capture_path=Path("inheritance/gen-005/genome-xyz.pt")` and `learn_episodes_per_eval=10`
- **WHEN** the K=10 train phase completes
- **THEN** `save_weights(brain, weight_capture_path)` SHALL be called BEFORE the L eval phase begins
- **AND** the file at `weight_capture_path` SHALL be a valid `.pt` checkpoint that round-trips via `load_weights` into a brain whose first-step action matches the captured brain's first-step action under identical sensory input
- **AND** the capture SHALL occur exactly once per genome evaluation, regardless of K (including K=1)
- **AND** when `weight_capture_path=None` (default), no file SHALL be written

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

### Requirement: Inheritance Strategy

The system SHALL provide an `InheritanceStrategy` Protocol in `quantumnematode/evolution/inheritance.py` with four methods (`select_parents`, `assign_parent`, `checkpoint_path`, `kind`) and at least three concrete implementations: `NoInheritance` (the default no-op), `LamarckianInheritance(elite_count: int = 1)` (per-genome weight inheritance), and `BaldwinInheritance` (per-genome trait inheritance — recorded in lineage, no weight checkpoints written).

The `kind()` method SHALL return one of three string literals so the loop can branch on intent rather than `isinstance` checks: `"none"` (no inheritance configured — `NoInheritance`), `"weights"` (per-genome trained-weight checkpoints flow between generations — `LamarckianInheritance`), or `"trait"` (per-genome elite-parent ID flows in lineage but no weight checkpoints are captured — `BaldwinInheritance`). The loop SHALL gate weight-IO code paths (capture, GC, warm-start) on `kind() == "weights"` and SHALL gate elite-ID lineage tracking on `kind() != "none"`.

When `kind() == "weights"` is active, the `EvolutionLoop` SHALL: (1) capture each genome's post-K-train brain weights to a per-genome `.pt` file via `LearnedPerformanceFitness`'s `weight_capture_path` kwarg; (2) after each generation completes its `optimizer.tell` call, ask the strategy to select parents from the prior generation's `(genome, fitness)` pairs; (3) before the next generation's fitness evaluation, warm-start each child's brain from its assigned parent's checkpoint via `LearnedPerformanceFitness`'s `warm_start_path_override` kwarg; (4) garbage-collect any per-genome checkpoint that is not selected as a parent for the next generation. Steady-state disk usage SHALL be at most `2 * inheritance_elite_count` `.pt` files across the entire run.

The `EvolutionLoop` SHALL persist the resolved `inheritance` value (the literal `"none"` / `"lamarckian"` / `"baldwin"` string, not the strategy instance) and the selected parent IDs in its checkpoint pickle dict, so that resume-time validation can reject mismatched inheritance settings (see "Resume rejects mismatched inheritance setting" scenario below).

The strategy SHALL be selectable via `evolution.inheritance: Literal["none", "lamarckian", "baldwin"]` in YAML and overridable via the `--inheritance` CLI flag on `scripts/run_evolution.py`. The `evolution.inheritance_elite_count` field is structurally `int >= 1` (default 1) but the validator SHALL reject any value other than 1 when `inheritance: lamarckian` (single-elite-broadcast only — multi-elite parent selection is reserved for future strategies). The `inheritance_elite_count` field is unused under `inheritance: baldwin` (Baldwin is conceptually single-elite by construction since trait inheritance flows through TPE's posterior, which biases sampling toward the prior elite). When `inheritance: none` (the default), the loop, fitness, and lineage code paths SHALL be byte-equivalent to a frozen-weight evolution baseline — no `inheritance/` directory created, no per-genome checkpoints written, no GC step performed, and `inherited_from` empty in lineage.

#### Scenario: Lamarckian inheritance is selectable via config and CLI

- **GIVEN** an `EvolutionConfig` with `inheritance: lamarckian` and `inheritance_elite_count: 1`
- **WHEN** the loop runs for ≥ 2 generations
- **THEN** every child genome of generation N (for N ≥ 1) SHALL have its brain loaded with the previous generation's selected-parent weights via `load_weights` BEFORE the K train phase begins
- **AND** the parent SHALL be the prior generation's single highest-fitness genome (broken by genome ID lexicographic order on ties)
- **AND** the CLI flag `--inheritance lamarckian` SHALL override the YAML field with the same behaviour
- **AND** `--inheritance none` SHALL force the no-op path even when the YAML sets `lamarckian`

#### Scenario: First generation runs from-scratch under any inheritance strategy

- **GIVEN** any inheritance config (`lamarckian`, `baldwin`, or `none`) with any `inheritance_elite_count`
- **WHEN** generation 0 evaluates
- **THEN** every gen-0 child SHALL be from-scratch — `LearnedPerformanceFitness.evaluate` SHALL be invoked with `warm_start_path_override=None` for every gen-0 genome
- **AND** the `lineage.csv` `inherited_from` column SHALL be empty for every gen-0 row regardless of strategy
- **AND** the gen-0 fitness scores SHALL be bit-for-bit identical between `inheritance: none` and `inheritance: baldwin` runs with the same seed (Baldwin only differs from no-op by recording `inherited_from` in gen-1+; gen-0 paths are identical)
- **AND** the gen-0 fitness scores SHALL be bit-for-bit identical between `inheritance: lamarckian` and `inheritance: none` runs with the same seed (modulo the side-effect `save_weights` write that captures each genome's post-train weights for gen-1 to inherit from — fitness arithmetic is unaffected by that write)

#### Scenario: Per-genome weight checkpoints are captured and garbage-collected

- **GIVEN** a Lamarckian run with `inheritance_elite_count: 1`, `population_size: 12`, `generations: 5` (i.e. generations 0 through 4 inclusive)
- **WHEN** the run completes
- **THEN** during generation N's evaluation, exactly 12 `.pt` files SHALL have been written under `<output_dir>/inheritance/gen-{N:03d}/`
- **AND** after generation N's `optimizer.tell` returns and the strategy selects the next-generation parent, the GC step SHALL delete all 11 non-selected files in `gen-{N:03d}/`; additionally when N ≥ 1 it SHALL delete all remaining files in `gen-{N-1:03d}/` (whose children have just finished evaluating, so those parent checkpoints are no longer needed). For N = 0 the second clause no-ops because no `gen-{-1}/` directory exists.
- **AND** at the moment generation N+1's evaluation begins, exactly one file SHALL exist in `gen-{N:03d}/` (the selected parent for the about-to-evaluate children)
- **AND** when the run completes after generation 4 (the final generation), the loop SHALL still run `select_parents` on gen 4's results and the GC step SHALL still delete gen 3's surviving parent — so the only surviving file SHALL be the selected parent of the final generation, under `inheritance/gen-004/`. This file is intentionally NOT deleted by GC: it is the final winner's trained policy, available for forensic inspection or downstream warm-start by future work.
- **AND** if the run terminates via `early_stop_on_saturation` (rather than reaching `generations`), the same invariant holds: the loop runs `select_parents` + GC for the final-evaluated generation BEFORE the early-stop break (per the early-stop ordering rule — break fires at end of body, after the GC guard), so the surviving file is the elite of the early-stop generation rather than `gen-004`

#### Scenario: Inheritance requires a training phase

- **GIVEN** a YAML with `evolution.inheritance: lamarckian` (or `baldwin`) AND `evolution.learn_episodes_per_eval: 0`
- **WHEN** the YAML is loaded via `load_simulation_config`
- **THEN** loading SHALL raise a Pydantic `ValidationError`
- **AND** the error message SHALL state that inheritance requires a non-zero train phase (Lamarckian needs trained weights to inherit; Baldwin's whole premise is that learning shapes the gen-N elite that becomes the prior for gen-N+1)
- **AND** the message SHALL point the user to either set `learn_episodes_per_eval > 0` or set `inheritance: none`

#### Scenario: Inheritance is mutually exclusive with static warm-start

- **GIVEN** a YAML with `evolution.inheritance: lamarckian` (or `baldwin`) AND `evolution.warm_start_path: /some/path.pt`
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a Pydantic `ValidationError`
- **AND** the error message SHALL state that `warm_start_path` (run-wide static checkpoint) and `inheritance` (per-genome dynamic checkpoint) both load weights into the same brain slot before the K train phase, and that exactly one MAY be set. For `inheritance: baldwin` the rule is enforced even though Baldwin doesn't use weight checkpoints — the rule prevents future Baldwin variants (which might combine static warm-start with trait inheritance) from being introduced silently.
- **AND** the message SHALL point the user to drop one of the two

#### Scenario: Inheritance requires hyperparameter encoding

- **GIVEN** a YAML with `evolution.inheritance: lamarckian` (or `baldwin`) AND `hyperparam_schema is None` (i.e. a weight-evolution config)
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a Pydantic `ValidationError`
- **AND** the error message SHALL state that inheritance over weight encoders would double-count weights as both genome and substrate (Lamarckian) or have nothing meaningful to "inherit" since the genome IS the weights (Baldwin)
- **AND** the message SHALL point the user to either drop `inheritance` (returning to weight evolution) or add a `hyperparam_schema` (switching to hyperparameter evolution + inheritance)

#### Scenario: Lamarckian inheritance incompatible with architecture-changing schema entries

- **GIVEN** a YAML with `evolution.inheritance: lamarckian` AND a `hyperparam_schema` containing at least one entry whose `name` references a brain-config field that changes tensor shapes (e.g. `actor_hidden_dim`, `lstm_hidden_dim`, `rnn_type`, `actor_num_layers` — the same set the existing static warm-start rejects)
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a Pydantic `ValidationError` naming the offending fields
- **AND** the error SHALL share the same `_ARCHITECTURE_CHANGING_FIELDS` denylist that the existing static warm-start uses (single source of truth)
- **AND** the message SHALL explain that per-genome checkpoints cannot be loaded into a child whose architecture differs from the parent's
- **AND** this rejection SHALL apply to `inheritance: lamarckian` ONLY; `inheritance: baldwin` does NOT load weights and therefore SHALL accept architecture-changing schema entries (so a future Baldwin arm can evolve `actor_hidden_dim` etc. if desired)

#### Scenario: Multi-elite inheritance is rejected for Lamarckian

- **GIVEN** a YAML with `evolution.inheritance: lamarckian` AND `evolution.inheritance_elite_count: 2` (or any value other than 1, including values that would also exceed `population_size`)
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a Pydantic `ValidationError` stating that `inheritance_elite_count` MUST be 1 when `inheritance: lamarckian`
- **AND** the message SHALL state that multi-elite parent selection (round-robin or tournament) is not currently supported for Lamarckian and that the field exists structurally so future strategies can populate it without a config-schema migration
- **AND** the field's `Field(default=1, ge=1)` constraint SHALL still permit values >1 in the schema (so a future strategy can lift this validator without breaking config-load semantics); the rejection is enforced by the model validator only when `inheritance == "lamarckian"`
- **AND** the rule SHALL NOT apply when `inheritance: baldwin` — Baldwin doesn't use the field; setting `inheritance_elite_count: 5` under Baldwin is silently ignored (documented in the field comment)
- **AND** a separate validator SHALL also reject `inheritance_elite_count > population_size` (e.g. 20 > 12) with a distinct error — this rule is independent of strategy and applies to all inheritance settings

#### Scenario: Inheritance defaults preserve frozen-weight evolution byte-equivalently

- **GIVEN** any existing evolution YAML with no `inheritance:` key under `evolution:`
- **WHEN** loaded and run via `scripts/run_evolution.py`
- **THEN** `EvolutionConfig.inheritance` SHALL be `"none"` and `EvolutionConfig.inheritance_elite_count` SHALL be `1`
- **AND** the loop SHALL construct a `NoInheritance` strategy whose `kind()` returns `"none"`
- **AND** no `inheritance/` directory SHALL be created under the output directory
- **AND** `LearnedPerformanceFitness.evaluate` SHALL be invoked with `warm_start_path_override=None` and `weight_capture_path=None` for every genome
- **AND** the `lineage.csv` `inherited_from` column SHALL be empty for every row

#### Scenario: Resume from checkpoint preserves selected parent IDs

- **GIVEN** a Lamarckian run that wrote a checkpoint at end of generation N with `_selected_parent_ids = [pid_a]`
- **WHEN** the run resumes via `--resume <checkpoint>`
- **THEN** the loaded loop's `_selected_parent_ids` SHALL equal `[pid_a]`
- **AND** generation N+1's children SHALL warm-start from `inheritance/gen-{N:03d}/genome-{pid_a}.pt`
- **AND** if that file is unexpectedly missing on resume, the affected children SHALL fall back to from-scratch with a `logger.warning` (the loop SHALL NOT crash)

#### Scenario: Checkpoint version compatibility

- **GIVEN** a checkpoint pickle file whose `checkpoint_version` field is older than the current `CHECKPOINT_VERSION`
- **WHEN** the loop attempts to load it via `--resume`
- **THEN** loading SHALL raise the existing version-mismatch error
- **AND** the user SHALL be advised to start the run fresh (no automated converter is provided)

#### Scenario: Resume rejects mismatched inheritance setting

- **GIVEN** a checkpoint produced under one inheritance setting (e.g. `inheritance: lamarckian`) AND a resume invocation whose resolved `EvolutionConfig.inheritance` is different (e.g. `inheritance: none` or `inheritance: baldwin`, whether via YAML or `--inheritance` CLI override)
- **WHEN** the loop attempts to resume
- **THEN** loading SHALL raise a clear error stating that the resumed run's inheritance setting differs from the original and that mid-run inheritance changes are not supported
- **AND** the message SHALL list both the checkpoint's recorded inheritance and the resolved current value so the user can decide which to keep
- **AND** the rejection SHALL fire BEFORE the loop reaches the first generation iteration (so an inadvertent CLI override doesn't waste compute on a corrupted run)
- **AND** this rejection SHALL apply to `--resume` invocations only — for fresh runs, `--inheritance` overrides the YAML field normally per the "Lamarckian inheritance is selectable via config and CLI" scenario above (the `--inheritance` flag itself is not broken, it just cannot change a run's inheritance mid-flight)

#### Scenario: CLI rejects inheritance + --fitness success_rate at startup

- **GIVEN** a YAML or CLI invocation with `evolution.inheritance != "none"` AND `--fitness success_rate` (the default when `--fitness` is omitted)
- **WHEN** `scripts/run_evolution.py` parses arguments and resolves the `EvolutionConfig`
- **THEN** the script SHALL exit with code 1 BEFORE constructing the optimizer or the loop
- **AND** the error message SHALL state that inheritance writes per-genome weight checkpoints (Lamarckian) or records elite-parent lineage from a trained-elite-fitness signal (Baldwin) after each train phase, and `EpisodicSuccessRate` is frozen-weight with no train phase
- **AND** the message SHALL point the user to `--fitness learned_performance` or to setting `inheritance: none`
- **AND** the guard fires in the CLI (not in `EvolutionConfig._validate_inheritance`) because the `--fitness` flag is not visible to the Pydantic validator — without this guard, the loop would compute `weight_capture_path` for every child (Lamarckian) or attempt to read fitnesses from a frozen-eval pass that doesn't reflect any learning (Baldwin), corrupting the signal in either case

#### Scenario: kind() Protocol method gates loop behaviour

- **GIVEN** an `InheritanceStrategy` instance
- **WHEN** the loop calls `strategy.kind()`
- **THEN** the return value SHALL be exactly one of `"none"`, `"weights"`, or `"trait"`
- **AND** `NoInheritance.kind()` SHALL return `"none"`
- **AND** `LamarckianInheritance.kind()` SHALL return `"weights"`
- **AND** `BaldwinInheritance.kind()` SHALL return `"trait"`
- **AND** the loop's `_inheritance_active()` helper (which decides whether to compute weight checkpoint paths and run the GC step) SHALL evaluate `strategy.kind() == "weights"`
- **AND** the loop's `_inheritance_records_lineage()` helper (which decides whether to write `inherited_from` in lineage rows AND whether to call `select_parents` to update `_selected_parent_ids`) SHALL evaluate `strategy.kind() != "none"`
- **AND** the post-`tell` block in the main loop SHALL use TWO distinct guards: the lineage-tracking guard (`_inheritance_records_lineage()`) wraps the `select_parents` call and the `_selected_parent_ids` assignment so it fires for both Lamarckian and Baldwin; the weight-IO GC guard (`_inheritance_active()`) wraps the two `_gc_inheritance_dir` calls so it fires only for Lamarckian

### Requirement: Baldwin Inheritance Strategy

The system SHALL provide a `BaldwinInheritance` implementation in `quantumnematode/evolution/inheritance.py` whose `kind()` returns `"trait"`. Baldwin inheritance is mechanically a no-op on the weight-IO path (no per-genome `.pt` files written, no GC, no warm-start) but the loop SHALL track the prior generation's elite genome ID (top fitness, lex-tie-broken — same selection rule as Lamarckian) and write it to the lineage CSV's `inherited_from` column for every child of the next generation. The hyperparameter genome continues to evolve via TPE; the elite-ID lineage trace exists so post-pilot analysis can identify which prior-gen elite each child shares hyperparameters with via TPE's posterior.

The `BaldwinInheritance` constructor SHALL take no required arguments. The `inheritance_elite_count` config field SHALL be ignored under Baldwin (the field exists for forward-compatibility with future multi-elite Baldwin variants but is unused in this milestone).

`BaldwinInheritance.select_parents()` SHALL return a single-element list `[best_genome_id]` containing the prior generation's elite genome ID (top fitness, lex-tie-broken on `genome_id` — the same selection rule as `LamarckianInheritance` with `elite_count=1`). The loop reuses this ID to populate the lineage CSV's `inherited_from` column for every child of the next generation, even though no on-disk checkpoint is created. `BaldwinInheritance.assign_parent()` SHALL return `None` (no per-child parent assignment for warm-start). `BaldwinInheritance.checkpoint_path()` SHALL return `None` (no on-disk checkpoint substrate).

The genetic-assimilation question Baldwin tests is whether evolution under TPE produces a hyperparameter genome that biases the brain to learn fast from random init. The post-pilot script `scripts/campaigns/baldwin_f1_postpilot_eval.py` (NOT part of the loop's runtime contract — a forensic script) SHALL re-evaluate the elite genome with K=0 to test whether the bias has been encoded into the genome itself.

#### Scenario: Baldwin is selectable via config and CLI

- **GIVEN** an `EvolutionConfig` with `inheritance: baldwin`
- **WHEN** the loop runs for ≥ 2 generations
- **THEN** every child genome of every generation SHALL have `LearnedPerformanceFitness.evaluate` invoked with `warm_start_path_override=None` AND `weight_capture_path=None` (no weight IO)
- **AND** no `inheritance/` directory SHALL be created under the output directory
- **AND** every child of generation N (for N ≥ 1) SHALL have its lineage row's `inherited_from` populated with the prior generation's elite genome ID
- **AND** the CLI flag `--inheritance baldwin` SHALL override the YAML field with the same behaviour
- **AND** `--inheritance none` SHALL force the no-op path even when the YAML sets `baldwin`

#### Scenario: Baldwin records lineage but creates no inheritance directory

- **GIVEN** a Baldwin run with `population_size: 12`, `generations: 5`
- **WHEN** the run completes
- **THEN** the output directory SHALL NOT contain an `inheritance/` subdirectory
- **AND** the output directory SHALL contain `lineage.csv`
- **AND** in `lineage.csv` every gen-0 row SHALL have empty `inherited_from`
- **AND** in `lineage.csv` every gen-1+ row SHALL have non-empty `inherited_from` equal to a single genome ID (the prior generation's elite, broadcast to all 12 children of the next generation)
- **AND** the elite ID for generation N's children SHALL be the gen-(N-1) genome ID with the highest fitness, with lexicographic tie-breaking on `genome_id`

#### Scenario: Baldwin requires a training phase

- **GIVEN** a YAML with `evolution.inheritance: baldwin` AND `evolution.learn_episodes_per_eval: 0`
- **WHEN** the YAML is loaded via `load_simulation_config`
- **THEN** loading SHALL raise a Pydantic `ValidationError`
- **AND** the error message SHALL state that Baldwin requires a non-zero train phase because the whole premise is that lifetime learning shapes the gen-N elite that becomes the prior for gen-N+1
- **AND** the message SHALL point the user to either set `learn_episodes_per_eval > 0` or set `inheritance: none`

#### Scenario: Baldwin is mutually exclusive with static warm-start

- **GIVEN** a YAML with `evolution.inheritance: baldwin` AND `evolution.warm_start_path: /some/path.pt`
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a Pydantic `ValidationError`
- **AND** the error message SHALL state that Baldwin under static warm-start would mean every child starts from the same fixed checkpoint — collapsing the Baldwin signal because all children share the same starting point regardless of the elite's evolved hyperparameters
- **AND** the message SHALL point the user to drop one of the two

#### Scenario: Baldwin requires hyperparameter encoding

- **GIVEN** a YAML with `evolution.inheritance: baldwin` AND `hyperparam_schema is None`
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a Pydantic `ValidationError`
- **AND** the error message SHALL state that Baldwin needs a hyperparameter genome to evolve — without one there is no trait substrate to inherit
- **AND** the message SHALL point the user to either drop `inheritance` or add a `hyperparam_schema`

#### Scenario: Baldwin permits architecture-changing schema entries

- **GIVEN** a YAML with `evolution.inheritance: baldwin` AND a `hyperparam_schema` containing an architecture-changing field (e.g. `actor_hidden_dim`)
- **WHEN** the YAML is loaded
- **THEN** loading SHALL succeed (no `ValidationError` raised)
- **AND** the loop SHALL run normally; each child constructs a fresh brain at the genome's evolved architecture (no weight checkpoints to load means no shape-mismatch concern)
- **AND** this is the documented difference between Baldwin and Lamarckian validators

### Requirement: Evolvable LSTMPPO weight_init_scale

The system SHALL expose a `weight_init_scale: float` field on `LSTMPPOBrainConfig` (default `1.0`, validator-bounded to `[0.1, 5.0]`) that scales the orthogonal-init `gain` for the actor's hidden Linear layers and ALL of the critic's Linear layers. The brain's `_initialize_weights` method SHALL compute `hidden_gain = sqrt(2) * weight_init_scale` and pass that gain to `nn.init.orthogonal_` for each in-scope layer. The actor's output Linear layer SHALL be initialised with a fixed `gain=0.01` (the standard PPO small-init trick for stable initial policy) regardless of `weight_init_scale`. The LSTM/GRU module (`self.rnn`) SHALL NOT be touched by `_initialize_weights`; it uses PyTorch's default initialisation.

The default `weight_init_scale=1.0` SHALL be byte-equivalent to the pre-existing standard PPO init under the same seed (i.e. `hidden_gain = sqrt(2) * 1.0 = sqrt(2)`, matching the prior fixed `gain=np.sqrt(2)` call). The field SHALL be addable to a Baldwin pilot's `hyperparam_schema` so TPE can evolve it within the schema-bounded range (typically a tighter sub-range of the validator bounds, e.g. `[0.5, 2.0]`).

The companion existing field `entropy_decay_episodes` (already on `LSTMPPOBrainConfig`, default 500) SHALL be likewise eligible for inclusion in a Baldwin pilot's `hyperparam_schema` without any brain-side changes — it is exposed for evolution by virtue of being a tagged config field, no new initialisation logic required.

#### Scenario: Default scale produces standard-PPO-init-equivalent tensors

- **GIVEN** an `LSTMPPOBrainConfig` with `weight_init_scale=1.0` (the default) and a fixed seed
- **WHEN** the brain is constructed
- **THEN** the actor's hidden Linear and critic Linear weight tensors SHALL be bit-identical to a direct call to `nn.init.orthogonal_(weight, gain=np.sqrt(2))` against the same seeded RNG state
- **AND** the actor's output Linear weight tensor SHALL be initialised with `gain=0.01` (small-init PPO trick) — its sample standard deviation SHALL be far below the hidden-layer std

#### Scenario: Larger scale doubles hidden-layer std without affecting output layer or RNN

- **GIVEN** two `LSTMPPOBrainConfig` instances under the same seed, one with `weight_init_scale=1.0` and one with `weight_init_scale=2.0`
- **WHEN** both brains are constructed
- **THEN** the ratio of the actor's hidden Linear weight std (2.0-config / 1.0-config) SHALL equal 2.0 within numerical tolerance for every hidden layer
- **AND** the ratio for the critic's Linear weights SHALL also equal 2.0
- **AND** the actor's output Linear weight tensors SHALL have approximately equal std (ratio ≈ 1.0) — the output layer is independent of `weight_init_scale`
- **AND** the LSTM/GRU module's parameter tensors SHALL be bit-identical between the two configs (the RNN module is outside the `_initialize_weights` scope)

#### Scenario: Validator rejects out-of-range scales

- **GIVEN** an `LSTMPPOBrainConfig` definition with `weight_init_scale=0.05` (below the 0.1 lower bound)
- **WHEN** the config is loaded
- **THEN** Pydantic SHALL raise a `ValidationError` mentioning `weight_init_scale`
- **AND** the same SHALL hold for `weight_init_scale=5.1` (above the 5.0 upper bound)
- **AND** the boundary values 0.1 and 5.0 SHALL be accepted (inclusive bounds)

### Requirement: Early Stop on Saturation

The system SHALL provide an `evolution.early_stop_on_saturation: int | None` config field (default `None`) that exits the evolution loop when `best_fitness` has not improved for the configured number of consecutive generations. The field SHALL be settable via YAML and overridable via the `--early-stop-on-saturation N` CLI flag on `scripts/run_evolution.py`.

When the field is `None` (the default), the loop SHALL run for the full `generations` budget regardless of fitness trajectory, preserving existing behaviour byte-equivalently.

When the field is set to a positive integer N, the loop SHALL track the previous generation's `best_fitness` after each `optimizer.tell` call. If the current generation's `best_fitness` is strictly greater than the prior generation's, the counter SHALL reset to 0; otherwise the counter SHALL increment. When the counter reaches N, the loop SHALL log "Early-stop: best_fitness has not improved for N generations (last improvement at gen X)" and break out of the main loop. The `lineage.csv`, `history.csv`, and final `best_params.json` SHALL reflect the truncated run (no padding); the aggregator handles cross-arm length normalisation at analysis time.

The early-stop state SHALL be persisted in the checkpoint pickle as **two** fields so resume preserves the saturation-tracking state byte-equivalently:

- `gens_without_improvement: int` — the consecutive non-improving-generation counter.
- `last_best_fitness: float | None` — the previous generation's recorded best (`None` until generation 1 completes its bootstrap).

The `CHECKPOINT_VERSION` SHALL be bumped (currently to `3`) to reflect the new pickle fields; older checkpoints (any `checkpoint_version` ≠ the current value) SHALL be rejected with a clear error per the existing version-mismatch scenario. On resume, the loader SHALL validate that BOTH `gens_without_improvement` and `last_best_fitness` are present in the payload (a v3 payload missing either is structurally inconsistent — possibly hand-edited or written by a buggy older revision claiming to be v3) and SHALL raise a descriptive `ValueError` naming the missing key rather than silently defaulting to `0` / `None`. Only after both keys are validated SHALL the loader assign them to `self._gens_without_improvement` and `self._last_best_fitness`.

#### Scenario: Default behaviour preserves full-budget runs

- **GIVEN** an evolution YAML with no `early_stop_on_saturation` key
- **WHEN** the loop runs
- **THEN** `EvolutionConfig.early_stop_on_saturation` SHALL be `None`
- **AND** the loop SHALL iterate exactly `generations` times regardless of fitness
- **AND** the lineage and history CSVs SHALL contain exactly `generations` rows (per arm) — byte-equivalent to runs where the field is absent

#### Scenario: Early-stop fires after N consecutive non-improving generations

- **GIVEN** an evolution YAML with `evolution.early_stop_on_saturation: 3` and `generations: 20`, on a run where `best_fitness` per generation is `[0.3, 0.5, 0.7, 0.85, 0.95, 0.95, 0.95, 0.95, ...]` (strict-increasing through gen 5, then plateauing at 0.95 from gen 6 onwards)
- **WHEN** the loop runs and the comparison rule is "current best_fitness is strictly greater than the previous generation's best_fitness"
- **THEN** the counter SHALL be 0 at gen 1 (no previous), 0 at gens 2-5 (all strict improvements), 1 at gen 6, 2 at gen 7, 3 at gen 8 (three consecutive non-improving generations after the last strict improvement at gen 5)
- **AND** when the counter reaches 3 at gen 8, the loop SHALL break AFTER gen 8's full evaluation completes (i.e. gen 8 IS recorded in lineage; gen 9 is NOT)
- **AND** the log SHALL contain "Early-stop: best_fitness has not improved for 3 generations (last improvement at gen 5)"
- **AND** the lineage CSV SHALL contain exactly `8 * population_size` rows
- **AND** the history CSV SHALL contain exactly 8 rows
- **AND** the existing post-loop final `_save_checkpoint()` SHALL persist the early-stopped state — no additional save call inside the loop is needed, because control flows out of the `while` via `break` and through the existing post-loop save site
- **AND** the persisted state SHALL include `_generation` (set to the post-evaluation increment value, i.e. 9 in this scenario), `_gens_without_improvement` (the value that triggered the break, i.e. 3), and `_last_best_fitness` (the plateau value, i.e. 0.95)
- **AND** subsequent `--resume` SHALL re-enter the main loop because `_generation (9) < cfg.generations (20)`; the run continues with the saturation counter intact, and if `--early-stop-on-saturation N` is passed again on resume the gate retriggers within zero or one further generations

#### Scenario: Counter resets on any strict improvement

- **GIVEN** an evolution YAML with `early_stop_on_saturation: 5` and `generations: 9`
- **WHEN** the loop runs and `best_fitness` per generation is `[0.3, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5]`
- **THEN** the counter SHALL be 0 at gen 1 (no previous), 0 at gen 2 (strict improvement), 1 at gen 3, 2 at gen 4, 0 at gen 5 (strict improvement, counter reset), 1 at gen 6, 2 at gen 7, 3 at gen 8, 4 at gen 9
- **AND** the loop SHALL NOT trigger early-stop because the counter never reaches 5 within this trajectory; the run completes the full `generations: 9` budget normally

#### Scenario: CLI override for early-stop

- **GIVEN** an evolution YAML with `early_stop_on_saturation: 3` AND a CLI invocation with `--early-stop-on-saturation 10`
- **WHEN** `scripts/run_evolution.py` parses arguments and resolves the config
- **THEN** the resolved `EvolutionConfig.early_stop_on_saturation` SHALL be `10` (CLI wins per the existing CLI-override convention)
- **AND** the YAML's `3` SHALL be ignored
- **AND** an explicit `--early-stop-on-saturation 0` (or any non-positive value) SHALL be rejected by argparse before reaching the loop

#### Scenario: Resume preserves the early-stop counter

- **GIVEN** a Lamarckian or Baldwin run with `early_stop_on_saturation: 3` that wrote a checkpoint at end of generation N with `gens_without_improvement: 2` and `last_best_fitness: 0.95`
- **WHEN** the run resumes via `--resume <checkpoint>`
- **THEN** the loaded loop's `_gens_without_improvement` SHALL equal `2` and `_last_best_fitness` SHALL equal `0.95`
- **AND** if generation N+1's `best_fitness <= 0.95` (no strict improvement), the counter SHALL increment to 3 and the loop SHALL early-stop after gen N+1
- **AND** if generation N+1's `best_fitness > 0.95` (strict improvement), the counter SHALL reset to 0 and the loop SHALL continue to the full `generations` budget

#### Scenario: Early-stop applies to all inheritance strategies

- **GIVEN** the same `early_stop_on_saturation: 3` setting under three different inheritance strategies (`none`, `lamarckian`, `baldwin`)
- **WHEN** all three runs use the same seed and trajectory
- **THEN** all three SHALL trigger early-stop at the same generation (the trigger is a function of `best_fitness` only, not of the inheritance strategy)
- **AND** the early-stop break SHALL fire AFTER both inheritance guards run for the final-evaluated generation
- **AND** under Baldwin, the lineage-tracking guard (`_inheritance_records_lineage()`) SHALL run `select_parents` and update `_selected_parent_ids` for the final-evaluated generation, preserving lineage CSV correctness AND keeping the resume invariant intact in case the user resumes the early-stopped run
- **AND** under Lamarckian, the weight-IO GC guard (`_inheritance_active()`) SHALL still run for the final-evaluated generation, preserving the surviving elite checkpoint per the "Per-genome weight checkpoints are captured and garbage-collected" scenario above
