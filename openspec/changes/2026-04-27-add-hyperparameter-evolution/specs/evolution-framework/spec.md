## ADDED Requirements

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

### Requirement: Learned-Performance Fitness

The system SHALL provide a `LearnedPerformanceFitness` that conforms to the existing `FitnessFunction` protocol and computes a genome's fitness by running K training episodes (where `brain.learn()` IS called) followed by L frozen eval episodes (using the existing `FrozenEvalRunner`). K is read from `evolution.learn_episodes_per_eval` (no CLI override). L is `evolution.eval_episodes_per_eval` if set in YAML, else falls back to the protocol's `episodes` kwarg (which the loop wires from the resolved `evolution_config.episodes_per_eval`, including any `--episodes` CLI override). The score SHALL be the eval-phase success ratio `eval_successes / L`. `LearnedPerformanceFitness` SHALL be a peer of M0's `EpisodicSuccessRate` and the choice between them SHALL be controllable from the CLI.

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
