## ADDED Requirements

### Requirement: Inheritance Strategy

The system SHALL provide an `InheritanceStrategy` Protocol in `quantumnematode/evolution/inheritance.py` with three methods (`select_parents`, `assign_parent`, `checkpoint_path`) and at least two concrete implementations: `NoInheritance` (the default no-op) and `LamarckianInheritance(elite_count: int = 1)`. When a non-`NoInheritance` strategy is active, the `EvolutionLoop` SHALL: (1) capture each genome's post-K-train brain weights to a per-genome `.pt` file via `LearnedPerformanceFitness`'s `weight_capture_path` kwarg; (2) after each generation completes its `optimizer.tell` call, ask the strategy to select parents from the prior generation's `(genome, fitness)` pairs; (3) before the next generation's fitness evaluation, warm-start each child's brain from its assigned parent's checkpoint via `LearnedPerformanceFitness`'s `warm_start_path_override` kwarg; (4) garbage-collect any per-genome checkpoint that is not selected as a parent for the next generation. Steady-state disk usage SHALL be at most `2 * inheritance_elite_count` `.pt` files across the entire run.

The strategy SHALL be selectable via `evolution.inheritance: Literal["none", "lamarckian"]` in YAML and overridable via the `--inheritance` CLI flag on `scripts/run_evolution.py`. The number of elites SHALL be configurable via `evolution.inheritance_elite_count: int >= 1` (default 1). When `inheritance: none` (the default), the loop, fitness, and lineage code paths SHALL be byte-equivalent to the M2.12 baseline — no `inheritance/` directory created, no per-genome checkpoints written, no GC step performed.

#### Scenario: Lamarckian inheritance is selectable via config and CLI

- **GIVEN** an `EvolutionConfig` with `inheritance: lamarckian` and `inheritance_elite_count: 1`
- **WHEN** the loop runs for ≥ 2 generations
- **THEN** every child genome of generation N (for N ≥ 1) SHALL have its brain loaded with the previous generation's selected-parent weights via `load_weights` BEFORE the K train phase begins
- **AND** the parent SHALL be the prior generation's single highest-fitness genome (broken by genome ID lexicographic order on ties)
- **AND** the CLI flag `--inheritance lamarckian` SHALL override the YAML field with the same behaviour
- **AND** `--inheritance none` SHALL force the no-op path even when the YAML sets `lamarckian`

#### Scenario: First generation runs from-scratch under any inheritance strategy

- **GIVEN** a Lamarckian config with any `inheritance_elite_count`
- **WHEN** generation 0 evaluates
- **THEN** every gen-0 child SHALL be from-scratch — `LearnedPerformanceFitness.evaluate` SHALL be invoked with `warm_start_path_override=None` for every gen-0 genome
- **AND** the `lineage.csv` `inherited_from` column SHALL be empty for every gen-0 row
- **AND** the gen-0 fitness scores SHALL be bit-for-bit identical to an `inheritance: none` run with the same seed (modulo the side-effect `save_weights` write that captures each genome's post-train weights for gen-1 to inherit from — fitness arithmetic is unaffected by that write)

#### Scenario: Per-genome weight checkpoints are captured and garbage-collected

- **GIVEN** a Lamarckian run with `inheritance_elite_count: 1`, `population_size: 12`, `generations: 5` (i.e. generations 0 through 4 inclusive)
- **WHEN** the run completes
- **THEN** during generation N's evaluation, exactly 12 `.pt` files SHALL have been written under `<output_dir>/inheritance/gen-{N:03d}/`
- **AND** after generation N's `optimizer.tell` returns and the strategy selects the next-generation parent, the GC step SHALL delete all 11 non-selected files in `gen-{N:03d}/`; additionally when N ≥ 1 it SHALL delete all remaining files in `gen-{N-1:03d}/` (whose children have just finished evaluating, so those parent checkpoints are no longer needed). For N = 0 the second clause no-ops because no `gen-{-1}/` directory exists.
- **AND** at the moment generation N+1's evaluation begins, exactly one file SHALL exist in `gen-{N:03d}/` (the selected parent for the about-to-evaluate children)
- **AND** when the run completes after generation 4 (the final generation), the loop SHALL still run `select_parents` on gen 4's results and the GC step SHALL still delete gen 3's surviving parent — so the only surviving file SHALL be the selected parent of the final generation, under `inheritance/gen-004/`. This file is intentionally NOT deleted by GC: it is the final winner's trained policy, available for forensic inspection or downstream warm-start by the next milestone (e.g. M4 Baldwin)

#### Scenario: Inheritance requires a training phase

- **GIVEN** a YAML with `evolution.inheritance: lamarckian` AND `evolution.learn_episodes_per_eval: 0`
- **WHEN** the YAML is loaded via `load_simulation_config`
- **THEN** loading SHALL raise a Pydantic `ValidationError`
- **AND** the error message SHALL state that Lamarckian inheritance requires a non-zero train phase to produce weights to inherit
- **AND** the message SHALL point the user to either set `learn_episodes_per_eval > 0` or set `inheritance: none`

#### Scenario: Inheritance is mutually exclusive with static warm-start

- **GIVEN** a YAML with `evolution.inheritance: lamarckian` AND `evolution.warm_start_path: /some/path.pt`
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a Pydantic `ValidationError`
- **AND** the error message SHALL state that `warm_start_path` (run-wide static checkpoint) and `inheritance` (per-genome dynamic checkpoint) both load weights into the same brain slot before the K train phase, and that exactly one MAY be set
- **AND** the message SHALL point the user to drop one of the two

#### Scenario: Inheritance requires hyperparameter encoding

- **GIVEN** a YAML with `evolution.inheritance: lamarckian` AND `hyperparam_schema is None` (i.e. a weight-evolution config)
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a Pydantic `ValidationError`
- **AND** the error message SHALL state that Lamarckian inheritance over weight encoders would double-count weights as both genome and substrate
- **AND** the message SHALL point the user to either drop `inheritance` (returning to weight evolution) or add a `hyperparam_schema` (switching to hyperparameter evolution + Lamarckian)

#### Scenario: Inheritance incompatible with architecture-changing schema entries

- **GIVEN** a YAML with `evolution.inheritance: lamarckian` AND a `hyperparam_schema` containing at least one entry whose `name` references a brain-config field that changes tensor shapes (e.g. `actor_hidden_dim`, `lstm_hidden_dim`, `rnn_type`, `actor_num_layers` — the same set the existing static warm-start rejects)
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a Pydantic `ValidationError` naming the offending fields
- **AND** the error SHALL share the same `_ARCHITECTURE_CHANGING_FIELDS` denylist that M2.10's static warm-start already uses (single source of truth)
- **AND** the message SHALL explain that per-genome checkpoints cannot be loaded into a child whose architecture differs from the parent's

#### Scenario: Elite count cannot exceed population size

- **GIVEN** a YAML with `evolution.inheritance: lamarckian` AND `evolution.inheritance_elite_count: 20` AND `evolution.population_size: 12`
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a Pydantic `ValidationError` stating that `inheritance_elite_count` MUST be ≤ `population_size`

#### Scenario: Inheritance defaults preserve M2.12 behaviour byte-equivalently

- **GIVEN** any existing M2 evolution YAML with no `inheritance:` key under `evolution:`
- **WHEN** loaded and run via `scripts/run_evolution.py`
- **THEN** `EvolutionConfig.inheritance` SHALL be `"none"` and `EvolutionConfig.inheritance_elite_count` SHALL be `1`
- **AND** the loop SHALL construct a `NoInheritance` strategy
- **AND** no `inheritance/` directory SHALL be created under the output directory
- **AND** `LearnedPerformanceFitness.evaluate` SHALL be invoked with `warm_start_path_override=None` and `weight_capture_path=None` for every genome
- **AND** the `lineage.csv` `inherited_from` column SHALL be empty for every row

#### Scenario: Resume from checkpoint preserves selected parent IDs

- **GIVEN** a Lamarckian run that wrote a checkpoint at end of generation N with `_selected_parent_ids = [pid_a]`
- **WHEN** the run resumes via `--resume <checkpoint>`
- **THEN** the loaded loop's `_selected_parent_ids` SHALL equal `[pid_a]`
- **AND** generation N+1's children SHALL warm-start from `inheritance/gen-{N:03d}/genome-{pid_a}.pt`
- **AND** if that file is unexpectedly missing on resume, the affected children SHALL fall back to from-scratch with a `logger.warning` (the loop SHALL NOT crash)

#### Scenario: Checkpoint version 1 cannot be resumed under M3

- **GIVEN** a checkpoint pickle file whose `version` field is 1 (M2-vintage)
- **WHEN** the M3 loop attempts to load it via `--resume`
- **THEN** loading SHALL raise the existing version-mismatch error
- **AND** the user SHALL be advised to start the run fresh (no automated converter is provided in this PR)

#### Scenario: Resume rejects mismatched inheritance setting

- **GIVEN** a checkpoint produced under one inheritance setting (e.g. `inheritance: lamarckian`) AND a resume invocation whose resolved `EvolutionConfig.inheritance` is different (e.g. `inheritance: none`, whether via YAML or `--inheritance` CLI override)
- **WHEN** the loop attempts to resume
- **THEN** loading SHALL raise a clear error stating that the resumed run's inheritance setting differs from the original and that mid-run inheritance changes are not supported
- **AND** the message SHALL list both the checkpoint's recorded inheritance and the resolved current value so the user can decide which to keep
- **AND** the rejection SHALL fire BEFORE the loop reaches the first generation iteration (so an inadvertent CLI override doesn't waste compute on a corrupted run)
- **AND** the loop SHALL persist `inheritance` (the literal `"none"` / `"lamarckian"` value, not the strategy instance) in the checkpoint pickle alongside `selected_parent_ids` so this comparison is possible at load time

## MODIFIED Requirements

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

### Requirement: Evolution Configuration Block

The `SimulationConfig` SHALL accept an optional `evolution` block; existing scenario configs without an `evolution` block SHALL load unchanged. The `evolution` block SHALL include `inheritance: Literal["none", "lamarckian"]` (default `"none"`) and `inheritance_elite_count: int >= 1` (default 1). Validators SHALL reject the following invalid combinations at YAML load time, before any optimiser code runs:

1. `inheritance != "none"` AND `learn_episodes_per_eval == 0`.
2. `inheritance != "none"` AND `warm_start_path is not None`.
3. `inheritance != "none"` AND `hyperparam_schema is None`.
4. `inheritance != "none"` AND `hyperparam_schema` contains any field in `_ARCHITECTURE_CHANGING_FIELDS` (the existing denylist that M2.10's static warm-start uses).
5. `inheritance_elite_count > population_size`.

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

### Requirement: Lineage Tracking

The system SHALL maintain an append-only `lineage.csv` per evolution run that records, for every evaluated genome, the generation index, genome ID, parent IDs, fitness score, brain type, and the parent genome ID this child inherited weights from (`inherited_from`). The `inherited_from` column SHALL be the empty string when the strategy is `NoInheritance` OR when the child is in generation 0 OR when the child fell back to from-scratch due to a missing parent file. CSV writes SHALL survive process kill and resume — appending continues from the last-written row on reload.

#### Scenario: Every genome gets a lineage row

- **GIVEN** an evolution run with `population_size=12, generations=5`
- **WHEN** the run completes
- **THEN** `lineage.csv` SHALL contain exactly 60 data rows (12 × 5)
- **AND** every row SHALL have non-empty `generation`, `genome_id`, `fitness`, `brain_type`
- **AND** `parent_ids` SHALL be empty for gen-0 rows and the prior-generation IDs (semicolon-separated) for gen ≥ 1 rows

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

#### Scenario: Resume continues lineage append

- **GIVEN** an evolution run that wrote 24 lineage rows, was killed, and resumes from a checkpoint
- **WHEN** the resumed run evaluates more generations
- **THEN** new rows SHALL append to the existing `lineage.csv` (no truncation, no header re-write)
- **AND** the row count after the resumed run completes SHALL equal `population_size × generations`
