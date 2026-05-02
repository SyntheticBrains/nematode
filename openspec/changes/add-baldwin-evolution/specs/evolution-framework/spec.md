## MODIFIED Requirements

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

- **GIVEN** a checkpoint pickle file whose `version` field is older than the current `CHECKPOINT_VERSION`
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
- **THEN** the script SHALL exit with code 1 BEFORE constructing the optimiser or the loop
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
- **AND** the loop's `_inheritance_records_lineage()` helper (which decides whether to write `inherited_from` in lineage rows) SHALL evaluate `strategy.kind() != "none"`

## ADDED Requirements

### Requirement: Baldwin Inheritance Strategy

The system SHALL provide a `BaldwinInheritance` implementation in `quantumnematode/evolution/inheritance.py` whose `kind()` returns `"trait"`. Baldwin inheritance is mechanically a no-op on the weight-IO path (no per-genome `.pt` files written, no GC, no warm-start) but the loop SHALL track the prior generation's elite genome ID (top fitness, lex-tie-broken — same selection rule as Lamarckian) and write it to the lineage CSV's `inherited_from` column for every child of the next generation. The hyperparameter genome continues to evolve via TPE; the elite-ID lineage trace exists so post-pilot analysis can identify which prior-gen elite each child shares hyperparameters with via TPE's posterior.

The `BaldwinInheritance` constructor SHALL take no required arguments. The `inheritance_elite_count` config field SHALL be ignored under Baldwin (the field exists for forward-compatibility with future multi-elite Baldwin variants but is unused in this milestone).

`BaldwinInheritance.select_parents()` SHALL return an empty list (no per-genome checkpoints to track). `BaldwinInheritance.assign_parent()` SHALL return `None` (no per-child parent assignment for warm-start). `BaldwinInheritance.checkpoint_path()` SHALL return `None` (no on-disk checkpoint substrate).

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

### Requirement: Early Stop on Saturation

The system SHALL provide an `evolution.early_stop_on_saturation: int | None` config field (default `None`) that exits the evolution loop when `best_fitness` has not improved for the configured number of consecutive generations. The field SHALL be settable via YAML and overridable via the `--early-stop-on-saturation N` CLI flag on `scripts/run_evolution.py`.

When the field is `None` (the default), the loop SHALL run for the full `generations` budget regardless of fitness trajectory, preserving existing behaviour byte-equivalently.

When the field is set to a positive integer N, the loop SHALL track the previous generation's `best_fitness` after each `optimizer.tell` call. If the current generation's `best_fitness` is strictly greater than the prior generation's, the counter SHALL reset to 0; otherwise the counter SHALL increment. When the counter reaches N, the loop SHALL log "Early-stop: best_fitness has not improved for N generations (last improvement at gen X)" and break out of the main loop. The `lineage.csv`, `history.csv`, and final `best_params.json` SHALL reflect the truncated run (no padding); the aggregator handles cross-arm length normalisation at analysis time.

The early-stop counter SHALL be persisted in the checkpoint pickle as `gens_without_improvement: int` so resume preserves the saturation-tracking state. The `CHECKPOINT_VERSION` SHALL be bumped to reflect the new pickle field; older checkpoints SHALL be rejected with a clear error per the existing version-mismatch scenario.

#### Scenario: Default behaviour preserves full-budget runs

- **GIVEN** an evolution YAML with no `early_stop_on_saturation` key
- **WHEN** the loop runs
- **THEN** `EvolutionConfig.early_stop_on_saturation` SHALL be `None`
- **AND** the loop SHALL iterate exactly `generations` times regardless of fitness
- **AND** the lineage and history CSVs SHALL contain exactly `generations` rows (per arm) — byte-equivalent to runs where the field is absent

#### Scenario: Early-stop fires after N consecutive non-improving generations

- **GIVEN** an evolution YAML with `evolution.early_stop_on_saturation: 3` and `generations: 20`, on a run where `best_fitness` is monotonically increasing through gen 5 (e.g. 0.3, 0.5, 0.7, 0.85, 0.95) and then plateaus exactly at 0.95 from gen 6 onwards
- **WHEN** the loop runs
- **THEN** the counter SHALL increment at gens 7, 8, 9 (three non-improving generations after the last improvement at gen 6 — wait, last improvement was gen 5 to gen 6 if 0.95 first appears at gen 6; counter resets at gen 6, then increments at 7, 8, 9; counter = 3 at gen 9, triggering early-stop after gen 9 evaluates)
- **AND** the loop SHALL break after gen 9's evaluation completes (i.e. gen 9 IS recorded in lineage; gen 10 is NOT)
- **AND** the log SHALL contain "Early-stop: best_fitness has not improved for 3 generations (last improvement at gen 6)"
- **AND** the lineage CSV SHALL contain exactly 9 generations × population_size rows
- **AND** the history CSV SHALL contain exactly 9 rows

#### Scenario: Counter resets on any positive improvement

- **GIVEN** an evolution YAML with `early_stop_on_saturation: 5`
- **WHEN** the loop runs and `best_fitness` follows the trajectory: 0.3, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5
- **THEN** the counter SHALL increment at gens 3, 4 (two non-improving), reset to 0 at gen 5 (improvement to 0.5), then increment at gens 6, 7, 8, 9 (four non-improving)
- **AND** the loop SHALL NOT trigger early-stop because the counter never reaches 5 within this trajectory

#### Scenario: CLI override for early-stop

- **GIVEN** an evolution YAML with `early_stop_on_saturation: 3` AND a CLI invocation with `--early-stop-on-saturation 10`
- **WHEN** `scripts/run_evolution.py` parses arguments and resolves the config
- **THEN** the resolved `EvolutionConfig.early_stop_on_saturation` SHALL be `10` (CLI wins per the existing CLI-override convention)
- **AND** the YAML's `3` SHALL be ignored
- **AND** an explicit `--early-stop-on-saturation 0` (or any non-positive value) SHALL be rejected by argparse before reaching the loop

#### Scenario: Resume preserves the early-stop counter

- **GIVEN** a Lamarckian or Baldwin run with `early_stop_on_saturation: 3` that wrote a checkpoint at end of generation N with `gens_without_improvement: 2`
- **WHEN** the run resumes via `--resume <checkpoint>`
- **THEN** the loaded loop's `_gens_without_improvement` SHALL equal `2`
- **AND** if generation N+1's `best_fitness` is no improvement, the counter SHALL increment to 3 and the loop SHALL early-stop after gen N+1
- **AND** if generation N+1 IS an improvement, the counter SHALL reset to 0 and the loop SHALL continue to the full `generations` budget

#### Scenario: Early-stop applies to all inheritance strategies

- **GIVEN** the same `early_stop_on_saturation: 3` setting under three different inheritance strategies (`none`, `lamarckian`, `baldwin`)
- **WHEN** all three runs use the same seed and trajectory
- **THEN** all three SHALL trigger early-stop at the same generation (the trigger is a function of `best_fitness` only, not of the inheritance strategy)
- **AND** Lamarckian's GC step SHALL still run for the final-evaluated generation (preserving the surviving elite checkpoint per existing behaviour)
