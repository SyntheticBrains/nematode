# Evolution Framework — Transgenerational Inheritance Delta

## ADDED Requirements

### Requirement: Transgenerational Inheritance Strategy

The system SHALL provide a `TransgenerationalInheritance` concrete implementation of the `InheritanceStrategy` Protocol in `quantumnematode/evolution/transgenerational_inheritance.py`. The strategy SHALL participate in the existing four-method Protocol (`select_parents`, `assign_parent`, `checkpoint_path`, `kind`) and SHALL expose a fourth `kind()` literal value `"transgenerational"` alongside the existing `"none"`, `"weights"`, and `"trait"` returns.

`TransgenerationalInheritance` SHALL implement single-elite-broadcast semantics:

- `select_parents(gen_ids, fitnesses, generation)` SHALL return the top-1 elite (highest fitness, ties broken lexicographically on `genome_id`).
- `assign_parent(child_index, parent_ids)` SHALL return `parent_ids[0]` for every `child_index` (single-elite broadcast).
- `checkpoint_path(output_dir, generation, genome_id)` SHALL return `output_dir / "inheritance" / f"gen-{generation:03d}" / f"genome-{genome_id}.tei.pt"` (note the `.tei.pt` extension distinguishing TEI substrates from Lamarckian weight checkpoints).
- `kind()` SHALL return the string literal `"transgenerational"`.

The `EvolutionLoop` SHALL extend its strategy-kind dispatch to gate substrate-IO code paths (capture, GC, load) on `kind() == "transgenerational"`. The loop SHALL ALSO gate elite-ID lineage tracking on `kind() != "none"` (the existing `_inheritance_records_lineage()` helper), so transgenerational runs SHALL populate `inherited_from` in lineage CSV from gen 1 onwards.

The strategy SHALL be selectable via `evolution.inheritance: Literal["none", "lamarckian", "baldwin", "transgenerational"]` in YAML and overridable via the `--inheritance` CLI flag on `scripts/run_evolution.py`. The `EvolutionLoop` SHALL persist `"transgenerational"` as the resolved inheritance value in its checkpoint pickle dict, and the resume-time validator SHALL reject mid-run inheritance changes (an existing safety guard SHALL apply equally to transgenerational).

#### Scenario: Transgenerational kind() literal is recognised

- **GIVEN** a `TransgenerationalInheritance` instance
- **WHEN** the loop calls `strategy.kind()`
- **THEN** the return value SHALL be exactly the string `"transgenerational"`
- **AND** the loop's substrate-IO gate (a new helper paralleling `_inheritance_active()`) SHALL evaluate `strategy.kind() == "transgenerational"`
- **AND** the loop's lineage-tracking guard `_inheritance_records_lineage()` SHALL return True for `"transgenerational"` (so `inherited_from` is populated from gen 1 onwards)
- **AND** the literal SHALL be one of exactly four values: `"none"`, `"weights"`, `"trait"`, `"transgenerational"`

#### Scenario: Transgenerational inheritance is selectable via config and CLI

- **GIVEN** an `EvolutionConfig` with `inheritance: transgenerational` and a `transgenerational` config block (per the configuration-system delta)
- **WHEN** the loop runs for ≥ 2 generations
- **THEN** the F0 generation SHALL train from random init (no warm-start, no parent substrate) for `ppo_train_episodes` episodes per the `lawn_schedule[0]` entry; after F0 evaluation completes, the F0-elite substrate is extracted via the F0 Substrate Extraction Pipeline (separate requirement below)
- **AND** every F1+ child's `tei_prior_source` SHALL be threaded through `fitness.evaluate` per the TEI Substrate Worker-to-Fitness Transport requirement; the substrate is loaded and decayed inside `fitness.evaluate`, not by the runner
- **AND** the CLI flag `--inheritance transgenerational` SHALL override the YAML field with the same behaviour
- **AND** `--inheritance none` SHALL force the no-op path even when the YAML sets `transgenerational`

#### Scenario: F0-elite substrate is captured once; F1+ substrates are mechanically derived (not stored)

- **GIVEN** a transgenerational run with `inheritance_elite_count: 1`, `population_size: 6`, `generations: 4` (F0 through F3 inclusive), schedule with F0 pathogen-on / F1-F3 pathogen-off
- **WHEN** the run completes
- **THEN** after F0 evaluation and elite identification, exactly 1 `.tei.pt` file SHALL exist under `<output_dir>/inheritance/gen-000/genome-{elite_id}.tei.pt` (the F0 elite's substrate, the single authoritative source for the cascade), produced by the F0 Substrate Extraction Pipeline (separate requirement below)
- **AND** at F1 evaluation, each of the 6 F1 children's worker tuple SHALL carry `tei_prior_source = (gen_000_substrate_path, decay_factor, 1)`; `fitness.evaluate` loads the F0 substrate and applies `inherit_from` once to produce an F1-depth substrate, then sets `brain.tei_prior` per the lstm-ppo-brain spec
- **AND** at F2 evaluation, each F2 child's `tei_prior_source` SHALL have `lineage_depth=2`; `fitness.evaluate` applies `inherit_from` TWICE (mechanically equivalent to `f0_substrate.logit_bias * decay_factor ** 2`)
- **AND** at F3 evaluation, each F3 child's `tei_prior_source` SHALL have `lineage_depth=3`; `fitness.evaluate` applies `inherit_from` THREE TIMES (mechanically equivalent to `f0_substrate.logit_bias * decay_factor ** 3`)
- **AND** no `.tei.pt` files SHALL be written under `gen-001/`, `gen-002/`, or `gen-003/` (F1/F2/F3 substrates are mechanically derived from the F0 source and do not require storage; this avoids any ambiguity about per-gen "elite substrate" since every member of F1+ shares the same depth-N substrate, mechanically)
- **AND** on completion, the only surviving substrate file SHALL be `gen-000/genome-{elite_id}.tei.pt`, intentionally retained for forensic inspection
- **AND** the lineage CSV `inherited_from` column SHALL be populated for all F1+ rows with the F0 elite's `genome_id` (so the lineage provenance is observable in the CSV even though F1+ substrates are not stored on disk)

#### Scenario: Resume rejects mid-run inheritance changes to/from transgenerational

- **GIVEN** a checkpoint produced under `inheritance: transgenerational` AND a resume invocation whose resolved `EvolutionConfig.inheritance` is different (e.g. `none`, `lamarckian`, or `baldwin`)
- **WHEN** the loop attempts to resume
- **THEN** loading SHALL raise the existing clear error stating that the resumed run's inheritance setting differs from the original
- **AND** the rejection SHALL fire BEFORE the loop reaches the first generation iteration
- **AND** the rejection rule SHALL apply symmetrically when resuming a non-transgenerational checkpoint with a transgenerational override

#### Scenario: Transgenerational requires hyperparameter encoding (same rule as lamarckian/baldwin)

- **GIVEN** a YAML with `evolution.inheritance: transgenerational` AND `hyperparam_schema is None`
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a Pydantic `ValidationError`
- **AND** the message SHALL state that the substrate would have no genome-identity anchor and SHALL point the user to add a `hyperparam_schema`

### Requirement: Per-Generation Lawn Schedule

The `EvolutionLoop` SHALL consume a per-generation `lawn_schedule` (the `LawnScheduleEntry` list defined in the configuration-system delta on `TransgenerationalConfig.lawn_schedule`) at the top of each generation, just before `optimizer.ask()`. The schedule entry for the current generation SHALL specify `pathogen_lawns_enabled: bool` and `ppo_train_episodes: int` (the number of training episodes for that generation, overriding `learn_episodes_per_eval` for this generation only).

The `pathogen_lawns_enabled` field of `LawnScheduleEntry` SHALL map onto the existing `PredatorConfig.enabled` field of the per-generation `sim_config` copy — no new env-schema field is introduced. "Pathogen lawns" is documentation vocabulary; the underlying storage is the existing `predators:` block configured with `predator_type: stationary`.

The loop SHALL implement the per-gen env toggle by creating a Pydantic `model_copy` of the run's base `sim_config` with `predators.enabled` set from the current schedule entry, and pass that per-gen copy to each worker tuple for that generation. The base `sim_config` SHALL remain unchanged across generations (no in-place mutation). This SHALL keep generations independent and allow resume from any generation boundary to reconstruct the correct env config for that generation.

When `transgenerational` config is absent (the default for all non-TEI runs), the loop SHALL NOT consult any schedule and SHALL pass the unmodified base `sim_config` to every worker. The no-schedule path SHALL be byte-equivalent to current behaviour for `inheritance: none|lamarckian|baldwin`.

#### Scenario: Schedule controls pathogen lawns and training episodes per generation

- **GIVEN** a config with `transgenerational.lawn_schedule: [{generation: 0, pathogen_lawns_enabled: true, ppo_train_episodes: 50}, {generation: 1, pathogen_lawns_enabled: false, ppo_train_episodes: 0}, {generation: 2, pathogen_lawns_enabled: false, ppo_train_episodes: 0}, {generation: 3, pathogen_lawns_enabled: false, ppo_train_episodes: 0}]`
- **WHEN** the loop reaches generation 0
- **THEN** the per-gen `sim_config` copy used to evaluate every gen-0 genome SHALL have `predators.enabled = True`
- **AND** the fitness invocation for every gen-0 genome SHALL use `learn_episodes_per_eval = 50` (overriding the default)
- **AND** when the loop reaches generation 1, the per-gen `sim_config` copy SHALL have `predators.enabled = False`
- **AND** the fitness invocation for every gen-1+ genome SHALL use `learn_episodes_per_eval = 0` (frozen-weight evaluation for inheriting generations)
- **AND** the run's base `sim_config` (the one resolved at YAML load) SHALL NOT be mutated; each generation's worker tuples SHALL receive distinct `model_copy` instances

#### Scenario: Schedule absence preserves current behaviour byte-equivalently

- **GIVEN** any evolution config without a `transgenerational` block
- **WHEN** the loop runs
- **THEN** each worker tuple SHALL receive the run's base `sim_config` unchanged (no per-gen copy)
- **AND** `learn_episodes_per_eval` SHALL be read from `evolution.learn_episodes_per_eval` for every generation
- **AND** the loop's observable behaviour SHALL be byte-equivalent to its current behaviour (no new generation-boundary branches taken)

#### Scenario: Schedule must cover every generation in the run

- **GIVEN** a config with `evolution.generations: 4` and a `lawn_schedule` whose entries do not cover all four generations (e.g. missing generation 3, or duplicate entry for generation 2)
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a Pydantic `ValidationError`
- **AND** the message SHALL state that each generation index in `[0, generations)` MUST appear exactly once in `lawn_schedule`

### Requirement: TEI Substrate Worker-to-Fitness Transport

When `EvolutionLoop` is running under `inheritance: transgenerational`, the worker tuple passed to `_evaluate_in_worker` SHALL be extended with one additional element: `tei_prior_source: tuple[Path, float, int] | None`, carrying the triple `(f0_substrate_path, decay_factor, lineage_depth)`. The element SHALL be `None` for all other inheritance strategies, preserving the existing worker contract for `none`/`lamarckian`/`baldwin` paths.

The loop SHALL compute `tei_prior_source` at the per-child dispatch step using the following rule:

- At F0 (`current_generation == 0`): `tei_prior_source = None` (no parent substrate exists yet; F0 trains from random init).
- At F1+: `tei_prior_source = (self._tei_f0_substrate_path, cfg.transgenerational.decay_factor, current_generation)`, where `self._tei_f0_substrate_path` is a new `EvolutionLoop` instance attribute populated by the F0 Substrate Extraction Pipeline (below) at the end of gen 0. The attribute SHALL also be persisted in the checkpoint pickle so resume from gen 1+ can recover the path without re-running F0.

The worker (`_evaluate_in_worker`) SHALL forward `tei_prior_source` as a keyword argument to `LearnedPerformanceFitness.evaluate(...)`. This mirrors the existing forwarding pattern for `warm_start_path_override` and `weight_capture_path`. The worker SHALL NOT construct the agent/brain itself nor set `tei_prior` directly — those responsibilities live inside `fitness.evaluate` (see lstm-ppo-brain spec "Fitness Evaluator Sets TEI Prior on Decoded Brain").

When `tei_prior_source is None`, the worker SHALL omit the kwarg entirely when calling `fitness.evaluate` (preserving byte-equivalent call shape for non-TEI runs).

#### Scenario: Worker tuple carries the TEI prior source for TEI runs

- **GIVEN** an `EvolutionLoop` running under `inheritance: transgenerational` at F1 (generation 1) with `decay_factor: 0.6`
- **WHEN** the loop dispatches a per-child worker tuple for an F1 genome
- **THEN** the tuple SHALL include `tei_prior_source = (f0_substrate_path, 0.6, 1)` where `f0_substrate_path` points to `<output_dir>/inheritance/gen-000/genome-{f0_elite_id}.tei.pt`
- **AND** at F2 the tuple SHALL have `lineage_depth == 2`; at F3, `lineage_depth == 3`
- **AND** for F0 the tuple SHALL have `tei_prior_source = None` (F0 has no parent substrate to inherit)

#### Scenario: Non-transgenerational worker tuples preserve existing contract

- **GIVEN** an `EvolutionLoop` running under any non-TEI inheritance strategy (`none` / `lamarckian` / `baldwin`)
- **WHEN** the loop dispatches a worker tuple
- **THEN** `tei_prior_source` SHALL be `None`
- **AND** the worker SHALL omit the `tei_prior_source` kwarg when calling `fitness.evaluate`
- **AND** the call shape SHALL be byte-equivalent to the pre-TEI worker contract for that strategy

#### Scenario: Worker forwards tei_prior_source to fitness.evaluate

- **GIVEN** a worker tuple under TEI with `tei_prior_source = (path, 0.6, 2)`
- **WHEN** `_evaluate_in_worker` invokes `fitness.evaluate`
- **THEN** the kwarg `tei_prior_source=(path, 0.6, 2)` SHALL be passed through
- **AND** `fitness.evaluate` SHALL be responsible for loading the substrate, applying decay, and setting `brain.tei_prior` per the lstm-ppo-brain spec's "Fitness Evaluator Sets TEI Prior on Decoded Brain" requirement
- **AND** the worker SHALL NOT directly touch any brain or substrate state

### Requirement: F0 Substrate Extraction Pipeline

Under `inheritance: transgenerational`, the F0 generation requires a substrate-extraction step that does not exist in any other inheritance mode. The pipeline SHALL:

1. **F0 weight capture (per-genome).** For every F0 genome, `EvolutionLoop` SHALL pass a `weight_capture_path` to `fitness.evaluate` (mirroring the Lamarckian pattern). This writes each F0 genome's post-train brain weights to `<output_dir>/inheritance/gen-000/genome-{gid}.pt`.
2. **F0 elite identification.** After F0's `optimizer.tell` call, the loop's strategy `select_parents(...)` returns the top-1 elite (by fitness, lex-tie-broken).
3. **F0 substrate extraction.** The loop SHALL load the F0 elite's captured weights into a fresh brain (decoded from the elite's genome via the same encoder), then invoke `TransgenerationalMemory.extract_from_brain(brain, env, probe_positions, rng_seed=transgenerational.extraction_seed)`. The resulting substrate SHALL be saved to `<output_dir>/inheritance/gen-000/genome-{elite_id}.tei.pt`.
4. **F0 weight GC.** The loop SHALL delete all F0 `.pt` weight files (both elite and non-elite) after substrate extraction completes — only the `.tei.pt` substrate file is retained for the cascade.
5. **F1+ flow.** For F1+, the loop SHALL compute `tei_prior_source = (gen_000/genome-{elite_id}.tei.pt, decay_factor, current_generation)` for every child's worker tuple. No F1+ weight capture or substrate file is created.

#### Scenario: F0 weight capture is enabled under transgenerational inheritance

- **GIVEN** a transgenerational run at generation 0 with `population_size: 6`
- **WHEN** the loop dispatches gen-0 worker tuples
- **THEN** each tuple's `weight_capture_path` SHALL be set to `<output_dir>/inheritance/gen-000/genome-{gid}.pt`
- **AND** after F0 evaluation completes, exactly 6 `.pt` weight files SHALL exist under `gen-000/`

#### Scenario: F0 substrate extraction reads the elite and writes the .tei.pt

- **GIVEN** F0 completion with 6 captured weight files and an identified elite `genome_id`
- **WHEN** the loop's post-gen-0 substrate-extraction step fires
- **THEN** the elite's `.pt` SHALL be loaded into a fresh brain via the same encoder + sim_config used in evaluation
- **AND** `TransgenerationalMemory.extract_from_brain(...)` SHALL be invoked with the configured `extraction_seed`
- **AND** the resulting substrate SHALL be saved to `<output_dir>/inheritance/gen-000/genome-{elite_id}.tei.pt`
- **AND** after extraction, all 6 `.pt` files SHALL be deleted (GC pass); the only surviving file under `gen-000/` SHALL be the `.tei.pt` substrate

#### Scenario: F1+ tei_prior_source is computed from the F0 substrate

- **GIVEN** F1, F2, F3 dispatches under transgenerational inheritance
- **WHEN** the loop computes per-child worker tuples for these generations
- **THEN** every gen-1 tuple SHALL have `tei_prior_source = (gen_000_substrate_path, decay_factor, 1)`
- **AND** every gen-2 tuple SHALL have `tei_prior_source = (gen_000_substrate_path, decay_factor, 2)`
- **AND** every gen-3 tuple SHALL have `tei_prior_source = (gen_000_substrate_path, decay_factor, 3)`
- **AND** no F1+ `weight_capture_path` or `.tei.pt` file SHALL be created (substrates are derived in-memory inside `fitness.evaluate`)

### Requirement: TEI Train-Phase Bypass for ppo_train_episodes=0

`LearnedPerformanceFitness.evaluate` currently rejects `learn_episodes_per_eval == 0` with a `ValueError` (per existing guard at `fitness.py:418-424`). Under TEI's experimental design, F1/F2/F3 generations explicitly use `ppo_train_episodes: 0` from the `lawn_schedule` because the milestone tests inheritance *without re-exposure or re-training*. The fitness evaluator SHALL therefore accept zero training episodes when called with `tei_prior_source is not None`, treating the train phase as a structural no-op (the `for ep_idx in range(0)` loop already produces no iterations; only the validator must be bypassed).

When `tei_prior_source is not None` AND `learn_episodes_per_eval == 0`, `fitness.evaluate` SHALL:

1. Skip the train phase entirely (no `train_runner.run(...)` calls, no `save_weights` capture).
2. Run the eval phase as normal (the substrate-biased brain is evaluated for L episodes).
3. Return the eval-phase fitness scalar.

The existing rejection SHALL remain in force for non-TEI calls (`tei_prior_source is None`) — only the TEI path relaxes the constraint.

#### Scenario: F1+ evaluation skips train phase but runs eval phase

- **GIVEN** a call to `fitness.evaluate(..., tei_prior_source=(path, 0.6, 1))` with `sim_config.evolution.learn_episodes_per_eval == 0`
- **WHEN** `fitness.evaluate` executes
- **THEN** the existing `learn_episodes_per_eval > 0` validator SHALL be bypassed (no `ValueError` raised)
- **AND** the train phase SHALL NOT execute (no `StandardEpisodeRunner.run` invocations, no weight mutations)
- **AND** the eval phase SHALL run for `episodes` episodes using `FrozenEvalRunner`
- **AND** `brain.tei_prior` SHALL be set (per the lstm-ppo-brain spec) BEFORE the eval phase begins, so eval-phase episodes use the substrate-biased policy
- **AND** the returned fitness scalar SHALL reflect the eval-phase success rate

#### Scenario: Non-TEI calls still reject learn_episodes_per_eval=0

- **GIVEN** a call to `fitness.evaluate(...)` with `tei_prior_source is None` (or omitted) AND `learn_episodes_per_eval == 0`
- **WHEN** `fitness.evaluate` executes
- **THEN** the existing `ValueError` SHALL be raised (current behaviour preserved)
- **AND** the error message SHALL be unchanged for non-TEI callers

### Requirement: CLI Guard for transgenerational + fitness pairing

`scripts/run_evolution.py` SHALL reject the combination of `evolution.inheritance: transgenerational` with `--fitness success_rate` at CLI parse time, mirroring the existing guard for `inheritance: lamarckian|baldwin`. The guard SHALL fire BEFORE the optimizer or loop is constructed, exiting with code 1 and a clear message.

The rationale: `EpisodicSuccessRate` does not accept the TEI `tei_prior_source` kwarg (nor the `warm_start_path_override` / `weight_capture_path` kwargs), and `inheritance: transgenerational` requires a learned-performance fitness signal because the F0 elite must be trained for its substrate-extraction telemetry pass to produce meaningful biases.

#### Scenario: CLI rejects inheritance: transgenerational + --fitness success_rate

- **GIVEN** an invocation of `scripts/run_evolution.py` with `evolution.inheritance: transgenerational` in YAML AND `--fitness success_rate` (the CLI default when `--fitness` is omitted)
- **WHEN** the script parses arguments and resolves the `EvolutionConfig`
- **THEN** the script SHALL exit with code 1 BEFORE constructing the optimizer or the loop
- **AND** the error message SHALL state that `inheritance: transgenerational` requires `--fitness learned_performance` because the F0 elite must be trained for substrate extraction
- **AND** the message SHALL point the user to set `--fitness learned_performance` or to set `inheritance: none`
