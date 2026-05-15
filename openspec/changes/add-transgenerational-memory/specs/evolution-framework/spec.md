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
- **THEN** the F0 generation SHALL run from-scratch and SHALL have its F0-elite substrate extracted via the telemetry pass AFTER fitness evaluation completes
- **AND** every F1+ child SHALL receive a substrate decayed from its assigned parent's substrate via the substrate's `inherit_from(parents, decay_factor)` call BEFORE the F1+ generation's episode evaluation begins
- **AND** the CLI flag `--inheritance transgenerational` SHALL override the YAML field with the same behaviour
- **AND** `--inheritance none` SHALL force the no-op path even when the YAML sets `transgenerational`

#### Scenario: F0-elite substrate is captured once; F1+ substrates are mechanically derived (not stored)

- **GIVEN** a transgenerational run with `inheritance_elite_count: 1`, `population_size: 6`, `generations: 4` (F0 through F3 inclusive), schedule with F0 pathogen-on / F1-F3 pathogen-off
- **WHEN** the run completes
- **THEN** during F0's post-fitness hook, exactly 1 `.tei.pt` file SHALL be written under `<output_dir>/inheritance/gen-000/genome-{elite_id}.tei.pt` (the F0 elite's substrate, the single authoritative source for the cascade)
- **AND** at F1 evaluation start, each of the 6 F1 children SHALL load the F0 elite substrate from `gen-000/`, apply `inherit_from([f0_substrate], decay_factor)` once to produce an F1-depth substrate, and the runner SHALL set `brain.tei_prior = f1_substrate.logit_bias` before each F1 episode
- **AND** at F2 evaluation start, each F2 child SHALL load the F0 elite substrate from `gen-000/` and apply `inherit_from` TWICE to produce an F2-depth substrate (mechanically equivalent to `f0_substrate.logit_bias * decay_factor ** 2`)
- **AND** at F3 evaluation start, each F3 child SHALL load the F0 elite substrate from `gen-000/` and apply `inherit_from` THREE TIMES to produce an F3-depth substrate (`f0_substrate.logit_bias * decay_factor ** 3`)
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

### Requirement: TEI Substrate Worker Transport

When `EvolutionLoop` is running under `inheritance: transgenerational`, the worker tuple passed to `_evaluate_in_worker` SHALL be extended with three additional elements: `(f0_substrate_path: Path | None, decay_factor: float | None, lineage_depth: int | None)`. These elements SHALL be `None` for all other inheritance strategies, preserving the existing worker contract for `none`/`lamarckian`/`baldwin` paths.

The worker SHALL, when `f0_substrate_path is not None`:

1. Load the F0 substrate via `TransgenerationalMemory.load(f0_substrate_path)`.
2. Apply `inherit_from([f0_substrate], decay_factor)` `lineage_depth` times to produce a depth-N substrate (F0 → 0 applications, F1 → 1, F2 → 2, F3 → 3).
3. Set `agent.brain.tei_prior = depth_n_substrate.logit_bias` via `hasattr`-gated dispatch (see lstm-ppo-brain delta "Worker Sets TEI Prior Before Runner Invocation") immediately after constructing the agent/brain and BEFORE invoking the episode runner. The runner code SHALL be unchanged.

When `f0_substrate_path is None`, the worker SHALL NOT set `tei_prior` on any brain — the default `None` attribute (or attribute absence on non-LSTMPPO brains) preserves pre-TEI baseline behaviour.

#### Scenario: Worker tuple is extended with TEI substrate transport elements

- **GIVEN** an `EvolutionLoop` running under `inheritance: transgenerational` at F1 (generation 1) with `decay_factor: 0.6`
- **WHEN** the loop dispatches a per-child worker tuple for an F1 genome
- **THEN** the tuple SHALL include the three TEI elements `(f0_substrate_path, decay_factor, lineage_depth)` where `f0_substrate_path` points to `<output_dir>/inheritance/gen-000/genome-{f0_elite_id}.tei.pt`, `decay_factor == 0.6`, and `lineage_depth == 1` (F1 = one decay application)
- **AND** at F2 the tuple SHALL have `lineage_depth == 2`; at F3, `lineage_depth == 3`
- **AND** for F0 the tuple SHALL have `f0_substrate_path = None` (F0 has no parent substrate to inherit)

#### Scenario: Non-transgenerational worker tuples preserve existing contract

- **GIVEN** an `EvolutionLoop` running under any non-TEI inheritance strategy (`none` / `lamarckian` / `baldwin`)
- **WHEN** the loop dispatches a worker tuple
- **THEN** the three TEI elements SHALL be `(None, None, None)`
- **AND** the worker SHALL NOT attempt to load any substrate file
- **AND** the worker SHALL NOT touch `brain.tei_prior` on any brain (preserving pre-TEI baseline byte-equivalence)

#### Scenario: Worker loads and decays substrate before runner invocation

- **GIVEN** a worker tuple with `f0_substrate_path` set to a valid `.tei.pt` file, `decay_factor = 0.6`, `lineage_depth = 2`
- **WHEN** the worker constructs the agent/brain
- **THEN** the worker SHALL load the F0 substrate (`load.logit_bias = b0`)
- **AND** apply `inherit_from([f0], 0.6)` twice to produce a depth-2 substrate with `logit_bias ≈ b0 * 0.36`
- **AND** if `hasattr(agent.brain, "tei_prior")` returns True, the worker SHALL set `agent.brain.tei_prior = depth2_substrate.logit_bias` BEFORE invoking `runner.run(agent, ...)`
- **AND** if the brain does not have a `tei_prior` attribute, the worker SHALL log a warning and proceed without setting (defensive: a future config selecting a non-LSTMPPO brain under TEI should not crash the worker; the operator gets a visible signal that the substrate was inert)
