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

The `EvolutionLoop` SHALL consume a per-generation `lawn_schedule` (provided via the `transgenerational` config block) at the top of each generation, just before `optimizer.ask()`. The schedule entry for the current generation SHALL specify `pathogen_lawns_enabled: bool` (toggling whether `STATIONARY` predator entities are spawned in the environment) and `ppo_train_episodes: int` (the number of training episodes for that generation, overriding `learn_episodes_per_eval` for this generation only).

When `transgenerational` config is absent (the default for all non-TEI runs), the loop SHALL NOT consult any schedule and the env-rebuild branch SHALL be skipped. The no-schedule path SHALL be byte-equivalent to current behaviour for `inheritance: none|lamarckian|baldwin`.

#### Scenario: Schedule controls pathogen lawns and training episodes per generation

- **GIVEN** a config with `transgenerational.lawn_schedule: [{generation: 0, pathogen_lawns_enabled: true, ppo_train_episodes: 50}, {generation: 1, pathogen_lawns_enabled: false, ppo_train_episodes: 0}, {generation: 2, pathogen_lawns_enabled: false, ppo_train_episodes: 0}, {generation: 3, pathogen_lawns_enabled: false, ppo_train_episodes: 0}]`
- **WHEN** the loop reaches generation 0
- **THEN** the env config used to evaluate every gen-0 genome SHALL have pathogen lawns enabled (at least one `STATIONARY` predator entity present)
- **AND** the fitness invocation for every gen-0 genome SHALL use `learn_episodes_per_eval = 50` (overriding the default)
- **AND** when the loop reaches generation 1, the env config SHALL have pathogen lawns disabled (no `STATIONARY` predator entities)
- **AND** the fitness invocation for every gen-1+ genome SHALL use `learn_episodes_per_eval = 0` (frozen-weight evaluation for inheriting generations)

#### Scenario: Schedule absence preserves current behaviour byte-equivalently

- **GIVEN** any evolution config without a `transgenerational` block
- **WHEN** the loop runs
- **THEN** the env config SHALL be built once from the static config (no per-gen rebuild)
- **AND** `learn_episodes_per_eval` SHALL be read from `evolution.learn_episodes_per_eval` for every generation
- **AND** the loop's observable behaviour SHALL be byte-equivalent to its current behaviour (no new generation-boundary branches taken)

#### Scenario: Schedule must cover every generation in the run

- **GIVEN** a config with `evolution.generations: 4` and a `lawn_schedule` whose entries do not cover all four generations (e.g. missing generation 3, or duplicate entry for generation 2)
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a Pydantic `ValidationError`
- **AND** the message SHALL state that each generation index in `[0, generations)` MUST appear exactly once in `lawn_schedule`
