## MODIFIED Requirements

### Requirement: Baldwin Inheritance Strategy

The system SHALL provide a `BaldwinInheritance` implementation in `quantumnematode/evolution/inheritance.py` whose `kind()` returns `"trait"`. Baldwin inheritance is mechanically a no-op on the weight-IO path (no per-genome `.pt` files written, no GC, no warm-start) but the loop SHALL track the prior generation's elite genome ID (top fitness, lex-tie-broken — same selection rule as Lamarckian) and write it to the lineage CSV's `inherited_from` column for every child of the next generation. The hyperparameter genome continues to evolve via TPE; the elite-ID lineage trace exists so post-pilot analysis can identify which prior-gen elite each child shares hyperparameters with via TPE's posterior.

The `BaldwinInheritance` constructor SHALL take no required arguments. The `inheritance_elite_count` config field SHALL be ignored under Baldwin (the field exists for forward-compatibility with future multi-elite Baldwin variants but is unused in this milestone).

`BaldwinInheritance.select_parents()` SHALL return a single-element list `[best_genome_id]` containing the prior generation's elite genome ID (top fitness, lex-tie-broken on `genome_id` — the same selection rule as `LamarckianInheritance` with `elite_count=1`). The loop reuses this ID to populate the lineage CSV's `inherited_from` column for every child of the next generation, even though no on-disk checkpoint is created. `BaldwinInheritance.assign_parent()` SHALL return `None` (no per-child parent assignment for warm-start). `BaldwinInheritance.checkpoint_path()` SHALL return `None` (no on-disk checkpoint substrate).

The genetic-assimilation question Baldwin tests is whether evolution under TPE produces a hyperparameter genome that biases the brain to learn fast from random init. The post-pilot script `scripts/campaigns/baldwin_f1_postpilot_eval.py` (NOT part of the loop's runtime contract — a forensic script) SHALL re-evaluate the elite genome via a **learning-acceleration test**: train the elite genome's brain for K' episodes (K' < K, where K is the pilot's `learn_episodes_per_eval`), measure success rate over L frozen-eval episodes, then repeat for a synthetic baseline genome drawn as a per-seed prior sample from `HyperparameterEncoder.initial_genome(sim_config, rng=np.random.default_rng(seed))` (stochastic per-seed sampling from the schema's prior — uniform-in-bounds for floats, log-uniform for log-scale floats, etc.; not deterministic defaults). Baldwin signal is the elite's L-eval success rate minus the baseline's L-eval success rate after equal K' training. Both arms include K'-train + L-eval; the comparison is apples-to-apples by construction. The script SHALL accept `--k-prime` (default `10`) and `--episodes` (default `25`, the L value) as CLI overrides so the evaluator can be re-run with different K' / L budgets without code changes.

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

#### Scenario: F1 evaluator runs a paired K'-train learning-acceleration test

- **GIVEN** a Baldwin pilot's session output directory with a valid `best_params.json` (containing the elite genome's params; birth metadata for schema reconstruction is rebuilt at re-eval time from the pilot YAML via `build_birth_metadata(sim_config)`, since `best_params.json` does not persist birth metadata)
- **WHEN** the operator runs `scripts/campaigns/baldwin_f1_postpilot_eval.py --baldwin-root <pilot_root> --config <pilot_yaml> --k-prime 10 --episodes 25 --output-dir <out>`
- **THEN** the script SHALL, for each seed in the pilot:
  - Load the seed's `best_params.json` and reconstruct the elite genome via `HyperparameterEncoder.decode` (birth metadata is rebuilt from the pilot YAML at re-eval time via `build_birth_metadata(sim_config)` — `best_params.json` itself does not persist birth metadata)
  - Construct a schema-prior baseline genome via `HyperparameterEncoder.initial_genome(sim_config, rng=np.random.default_rng(seed))` (a deterministic per-seed random sample from the schema's prior distribution)
  - Build a per-evaluation `sim_config` copy whose `evolution.learn_episodes_per_eval` is set to K' and whose `evolution.eval_episodes_per_eval` is set to L (`LearnedPerformanceFitness.evaluate` reads K from `sim_config.evolution.learn_episodes_per_eval` directly; L resolves from `sim_config.evolution.eval_episodes_per_eval` if set, else falls back to the protocol's `episodes` kwarg)
  - Run `LearnedPerformanceFitness.evaluate(elite_genome, sim_config_for_kprime, encoder, episodes=L, seed=seed)`; record the elite's success rate
  - Run `LearnedPerformanceFitness.evaluate(baseline_genome, sim_config_for_kprime, encoder, episodes=L, seed=seed)`; record the baseline's success rate
  - Both runs SHALL use the same per-seed RNG seed so the only difference between them is the genome
- **AND** the script SHALL write `f1_learning_acceleration.csv` to the output directory with columns `seed, k_prime, episodes, elite_success_rate, baseline_success_rate, signal_delta` (where `signal_delta = elite − baseline`). The `k_prime` and `episodes` columns SHALL record the K' and L values used for that row's evaluation so multiple K' / L runs can coexist in the same CSV via append-mode.
- **AND** if `--k-prime` is omitted it SHALL default to `10`; if `--episodes` is omitted it SHALL default to `25` (matches the pilot's default L)
- **AND** the script SHALL reject `--k-prime <= 0` and `--episodes <= 0` at argparse-time with a clear error message
- **AND** the script SHALL be safe to re-run with different K' / L budgets without re-running the pilot — re-runs append rows to the existing CSV (creating it if absent), so an operator can run K' = 10 first then K' = 25 (or any other budget) and have both sets of measurements available for analysis without losing the prior data
