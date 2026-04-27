# Tasks: M2 — Hyperparameter Evolution Framework + MLPPPO Pilot

## Phase 1: ParamSchemaEntry + SimulationConfig field

**Dependencies**: None
**Parallelizable**: No

- [ ] 1.1 Add `ParamSchemaEntry` Pydantic model to [packages/quantum-nematode/quantumnematode/utils/config_loader.py](packages/quantum-nematode/quantumnematode/utils/config_loader.py): fields `name: str`, `type: Literal["float", "int", "bool", "categorical"]`, `bounds: tuple[float, float] | None = None`, `values: list[str] | None = None`, `log_scale: bool = False`. Add a `@model_validator(mode="after")` that enforces type-conditional metadata (float/int require `bounds`; categorical requires `values` with len ≥ 2; bool/categorical reject `log_scale=True`; bool/categorical reject `bounds`; float/int/bool reject `values`). See Decision 4 in design.md
- [ ] 1.2 Add `hyperparam_schema: list[ParamSchemaEntry] | None = None` field to `SimulationConfig`
- [ ] 1.3 Add a `@model_validator(mode="after")` on `SimulationConfig` that runs when `hyperparam_schema is not None`. It SHALL handle three defensive cases up-front before walking the schema:
  - **Brain block missing**: if `sim_config.brain is None`, raise `ValidationError` with "`hyperparam_schema` requires a `brain:` block in the YAML to resolve the field-name validation against".
  - **Brain name unknown**: if `sim_config.brain.name not in BRAIN_CONFIG_MAP`, raise `ValidationError` mirroring the existing `_resolve_brain_config` error style, listing the brain name and pointing the user at the registered brains. Don't crash with raw `KeyError`.
  - **Brain name valid**: resolve the concrete brain config class via `BRAIN_CONFIG_MAP[sim_config.brain.name]` and walk every schema entry. For any `entry.name` not in `<brain_config>.model_fields`, raise `ValidationError` with the bogus `name` plus 3+ valid alternatives. See Decision 2.
    Note: the validator does NOT check whether the brain has an *encoder* registered (e.g. `qvarcircuit` is a valid brain name in `BRAIN_CONFIG_MAP` but has no encoder in `ENCODER_REGISTRY`). That's a runtime concern the encoder-dispatch path handles separately at startup
- [ ] 1.4 Unit test: `test_param_schema_entry_validates_type_metadata` — float-without-bounds, categorical-with-1-value, bool-with-log_scale, etc. all raise
- [ ] 1.5 Unit test: `test_hyperparam_schema_yaml_parses` — load a fixture YAML with mixed-type schema, assert `SimulationConfig.hyperparam_schema` is populated with `ParamSchemaEntry` instances
- [ ] 1.6 Unit test: `test_hyperparam_schema_rejects_typo` — schema with `name: "actor_hidden_dimm"` against brain `mlpppo` raises `ValidationError` whose message contains the typo and at least 3 valid alternatives from `MLPPPOBrainConfig.model_fields`
- [ ] 1.7 Unit test: `test_hyperparam_schema_absence_preserves_m0_behaviour` — existing scenario configs with no `hyperparam_schema` load with `cfg.hyperparam_schema is None` (back-compat)
- [ ] 1.8 Unit test: `test_hyperparam_schema_requires_brain_block` — load a YAML with `hyperparam_schema:` but NO `brain:` block; assert `ValidationError` whose message names "brain:" as the missing block. Locks in task 1.3's first defensive branch
- [ ] 1.9 Unit test: `test_hyperparam_schema_unknown_brain_name` — load a YAML with `hyperparam_schema:` and `brain.name: bogus_brain`; assert `ValidationError` whose message names "bogus_brain" and lists registered brains. Locks in task 1.3's second defensive branch

## Phase 2: EvolutionConfig fields for learned-performance fitness

**Dependencies**: None (can land in parallel with Phase 1)
**Parallelizable**: Yes

- [ ] 2.1 Add `learn_episodes_per_eval: int = Field(default=0, ge=0)` to `EvolutionConfig`. The default of 0 means `LearnedPerformanceFitness.evaluate` raises (M0 default behaviour preserved)
- [ ] 2.2 Add `eval_episodes_per_eval: int | None = Field(default=None, ge=1)`. None means "fall back to `episodes_per_eval`" (M0 default behaviour for that field is unchanged)
- [ ] 2.3 Unit test: `test_evolution_config_learn_eval_defaults` — verify defaults are `0` and `None`
- [ ] 2.4 Unit test: `test_evolution_config_learn_eval_bounds` — Pydantic raises on `learn_episodes_per_eval=-1`, `eval_episodes_per_eval=0`

## Phase 3: HyperparameterEncoder

**Dependencies**: Phase 1 (needs `ParamSchemaEntry` and `hyperparam_schema` on `SimulationConfig`)
**Parallelizable**: No

- [ ] 3.1 Add `HyperparameterEncoder` class to [packages/quantum-nematode/quantumnematode/evolution/encoders.py](packages/quantum-nematode/quantumnematode/evolution/encoders.py). The encoder is **brain-agnostic** — the actual brain comes from `sim_config.brain` — so it sets `brain_name: str = ""` (the empty string, NOT `None` and NOT a missing attribute). Rationale: the `GenomeEncoder` protocol at [encoders.py:68](packages/quantum-nematode/quantumnematode/evolution/encoders.py#L68) declares `brain_name: str` and is decorated `@runtime_checkable`, which means `isinstance(encoder, GenomeEncoder)` checks attribute presence AND type at runtime. Setting `None` would violate the `str` annotation; omitting the attribute would fail the `isinstance` check. The empty string preserves protocol conformance and signals "brain-agnostic" without colliding with any real brain name. Per Decision 0, this encoder is selected via `select_encoder(sim_config)` based on `hyperparam_schema` presence, not via `ENCODER_REGISTRY` lookup, so it does not need a registry-key identity
- [ ] 3.2 Implement `initial_genome(sim_config, *, rng)`. Reads `sim_config.hyperparam_schema`, samples one float per slot using the per-type initial-sampling rules from Decision 3 (float: uniform-in-bounds or log-uniform when `log_scale=True`; int: uniform-in-bounds; bool: ±1 then `> 0` threshold; categorical: random index in `[0, len(values))`). Constructs a `Genome` with `params` of length `len(schema)` and `birth_metadata` populated by calling the shared `build_birth_metadata(sim_config)` helper (added in task 4.5.2 — a single source of truth used by both `initial_genome` AND `EvolutionLoop`'s genome construction sites, preventing format drift between test/programmatic and production paths)
- [ ] 3.3 Implement `decode(genome, sim_config, *, seed=None)`. Reads `birth_metadata["param_schema"]` (NOT `sim_config.hyperparam_schema` — the genome is the source of truth in worker processes). The metadata is a list of plain dicts (per task 3.2 / Decision 4 / Phase 4.5); reconstitute each entry via `ParamSchemaEntry(**entry_dict)` if Pydantic-instance access is needed, OR consume the dict directly (`entry_dict["type"]`, `entry_dict["bounds"]`, etc.) — both work. For each schema entry, calls a private `_decode_one(entry, value)` helper that dispatches by `entry.type` and returns the decoded Python value (per Decision 3's per-type transforms). Accumulates the results into an `updates: dict[str, Any]`. Then **constructs a fresh `SimulationConfig` with the patched brain config — NEVER mutates the input `sim_config` in place**. Use the M0 idiom: `new_brain_cfg = sim_config.brain.config.model_copy(update=updates); new_container = sim_config.brain.model_copy(update={"config": new_brain_cfg}); new_sim_config = sim_config.model_copy(update={"brain": new_container})`. Dispatches to `instantiate_brain_from_sim_config(new_sim_config, seed=seed)`. **Does NOT call `load_weight_components`** — fresh weights every evaluation
- [ ] 3.3a Implement private `_decode_one(entry: ParamSchemaEntry, value: float) -> Any` on `HyperparameterEncoder`. Single-method dispatch by `entry.type` returning the typed Python value. Concentrates the per-type logic so that adding a new param type (`set`, `range`, etc.) is a one-method extension. See Decision 3 for the per-type transforms
- [ ] 3.4 Implement `genome_dim(sim_config)`. Returns `len(sim_config.hyperparam_schema)` directly. Does NOT construct a brain
- [ ] 3.5 Do NOT add `HyperparameterEncoder` to `ENCODER_REGISTRY`. The registry is keyed by **brain name** (`MLPPPOEncoder.brain_name = "mlpppo"`, etc.) and reserves brain-keyed dispatch for brain-specific encoders. `HyperparameterEncoder` is brain-agnostic and is selected at the dispatch layer (see task 5.4's `select_encoder` helper) based on `hyperparam_schema` presence. Adding it to the registry would mean (a) `get_encoder("hyperparam")` returns it for any user who happens to write `brain.name: hyperparam` in YAML — confusing and wrong; (b) `get_encoder`'s error message would list `"hyperparam"` alongside real brain names, suggesting it as a brain-name option to misled users. The encoder is still importable from `quantumnematode.evolution.encoders` for programmatic callers who want to construct it directly
- [ ] 3.6 Unit test: `test_hyperparam_encoder_round_trip_float` — author a 1-slot float schema, encode → decode → verify the resulting `BrainConfig.<name>` equals the genome value within float tolerance
- [ ] 3.7 Unit test: `test_hyperparam_encoder_round_trip_int` — same but with int slot, verify rounding
- [ ] 3.8 Unit test: `test_hyperparam_encoder_decode_bool` — verify the `> 0` decode rule (positive sample → True, negative → False). Note: bool decode is intentionally lossy (the original genome float is not recoverable from the bool), so this test asserts the threshold semantics rather than a full round-trip
- [ ] 3.9 Unit test: `test_hyperparam_encoder_round_trip_categorical` — verify `int(round(value)) mod len(values)` indexing (and that boundary values like `1.5` between bins decode deterministically)
- [ ] 3.10 Unit test: `test_hyperparam_encoder_log_scale` — float slot with `log_scale: true`, verify the genome value is in log-space and decode applies `exp(value)`
- [ ] 3.11 Unit test: `test_hyperparam_encoder_genome_dim_matches_schema_length` — verify `genome_dim(sim_config) == len(sim_config.hyperparam_schema)` for a multi-entry schema; also verify the encoder does NOT construct a brain at this step (fast)
- [ ] 3.12 Unit test: `test_hyperparam_encoder_unspecified_fields_unchanged` — schema only mentions `learning_rate`; decode produces a brain whose `actor_hidden_dim` (not in schema) equals the YAML-configured value
- [ ] 3.13 Unit test: `test_hyperparam_encoder_does_not_load_weights` — patch `WeightPersistence.load_weight_components` and verify `decode()` does NOT call it (the genome carries no weights)
- [ ] 3.14 Unit test: `test_hyperparam_encoder_not_in_brain_registry` — assert `"hyperparam"` is NOT a key in `ENCODER_REGISTRY` (the registry is reserved for brain-keyed encoders); also assert `HyperparameterEncoder` is importable directly from `quantumnematode.evolution.encoders` for programmatic callers. Locks Decision 0's "no registry pollution" invariant in. The actual dispatch path is unit-tested separately in task 6.5 via `select_encoder`
- [ ] 3.15 Unit test: `test_hyperparam_encoder_pickles_with_schema` — pickle a `Genome` containing a populated `birth_metadata["param_schema"]` and verify it round-trips (workers receive the schema this way)

## Phase 4: LearnedPerformanceFitness

**Dependencies**: Phase 2 (needs `learn_episodes_per_eval` / `eval_episodes_per_eval` on `EvolutionConfig`); Phase 3 helpful but not strict (can use mocked encoder for tests)
**Parallelizable**: Yes

- [ ] 4.1 Add `LearnedPerformanceFitness` class to [packages/quantum-nematode/quantumnematode/evolution/fitness.py](packages/quantum-nematode/quantumnematode/evolution/fitness.py) as a peer of `EpisodicSuccessRate`
- [ ] 4.2 Implement `evaluate(genome, sim_config, encoder, *, episodes, seed)` per Decision 5. The `episodes` kwarg from the `FitnessFunction` protocol is **the eval-phase fallback** when `sim_config.evolution.eval_episodes_per_eval is None` — see task 4.7 and Decision 5's "Asymmetry" note. K (the train-episode count) is read directly from `sim_config.evolution.learn_episodes_per_eval` (no CLI override exists). Document this asymmetry in the docstring. **First**, defensively guard against the three sim_config blocks the rest of the function dereferences — mirroring M0's `EpisodicSuccessRate.evaluate` ([fitness.py:207-221](packages/quantum-nematode/quantumnematode/evolution/fitness.py#L207)) which guards `environment` and `reward` similarly:
  - **`evolution` is None**: raise `ValueError` whose message says `LearnedPerformanceFitness` requires an `evolution:` block in the YAML to set `learn_episodes_per_eval`. The loop forwards the raw `sim_config` to fitness (see [evolution/loop.py:130,297](packages/quantum-nematode/quantumnematode/evolution/loop.py#L130)), and `sim_config.evolution` is typed `EvolutionConfig | None`.
  - **`environment` is None**: raise `ValueError` mirroring M0's `EpisodicSuccessRate.evaluate` style — needed because `create_env_from_config(sim_config.environment, ...)` and `_build_agent` (which reads `sim_config.environment.sensing`) both blow up with raw `AttributeError` otherwise.
  - **`reward` is None**: raise `ValueError` mirroring M0's `EpisodicSuccessRate.evaluate` style — needed because `runner.run(agent, sim_config.reward, max_steps)` is called per-episode in both train and eval phases.
    After the guards, alias `evolution_config = sim_config.evolution` for ergonomic reads in subsequent tasks (4.3, 4.5, 4.7)
- [ ] 4.3 After the None guard, reject `evolution_config.learn_episodes_per_eval == 0` with a `ValueError` whose message mentions `EpisodicSuccessRate` as the correct alternative
- [ ] 4.4 Build the train env via `create_env_from_config(sim_config.environment, seed=seed, theme=Theme.HEADLESS)` and a `train_agent = _build_agent(brain, train_env, sim_config)`. Use M0's `_build_agent` helper unchanged
- [ ] 4.5 Train phase: `StandardEpisodeRunner()` × K episodes — standard contract, brain.learn fires per-step
- [ ] 4.6 Build a SECOND env for the eval phase via `create_env_from_config(sim_config.environment, seed=seed, theme=Theme.HEADLESS)` (same seed) and a `eval_agent = _build_agent(brain, eval_env, sim_config)`. Critical: this is NOT the same env that was used for training. Per Decision 5, post-training env state is arbitrary (food consumed, agent in some corner) and would corrupt the eval measurement. The brain carries over with its learned weights; the env starts clean
- [ ] 4.7 Resolve eval-episode count: `eval_eps = sim_config.evolution.eval_episodes_per_eval; L = eval_eps if eval_eps is not None else episodes`. **Read the loop's `episodes` kwarg, NOT `sim_config.evolution.episodes_per_eval`** — the loop passes the resolved (CLI-override-aware) value as the kwarg, while `sim_config.evolution.episodes_per_eval` would be the un-overridden YAML value. This makes `--episodes` work consistently for both `EpisodicSuccessRate` and `LearnedPerformanceFitness`. See Decision 5's "Asymmetry" paragraph for the rationale on why K reads from `sim_config.evolution` directly while L falls through the protocol kwarg
- [ ] 4.8 Eval phase: `FrozenEvalRunner()` × L episodes — M0's existing dual-override; success counted via `result.termination_reason == TerminationReason.COMPLETED_ALL_FOOD`
- [ ] 4.9 Unit test: `test_learned_performance_smoke_K2_L1` — call `evaluate` with K=2, L=1; verify it returns a float in `[0, 1]` without error
- [ ] 4.10 Unit test: `test_learned_performance_eval_env_is_fresh` — patch `create_env_from_config` with a counting mock; assert it's called exactly twice during a single `evaluate()` invocation (once for train env, once for eval env). This locks Decision 5's invariant in
- [ ] 4.10a Unit test: `test_learned_performance_no_evolution_block_raises` — invoke `evaluate` with a `SimulationConfig` whose `evolution` field is `None`; assert `ValueError` is raised whose message mentions the missing `evolution:` block (per task 4.2's None guard). Locks Decision 5's defensive contract in
- [ ] 4.10b Unit test: `test_learned_performance_no_environment_raises` — invoke `evaluate` with a `SimulationConfig` whose `environment` field is `None` (and `evolution` populated, so the first guard passes); assert `ValueError` whose message mirrors M0's `EpisodicSuccessRate.evaluate` ("requires sim_config.environment to be set"). Locks the second defensive branch from task 4.2
- [ ] 4.10c Unit test: `test_learned_performance_no_reward_raises` — invoke `evaluate` with a `SimulationConfig` whose `reward` field is `None` (and both `evolution` and `environment` populated); assert `ValueError` whose message mirrors M0's `EpisodicSuccessRate.evaluate` ("requires sim_config.reward to be set"). Locks the third defensive branch from task 4.2
- [ ] 4.11 Unit test: `test_learned_performance_K0_raises` — `evolution_config.learn_episodes_per_eval=0` raises `ValueError` mentioning `EpisodicSuccessRate`
- [ ] 4.12 Unit test: `test_learned_performance_eval_episodes_falls_back_to_kwarg` — `evolution.eval_episodes_per_eval=None`; invoke `evaluate(..., episodes=5, ...)`; eval phase runs exactly 5 episodes (use a mocked runner to count calls). Critically, this asserts the fallback uses the protocol's `episodes` kwarg (which the loop wires from the CLI-resolved `evolution_config.episodes_per_eval`), NOT `sim_config.evolution.episodes_per_eval` directly. Add a sibling test `test_learned_performance_eval_episodes_yaml_wins_over_kwarg` — `evolution.eval_episodes_per_eval=3`; invoke `evaluate(..., episodes=99, ...)`; eval runs exactly 3 episodes (YAML override wins)
- [ ] 4.13 Unit test: `test_learned_performance_train_phase_calls_learn` — patch `MLPPPOBrain.learn` with a `Mock`; assert it's called > 0 times during the K train episodes
- [ ] 4.14 Unit test: `test_learned_performance_eval_phase_does_not_call_learn` — same approach as M0's `test_frozen_eval_runner_never_calls_learn`; assert `learn` is NOT called during the L eval episodes (verifies eval phase actually uses `FrozenEvalRunner`)
- [ ] 4.15 Unit test: `test_learned_performance_score_uses_termination_reason` — mock the eval runner to return controlled `EpisodeResult` objects with `COMPLETED_ALL_FOOD` and `MAX_STEPS`; verify the score counts only `COMPLETED_ALL_FOOD`

## Phase 4.5: Loop wires `birth_metadata` for hyperparameter genomes

**Dependencies**: Phase 1 (the `hyperparam_schema` field on `SimulationConfig`) AND Phase 3 (the `HyperparameterEncoder` exists for the loop tests in 4.5.3 + 4.5.5 to actually run end-to-end).
**Parallelizable**: Independent of Phases 2 and 4; must follow Phase 3.

**Implementation-ordering note**: task 4.5.2 (the `build_birth_metadata` helper in `encoders.py`) is consumed by task 3.2 (`HyperparameterEncoder.initial_genome`). In practice the helper should be authored alongside or before task 3.2 — Phase 4.5 only sequences after Phase 3 because 4.5.1 and 4.5.3-4.5.5 need the encoder, not the helper. A pragmatic implementation order: 4.5.2 (helper) → 3.2 (encoder uses helper) → 4.5.1 (loop uses helper) → 4.5.3-4.5.5 (loop integration tests).

The `EvolutionLoop` currently constructs `Genome` instances directly from optimiser-sampled vectors at two sites: the worker-handoff path ([evolution/loop.py:124-128](packages/quantum-nematode/quantumnematode/evolution/loop.py#L124)) and the lineage-record path ([evolution/loop.py:316-321](packages/quantum-nematode/quantumnematode/evolution/loop.py#L316)). Neither populates `birth_metadata`. M0's `MLPPPOEncoder.decode` works around this with a fallback that derives `shape_map` from the brain's actual structure ([encoders.py:252-257](packages/quantum-nematode/quantumnematode/evolution/encoders.py#L252)). `HyperparameterEncoder` cannot use the same fallback — the schema lives only in `sim_config.hyperparam_schema` and would need a side-channel from the loop without this wiring.

- [ ] 4.5.1 Modify [evolution/loop.py](packages/quantum-nematode/quantumnematode/evolution/loop.py) to populate `Genome.birth_metadata` at BOTH construction sites (worker eval args at line 124-128 AND lineage record at line 316-321). Both sites SHALL import `build_birth_metadata` from `quantumnematode.evolution.encoders` and pass `build_birth_metadata(sim_config)` directly to the `Genome(birth_metadata=...)` constructor. The schema-or-empty branching SHALL live only in `build_birth_metadata` (added in task 4.5.2 — see also the storage-format contract in Decision 4 and the spec scenario "Schema travels with the genome to workers"). When `sim_config.hyperparam_schema is None`, the helper returns `{}` and M0's empty-`birth_metadata` behaviour is preserved verbatim
- [ ] 4.5.2 Add a public module-level helper `build_birth_metadata(sim_config: SimulationConfig) -> dict[str, Any]` to [packages/quantum-nematode/quantumnematode/evolution/encoders.py](packages/quantum-nematode/quantumnematode/evolution/encoders.py) (peer of `select_encoder` from task 5.4). Returns `{"param_schema": [entry.model_dump() for entry in sim_config.hyperparam_schema]}` when the schema is set, otherwise an empty dict. Living in `encoders.py` (not `loop.py`) makes it (a) the single source of truth shared between `HyperparameterEncoder.initial_genome` (task 3.2) AND `EvolutionLoop`'s two genome construction sites (task 4.5.1) — preventing format drift between test/programmatic and production paths; (b) unit-testable in isolation; (c) co-located with the Genome-construction concern. The dependency direction (`loop.py` imports from `encoders.py`) is already established in the M0 codebase
- [ ] 4.5.3 Unit test: `test_loop_populates_param_schema_in_birth_metadata` — construct an `EvolutionLoop` with a sim_config that has `hyperparam_schema`, run one generation with a population of 2, intercept the genomes (via a recording fitness that captures the genome arg before delegating), assert `genome.birth_metadata["param_schema"]` is a list of dicts whose names match the schema
- [ ] 4.5.4 Unit test: `test_loop_birth_metadata_empty_when_no_hyperparam_schema` — same as above but with `hyperparam_schema=None` (M0 weight-evolution config); assert `genome.birth_metadata` is empty (M0 back-compat). Pairs with the Decision 1/2 guarantee that hyperparameter evolution is fully opt-in
- [ ] 4.5.5 Unit test: `test_lineage_genome_carries_param_schema` — after a 1-gen run with a populated `hyperparam_schema`, the in-memory `Genome` instance passed to `LineageTracker.record(...)` SHALL have populated `birth_metadata["param_schema"]`. Use a recording lineage tracker mock to capture the `Genome` argument. Note: the lineage CSV's flat row format is unchanged from M0 (the assertion targets only the in-memory genome instance, not the on-disk CSV)
- [ ] 4.5.6 Unit test: `test_build_birth_metadata_with_schema_returns_dump` — call `build_birth_metadata(sim_config)` with a populated `hyperparam_schema`, assert the result is `{"param_schema": [<list of dicts>]}` where each dict has the expected `name`/`type`/etc. fields from `entry.model_dump()`. Sibling test `test_build_birth_metadata_no_schema_returns_empty` — same call with `sim_config.hyperparam_schema is None`, assert `{}`. Locks the helper's contract independent of any caller

## Phase 5: CLI dispatch

**Dependencies**: Phase 3, Phase 4
**Parallelizable**: No

- [ ] 5.1 Add `--fitness` flag to `scripts/run_evolution.py` with choices `{"success_rate", "learned_performance"}`, default `"success_rate"`
- [ ] 5.2 Pick fitness class in `main()` based on the flag — instantiate `EpisodicSuccessRate()` or `LearnedPerformanceFitness()`
- [ ] 5.3 When `--fitness learned_performance`, validate (a) that `sim_config.hyperparam_schema is not None` (per Decision 0 — combining weight encoder + learned-performance fitness would silently be Lamarckian inheritance, which is M3 scope); (b) that `evolution_config.learn_episodes_per_eval > 0`. If either check fails, log a clear error pointing the user to the right next step and `return 1`
- [ ] 5.4 Add a public helper `select_encoder(sim_config: SimulationConfig) -> GenomeEncoder` in [packages/quantum-nematode/quantumnematode/evolution/encoders.py](packages/quantum-nematode/quantumnematode/evolution/encoders.py) (peer of M0's `get_encoder`). Logic: when `sim_config.hyperparam_schema is not None`, return `HyperparameterEncoder()` directly (NOT via registry — see task 3.5); otherwise return `get_encoder(sim_config.brain.name)` (the existing M0 path). Per Decision 0, no separate `--encoder` flag. `scripts/run_evolution.py:main()` imports and calls `select_encoder(sim_config)` instead of inlining the dispatch logic. Locating the helper in `evolution/encoders.py` (rather than `scripts/run_evolution.py`) makes it (a) unit-testable in isolation without subprocess-importing the script, and (b) available to any programmatic caller of `EvolutionLoop` — not just the CLI
- [ ] 5.5 Subprocess test: `test_run_evolution_cli_fitness_flag_default_is_success_rate` — invoke without `--fitness`, verify it runs (back-compat)
- [ ] 5.6 Subprocess test: `test_run_evolution_cli_fitness_learned_performance_requires_K` — invoke `--fitness learned_performance` against a config with `learn_episodes_per_eval=0`, verify exit code is 1 and stderr mentions the field
- [ ] 5.7 Subprocess test: `test_run_evolution_cli_fitness_learned_performance_requires_hyperparam_schema` — invoke `--fitness learned_performance` against `configs/evolution/mlpppo_foraging_small.yml` (which has no `hyperparam_schema`), verify exit code is 1 and stderr names "hyperparam_schema" and "Lamarckian inheritance" / "M3" in the rationale

## Phase 6: MLPPPO pilot config

**Dependencies**: Phase 1
**Parallelizable**: Yes

- [ ] 6.1 Create `configs/evolution/hyperparam_mlpppo_pilot.yml`. Brain block mirrors `configs/scenarios/foraging/mlpppo_small_oracle.yml` (the source of the M2 baseline). 7 evolved hyperparameters (per the approved plan):
  - `actor_hidden_dim` (int, [32, 256])
  - `critic_hidden_dim` (int, [32, 256])
  - `num_hidden_layers` (int, [1, 3])
  - `learning_rate` (float, [1e-5, 1e-2], log_scale: true)
  - `gamma` (float, [0.9, 0.999])
  - `entropy_coef` (float, [1e-4, 0.1], log_scale: true)
  - `num_epochs` (int, [1, 8])
- [ ] 6.2 `evolution:` block — `algorithm: cmaes`, `population_size: 12`, `generations: 20`, `learn_episodes_per_eval: 30`, `eval_episodes_per_eval: 5`, `parallel_workers: 4`, `checkpoint_every: 2`, `cma_diagonal: false` (n=7 is well within full-cov tractability — opting OUT of diagonal here is the intentional M2 hyperparam-evolution default)
- [ ] 6.3 Pilot-config YAML comment header: explains the schema rationale, the cma_diagonal opt-out, and the decision-gate criteria
- [ ] 6.4 Unit test: `test_mlpppo_pilot_config_loads` — `load_simulation_config` against the pilot YAML succeeds; `cfg.hyperparam_schema` is populated; `cfg.evolution.learn_episodes_per_eval == 30`
- [ ] 6.5 Unit test: `test_mlpppo_pilot_config_dispatch_routes_to_hyperparam_encoder` — load the pilot YAML, call `select_encoder(cfg)` (the public helper added in task 5.4 at `quantumnematode.evolution.encoders.select_encoder`), assert it returns a `HyperparameterEncoder` instance. Also assert that loading `configs/evolution/mlpppo_foraging_small.yml` (no `hyperparam_schema`) and calling `select_encoder` returns an `MLPPPOEncoder` instance — back-compat with M0 dispatch

## Phase 7: Smoke test

**Dependencies**: Phase 5, Phase 6
**Parallelizable**: No

- [ ] 7.1 Add `@pytest.mark.smoke test_run_evolution_smoke_hyperparam_mlpppo` to `packages/quantum-nematode/tests/quantumnematode_tests/test_smoke.py`. Invokes `scripts/run_evolution.py --config configs/evolution/hyperparam_mlpppo_pilot.yml --fitness learned_performance --generations 1 --population 4 --seed 42` with overrides for `learn_episodes_per_eval=2` and `eval_episodes_per_eval=1` via CLI flags (or via a tmp YAML if CLI overrides for those fields aren't added — the spec doesn't require them; happy with overriding via a tmp config copy if simpler). Asserts exit 0 and no `Traceback` in stderr
- [ ] 7.2 Verify the smoke completes in \<30 s on a typical dev machine (run locally; document in PR body)

## Phase 8: Campaign script + pilot run

**Dependencies**: Phases 5-7 (everything green)
**Parallelizable**: No (this IS the pilot)

- [ ] 8.1 Create `scripts/campaigns/phase5_m2_hyperparam_mlpppo.sh`. Loops 4 seeds (42, 43, 44, 45), invokes `run_evolution.py --config configs/evolution/hyperparam_mlpppo_pilot.yml --fitness learned_performance --seed $SEED --output-dir evolution_results` for each, and tags the runs with the seed in the session id pathway
- [ ] 8.2 Run the pilot: 4 seeds × 20 gens × pop 12. Wall time estimate: 30 train + 5 eval = 35 ep × 12 genomes × 20 gens × 4 seeds × ~50 ms/ep / parallel 4 ≈ 1.75 hours. Background it; check back when done
- [ ] 8.3 Run the hand-tuned MLPPPO baseline for comparison: 4 seeds of `configs/scenarios/foraging/mlpppo_small_oracle.yml` for matching number of episodes (eval-only, e.g. 100 episodes per seed). Use the existing `scripts/run_simulation.py` and `nematode-run-experiments` skill convention
- [ ] 8.4 Capture both result sets and compute mean success rates per seed and per generation; plot the convergence curve

## Phase 9: Logbook

**Dependencies**: Phase 8 results
**Parallelizable**: No

- [ ] 9.1 Use the `nematode-logbook` skill to draft `artifacts/logbooks/012/hyperparam_pilot_mlpppo.md`. Include: pilot config summary (table of evolved hyperparams + bounds); hand-tuned baseline reference; convergence-curve plot; per-seed final fitness; mean+std over seeds; the GO/PIVOT/STOP decision against the gate (≥3pp over baseline AND fitness still rising at gen 20)
- [ ] 9.2 If the decision is GO/PIVOT, the logbook explicitly states what PR 3's LSTMPPO arm should test (e.g. "evolve the same 6 LSTMPPO-equivalent hyperparams under the same pilot shape"). If STOP, the logbook states the reason and what an alternate Phase 5 strategy might look like
- [ ] 9.3 Move the actual `evolution_results/<seed>/` directories somewhere committable or referenced via Git LFS — follow the same pattern as logbook 011 (klinotaxis evaluation)

## Phase 10: Pre-PR Verification + M-1 Invariant

**Dependencies**: Phases 1-9 complete
**Parallelizable**: No (final checks before opening the PR)

- [ ] 10.1 Update `openspec/changes/2026-04-26-phase5-tracking/tasks.md` — tick `M2.1`, `M2.2`, `M2.3`, `M2.5` (only MLPPPO arm shipped here), and `M2.7` (logbook published). Leave `M2.4` (LSTMPPO pilot config) and `M2.6` LSTMPPO arm for PR 3. Also: the Phase 5 Research Questions section + RQ1 (optimiser portfolio re-evaluation) are added in this same edit — recorded by Decision N (Considered Alternatives — Optimiser choice) in this PR's `design.md`. Already done as part of drafting; verify the section is committed
- [ ] 10.2 Update `docs/roadmap.md` Phase 5 Milestone Tracker — flip M2 row from `🔲 not started` to `🟡 in progress` (LSTMPPO arm pending in PR 3). Will flip to `✅ complete` in PR 3. Also: the "Phase 5 research questions" paragraph after the milestone tracker is added in this same edit — already done as part of drafting; verify present
- [ ] 10.3 Run `openspec validate 2026-04-27-add-hyperparameter-evolution --strict` — passes
- [ ] 10.4 Run `uv run pre-commit run -a` — clean
- [ ] 10.5 Run `uv run pytest -m "not nightly"` — all green. M0+perf baseline is 2193; this PR adds ~37 new unit/subprocess tests (Phases 1-7 + 4 in Phase 4.5 [4.5.3-4.5.6] + 1 sibling test in 4.12 + 2 sibling tests in 4.10b/4.10c) so the new total is ≈ 2230
- [ ] 10.6 Archive in-branch: run `openspec archive 2026-04-27-add-hyperparameter-evolution -y`. The change moves into `openspec/changes/archive/` and the spec deltas are merged into `openspec/specs/evolution-framework/spec.md`. Same pattern as M0's PR
- [ ] 10.7 Manually verify: pilot logbook references the right session directories; convergence-curve plot is committed (or LFS-tracked)

> **Out-of-task actions** (performed after all 10.x tasks are checked, NOT tracked as tasks in this change):
>
> 1. Open the PR with Conventional Commits prefix per AGENTS.md: `feat: hyperparameter evolution + MLPPPO pilot (M2)`
> 2. Merge the PR
>
> No follow-up commit on `main` is required — the archive and roadmap-row update both ship inside the PR.

## Deferred to PR 3 (LSTMPPO+klinotaxis pilot)

These items belong to M2 but are out of scope for this PR by design:

- LSTMPPO+klinotaxis pilot config (`configs/evolution/hyperparam_lstmppo_klinotaxis_pilot.yml`)
- LSTMPPO arm of the M2 campaign (2 seeds, smaller-than-MLPPPO due to wall-time budget)
- LSTMPPO logbook (`artifacts/logbooks/012/hyperparam_pilot_lstmppo.md`)
- Final M2 row flip on `docs/roadmap.md` from `🟡 in progress` to `✅ complete`
- Tick of `M2.4`, `M2.6` (LSTMPPO arm), and `M2.8` (final checklist + roadmap)

When authoring the LSTMPPO pilot YAML, watch for the **cross-field brain-config constraints** flagged in `design.md` Decision 7 — most relevantly `feature_gating: True` requiring `feature_expansion != "none"`, and the `rnn_type` categorical (LSTM vs GRU) interacting with `lstm_hidden_dim` (some sizes work better for one rnn_type than the other). The schema validator does NOT enforce these — it's the schema author's responsibility to either hold conflicting fields constant or co-evolve them with awareness of the constraint.

PR 3 is gated on this PR's MLPPPO decision — if STOP, PR 3 may be skipped or restructured.
