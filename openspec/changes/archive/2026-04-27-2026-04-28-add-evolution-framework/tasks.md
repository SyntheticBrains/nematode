# Tasks: M0 — Brain-Agnostic Evolution Framework

## Phase 1: New Module Skeleton

**Dependencies**: None
**Parallelizable**: No (foundational)

- [x] 1.1 Create `packages/quantum-nematode/quantumnematode/evolution/` directory with `__init__.py` exporting public API
- [x] 1.2 Create `genome.py`: `Genome` dataclass (`params: np.ndarray`, `genome_id: str`, `parent_ids: list[str]`, `generation: int`, `birth_metadata: dict[str, Any]`)
- [x] 1.3 Add `genome_id_for(generation: int, index: int, parent_ids: list[str]) -> str` helper. **Implementation**: use `uuid.uuid5(uuid.NAMESPACE_OID, f"gen{generation}-idx{index}-parents{','.join(sorted(parent_ids))}")` so identical inputs produce identical IDs and any input change produces a different ID
- [x] 1.4 Unit tests: `test_genome_id_deterministic` (same inputs → same UUID); `test_genome_id_distinct_for_distinct_inputs` (vary `generation`, `index`, OR `parent_ids` independently — each variation produces a different UUID. Note: under Decision 5a, all children in the same generation share the same `parent_ids` list, so within-generation `parent_ids` variation is rare in practice, but the helper itself remains general for M3 Lamarckian inheritance which may pass per-child parent IDs); `test_genome_id_format_is_uuid` (returned string parses as a valid UUID); `test_genome_id_parent_ids_order_independent` (sorting internally means parent_ids order doesn't affect result)

## Phase 2: Encoder Protocol, Brain Factory Wrapper, and Concrete Encoders

**Dependencies**: Phase 1
**Parallelizable**: No

- [x] 2.1 Create `evolution/brain_factory.py` with `instantiate_brain_from_sim_config(sim_config: SimulationConfig, *, seed: int | None = None) -> Brain`. Body (matches Decision 0 sketch verbatim): (1) extract `BrainConfig` via `configure_brain(sim_config)`; (2) build `overrides = {"weights_path": None}` (force `weights_path=None` so the genome is the sole weight source — see Decision 0); if `seed is not None`, add `overrides["seed"] = seed`; apply via `brain_config = brain_config.model_copy(update=overrides)`; (3) **convert config-shaped fields to runtime objects via the existing helpers in [utils/config_loader.py](packages/quantum-nematode/quantumnematode/utils/config_loader.py)**, matching the canonical pattern in [scripts/run_simulation.py](scripts/run_simulation.py): `learning_rate = configure_learning_rate(sim_config)` (single arg), `gradient_method, gradient_max_norm = configure_gradient_method(GradientCalculationMethod.RAW, sim_config)` (**TWO args, returns a tuple** — `RAW` is a no-op default since classical brains in evolution don't use gradient methods, the helper extracts any user-configured `max_norm`), `parameter_initializer_config = configure_parameter_initializer(sim_config)`. Do NOT pass `sim_config.learning_rate` directly — that's a `LearningRateConfig` Pydantic model and `setup_brain_model` expects a runtime object like `DynamicLearningRate`. Do NOT compute `gradient_max_norm` from `sim_config.gradient.max_norm` directly — it's part of the `configure_gradient_method` tuple return; (4) coerce brain name to enum via `brain_type = BrainType(sim_config.brain.name)`; (5) dispatch to `setup_brain_model(brain_type=brain_type, brain_config=brain_config, shots=sim_config.shots or 1024, qubits=sim_config.qubits or 0, device=DeviceType.CPU, learning_rate=learning_rate, gradient_method=gradient_method, gradient_max_norm=gradient_max_norm, parameter_initializer_config=parameter_initializer_config)`. The seed-patching pattern (`BrainConfig.seed`, NOT `SimulationConfig.seed`) matches [scripts/run_simulation.py](scripts/run_simulation.py)
- [x] 2.2 Unit test: `test_instantiate_mlpppo_from_sim_config` (load a fixture sim_config, call wrapper, assert `isinstance(brain, MLPPPOBrain)` and `brain._episode_count == 0`)
- [x] 2.3 Unit test: `test_instantiate_lstmppo_from_sim_config` (same with LSTMPPO)
- [x] 2.3a Unit test: `test_instantiate_brain_seed_patches_brain_config_not_sim_config` — load sim_config with `brain.config.seed = 0`, call `instantiate_brain_from_sim_config(sim_config, seed=42)`, assert `brain.seed == 42` (proves the wrapper patches `BrainConfig.seed`, not `SimulationConfig.seed`)
- [x] 2.3b Unit test: `test_instantiate_brain_forces_weights_path_none` — load sim_config with `brain.config.weights_path = "./fake.pt"` (file does not exist; the path being set is what matters), call wrapper, assert no `FileNotFoundError` is raised (proves `weights_path` was forced to `None` before brain construction)
- [x] 2.4 Create `encoders.py` with `GenomeEncoder` protocol whose methods take **the full `SimulationConfig`**. `decode` also accepts an optional `seed` kwarg so the fitness function can override the brain's RNG seed for the current evaluation: `initial_genome(sim_config: SimulationConfig, *, rng) -> Genome`, `decode(genome, sim_config, *, seed: int | None = None) -> Brain`, `genome_dim(sim_config) -> int`. Plus module-level constant `NON_GENOME_COMPONENTS = {"optimizer", "actor_optimizer", "critic_optimizer", "training_state"}` (the denylist)
- [x] 2.5 Implement private `_flatten_components(components: dict[str, WeightComponent]) -> tuple[np.ndarray, dict]` (returns flat array + shape map for unflatten — walks components in deterministic key-sorted order)
- [x] 2.6 Implement private `_unflatten_components(params: np.ndarray, shape_map: dict) -> dict[str, WeightComponent]`
- [x] 2.7 Implement `MLPPPOEncoder`: `brain_name = "mlpppo"`. `decode(genome, sim_config, *, seed=None)` calls `instantiate_brain_from_sim_config(sim_config, seed=seed)` to get a fresh brain (with `BrainConfig.seed` patched to `seed` when supplied), then `load_weight_components()` to apply the genome. **Dynamic discovery**: filters `get_weight_components()` output by `NON_GENOME_COMPONENTS`. Picks up `policy`, `value`, and conditional `gate_weights`. After `load_weight_components`, sets `brain._episode_count = 0` AND calls `brain._update_learning_rate()` (so the LR matches the reset count — see Decision 2)
- [x] 2.8 Implement `LSTMPPOEncoder`: `brain_name = "lstmppo"`. `decode(genome, sim_config, *, seed=None)` follows the same pattern as MLPPPOEncoder. Same dynamic-discovery + denylist pattern. Picks up `lstm`, `layer_norm`, `policy`, `value`. Same `_episode_count = 0` + `_update_learning_rate()` after load. Per-episode hidden state (`_pending_h_state`, `_pending_c_state`) AND `_step_count` both reset in `prepare_episode()` ([lstmppo.py:873-877](packages/quantum-nematode/quantumnematode/brain/arch/lstmppo.py#L873)) which is called by the runner before the first step of every episode — encoder doesn't need to handle these explicitly
- [x] 2.9 Define `ENCODER_REGISTRY: dict[str, type[GenomeEncoder]] = {"mlpppo": MLPPPOEncoder, "lstmppo": LSTMPPOEncoder}`
- [x] 2.10 Unit test: `test_mlpppo_encoder_roundtrip` (encode → decode → identical first-step action on seeded input)
- [x] 2.11 Unit test: `test_lstmppo_encoder_roundtrip` (same with recurrent brain — verifies all four components round-trip including `layer_norm`)
- [x] 2.12 Unit test: `test_genome_dim_matches_flattened_state` (for both encoders)
- [x] 2.13 Unit test: `test_episode_count_resets_and_lr_synced_on_decode` (verify both `_episode_count == 0` AND the LR matches what a freshly constructed brain has, for both brain types)
- [x] 2.15 Unit test: `test_encoder_registry_membership` (asserts `"mlpppo" in ENCODER_REGISTRY` and `"lstmppo" in ENCODER_REGISTRY` and that `ENCODER_REGISTRY[name]()` produces a working encoder for both)
- [x] 2.16 Unit test: `test_encoder_excludes_denylist_components` (verifies `optimizer`, `actor_optimizer`, `critic_optimizer`, `training_state` are NEVER in the genome regardless of brain)

## Phase 3: Fitness Function

**Dependencies**: Phase 2
**Parallelizable**: Can start in parallel with Phase 4

- [x] 3.1 Create `fitness.py` with `FitnessFunction` protocol (single `evaluate(genome: Genome, sim_config: SimulationConfig, encoder: GenomeEncoder, *, episodes: int, seed: int) -> float` method). Note: signature takes the full `SimulationConfig`, matching the encoder API
- [x] 3.2 Implement `FrozenEvalRunner(StandardEpisodeRunner)` in `fitness.py`: subclass of [`StandardEpisodeRunner`](packages/quantum-nematode/quantumnematode/agent/runners.py#L599) that overrides BOTH `run()` and `_terminate_episode()`. The two override points are needed because the standard runner calls `learn` in two places: per-step at [runners.py:747](packages/quantum-nematode/quantumnematode/agent/runners.py#L747) (fires every step, no kwarg override possible) and per-termination via `_terminate_episode` (defaults `learn=True` on success). (1) `run(self, agent, reward_config, max_steps, **kwargs)` saves `agent.brain.learn` and `agent.brain.update_memory`, replaces them with no-op lambdas, calls `super().run(...)` inside a `try`, and restores the originals in `finally`. This neutralises the per-step call. (2) `_terminate_episode(self, agent, params, reward, **kwargs)` forces `kwargs["learn"] = False; kwargs["update_memory"] = False` then calls `super()._terminate_episode(agent, params, reward, **kwargs)`. **All other kwargs (including the `food_history=...` Ellipsis sentinel) MUST pass through unchanged** — do NOT use `kwargs.get("food_history")` because that converts the sentinel to `None` and breaks the parent's `food_history is ...` fallback to `agent.food_history`. This is belt-and-braces — `run()` already neutered the brain methods, but the kwarg override keeps the success path ([runners.py:817-823](packages/quantum-nematode/quantumnematode/agent/runners.py#L817)) honest. All step-loop logic is inherited unchanged
- [x] 3.3 Implement private helper `_build_agent(brain: Brain, env: DynamicForagingEnvironment, sim_config: SimulationConfig) -> QuantumNematodeAgent` in `fitness.py`: passes `brain`, `env`, `satiety_config=sim_config.satiety`, and `sensing_config=sim_config.environment.sensing if sim_config.environment else None` (inline access, no separate helper). Other `QuantumNematodeAgent.__init__` args use defaults (theme/rich_style_config are presentation-only; agent_id defaults to `"default"`)
- [x] 3.4 Implement `EpisodicSuccessRate.evaluate()`. **Apply `seed` by calling `encoder.decode(genome, sim_config, seed=seed)`** (per Decision 3a). The encoder forwards `seed` to `instantiate_brain_from_sim_config(sim_config, seed=seed)`, which patches `BrainConfig.seed` (the field the brain's `__init__` actually reads — NOT `SimulationConfig.seed`); the brain's constructor then calls `set_global_seed(seed)` ([mlpppo.py:170](packages/quantum-nematode/quantumnematode/brain/arch/mlpppo.py#L170)) seeding numpy global, torch global, and the brain's local RNG via `self.rng = get_rng(seed)`. **Do NOT call `torch.manual_seed(seed)` or assign `brain.rng` directly, AND do NOT call `sim_config.model_copy(update={"seed": seed})` at the fitness layer** — none of those propagate correctly. Then `env = create_env_from_config(sim_config.environment, seed=seed)`, `agent = _build_agent(brain, env, sim_config)`, instantiate `FrozenEvalRunner()`, loop `episodes` calls to `runner.run(agent, sim_config.reward, sim_config.max_steps)`, count successes via `result.termination_reason == TerminationReason.COMPLETED_ALL_FOOD`, return `successes / episodes`. Success detection uses `TerminationReason.COMPLETED_ALL_FOOD` (codebase convention, see [experiment/tracker.py:304](packages/quantum-nematode/quantumnematode/experiment/tracker.py#L304)) — **NOT** a non-existent `result.episode_success` attribute. The `TerminationReason` import is `from quantumnematode.report.dtypes import TerminationReason` (NOT `quantumnematode.agent.dtypes`)
- [x] 3.5 **Frozen-weight contract**: with `FrozenEvalRunner`, `brain.learn()` and `brain.update_memory()` are never called regardless of episode outcome. `prepare_episode()` and `post_process_episode()` are still called by the inherited per-step logic — this advances `_episode_count` but does not change weights. (LearnedPerformanceFitness is M2.)
- [x] 3.6 Unit test: `test_episodic_success_rate_returns_float_in_unit_interval` (fitness is a finite float in `[0.0, 1.0]` for an arbitrary genome — no assumption that random brains fail)
- [x] 3.7 Unit test: `test_episodic_success_rate_deterministic_for_seeded_genome` (same genome + seed → byte-identical fitness across two invocations; this is the test that fails if seeding is incomplete)
- [x] 3.8 Unit test: `test_frozen_eval_runner_never_calls_learn` (mock `brain.learn` and `brain.update_memory`; run a successful episode via `FrozenEvalRunner.run()`; assert neither was called)
- [x] 3.9 Unit test: `test_frozen_eval_runner_never_calls_learn_on_success` (specifically construct a scenario where the standard runner *would* default to `learn=True` — i.e. `TerminationReason.COMPLETED_ALL_FOOD` — and verify the override forces `learn=False`)
- [x] 3.10 Unit test: `test_episodic_success_rate_uses_termination_reason_for_success` (mock the runner to return `EpisodeResult` with `COMPLETED_ALL_FOOD`; assert fitness counts it as success. Repeat with `MAX_STEPS`; assert it's not counted)
- [x] 3.11 Unit test: `test_frozen_eval_runner_preserves_food_history_sentinel` (verify `EpisodeResult.food_history` returned by `FrozenEvalRunner.run()` matches what `StandardEpisodeRunner.run()` returns on the same episode — i.e. the override doesn't silently drop `food_history` to `None`)
- [x] 3.12 Unit test: `test_evaluate_passes_seed_to_encoder_decode` — instrument `encoder.decode` to record the `seed` kwarg it received; instrument `create_env_from_config` to record its `seed` arg; call `evaluate(sim_config_with_brain_config_seed=99, seed=42)`; assert the encoder.decode saw `seed=42` (not 99) AND the env saw `seed=42`. This verifies the encoder receives `seed` as a kwarg (not via `sim_config.model_copy`) and that the env factory gets the same value
- [x] 3.13 Unit test: `test_evaluate_seed_overrides_brain_config_seed_changes_fitness` — call `evaluate(sim_config_with_brain_config_seed=0, seed=1)` then `evaluate(sim_config_with_brain_config_seed=0, seed=2)`; assert the two fitnesses differ (proves the fitness function's `seed` parameter actually patches `BrainConfig.seed` and changes behaviour — independent of the YAML-configured `BrainConfig.seed`)

## Phase 4: Lineage Tracker

**Dependencies**: Phase 1
**Parallelizable**: Yes

- [x] 4.1 Create `lineage.py` with `LineageTracker(output_path: Path)` class. The tracker is owned by the **parent process only** — workers report fitnesses back to the parent, and the parent calls `record()`. Workers MUST NOT instantiate or write to the tracker directly (no concurrent-write hazard)
- [x] 4.2 `record(genome: Genome, fitness: float, brain_type: str)` appends a CSV row
- [x] 4.3 CSV columns: `generation, child_id, parent_ids, fitness, brain_type` (parent_ids `;`-joined for CSV-safety)
- [x] 4.4 Append mode (not write mode) so resume works without recreating. On `__init__`, check whether the file exists: if yes, skip writing the header; if no, write it
- [x] 4.5 Unit test: `test_lineage_csv_appends_across_generations` (record gens `0..4` inclusive — i.e. 5 generations × population 4 = 20 data rows + 1 header row; parent_ids correctly populated for gen >= 1)
- [x] 4.6 Unit test: `test_lineage_csv_gen_zero_has_empty_parent_ids` (gen 0 rows have empty `parent_ids` field)
- [x] 4.7 Unit test: `test_lineage_csv_header_only_written_once` (instantiate tracker → record → close; reinstantiate same path → record → close; assert header appears exactly once)
- [x] 4.8 Unit test: `test_lineage_generation_indexing_is_zero_based` (a run with `generations: G` produces rows whose `generation` column takes values in `[0, G-1]`, each appearing exactly P times)

## Phase 5: Evolution Loop

**Dependencies**: Phases 2, 3, 4
**Parallelizable**: No

- [x] 5.1 Create `loop.py` with `EvolutionLoop` class taking `optimizer`, `encoder`, `fitness`, `sim_config`, `evolution_config`, `output_dir`, `rng`
- [x] 5.2 Implement `run(*, resume_from: Path | None = None) -> EvolutionResult`
- [x] 5.3 Generation loop: `optimizer.ask()` → wrap each candidate as a `Genome` whose `parent_ids` is the list of ALL genome IDs from the previous generation (per Decision 5a — uniform convention for both CMA-ES and GA, since neither optimiser exposes per-child parent provenance). For generation 0, `parent_ids = []`. Then parallel fitness eval → `optimizer.tell()` → record lineage → checkpoint every N gens
- [x] 5.4 Multiprocessing: reuse the worker pattern from legacy `run_evolution.py:452-470` (SIGINT-ignore, per-worker logger config)
- [x] 5.5 Worker function takes picklable args (`params: np.ndarray`, `sim_config: SimulationConfig` Pydantic model, `episodes: int`, `seed: int`) and reconstructs the brain inside the worker via `encoder.decode(genome, sim_config)`. **Pass the Pydantic model directly, not a dict** — Pydantic v2 models pickle cleanly via `__getstate__`/`__setstate__`, so workers get a typed `SimulationConfig` without re-parsing. No separate `brain_config` arg — sim_config carries everything `instantiate_brain_from_sim_config` needs
- [x] 5.6 Pickle checkpoint: dump `{optimizer, generation, rng_state, lineage_path, checkpoint_version: 1}` to `output_dir/checkpoint.pkl`
- [x] 5.7 Resume: load pickle, validate `checkpoint_version`, restore optimizer state, continue from saved generation
- [x] 5.8 On completion: write `best_params.json` (compatible with legacy artifact contract) and `history.csv` to `output_dir`
- [x] 5.9 Unit test: `test_loop_runs_3_generations_mlpppo` (minimal config, 3 gens, pop 4, 1 episode each — completes and produces best_params.json)
- [x] 5.10 Unit test: `test_loop_resume_from_checkpoint` (run 2 gens → kill → resume → run 3 more — total 5 gens in lineage CSV)
- [x] 5.11 Unit test: `test_loop_rejects_incompatible_checkpoint_version`
- [x] 5.12 Unit test: `test_checkpoint_contains_required_keys` (load a written checkpoint pickle and assert exact keys: `{"optimizer", "generation", "rng_state", "lineage_path", "checkpoint_version"}`)
- [x] 5.13 Unit test: `test_unknown_brain_name_raises_with_helpful_message` (instantiate `EvolutionLoop` with a brain config whose `name` is not in `ENCODER_REGISTRY`; assert `ValueError` whose message lists the registered brain names and notes Phase 6 deferral)
- [x] 5.14 Unit test: `test_lineage_parent_ids_lists_all_prev_generation_ids` — run a 3-gen × pop 4 loop, parse the resulting lineage CSV, assert that for every gen-1 row `parent_ids` is the `;`-joined set of all 4 gen-0 `child_id`s, and for every gen-2 row `parent_ids` is the set of all 4 gen-1 `child_id`s. Verifies Decision 5a's convention

## Phase 6: Configuration Schema Extension

**Dependencies**: None (can be done in parallel with Phases 1-5)
**Parallelizable**: Yes

- [x] 6.1 In `packages/quantum-nematode/quantumnematode/utils/config_loader.py`, add `EvolutionConfig` Pydantic model
- [x] 6.2 Fields: `algorithm: Literal["cmaes", "ga"] = "cmaes"`, `population_size: int = 20`, `generations: int = 50`, `episodes_per_eval: int = 15`, `sigma0: float = math.pi/2`, `elite_fraction: float = 0.2`, `mutation_rate: float = 0.1`, `crossover_rate: float = 0.8`, `parallel_workers: int = 1`, `checkpoint_every: int = 10`
- [x] 6.3 Add `evolution: EvolutionConfig | None = None` field to `SimulationConfig`
- [x] 6.4 Unit test: `test_existing_scenario_config_loads_without_evolution_block` (e.g. load `configs/scenarios/foraging/mlpppo_small_oracle.yml`; assert `SimulationConfig.evolution is None`)
- [x] 6.5 Unit test: `test_evolution_block_parses_into_populated_config` (load a fixture with `evolution:` block; assert all fields populated; assert unspecified fields use defaults)

## Phase 7: New CLI Script (legacy deleted, not preserved)

**Dependencies**: Phases 5, 6
**Parallelizable**: No

- [x] 7.1 **Delete** existing `scripts/run_evolution.py` (per Phase 5 decision: no `scripts/legacy/` fallback). Git history preserves it via `git log -- scripts/run_evolution.py`
- [x] 7.2 **Delete** existing `configs/evolution/qvarcircuit_foraging_small.yml` (no consumer remains)
- [x] 7.3 **Delete** existing `test_run_evolution_smoke` from `packages/quantum-nematode/tests/quantumnematode_tests/test_smoke.py` (CI smoke test for the deleted script)
- [x] 7.4 In the same `test_smoke.py` edit, add `test_run_evolution_smoke_mlpppo`: `@pytest.mark.smoke`, runs `scripts/run_evolution.py --config configs/evolution/mlpppo_foraging_small.yml --generations 1 --population 4 --episodes 2 --seed 42 --output-dir <tmp>` via `subprocess.run`, asserts `returncode == 0` and no `Traceback` in stderr. Same pattern as the deleted test. (Smoke test edits live alongside script deletion since they touch the same file.) Note: this test exercises the new script which is created in tasks 7.5+, so the test will fail until 7.5 lands — fine within the same PR, but order accordingly when running locally
- [x] 7.5 Create new `scripts/run_evolution.py` (~150 LOC): argparse → load config → instantiate encoder via registry → instantiate optimiser → instantiate `EvolutionLoop` → run
- [x] 7.6 CLI flags: `--config`, `--generations`, `--population`, `--episodes`, `--algorithm`, `--sigma`, `--parallel`, `--seed`, `--resume`, `--output-dir`, `--log-level`
- [x] 7.7 CLI flags default to `None` and only override YAML when explicitly passed (single-source-of-truth = `EvolutionConfig` defaults)
- [x] 7.8 On startup, log prominently: `Brain type: <name>`, `Algorithm: <cmaes|ga>`, `Population: <N>`, `Generations: <M>`
- [x] 7.9 Error message if `brain.name` is not in `ENCODER_REGISTRY`: `f"No encoder for brain '{name}'. Registered: {sorted(ENCODER_REGISTRY)}. Quantum brain support is deferred to a Phase 6 re-evaluation."`
- [x] 7.10 Subprocess test: `test_cli_overrides_yaml_for_generations` — invoke the new script via `subprocess.run` with `--config <yaml-with-generations:50> --generations 2`, assert run completes with exactly 2 generations recorded in lineage CSV

## Phase 8: Pilot Configs

**Dependencies**: Phase 6
**Parallelizable**: Yes

- [x] 8.1 Create `configs/evolution/mlpppo_foraging_small.yml`: `mlpppo` brain, 20×20 grid, oracle sensing, target_foods_to_collect ≈ 5, `evolution: {generations: 10, population_size: 8, episodes_per_eval: 3}`. **Do NOT specify `brain.config.weights_path`** — the wrapper forces it to `None` (genome is the sole weight source), so any value would be ignored. **Do NOT rely on `brain.config.seed`** — the fitness function's `seed` parameter overrides it per evaluation
- [x] 8.2 Create `configs/evolution/lstmppo_foraging_small_klinotaxis.yml`: copy brain block from `configs/scenarios/foraging/lstmppo_small_klinotaxis.yml` (gru, lstm_hidden_dim=64, klinotaxis sensing, STAM enabled), same minimal `evolution:` block. Same `weights_path` and `seed` notes as 8.1

## Phase 9: End-to-End Smoke Verification

**Dependencies**: Phases 7, 8
**Parallelizable**: No

- [x] 9.1 Run: `uv run python scripts/run_evolution.py --config configs/evolution/mlpppo_foraging_small.yml --generations 1 --population 4 --episodes 1` — completes without error (verified manually; full 10-gen × pop 8 deferred to M2 when timing matters)
- [x] 9.2 Verify `evolution_results/<session>/best_params.json` exists; verified it round-trips back into a working `MLPPPOBrain` (covered by `test_loop_best_params_json_round_trips_back_to_brain`)
- [x] 9.3 Verify lineage.csv shape — covered by `test_loop_writes_p_times_g_lineage_rows` for the 3-gen × pop 4 case (13 rows = header + 12 data); the full 10-gen × pop 8 = 80 rows case is by extension since the test covers the row-count formula
- [x] 9.4 Run LSTMPPO+klinotaxis smoke — verified end-to-end via background CLI run with `--generations 1 --population 2 --episodes 1`. Earlier attempts got SIGKILL'd at the Bash tool's 120 s timeout because the GRU forward pass + 1000-step episode is slow; given enough time (a few minutes) and left undisturbed, the run completes cleanly. Full 10-gen × pop 8 is left as manual verification when M2/M3 require it. Note for CI: an LSTMPPO smoke test with these minimal params would be ~2–3 min — acceptable for nightly but probably too slow for per-PR smoke
- [x] 9.5 Same artefact verification for LSTMPPO smoke — confirmed all 4 artefacts (best_params.json, history.csv, lineage.csv, checkpoint.pkl) present after the 9.4 run. best_params.json decodes back to a 46,989-param vector matching `LSTMPPOEncoder.genome_dim` for the pilot config; brain_type field reads "lstmppo"
- [x] 9.6 CLI resume test — `test_run_evolution_smoke_mlpppo_resume` in `test_smoke.py` exercises the `--resume <path>` CLI flag end-to-end via subprocess. Sequence: run 1 gen, force checkpoint via checkpoint_every=1, resume from the checkpoint and run 1 more gen, verify both subprocess invocations exit 0. Passes in ~10 s
- [x] 9.7 Run `uv run pytest -m smoke -v` — all green including the new MLPPPO smoke (`test_run_evolution_smoke_mlpppo` was added in task 7.4)

## Phase 10: Pre-PR Verification + M-1 Invariant Updates

**Dependencies**: Phases 1-9 complete
**Parallelizable**: No (these are the final checks before the PR is opened — every task here MUST be `[x]` before opening the PR or archiving the change)

- [x] 10.1 Update `openspec/changes/2026-04-26-phase5-tracking/tasks.md`: marked M0.1 → M0.14 complete (`[x]`); M0 status header set to `complete` in the milestone scaffold
- [x] 10.2 Update `docs/roadmap.md` Phase 5 Milestone Tracker table: M0 row set to `✅ complete` in this PR (the PR's merge IS the M0 completion event, so the row flip ships in the same diff as the framework code)
- [x] 10.3 Run `openspec validate --changes 2026-04-28-add-evolution-framework --strict` — passes
- [x] 10.4 Run `uv run pre-commit run -a` — clean (all 10 hooks passed: large files, EOF fix, YAML/TOML check, mdformat, markdownlint-cli2, ruff check, ruff format, pyright, tests)
- [x] 10.5 Run `uv run pytest -m "not nightly"` — all green (2188 passed in 95.37 s)
- [x] 10.6 Run `openspec archive 2026-04-28-add-evolution-framework -y` on this branch. This moves the change into `openspec/changes/archive/` and applies the `evolution-framework` capability deltas to the project's main specs under `openspec/specs/`. Archiving on the feature branch (not on `main` post-merge) keeps the PR a single self-contained unit: code + spec + archive land in one diff

> **Out-of-task actions** (performed after all 10.x tasks are checked, NOT tracked as tasks in this change):
>
> 1. Open the PR with Conventional Commits prefix per AGENTS.md: `feat: add brain-agnostic evolution framework (M0)`
> 2. Merge the PR — no follow-up commits on `main` are required; the archive and the roadmap row flip already shipped inside the PR

## Deferred to M2

These items belong to the brain-agnostic evolution framework's spec but are deliberately deferred to a later milestone, NOT left undone:

- **Test for "Conditional weight components are picked up automatically"** (the `_feature_gating: true` round-trip case from `specs/evolution-framework/spec.md`). The dynamic-discovery code path is already in place (`MLPPPOEncoder` uses the denylist filter, and the unrelated `test_encoder_excludes_denylist_components` exercises it on the standard config). The missing piece is a fixture config with `_feature_gating: true` plus an assertion that the genome dim is larger and the gated/ungated brains produce identical post-decode actions. Standard MLPPPO configs have `feature_gating: False` and there is no consumer of feature-gated MLPPPO in M0. The test will be added as part of M2 (hyperparameter pilot), where feature_gating becomes an evolvable hyperparam and the fixture config becomes load-bearing.
