# Tasks: M0 — Brain-Agnostic Evolution Framework

## Phase 1: New Module Skeleton

**Dependencies**: None
**Parallelizable**: No (foundational)

- [ ] 1.1 Create `packages/quantum-nematode/quantumnematode/evolution/` directory with `__init__.py` exporting public API
- [ ] 1.2 Create `genome.py`: `Genome` dataclass (`params: np.ndarray`, `genome_id: str`, `parent_ids: list[str]`, `generation: int`, `birth_metadata: dict[str, Any]`)
- [ ] 1.3 Add `genome_id_for(generation: int, index: int, parent_ids: list[str]) -> str` helper. **Implementation**: use `uuid.uuid5(uuid.NAMESPACE_OID, f"gen{generation}-idx{index}-parents{','.join(sorted(parent_ids))}")` so identical inputs produce identical IDs and any input change produces a different ID
- [ ] 1.4 Unit tests: `test_genome_id_deterministic` (same inputs → same UUID), `test_genome_id_distinct_for_distinct_inputs` (any input change → different UUID), `test_genome_id_format_is_uuid` (returned string parses as a valid UUID)

## Phase 2: Encoder Protocol, Brain Factory Wrapper, and Concrete Encoders

**Dependencies**: Phase 1
**Parallelizable**: No

- [ ] 2.1 Create `evolution/brain_factory.py` with `instantiate_brain_from_sim_config(sim_config: SimulationConfig) -> Brain`. Pulls `shots`, `qubits`, `device`, `learning_rate`, `gradient_method`, `gradient_max_norm`, `parameter_initializer_config` out of `sim_config` and dispatches to `quantumnematode.utils.brain_factory.setup_brain_model()`. Hardcodes `device=DeviceType.CPU` for evolution fitness eval. Default fallbacks for fields the YAML omits (e.g. `parameter_initializer_config=None`)
- [ ] 2.2 Unit test: `test_instantiate_mlpppo_from_sim_config` (load a fixture sim_config, call wrapper, assert `isinstance(brain, MLPPPOBrain)` and `brain._episode_count == 0`)
- [ ] 2.3 Unit test: `test_instantiate_lstmppo_from_sim_config` (same with LSTMPPO)
- [ ] 2.4 Create `encoders.py` with `GenomeEncoder` protocol whose methods take **the full `SimulationConfig`**: `initial_genome(sim_config: SimulationConfig, *, rng) -> Genome`, `decode(genome, sim_config) -> Brain`, `genome_dim(sim_config) -> int`. Plus module-level constant `NON_GENOME_COMPONENTS = {"optimizer", "actor_optimizer", "critic_optimizer", "training_state"}` (the denylist)
- [ ] 2.5 Implement private `_flatten_components(components: dict[str, WeightComponent]) -> tuple[np.ndarray, dict]` (returns flat array + shape map for unflatten — walks components in deterministic key-sorted order)
- [ ] 2.6 Implement private `_unflatten_components(params: np.ndarray, shape_map: dict) -> dict[str, WeightComponent]`
- [ ] 2.7 Implement `MLPPPOEncoder`: `brain_name = "mlpppo"`. `decode()` calls `instantiate_brain_from_sim_config(sim_config)` to get a fresh brain, then `load_weight_components()` to apply the genome. **Dynamic discovery**: filters `get_weight_components()` output by `NON_GENOME_COMPONENTS`. Picks up `policy`, `value`, and conditional `gate_weights`. After `load_weight_components`, sets `brain._episode_count = 0` AND calls `brain._update_learning_rate()` (so the LR matches the reset count — see Decision 2)
- [ ] 2.8 Implement `LSTMPPOEncoder`: `brain_name = "lstmppo"`. Same dynamic-discovery + denylist pattern. Picks up `lstm`, `layer_norm`, `policy`, `value`. Same `_episode_count = 0` + `_update_learning_rate()` after load. (Per-episode hidden state resets on `prepare_episode()` already.)
- [ ] 2.9 Define `ENCODER_REGISTRY: dict[str, type[GenomeEncoder]] = {"mlpppo": MLPPPOEncoder, "lstmppo": LSTMPPOEncoder}`
- [ ] 2.10 Unit test: `test_mlpppo_encoder_roundtrip` (encode → decode → identical first-step action on seeded input)
- [ ] 2.11 Unit test: `test_lstmppo_encoder_roundtrip` (same with recurrent brain — verifies all four components round-trip including `layer_norm`)
- [ ] 2.12 Unit test: `test_genome_dim_matches_flattened_state` (for both encoders)
- [ ] 2.13 Unit test: `test_episode_count_resets_and_lr_synced_on_decode` (verify both `_episode_count == 0` AND the LR matches what a freshly constructed brain has, for both brain types)
- [ ] 2.14 Unit test: `test_mlpppo_with_feature_gating_includes_gate_weights` (sanity: when `_feature_gating: true`, the genome dim is larger than without, and round-trip preserves gate behaviour)
- [ ] 2.15 Unit test: `test_encoder_registry_membership` (asserts `"mlpppo" in ENCODER_REGISTRY` and `"lstmppo" in ENCODER_REGISTRY` and that `ENCODER_REGISTRY[name]()` produces a working encoder for both)
- [ ] 2.16 Unit test: `test_encoder_excludes_denylist_components` (verifies `optimizer`, `actor_optimizer`, `critic_optimizer`, `training_state` are NEVER in the genome regardless of brain)

## Phase 3: Fitness Function

**Dependencies**: Phase 2
**Parallelizable**: Can start in parallel with Phase 4

- [ ] 3.1 Create `fitness.py` with `FitnessFunction` protocol (single `evaluate(genome: Genome, sim_config: SimulationConfig, encoder: GenomeEncoder, *, episodes: int, seed: int) -> float` method). Note: signature takes the full `SimulationConfig`, matching the encoder API
- [ ] 3.2 Implement `FrozenEvalRunner(StandardEpisodeRunner)` in `fitness.py`: subclass of [`StandardEpisodeRunner`](packages/quantum-nematode/quantumnematode/agent/runners.py#L599) that overrides `_terminate_episode` to always pass `learn=False, update_memory=False` to its parent `super()._terminate_episode(...)`, regardless of what the caller's kwargs say. This bypasses the standard runner's default `learn=True` on the success path ([runners.py:817-823](packages/quantum-nematode/quantumnematode/agent/runners.py#L817)). All step-loop logic is inherited unchanged
- [ ] 3.3 Implement private helper `_build_agent(brain: Brain, env: DynamicForagingEnvironment, sim_config: SimulationConfig) -> QuantumNematodeAgent` in `fitness.py`: passes `brain`, `env`, `satiety_config=sim_config.satiety`, and `sensing_config=sim_config.environment.sensing` (or `None` if absent). Other `QuantumNematodeAgent.__init__` args use defaults (theme/rich_style_config are presentation-only; agent_id defaults to `"default"`)
- [ ] 3.4 Implement `EpisodicSuccessRate.evaluate()`: decode genome via encoder → build env via `create_env_from_config(sim_config.environment)` → call `_build_agent(brain, env, sim_config)` → instantiate `FrozenEvalRunner` → loop `episodes` calls to `runner.run(agent, sim_config.reward, sim_config.max_steps)` → count successes via `result.termination_reason == TerminationReason.COMPLETED_ALL_FOOD` → return `successes / episodes`. Success detection uses `TerminationReason.COMPLETED_ALL_FOOD` (codebase convention, see [experiment/tracker.py:304](packages/quantum-nematode/quantumnematode/experiment/tracker.py#L304)) — **NOT** a non-existent `result.episode_success` attribute
- [ ] 3.5 **Frozen-weight contract**: with `FrozenEvalRunner`, `brain.learn()` and `brain.update_memory()` are never called regardless of episode outcome. `prepare_episode()` and `post_process_episode()` are still called by the inherited per-step logic — this advances `_episode_count` but does not change weights. (LearnedPerformanceFitness is M2.)
- [ ] 3.6 Unit test: `test_episodic_success_rate_returns_float_in_unit_interval` (fitness is a finite float in `[0.0, 1.0]` for an arbitrary genome — no assumption that random brains fail)
- [ ] 3.7 Unit test: `test_episodic_success_rate_deterministic_for_seeded_genome` (same genome + seed → same fitness across two invocations)
- [ ] 3.8 Unit test: `test_frozen_eval_runner_never_calls_learn` (mock `brain.learn` and `brain.update_memory`; run a successful episode via `FrozenEvalRunner.run()`; assert neither was called)
- [ ] 3.9 Unit test: `test_frozen_eval_runner_never_calls_learn_on_success` (specifically construct a scenario where the standard runner *would* default to `learn=True` — i.e. `TerminationReason.COMPLETED_ALL_FOOD` — and verify the override forces `learn=False`)
- [ ] 3.10 Unit test: `test_episodic_success_rate_uses_termination_reason_for_success` (mock the runner to return `EpisodeResult` with `COMPLETED_ALL_FOOD`; assert fitness counts it as success. Repeat with `MAX_STEPS`; assert it's not counted)

## Phase 4: Lineage Tracker

**Dependencies**: Phase 1
**Parallelizable**: Yes

- [ ] 4.1 Create `lineage.py` with `LineageTracker(output_path: Path)` class. The tracker is owned by the **parent process only** — workers report fitnesses back to the parent, and the parent calls `record()`. Workers MUST NOT instantiate or write to the tracker directly (no concurrent-write hazard)
- [ ] 4.2 `record(genome: Genome, fitness: float, brain_type: str)` appends a CSV row
- [ ] 4.3 CSV columns: `generation, child_id, parent_ids, fitness, brain_type` (parent_ids `;`-joined for CSV-safety)
- [ ] 4.4 Append mode (not write mode) so resume works without recreating. On `__init__`, check whether the file exists: if yes, skip writing the header; if no, write it
- [ ] 4.5 Unit test: `test_lineage_csv_appends_across_generations` (record gens `0..4` inclusive — i.e. 5 generations × population 4 = 20 data rows + 1 header row; parent_ids correctly populated for gen >= 1)
- [ ] 4.6 Unit test: `test_lineage_csv_gen_zero_has_empty_parent_ids` (gen 0 rows have empty `parent_ids` field)
- [ ] 4.7 Unit test: `test_lineage_csv_header_only_written_once` (instantiate tracker → record → close; reinstantiate same path → record → close; assert header appears exactly once)
- [ ] 4.8 Unit test: `test_lineage_generation_indexing_is_zero_based` (a run with `generations: G` produces rows whose `generation` column takes values in `[0, G-1]`, each appearing exactly P times)

## Phase 5: Evolution Loop

**Dependencies**: Phases 2, 3, 4
**Parallelizable**: No

- [ ] 5.1 Create `loop.py` with `EvolutionLoop` class taking `optimizer`, `encoder`, `fitness`, `sim_config`, `evolution_config`, `output_dir`, `rng`
- [ ] 5.2 Implement `run(*, resume_from: Path | None = None) -> EvolutionResult`
- [ ] 5.3 Generation loop: `optimizer.ask()` → wrap each candidate as a `Genome` with proper parent_ids → parallel fitness eval → `optimizer.tell()` → record lineage → checkpoint every N gens
- [ ] 5.4 Multiprocessing: reuse the worker pattern from legacy `run_evolution.py:452-470` (SIGINT-ignore, per-worker logger config)
- [ ] 5.5 Worker function takes picklable args (`params: np.ndarray`, `sim_config: SimulationConfig` Pydantic model, `episodes: int`, `seed: int`) and reconstructs the brain inside the worker via `encoder.decode(genome, sim_config)`. **Pass the Pydantic model directly, not a dict** — Pydantic v2 models pickle cleanly via `__getstate__`/`__setstate__`, so workers get a typed `SimulationConfig` without re-parsing. No separate `brain_config` arg — sim_config carries everything `instantiate_brain_from_sim_config` needs
- [ ] 5.6 Pickle checkpoint: dump `{optimizer, generation, rng_state, lineage_path, checkpoint_version: 1}` to `output_dir/checkpoint.pkl`
- [ ] 5.7 Resume: load pickle, validate `checkpoint_version`, restore optimizer state, continue from saved generation
- [ ] 5.8 On completion: write `best_params.json` (compatible with legacy artifact contract) and `history.csv` to `output_dir`
- [ ] 5.9 Unit test: `test_loop_runs_3_generations_mlpppo` (minimal config, 3 gens, pop 4, 1 episode each — completes and produces best_params.json)
- [ ] 5.10 Unit test: `test_loop_resume_from_checkpoint` (run 2 gens → kill → resume → run 3 more — total 5 gens in lineage CSV)
- [ ] 5.11 Unit test: `test_loop_rejects_incompatible_checkpoint_version`
- [ ] 5.12 Unit test: `test_checkpoint_contains_required_keys` (load a written checkpoint pickle and assert exact keys: `{"optimizer", "generation", "rng_state", "lineage_path", "checkpoint_version"}`)
- [ ] 5.13 Unit test: `test_unknown_brain_name_raises_with_helpful_message` (instantiate `EvolutionLoop` with a brain config whose `name` is not in `ENCODER_REGISTRY`; assert `ValueError` whose message lists the registered brain names and notes Phase 6 deferral)

## Phase 6: Configuration Schema Extension

**Dependencies**: None (can be done in parallel with Phases 1-5)
**Parallelizable**: Yes

- [ ] 6.1 In `packages/quantum-nematode/quantumnematode/utils/config_loader.py`, add `EvolutionConfig` Pydantic model
- [ ] 6.2 Fields: `algorithm: Literal["cmaes", "ga"] = "cmaes"`, `population_size: int = 20`, `generations: int = 50`, `episodes_per_eval: int = 15`, `sigma0: float = math.pi/2`, `elite_fraction: float = 0.2`, `mutation_rate: float = 0.1`, `crossover_rate: float = 0.8`, `parallel_workers: int = 1`, `checkpoint_every: int = 10`
- [ ] 6.3 Add `evolution: EvolutionConfig | None = None` field to `SimulationConfig`
- [ ] 6.4 Unit test: `test_existing_scenario_config_loads_without_evolution_block` (e.g. load `configs/scenarios/foraging/mlpppo_small_oracle.yml`; assert `SimulationConfig.evolution is None`)
- [ ] 6.5 Unit test: `test_evolution_block_parses_into_populated_config` (load a fixture with `evolution:` block; assert all fields populated; assert unspecified fields use defaults)

## Phase 7: New CLI Script (legacy deleted, not preserved)

**Dependencies**: Phases 5, 6
**Parallelizable**: No

- [ ] 7.1 **Delete** existing `scripts/run_evolution.py` (per Phase 5 decision: no `scripts/legacy/` fallback). Git history preserves it via `git log -- scripts/run_evolution.py`
- [ ] 7.2 **Delete** existing `configs/evolution/qvarcircuit_foraging_small.yml` (no consumer remains)
- [ ] 7.3 **Delete** existing `test_run_evolution_smoke` from `packages/quantum-nematode/tests/quantumnematode_tests/test_smoke.py` (CI smoke test for the deleted script)
- [ ] 7.4 In the same `test_smoke.py` edit, add `test_run_evolution_smoke_mlpppo`: `@pytest.mark.smoke`, runs `scripts/run_evolution.py --config configs/evolution/mlpppo_foraging_small.yml --generations 1 --population 4 --episodes 2 --seed 42 --output-dir <tmp>` via `subprocess.run`, asserts `returncode == 0` and no `Traceback` in stderr. Same pattern as the deleted test. (Grouping deletion + replacement in one task avoids a window of reduced smoke coverage between Phase 7 and Phase 9.) Note: this test exercises the new script which is created in tasks 7.5+, so verify it's wired up before running it locally
- [ ] 7.5 Create new `scripts/run_evolution.py` (~150 LOC): argparse → load config → instantiate encoder via registry → instantiate optimiser → instantiate `EvolutionLoop` → run
- [ ] 7.6 CLI flags: `--config`, `--generations`, `--population`, `--episodes`, `--algorithm`, `--sigma`, `--parallel`, `--seed`, `--resume`, `--output-dir`, `--log-level`
- [ ] 7.7 CLI flags default to `None` and only override YAML when explicitly passed (single-source-of-truth = `EvolutionConfig` defaults)
- [ ] 7.8 On startup, log prominently: `Brain type: <name>`, `Algorithm: <cmaes|ga>`, `Population: <N>`, `Generations: <M>`
- [ ] 7.9 Error message if `brain.name` is not in `ENCODER_REGISTRY`: `f"No encoder for brain '{name}'. Registered: {sorted(ENCODER_REGISTRY)}. Quantum brain support is deferred to a Phase 6 re-evaluation."`
- [ ] 7.10 Subprocess test: `test_cli_overrides_yaml_for_generations` — invoke the new script via `subprocess.run` with `--config <yaml-with-generations:50> --generations 2`, assert run completes with exactly 2 generations recorded in lineage CSV

## Phase 8: Pilot Configs

**Dependencies**: Phase 6
**Parallelizable**: Yes

- [ ] 8.1 Create `configs/evolution/mlpppo_foraging_small.yml`: `mlpppo` brain, 20×20 grid, oracle sensing, target_foods_to_collect ≈ 5, `evolution: {generations: 10, population_size: 8, episodes_per_eval: 3}`
- [ ] 8.2 Create `configs/evolution/lstmppo_foraging_small_klinotaxis.yml`: copy brain block from `configs/scenarios/foraging/lstmppo_small_klinotaxis.yml` (gru, lstm_hidden_dim=64, klinotaxis sensing, STAM enabled), same minimal `evolution:` block

## Phase 9: End-to-End Smoke Verification

**Dependencies**: Phases 7, 8
**Parallelizable**: No

- [ ] 9.1 Run: `uv run python scripts/run_evolution.py --config configs/evolution/mlpppo_foraging_small.yml` — completes without error
- [ ] 9.2 Verify `evolution_results/<session>/best_params.json` exists; load it and decode back into a working `MLPPPOBrain` (test or one-off script)
- [ ] 9.3 Verify `evolution_results/<session>/lineage.csv` has 80 data rows + 1 header row (10 generations × pop 8 = 80; generation column takes values 0..9 inclusive) with `parent_ids` empty for gen 0 and populated for gens 1..9
- [ ] 9.4 Run: `uv run python scripts/run_evolution.py --config configs/evolution/lstmppo_foraging_small_klinotaxis.yml` — completes without error
- [ ] 9.5 Same artifact verification for the LSTMPPO smoke
- [ ] 9.6 Resume test: run smoke #1, kill at gen ~5, resume with `--resume evolution_results/<session>/checkpoint.pkl`, verify completes 10 gens total
- [ ] 9.7 Run `uv run pytest -m smoke -v` — all green including the new MLPPPO smoke (`test_run_evolution_smoke_mlpppo` was added in task 7.4)

## Phase 10: M-1 Invariant — Cross-Phase Tracking Updates

**Dependencies**: Phases 1-9 complete
**Parallelizable**: No (final step before PR)

- [ ] 10.1 Update `openspec/changes/2026-04-26-phase5-tracking/tasks.md`: mark M0.1 → M0.14 complete (`[x]`); update M0 status header to `complete`
- [ ] 10.2 Update `docs/roadmap.md` Phase 5 Milestone Tracker table: change M0 row status from `🔲 not started` to `✅ complete` (with one-line summary)
- [ ] 10.3 Run `openspec validate --changes 2026-04-28-add-evolution-framework --strict` — passes
- [ ] 10.4 Run `uv run pre-commit run -a` — clean
- [ ] 10.5 Run `uv run pytest -m "not nightly"` — all green
- [ ] 10.6 Open PR with Conventional Commits prefix (per AGENTS.md): `feat: add brain-agnostic evolution framework (M0)`
